"""Training of one fold of the dense model, then its out-of-fold maps.

Usage: ``python -m solarfil.train --config configs/e1_r34.json --fold 0 [--out DIR]``

Outputs in ``<out>/<config name>/fold<k>/``: ``model.pt`` (EMA weights), ``log.jsonl``, and ``oof/<stem>.png``
for every held-out frame (an RGB PNG at the model scale: channel 0 the filament probability, 1 the spine, x 255).
With ``swa`` > 0 the uniform average of the weights over that last share of the steps is kept beside them, as
``model_swa.pt`` and ``oof_swa/``, a paired comparison within the same run.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from .annotators import GroupAnnotators
from .cache import DEFAULT_CACHE, load_disks, record_planes, soft_contact
from .checkpoints import save as save_checkpoint
from .data import Dataset, read_image
from .folds import load_folds
from .frames import CropSampler, downsample, load_frame, shift_rows
from .infer import predict_frame, write_map_png
from .model import EMA, WeightAverage, device, io_channels, model_from_cfg
from .preprocess import global_field, to_uint8

DEFAULTS = dict(
    name="e1_r34", arch="unet", encoder="resnet34", weights="imagenet", scheme="obs", scale=2, crop=512,
    batch=8, steps=5000, lr=5e-4, weight_decay=1e-4, warmup=250, ema=0.999, dice_w=0.5, spine_w=0.3,
    p_on_filament=0.5, seed=0, amp="bf16", grad_clip=1.0, oof_flips=4,
    prefetch_threads=1, grad_accum=1, ckpt_every=1000,
    coords=True,  # disk-coordinate input planes x/R, y/R (False: the flat field is the only input plane)
    input="flat",  # the flat field, or "raw": the frame over its on-disk median (an ablation of the flat field)
    affine=0.0, max_zoom=1.2,  # probability of a random rotation and zoom of a crop, and the largest zoom
    label_dy=0.0,  # px (at 2048): targets move down by this onto the image features, predicted maps back up by it
    dice="soft",  # the Dice term: "soft" 1 - 2<p,t>/(|p|+|t|), or "dml1" |p - t|/(|p|+|t|) (arXiv 2303.16296)
    swa=0.0,  # share of the last steps whose weights are also averaged uniformly (0: off); training is unchanged
    branch_at=0.0,  # share of the steps at which the checkpoint is also kept as ckpt_branch.pt (0: off): a trunk
    init_from="",  # a checkpoint under the runs folder to start from ("{dir}": the fold's folder), e.g. a branch
    affine_until=1.0,  # share of the steps after which crops are no longer rotated or zoomed
    inner_scale=1.0,  # the network sees its input enlarged by this inside itself (`model.InnerScale`); maps stay native
    hier_em=False,  # train on the latent mask under each annotation group's sensitivity and specificity (HierEM,
                    # `annotators.GroupAnnotators`); the saved network returns the population annotator's map
    em_warm=0.1,  # HierEM: share of the steps over which the soft-mean target's weight falls from 1 to 0
    em_every=250,  # HierEM: steps between refits of the groups' sensitivity and specificity
    em_pool=10.0,  # HierEM: records at which a group's own estimate weighs as much as the population's
    infer_tile=0,  # px at the model scale: maps from windows of this size (`infer.tiled_sigmoid`; 0: the whole frame)
    # Exploratory options (all off by default, leaving older recipes unchanged):
    aux_target="spine",  # the second target plane: "spine", or "contact" (`cache.soft_contact`; line 6)
    contact_px=3,  # contact borders: filament pixels within this many px (at 2048) of another filament of the record
    bce_edge_w=0.0,  # boundary-weighted BCE (SenNet 5th, 475288): weight 1 + this on the target's 2-3 px boundary
    copy_paste=0.0,  # probability of pasting one filament of another training frame into a crop (line 6)
    bdou_alpha_max=0.8,  # dice "bdou": Boundary DoU (arXiv 2308.00220) averaged over the records; alpha's cap
    ds_weights=[0.4, 0.2],  # arch "unet_ds": loss weights of the deep-supervision heads at 1/2 and 1/4 scale
    cutmix=0.0,  # probability that a crop gets a box from another crop of the batch (line 9)
    mixup=0.0,  # probability that a batch is mixed up with itself permuted, Beta(0.4, 0.4), until `mixup_until`
    mixup_until=0.25,  # share of the main (pre-finish) steps after which MixUp is off
    elastic=0.0, grid=0.0,  # probabilities of an elastic and a grid distortion of a crop (line 9)
    finish_steps=0,  # the last steps of `steps` train with the symmetric Lovasz hinge on the majority mask (line 9)
    finish_lr=0.1,  # their learning rate, as a share of `lr` (constant)
    pretrain_steps=0,  # line 5: steps of pretraining on a strong teacher's pseudo-labels (`pseudo.py`) before `steps`
    pool_frames=96,  # line 5: pseudo-labelled frames each sampling thread holds in memory (`pseudo.PoolSampler`)
)


def load_config(path: Path) -> dict:
    """A recipe: the defaults updated by a JSON file. Unknown keys are refused, so a stale recipe fails loudly
    instead of silently training something else."""
    extra = json.loads(Path(path).read_text())
    unknown = sorted(set(extra) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown recipe keys {unknown}")
    return dict(DEFAULTS, **extra)


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss 1 - (2 sum(prob target) + eps) / (sum prob + sum target + eps) over the whole batch."""
    inter = (prob * target).sum()
    return 1 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def dml1_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice semimetric loss DML1 (Wang et al., arXiv 2303.16296): the soft Dice loss on hard targets, but with its
    minimum at prob = target for soft ones, where the soft Dice loss pushes predictions towards 0 and 1. Our target
    is the mean of the annotators' masks, so it is soft wherever they disagree."""
    return (prob - target).abs().sum() / (prob.sum() + target.sum() + eps)


def bdou_loss(prob: torch.Tensor, recs: torch.Tensor, alpha_max: float = 0.8, eps: float = 1.0) -> torch.Tensor:
    """Boundary DoU loss (Sun et al., arXiv 2308.00220), per crop and record, averaged over the records present.

    `prob` (N, H, W); `recs` (N, K, H, W) each record's mask, -1 where the crop's frame has fewer than K records.
    L = (|G u P| - |G n P|) / (|G u P| - alpha |G n P|), alpha = min(1 - 2 C / S, alpha_max) with C the target's
    boundary pixels and S its area (alpha_max when the record draws nothing in the crop)."""
    valid = (recs[:, :, 0, 0] >= 0).float()
    t = recs.clamp(min=0)
    tb = (t > 0.5).float()
    boundary = tb + F.max_pool2d(-tb, 3, 1, 1)  # tb minus its erosion
    c, a = boundary.sum((2, 3)), tb.sum((2, 3))
    alpha = torch.where(a > 0, 1 - 2 * c / a.clamp(min=1), torch.ones_like(a)).clamp(0, alpha_max)
    p = prob[:, None]
    inter = (p * t).sum((2, 3))
    union = (p + t - p * t).sum((2, 3))
    loss = (union - inter + eps) / (union - alpha * inter + eps)
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss with respect to the sorted errors (Berman et al.)."""
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1 - inter / union
    jac[1:] = jac[1:] - jac[:-1].clone()
    return jac


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz hinge loss of binary `labels` (N, H, W), per image, averaged over the batch."""
    out = []
    for lg, lb in zip(logits.flatten(1), labels.flatten(1).float()):
        errors = 1 - lg * (2 * lb - 1)
        e, perm = torch.sort(errors, descending=True)
        out.append(torch.dot(F.relu(e), _lovasz_grad(lb[perm])))
    return torch.stack(out).mean()


def symmetric_lovasz(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The Lovasz hinge of the mask and of its complement, averaged (Severstal 9th, 114297)."""
    return 0.5 * (lovasz_hinge(logits, labels) + lovasz_hinge(-logits, 1 - labels.float()))


def edge_weights(target: torch.Tensor, w: float, px: int = 1) -> torch.Tensor:
    """BCE weights 1 + w on the boundary band of a soft target (N, H, W): where its local max and min over a
    (2 px + 1) window differ (SenNet 5th's x2 boundary weights at w = 1, 475288)."""
    t = target[:, None]
    k = 2 * px + 1
    band = (F.max_pool2d(t, k, 1, px) + F.max_pool2d(-t, k, 1, px)) > 0.01
    return 1 + w * band[:, 0].float()


def _cutmix_mixup(x: torch.Tensor, y: torch.Tensor, cfg: dict, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Line 9's batch mixing: MixUp of the whole batch (probability `mixup`, early steps only), else CutMix of each
    crop (probability `cutmix`), a box of area share 1 - lam, lam ~ U(0, 1), from a permuted crop."""
    n = x.shape[0]
    main = cfg["steps"] - cfg["finish_steps"]
    perm = torch.randperm(n, device=x.device)
    if cfg["mixup"] and step < main * cfg["mixup_until"] and torch.rand(()) < cfg["mixup"]:
        lam = float(np.random.default_rng(step).beta(0.4, 0.4))
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    if not cfg["cutmix"]:
        return x, y
    x, y = x.clone(), y.clone()
    h, w = x.shape[-2:]
    for i in range(n):
        if torch.rand(()) >= cfg["cutmix"]:
            continue
        r = math.sqrt(1 - float(torch.rand(())))
        bh, bw = int(h * r), int(w * r)
        y0, x0 = int(torch.randint(0, h - bh + 1, ())), int(torch.randint(0, w - bw + 1, ()))
        x[i, :, y0:y0 + bh, x0:x0 + bw] = x[perm[i], :, y0:y0 + bh, x0:x0 + bw]
        y[i, :, y0:y0 + bh, x0:x0 + bw] = y[perm[i], :, y0:y0 + bh, x0:x0 + bw]
    return x, y


def lr_at(step: int, cfg: dict) -> float:
    """Learning rate of a step: linear warm-up over `warmup` steps, then a cosine decay from `lr` to 1% of it at
    `steps` (at the end of the main steps, and `finish_lr` x `lr` over the `finish_steps` after them)."""
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    main = cfg["steps"] - cfg.get("finish_steps", 0)
    if step >= main:
        return cfg["lr"] * cfg["finish_lr"]
    t = (step - cfg["warmup"]) / max(1, main - cfg["warmup"])
    return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))


def _prefetch(sampler: CropSampler, n: int, steps: int, q: queue.Queue, affine_off_after: int | None = None) -> None:
    for i in range(steps):
        if i == affine_off_after:  # the recipe's last steps see plain crops
            sampler.affine = 0.0
        q.put(sampler.batch(n, frames=True))


def _samplers(frames, cfg: dict, fold: int) -> list:
    """One sampler per prefetch thread, each with its own seed, sharing the filament-pixel index."""
    aug = dict(p_on_filament=cfg["p_on_filament"], affine=cfg["affine"], max_zoom=cfg["max_zoom"])
    extra = dict(copy_paste=cfg["copy_paste"], elastic=cfg["elastic"], grid=cfg["grid"],
                 n_rec=3 if cfg["dice"] == "bdou" else 0)
    aug.update({k: v for k, v in extra.items() if v})  # off: the sampler's defaults
    first = CropSampler(frames, cfg["crop"], seed=cfg["seed"] * 100 + fold, **aug)
    return [first] + [CropSampler(frames, cfg["crop"], seed=cfg["seed"] * 100 + fold + 1000 * t, fil_px=first.fil_px,
                                  **aug)
                      for t in range(1, cfg["prefetch_threads"])]


def _set_input(fr, cfg: dict, ds: Dataset) -> None:
    """Apply the recipe's input ablations to a loaded frame."""
    fr.coords = cfg["coords"]
    if cfg["input"] == "raw":
        fr.flat = downsample(to_uint8(global_field(read_image(ds.image_path(fr.stem)), fr.disk)), cfg["scale"])


def fold_dir(fold: int) -> str:
    """Run subfolder of a fold model, or `full` for the model trained on all frames (fold < 0)."""
    return "full" if fold < 0 else f"fold{fold}"


@dataclass
class _Trainee:
    """What a training step updates: the network (`net` is the part that is trained: the latent network under
    HierEM, else the model itself), the optimiser, the EMA, the optional uniform weight average from step
    `swa_from` on, and the optional HierEM annotator model with each training frame's group and record count."""
    model: torch.nn.Module
    net: torch.nn.Module
    opt: torch.optim.Optimizer
    ema: EMA
    swa: WeightAverage | None
    swa_from: int | None
    annot: GroupAnnotators | None = None
    frame_group: torch.Tensor | None = None
    frame_records: torch.Tensor | None = None
    warm_steps: int | None = None

    def extras(self) -> dict:
        """What a checkpoint keeps beside the model, the optimiser and the EMA: the weight average and the annotator
        model, when the recipe uses them."""
        return ((dict(swa=self.swa.state_dict(), swa_n=self.swa.n) if self.swa is not None else {})
                | ({"annot": self.annot.state_dict()} if self.annot is not None else {}))

    def checkpoint(self, step: int) -> dict:
        """The state a run resumes or branches from, after `step` steps."""
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), ema=self.ema.state_dict(), step=step,
                    **self.extras())


def _prepare_out(cfg: dict, fold: int, out_root: Path) -> Path:
    """The run folder of a fold (with its `oof/` when the fold holds frames out) and the recipe saved in it."""
    out = out_root / cfg["name"] / fold_dir(fold)
    (out / "oof" if fold >= 0 else out).mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(cfg, indent=1))
    return out


def _load_train_frames(cfg: dict, tr: list, cache: Path, disks: dict, ds: Dataset) -> list:
    """The training frames at the model scale, with the recipe's input ablations and label shift applied."""
    frames = [load_frame(s, cfg["scale"], cache, disks, n_records=ds.observations[s].n_records) for s in tr]
    for fr in frames:
        obs = ds.observations[fr.stem]
        if cfg["aux_target"] == "contact":  # the second plane holds the records' contact borders
            fr.spine = downsample(soft_contact(obs, cfg["contact_px"]), cfg["scale"])
        if cfg["dice"] == "bdou":  # each record's own mask, for the loss averaged over the records
            fr.recs = np.stack([downsample(a, cfg["scale"]) for a in record_planes(obs)])
        _set_input(fr, cfg, ds)
        if cfg["label_dy"]:  # the annotators' masks sit about 0.24 px above the image (scripts/label_offset.py)
            fr.p, fr.spine = (shift_rows(a, cfg["label_dy"] / cfg["scale"]) for a in (fr.p, fr.spine))
            if fr.recs is not None:
                fr.recs = shift_rows(fr.recs, cfg["label_dy"] / cfg["scale"])
    return frames


def _annotator_model(cfg: dict, ds: Dataset, tr: list) -> dict:
    """HierEM: the annotator model of the training frames' groups, each frame's group and record count (for the
    posterior of the mask), and the steps over which the soft-mean target fades out."""
    obs = [ds.observations[s] for s in tr]
    groups = sorted({o.group for o in obs})
    frame_group = torch.tensor([groups.index(o.group) for o in obs])
    frame_records = torch.tensor([o.n_records for o in obs])
    annot = GroupAnnotators(groups, [sum(o.n_records for o in obs if o.group == g) for g in groups],
                            pool=cfg["em_pool"])
    warm_steps = max(1, int(cfg["steps"] * cfg["em_warm"]))
    return dict(annot=annot, frame_group=frame_group, frame_records=frame_records, warm_steps=warm_steps)


def _trainee(cfg: dict, ds: Dataset, tr: list, dev: torch.device) -> _Trainee:
    """A fresh network on `dev` with its optimiser, EMA, weight average and (HierEM) annotator model."""
    em = _annotator_model(cfg, ds, tr) if cfg["hier_em"] else {}
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model_from_cfg(cfg, cfg["weights"]).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    net = model.net if em else model  # HierEM trains the latent logits, without the population map
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(model, cfg["ema"])
    swa_from = int(cfg["steps"] * (1 - cfg["swa"])) if cfg["swa"] else None
    swa = WeightAverage() if swa_from is not None else None
    return _Trainee(model, net, opt, ema, swa, swa_from, **em)


def _load_weights(tn: _Trainee, st: dict, dev: torch.device) -> None:
    """Load a checkpoint's model, optimiser, EMA and (when kept) weight average."""
    tn.model.load_state_dict(st["model"])
    tn.opt.load_state_dict(st["opt"])
    tn.ema.shadow = {k: v.to(dev) for k, v in st["ema"].items()}
    if tn.swa is not None and st.get("swa"):
        tn.swa.shadow, tn.swa.n = {k: v.to(dev) for k, v in st["swa"].items()}, st["swa_n"]


def _restore(tn: _Trainee, cfg: dict, fold: int, out_root: Path, ckpt: Path, dev: torch.device) -> int:
    """Resume from the fold's checkpoint after an interruption, else start from the recipe's `init_from` branch,
    else from scratch; returns the first step to train."""
    if ckpt.exists():  # resume after an interruption
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        _load_weights(tn, st, dev)
        if tn.annot is not None:
            tn.annot.load_state_dict(st["annot"])
        start = st["step"]
        print(json.dumps(dict(event="resumed", step=start)), flush=True)
        return start
    if cfg["init_from"]:  # a variant that continues another run from its kept checkpoint
        src = out_root / cfg["init_from"].format(dir=fold_dir(fold))
        st = torch.load(src, map_location="cpu", weights_only=True)
        _load_weights(tn, st, dev)
        start = st["step"]
        print(json.dumps(dict(event="branched", source=str(src), step=start)), flush=True)
        return start
    return 0


def _start_prefetch(frames: list, cfg: dict, seed_key: int, start: int, samplers: list | None = None) -> queue.Queue:
    """Start the crop-sampling threads for the steps from `start` on; returns the queue they fill."""
    accum = cfg["grad_accum"]
    q: queue.Queue = queue.Queue(maxsize=4 + 2 * cfg["prefetch_threads"])
    per_thread = -(-(cfg["steps"] - start) * accum // cfg["prefetch_threads"])
    plain_from = int(cfg["steps"] * cfg["affine_until"])
    off_after = (max(0, -(-(plain_from - start) * accum // cfg["prefetch_threads"]))
                 if plain_from < cfg["steps"] else None)
    for smp_ in samplers or _samplers(frames, cfg, seed_key):
        threading.Thread(target=_prefetch, args=(smp_, cfg["batch"], per_thread, q, off_after), daemon=True).start()
    return q


def _micro_step(tn: _Trainee, batch: tuple, cfg: dict, step: int, dev: torch.device, amp) -> list:
    """Forward and backward of one micro-batch (its gradient scaled by 1 / `grad_accum`); returns its BCE, Dice and
    spine losses."""
    xb, yb, ib = batch
    x = torch.from_numpy(xb).to(dev, non_blocking=True)
    y = torch.from_numpy(yb).to(dev, non_blocking=True)
    if dev.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    if cfg["cutmix"] or cfg["mixup"]:
        x, y = _cutmix_mixup(x, y, cfg, step)
    with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
        out = tn.net(x)
    aux = []
    if isinstance(out, tuple):  # deep supervision: the heads at 1/2 and 1/4 scale
        out, *aux = out
    logits = out.float()
    target = y[:, 0]
    if cfg["finish_steps"] and step >= cfg["steps"] - cfg["finish_steps"]:  # the Lovasz finish
        loss = symmetric_lovasz(logits[:, 0], target >= 0.5)
        (loss / cfg["grad_accum"]).backward()
        return [0.0, loss.item(), 0.0]
    wts = edge_weights(target, cfg["bce_edge_w"]) if cfg["bce_edge_w"] else None
    if cfg["arch"] == "unet_sm3":  # background, body without contact, contact: soft CE + Dice on filament, contact
        c = y[:, 1]
        t3 = torch.stack([1 - target, (target - c).clamp(min=0), c], 1)
        t3 = t3 / t3.sum(1, keepdim=True).clamp(min=1e-6)
        ce = -(t3 * F.log_softmax(logits, 1)).sum(1)
        loss_p = (ce * wts).mean() if wts is not None else ce.mean()
        pr = logits.softmax(1)
        loss_d = soft_dice_loss(1 - pr[:, 0], target) + soft_dice_loss(pr[:, 2], c)
        loss = loss_p + cfg["dice_w"] * loss_d
        (loss / cfg["grad_accum"]).backward()
        return [loss_p.item(), loss_d.item(), 0.0]
    if tn.annot is not None:  # the posterior of the latent mask, mixed with the soft mean at first
        ib = torch.from_numpy(ib)
        post = tn.annot.posterior(logits[:, 0].detach(), target, tn.frame_records[ib].to(dev),
                                  tn.frame_group[ib].to(dev))
        w = max(0.0, 1 - step / tn.warm_steps)
        target = w * target + (1 - w) * post
    loss_p = F.binary_cross_entropy_with_logits(logits[:, 0], target, weight=wts)
    if cfg["dice"] == "bdou":
        loss_d = bdou_loss(logits[:, 0].sigmoid(), y[:, 2:], cfg["bdou_alpha_max"])
    else:
        loss_d = (dml1_loss if cfg["dice"] == "dml1" else soft_dice_loss)(logits[:, 0].sigmoid(), target)
    if cfg["classes"] > 1:
        loss_s = F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1], weight=wts)
        if cfg["aux_target"] == "contact":  # DSB 1st: BCE + soft Dice per channel
            loss_s = loss_s + cfg["dice_w"] * soft_dice_loss(logits[:, 1].sigmoid(), y[:, 1])
    else:
        loss_s = logits.new_zeros(())
    loss = loss_p + cfg["dice_w"] * loss_d + cfg["spine_w"] * loss_s
    for a, wt in zip(aux, cfg["ds_weights"]):  # deep supervision against the area-averaged target
        a = a.float()[:, 0]
        t = F.adaptive_avg_pool2d(target[:, None], a.shape[-2:])[:, 0]
        loss = loss + wt * (F.binary_cross_entropy_with_logits(a, t) + cfg["dice_w"] * soft_dice_loss(a.sigmoid(), t))
    (loss / cfg["grad_accum"]).backward()
    return [loss_p.item(), loss_d.item(), loss_s.item()]


def _after_step(tn: _Trainee, cfg: dict, step: int, out: Path, ckpt: Path, t_begin: float,
                stop_after: float | None) -> bool:
    """The updates that follow an optimiser step: EMA, annotator refit, weight average, the branch and periodic
    checkpoints. Returns True when the run pauses here (`stop_after` minutes have passed at a checkpoint)."""
    tn.ema.update(tn.model)
    if tn.annot is not None and (step + 1) % cfg["em_every"] == 0:
        tn.annot.refit()
    if tn.swa is not None and step >= tn.swa_from:
        tn.swa.update(tn.model)
    if cfg["branch_at"] and step + 1 == int(cfg["steps"] * cfg["branch_at"]):
        save_checkpoint(tn.checkpoint(step + 1), out / "ckpt_branch.pt")
    if cfg["ckpt_every"] and (step + 1) % cfg["ckpt_every"] == 0 and step + 1 < cfg["steps"]:
        save_checkpoint(tn.checkpoint(step + 1), ckpt)
        if stop_after is not None and (time.time() - t_begin) / 60 >= stop_after:
            print(json.dumps(dict(event="paused", step=step + 1, min=round((time.time() - t_begin) / 60, 1))),
                  flush=True)
            return True
    return False


def _log_record(tn: _Trainee, cfg: dict, step: int, start: int, run_loss: list, t0: float) -> dict:
    """The log line of the steps since the last one: mean losses, learning rate, speed and (HierEM) the population
    annotator's sensitivity and specificity."""
    m = np.mean(run_loss, axis=0)
    rec = dict(step=step + 1, bce=round(m[0], 5), dice=round(m[1], 4), spine=round(m[2], 5),
               lr=round(lr_at(step, cfg), 7), sec_per_step=round((time.time() - t0) / (step + 1 - start), 3))
    if tn.annot is not None:
        rec.update(zip(("alpha", "beta"), (round(v, 6) for v in tn.annot.population())))
    return rec


def _train(tn: _Trainee, q: queue.Queue, cfg: dict, start: int, dev: torch.device, amp, out: Path, ckpt: Path,
           t_begin: float, stop_after: float | None) -> float | None:
    """The training loop from step `start`, logging to `log.jsonl` every 50 steps; returns the time it began, or
    None when it paused at a checkpoint."""
    log = (out / "log.jsonl").open("a")
    t0, run_loss = time.time(), []
    tn.model.train()
    for step in range(start, cfg["steps"]):
        for g in tn.opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        tn.opt.zero_grad(set_to_none=True)
        for _ in range(cfg["grad_accum"]):
            run_loss.append(_micro_step(tn, q.get(), cfg, step, dev, amp))
        torch.nn.utils.clip_grad_norm_(tn.model.parameters(), cfg["grad_clip"])
        tn.opt.step()
        if _after_step(tn, cfg, step, out, ckpt, t_begin, stop_after):
            return None
        if (step + 1) % 50 == 0 or step == cfg["steps"] - 1:
            rec = _log_record(tn, cfg, step, start, run_loss, t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (step + 1) % 250 == 0:
                print(json.dumps(rec), flush=True)
            run_loss = []
    return t0


def _save_final(tn: _Trainee, out: Path, ckpt: Path) -> None:
    """Keep the EMA weights as `model.pt` (with the population annotator under HierEM), the weight average as
    `model_swa.pt`, and drop the resume checkpoint."""
    tn.model.load_state_dict(tn.ema.state_dict())
    if tn.annot is not None:  # the saved network returns the population annotator's map
        tn.model.annotator.copy_(torch.tensor(tn.annot.population()))
        (out / "annotators.json").write_text(json.dumps(tn.annot.summary(), indent=1))
    save_checkpoint({k: v.cpu() for k, v in tn.model.state_dict().items()}, out / "model.pt")
    if tn.swa is not None:
        save_checkpoint({k: v.cpu() for k, v in tn.swa.state_dict().items()}, out / "model_swa.pt")
    ckpt.unlink(missing_ok=True)


def _write_oof(tn: _Trainee, cfg: dict, out: Path, va: list, cache: Path, disks: dict, ds: Dataset,
               dev: torch.device, amp) -> None:
    """The out-of-fold maps of the held-out frames `va` in `oof/`, and of the weight average in `oof_swa/`."""
    for sub, state in [("oof", None)] + ([("oof_swa", tn.swa.state_dict())] if tn.swa is not None and va else []):
        if state is not None:
            tn.model.load_state_dict(state)
            if tn.annot is not None:
                tn.model.annotator.copy_(torch.tensor(tn.annot.population()))
            (out / sub).mkdir(exist_ok=True)
        for s in va:
            fr = load_frame(s, cfg["scale"], cache, disks, targets=False)
            _set_input(fr, cfg, ds)
            pm = predict_frame([tn.model], fr, dev, n_flips=cfg["oof_flips"], amp_dtype=amp,
                               label_dys=[cfg["label_dy"] / cfg["scale"]], tiles=[cfg["infer_tile"]])
            write_map_png(out / sub / f"{s}.png", pm)


def _pretrain(cfg: dict, fold: int, out: Path, pseudo: dict, ds: Dataset, tr: list, dev: torch.device, amp,
              t_begin: float, stop_after: float | None) -> bool:
    """Line 5's first phase: `pretrain_steps` steps of the recipe on the teacher's pseudo-labelled frames alone
    (no weight average, no finish), in `<out>/pretrain/`; its EMA weights become `<out>/pretrain.pt`, the start of
    the training on the records. Returns False when it paused at a checkpoint."""
    from .pseudo import PoolSampler, PseudoLoader, load_disks_json

    pcfg = dict(cfg, steps=cfg["pretrain_steps"], swa=0.0, finish_steps=0, branch_at=0.0, hier_em=False,
                cutmix=0.0, mixup=0.0, init_from="")
    pout = out / "pretrain"
    pout.mkdir(exist_ok=True)
    loader = PseudoLoader(pseudo["flat_dir"], pseudo["map_dir"], load_disks_json(pseudo["disks"]), cfg)
    stems = sorted(pseudo["stems"])
    aug = dict(p_on_filament=cfg["p_on_filament"], affine=cfg["affine"], max_zoom=cfg["max_zoom"])
    seed_key = fold if fold >= 0 else 99
    smps = [PoolSampler(loader, stems, cfg["crop"], seed=cfg["seed"] * 100 + seed_key + 1000 * t + 7,
                        pool=cfg["pool_frames"], **aug) for t in range(cfg["prefetch_threads"])]
    print(json.dumps(dict(event="pretrain_loaded", frames=len(stems), pool=cfg["pool_frames"])), flush=True)
    tn = _trainee(pcfg, ds, tr, dev)
    ckpt = pout / "ckpt.pt"
    start = _restore(tn, pcfg, fold, out.parent.parent, ckpt, dev)
    q = _start_prefetch([], pcfg, seed_key, start, samplers=smps)
    if _train(tn, q, pcfg, start, dev, amp, pout, ckpt, t_begin, stop_after) is None:
        return False
    save_checkpoint({k: v.cpu() for k, v in tn.ema.state_dict().items()}, out / "pretrain.pt")
    ckpt.unlink(missing_ok=True)
    return True


def run(cfg: dict, fold: int, out_root: Path, cache: Path = DEFAULT_CACHE, max_train: int | None = None,
        stop_after: float | None = None, pseudo: dict | None = None) -> Path | None:
    """Train the model of one fold (its out-of-fold maps go to `oof/`), or on all frames when fold < 0.

    `stop_after` (minutes): stop at the first checkpoint saved after that long and return None; a later call
    resumes from the checkpoint (for jobs with a time cap). `pseudo` (with `pretrain_steps`): the pseudo-labelled
    frames of line 5 (flat_dir, map_dir, disks, stems), pretrained on first."""
    t_begin = time.time()
    cfg = dict(cfg, **io_channels(cfg))
    out = _prepare_out(cfg, fold, out_root)
    seed_key = fold if fold >= 0 else 99  # seeds must be non-negative
    torch.manual_seed(cfg["seed"] * 100 + seed_key)
    dev = device()
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["amp"]) if dev.type == "cuda" else None

    ds = Dataset()
    folds = load_folds(cfg["scheme"])
    disks = load_disks(cache)
    tr = [s for s, k in folds.items() if k != fold][:max_train]
    va = [s for s, k in folds.items() if k == fold]
    t0 = time.time()
    frames = _load_train_frames(cfg, tr, cache, disks, ds)
    print(json.dumps(dict(event="loaded", n_train=len(frames), n_val=len(va), sec=round(time.time() - t0, 1),
                          device=str(dev))), flush=True)
    ckpt = out / "ckpt.pt"
    if cfg["pretrain_steps"] and not ckpt.exists() and not (out / "pretrain.pt").exists():
        if not _pretrain(cfg, fold, out, pseudo, ds, tr, dev, amp, t_begin, stop_after):
            return None
    tn = _trainee(cfg, ds, tr, dev)
    start = _restore(tn, cfg, fold, out_root, ckpt, dev)
    if start == 0 and cfg["pretrain_steps"]:  # the records' training starts from the pretrained EMA weights
        tn.model.load_state_dict(torch.load(out / "pretrain.pt", map_location="cpu", weights_only=True))
        tn.ema = EMA(tn.model, cfg["ema"])
        print(json.dumps(dict(event="from_pretrain")), flush=True)
    q = _start_prefetch(frames, cfg, seed_key, start)
    t0 = _train(tn, q, cfg, start, dev, amp, out, ckpt, t_begin, stop_after)
    if t0 is None:
        return None
    _save_final(tn, out, ckpt)

    tn.model.eval()
    t1 = time.time()
    _write_oof(tn, cfg, out, va, cache, disks, ds, dev, amp)
    print(json.dumps(dict(event="done", train_min=round((t1 - t0) / 60, 1),
                          oof_sec_per_frame=round((time.time() - t1) / max(1, len(va) * (1 + (tn.swa is not None))),
                                                  2))), flush=True)
    return out


def main():
    """Command line: train one fold (or every frame with ``--fold all``) of the recipe in ``--config``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", required=True, help="fold number, or 'all' to train on every frame")
    ap.add_argument("--out", default=str(paths.RUNS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--max-train", type=int, default=None, help="limit training frames (smoke tests)")
    ap.add_argument("--stop-after", type=float, default=None,
                    help="minutes: stop at the first checkpoint after this long; running again resumes")
    ap.add_argument("--ckpt-every", type=int, default=None, help="override (smoke tests of pause and resume)")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    if a.steps:
        cfg["finish_steps"] = round(cfg["finish_steps"] * a.steps / cfg["steps"])  # a smoke keeps both phases
        cfg["steps"] = a.steps
        cfg["name"] += "_smoke"
    if a.ckpt_every:
        cfg["ckpt_every"] = a.ckpt_every
    run(cfg, -1 if a.fold == "all" else int(a.fold), Path(a.out), Path(a.cache), a.max_train, a.stop_after)


if __name__ == "__main__":
    main()

"""Train the instance model (Mask R-CNN, `solarfil.instances`) on the frames outside one fold and write its detections
for the fold's held-out frames.

Each sample is a frame's flat field at 1024 px with the instances of one of its annotator records, drawn at random,
under a random flip or transpose. AdamW with warm-up and cosine decay; checkpoints at epoch ends, so a job with
`--stop-after` minutes resumes where the last one stopped, as the dense training does.

Usage: train_instances.py --fold 0|all [--epochs 24] [--batch 4] [--stop-after 150] [--max-train 4 --epochs 1 (smoke)]
Outputs in <runs>/inst_maskrcnn/fold<k>/ (or full/ for every frame): model.pt, log.jsonl, oof/<stem>.npz (scores
>= 0.05, maskness and bit-packed masks). With `--scheme group` the folds hold out whole annotation groups and the
run is `inst_maskrcnn_grp`: its out-of-fold detections come from detectors that never saw the frame's annotators.
`--labels union|consensus` trains on agreement levels (runs `inst_maskrcnn_union`, `_consensus`), and `--anchor 16`
halves the anchor boxes for the smallest filaments (run `inst_maskrcnn_a16`).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from solarfil import paths
from solarfil.cache import DEFAULT_CACHE, load_png
from solarfil.data import Dataset
from solarfil.folds import load_folds
from solarfil.instances import build, half, save_instances, targets, to_input
from solarfil.model import device

NAME = "inst_maskrcnn"


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """One of the eight flips and transposes of a square array (k = 0: unchanged)."""
    a = a.T if k >= 4 else a
    a = a[:, ::-1] if k % 4 in (1, 3) else a
    a = a[::-1] if k % 4 in (2, 3) else a
    return np.ascontiguousarray(a)


def clusters(obs) -> list[list[tuple[int, int]]]:
    """The frame's filaments grouped across records: two filaments of different records with IoU above 0.5 (at
    1024 px) are the same filament. Each cluster lists its members as (record, filament)."""
    members = [(r, f) for r, rec in enumerate(obs.records) for f in range(len(rec.filaments))]
    masks = [half(obs.records[r].filaments[f].mask()) for r, f in members]
    boxes = [(np.nonzero(m.any(1))[0][[0, -1]], np.nonzero(m.any(0))[0][[0, -1]]) if m.any() else None for m in masks]
    parent = list(range(len(members)))

    def root(i):
        """The root of member i's cluster (union-find with path halving)."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if members[i][0] == members[j][0] or boxes[i] is None or boxes[j] is None:
                continue
            (ya, xa), (yb, xb) = boxes[i], boxes[j]
            if ya[1] < yb[0] or yb[1] < ya[0] or xa[1] < xb[0] or xb[1] < xa[0]:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter and inter / np.logical_or(masks[i], masks[j]).sum() > 0.5:
                parent[root(i)] = root(j)
    groups: dict[int, list] = {}
    for i, m in enumerate(members):
        groups.setdefault(root(i), []).append(m)
    return list(groups.values())


class Samples(torch.utils.data.Dataset):
    """`labels`: "record" shows one random annotator record's filaments (the score then estimates how often an
    annotator draws a filament); "union" one instance per filament any annotator drew, and "consensus" one per
    filament two or more drew (multi-record frames only): detectors of agreement levels (VinBigData 4th, 229786)."""

    def __init__(self, stems: list[str], cache: Path, labels: str = "record"):
        """Samples of `stems` from the flat-field cache; "consensus" keeps only the frames with two or more records."""
        self.ds = Dataset()
        if labels == "consensus":
            stems = [s for s in stems if self.ds.observations[s].n_records > 1]
        self.stems, self.cache, self.labels = stems, Path(cache), labels
        self._clusters: dict = {}

    def __len__(self) -> int:
        """The number of frames."""
        return len(self.stems)

    def __getitem__(self, i: int):
        """Frame i's flat field at 1024 px and its instance targets under one of the eight flips and transposes, the
        labels drawn afresh on each call."""
        stem = self.stems[i]  # the loader seeds `random` in each worker from torch's generator
        img = half(load_png(self.cache / "flat" / f"{stem}.png"))
        obs = self.ds.observations[stem]
        if self.labels == "record":
            chosen = random.choice(obs.records).filaments
        else:
            if stem not in self._clusters:
                self._clusters[stem] = clusters(obs)
            need = 2 if self.labels == "consensus" else 1
            chosen = [obs.records[r].filaments[f] for r, f in
                      (random.choice(c) for c in self._clusters[stem] if len({r for r, _ in c}) >= need)]
        k = random.randrange(8)
        return to_input(dihedral(img, k)), targets([dihedral(half(f.mask()), k) for f in chosen])


def collate(batch):
    """A batch as a tuple of images and a tuple of targets, the lists the detector takes."""
    return tuple(zip(*batch))


def lr_at(step: int, total: int, lr: float, warmup: int = 200) -> float:
    """Learning rate at `step`: linear warm-up over `warmup` steps, then cosine decay from `lr` to 1% of it at
    `total`."""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    """Train (or resume) one fold's instance model and write its detections on the fold's held-out frames."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="fold number, or 'all' to train on every frame")
    ap.add_argument("--out", default=str(paths.RUNS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-after", type=float, default=None, help="minutes; stop at the next epoch end after that")
    ap.add_argument("--max-train", type=int, default=None, help="first N training frames only (smoke tests)")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (smoke tests on a Mac)")
    ap.add_argument("--scheme", default="obs", choices=["obs", "group"])
    ap.add_argument("--labels", default="record", choices=["record", "union", "consensus"])
    ap.add_argument("--anchor", type=int, default=32, help="smallest anchor side at 1024 px (run name _a<side>)")
    a = ap.parse_args()
    t_begin = time.time()
    a.fold = -1 if a.fold == "all" else int(a.fold)
    name = (NAME + ("" if a.labels == "record" else f"_{a.labels}") + ("" if a.anchor == 32 else f"_a{a.anchor}")
            + ("_grp" if a.scheme == "group" else "") + ("" if a.seed == 0 else f"_s{a.seed}"))  # _s<k>: another draw
    out = Path(a.out) / name / ("full" if a.fold < 0 else f"fold{a.fold}")
    (out / "oof").mkdir(parents=True, exist_ok=True)
    folds = load_folds(a.scheme)
    tr = [s for s, k in folds.items() if k != a.fold][:a.max_train]
    va = [s for s, k in folds.items() if k == a.fold][:a.max_val]
    (out / "cfg.json").write_text(json.dumps(dict(vars(a), n_train=len(tr), n_val=len(va)), indent=1))
    torch.manual_seed(a.seed * 100 + (a.fold if a.fold >= 0 else 99))  # as the dense training seeds its folds
    dev = torch.device("cpu") if a.cpu else device()
    amp = torch.bfloat16 if dev.type == "cuda" else None
    model = build(anchor=a.anchor).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    samples = Samples(tr, Path(a.cache), a.labels)
    per_epoch = math.ceil(len(samples) / a.batch)
    total = a.epochs * per_epoch
    start_epoch = 0
    ckpt = out / "ckpt.pt"
    trained = (out / "model.pt").exists()
    if trained and all((out / "oof" / f"{s}.npz").exists() for s in va):
        print(json.dumps(dict(event="fold done")), flush=True)
        return
    if trained:  # trained, but its detections are missing: predict only
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
        print(json.dumps(dict(event="predicting only")), flush=True)
    if ckpt.exists() and not trained:
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"]
        torch.set_rng_state(st["rng"])
        print(json.dumps(dict(event="resumed", epoch=start_epoch)), flush=True)
    loader = torch.utils.data.DataLoader(samples, batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, collate_fn=collate, drop_last=False,
                                         persistent_workers=a.workers > 0)
    log = (out / "log.jsonl").open("a")
    model.train()
    for epoch in range(start_epoch, 0 if trained else a.epochs):
        t0, run = time.time(), []
        for i, (imgs, tgts) in enumerate(loader):
            step = epoch * per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr)
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run.append([float(v) for v in losses.values()])
        rec = dict(epoch=epoch + 1, **{k: round(float(v), 4) for k, v in zip(losses, np.mean(run, 0))},
                   lr=round(lr_at(step, total, a.lr), 7), sec=round(time.time() - t0, 1))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if epoch + 1 < a.epochs:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch + 1,
                            rng=torch.get_rng_state()), ckpt)
            if a.stop_after is not None and (time.time() - t_begin) / 60 >= a.stop_after:
                print(json.dumps(dict(event="paused", epoch=epoch + 1)), flush=True)
                return
    if not trained:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    ckpt.unlink(missing_ok=True)

    model.eval()
    t1, kept = time.time(), []
    with torch.no_grad():
        for s in va:
            x = to_input(half(load_png(Path(a.cache) / "flat" / f"{s}.png"))).to(dev)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                det = model([x])[0]
            prob = det["masks"][:, 0].float()
            on = prob > 0.5
            maskness = (prob * on).sum((1, 2)) / on.sum((1, 2)).clamp(min=1)
            save_instances(out / "oof" / f"{s}.npz", det["scores"].float().cpu().numpy(), on.cpu().numpy(),
                           maskness.cpu().numpy())
            kept.append(len(on))
    print(json.dumps(dict(event="done", train_min=round((t1 - t_begin) / 60, 1), val=len(va),
                          detections_per_frame=round(float(np.mean(kept)) if kept else 0.0, 1),
                          sec_per_frame=round((time.time() - t1) / max(1, len(va)), 2))), flush=True)


if __name__ == "__main__":
    main()

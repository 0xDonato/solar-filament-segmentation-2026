"""End-to-end inference: from a test JPEG to its submission rows, with the validated settings.

A weights folder holds `manifest.json`, the dense models (`dense_*.pt`, float32 state dicts) and the node scorer
(`scorer.npz`, the fitted trees as plain arrays); nothing in it is a pickle::

    {"scale": 1, "flips": 4, "tau": 0.26, "splits": true, "scorer": "scorer.npz", "alpha": 0.5,
     "dense": [{"arch": "unet", "encoder": "tu-tf_efficientnetv2_s", "file": "dense_full.pt", "in_channels": 1,
                "classes": 1}]}

A dense entry's `weight`, when present, is its network's share of the averaged map (equal shares without it).

`alpha`, when present, redraws each selected filament at its own level after the selection (`tree.relevel`).
`Pipeline.outline`, when set (an option; our final selection sets t_low 0.3), grows each selected filament by the thin
protrusions of its map after the selection (`morph.grow_thin`, with its keyword arguments, such as the map level
`t_low`; `{}` takes its defaults).
`instances`, when present, names the instance model's state dict (`instances.pt`, Mask R-CNN, `solarfil.instances`),
or a list of them (the fold models): their detections give every candidate six more columns, and the scorer was
fitted with them. With several, each one's columns are scored on their own and the utilities averaged, so every
prediction sees columns from a detector trained as the scorer's training columns were (on four folds of frames).
An entry of the list may itself be a list, one model of each of several instance models whose columns the scorer
read side by side (`[["instances_a_fold0.pt", "instances_b_fold0.pt"], ...]`): each such set is scored on its
own, and the sets' utilities averaged. `detections`, when present
(with `instances`), admits the detections as candidates beside the tree's nodes (`solarfil.detcands`): its `scorer`
file rates them, `margin`, `min_score`, `overlap`, `gated` and `candidate_min_score` set the exact selection over
both kinds, and the chosen masks are painted from the highest utility down.

The flat field and final dense map use 8-bit values. Live ensembles average floating model outputs before
rounding. Historical validation blends average already rounded member maps, so their quantisation path differs.
Backend and rounding effects require a matched comparison.
`Pipeline.run` overlaps the CPU stages (flat field; tree, scorer and selection) of other frames with the dense
maps on the GPU, in worker processes, and returns the same instances as `Pipeline.instances` frame by frame.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import detcands, preprocess
from . import instances as detector
from .bank import describe
from .data import read_image
from .frames import Frame, downsample
from .infer import predict_frame, to_native, to_u8
from .model import build_model, device, inference_dtype
from .morph import grow_thin
from .parallel import pool
from .postprocess import labels_to_rles
from .scorer import FrozenHurdle
from .tree import build_tree, relevel, select

STAGES = ("flat", "dense", "detect", "tree", "score", "select")
_WORKER: dict = {}


def load_state(path: Path) -> dict:
    """A state dict from a `.pt` file or a `.safetensors` file; a safetensors file over GitHub's 100 MB limit is
    stored as shards `<name>-00001-of-0000N.safetensors` beside it (`scripts/export_compact.py`), read back as one."""
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=True)
    from safetensors.torch import load_file
    files = [path] if path.exists() else sorted(path.parent.glob(f"{path.stem}-*-of-*.safetensors"))
    assert files, f"no weights at {path}"
    state = {}
    for f in files:
        state.update(load_file(f, device="cpu"))
    return state


def flat_stage(img: np.ndarray) -> tuple[np.ndarray, preprocess.Disk, float]:
    """The 8-bit flat field of a frame, its disk, and the seconds it took."""
    t0 = time.perf_counter()
    flat, disk = preprocess.flat_field(img)
    return preprocess.to_uint8(flat), disk, time.perf_counter() - t0


def decision_stage(flat_u8: np.ndarray, p_u8: np.ndarray, disk: preprocess.Disk, scale: int, scorer, tau: float,
                   splits: bool, alpha: float | None = None, dets: tuple | None = None,
                   det: tuple | None = None, outline: dict | None = None) -> tuple[list[str], dict]:
    """Submission RLE strings of the filaments selected from one frame's 8-bit filament map (model scale), and the
    seconds of each step. `dets`: the instance model's detections (scores, packed masks, maskness), if any, or a
    list of sets of them (each set's columns side by side, the sets' utilities averaged); `det`: (detection scorer,
    its settings) to admit them as candidates; `outline`: `morph.grow_thin`'s keyword arguments, to grow the selected
    filaments by thin protrusions."""
    t0 = time.perf_counter()
    flat = preprocess.from_uint8(flat_u8)
    p = to_native(p_u8.astype(np.float32) / 255.0, scale)
    tree = build_tree(p, splits=splits)
    t1 = time.perf_counter()
    if tree.n == 0:
        return [], dict(tree=t1 - t0, score=0.0, select=0.0)
    X = describe(tree, p, flat, disk)
    masks = None
    if dets is None:
        u = scorer.predict(X)[0]
    else:
        us = []
        for group in (dets if isinstance(dets, list) else [dets]):
            cols = []
            for scores, packed, maskness in (group if isinstance(group, list) else [group]):
                masks = detector.unpack(packed)
                cols.append(detector.describe(tree, scores, masks, maskness))
            us.append(scorer.predict(np.concatenate([X, *cols], 1))[0])
        u = np.mean(us, 0)
    t2 = time.perf_counter()
    if det is not None and masks is not None:
        assert outline is None, "the outline step goes with the tree's selection only"
        assert not isinstance(dets, list) or (len(dets) == 1 and len(cols) == 1), \
            "detection candidates take one instance model"
        det_scorer, cfg = det
        c = detcands.candidates(tree, p, flat, disk, scores, masks, maskness, cfg["candidate_min_score"],
                                cfg["gated"])
        u_d = det_scorer.predict(c["X"])[0] if len(c["X"]) else np.zeros(0)
        nodes, chosen = detcands.select_mixed(tree, u, c, u_d, tau, cfg["overlap"], cfg["margin"], cfg["min_score"])
        shown = relevel(tree, p, nodes, alpha) if alpha else nodes
        y0, _, x0, _ = tree.box
        sets = []
        for v in shown:
            r, cc = np.divmod(tree.node_pixels(v), tree.crop_shape[1])
            sets.append(((y0 + r) * tree.shape[1] + x0 + cc).astype(np.int64))
        sets += [c["pix"][j] for j in chosen]
        lab = detcands.paint(sets, [u[v] for v in nodes] + [u_d[j] for j in chosen], tree.shape)
        return labels_to_rles(lab), dict(tree=t1 - t0, score=t2 - t1, select=time.perf_counter() - t2)
    nodes = select(tree, u, tau)
    if alpha:
        nodes = relevel(tree, p, nodes, alpha)
    lab = tree.labels(nodes)
    if outline is not None:
        lab = grow_thin(lab, p, **outline)
    rles = labels_to_rles(lab)
    return rles, dict(tree=t1 - t0, score=t2 - t1, select=time.perf_counter() - t2)


def _worker_init(scorer: FrozenHurdle, scale: int, tau: float, splits: bool, alpha: float | None,
                 det: tuple | None = None, outline: dict | None = None) -> None:
    _WORKER.update(scorer=scorer, scale=scale, tau=tau, splits=splits, alpha=alpha, det=det, outline=outline)


def _flat_job(path: str):
    return flat_stage(read_image(path))


def _decision_job(flat_u8, p_u8, disk, dets=None):
    w = _WORKER
    return decision_stage(flat_u8, p_u8, disk, w["scale"], w["scorer"], w["tau"], w["splits"], w["alpha"], dets,
                          w["det"], w["outline"])


@dataclass
class Pipeline:
    """The released inference pipeline: the dense networks and their shares, the frozen scorer and tau, the
    detector sets, and the settings `load` reads from a weights folder's manifest."""
    models: list
    scorer: FrozenHurdle
    tau: float
    scale: int = 1
    flips: int = 4
    splits: bool = True
    coords: bool = True  # the networks read the coordinate planes
    label_dys: list | None = None  # each network's recipe label_dy at the model scale (`infer.predict_frame`)
    tiles: list | None = None  # each network's recipe infer_tile (`infer.tiled_sigmoid`), when any is set
    alpha: float | None = None  # each selected filament redrawn at alpha times its own peak (`tree.relevel`)
    dev: torch.device = field(default_factory=device)
    timings: dict = field(default_factory=lambda: {k: 0.0 for k in STAGES})
    detector: list | None = None  # the instance models in sets, whose detections the scorer reads
    det: tuple | None = None  # (detection scorer, settings) when the detections are candidates too
    outline: dict | None = None  # `morph.grow_thin`'s arguments: the selected filaments grown by thin protrusions
    weights: list | None = None  # each network's share of the dense average, when the shares differ

    @classmethod
    def load(cls, weights: Path | str, dev: torch.device | None = None) -> Pipeline:
        """The pipeline of a weights folder from its `manifest.json`: the dense networks, the frozen scorer and tau, the
        inference settings, and the instance models and detection scorer when the manifest lists them."""
        weights = Path(weights)
        man = json.loads((weights / "manifest.json").read_text())
        dev = dev or device()
        models = []
        for d in man["dense"]:
            m = build_model(d["arch"], d["encoder"], weights=None, in_channels=d.get("in_channels", 3),
                            classes=d.get("classes", 2), inner_scale=d.get("inner_scale", 1.0),
                            population=d.get("hier_em", False))
            m.load_state_dict(load_state(weights / d["file"]))
            if dev.type == "cpu":  # 1.6x faster on the CPU, with the same 8-bit maps
                m = m.to(memory_format=torch.channels_last)
            models.append(m.to(dev).eval())
        scorer = FrozenHurdle.load(weights / man["scorer"])
        coords = all(d.get("in_channels", 3) > 1 for d in man["dense"])
        scale = man.get("scale", 1)
        label_dys = [d.get("label_dy", 0.0) / scale for d in man["dense"]]
        tiles = [d.get("infer_tile", 0) for d in man["dense"]]
        pipe = cls(models, scorer, man["tau"], scale, man.get("flips", 4), man.get("splits", True), coords,
                   label_dys if any(label_dys) else None, tiles if any(tiles) else None, man.get("alpha"), dev)
        shares = [d.get("weight", 1.0) for d in man["dense"]]
        pipe.weights = shares if len(set(shares)) > 1 else None
        if man.get("instances"):
            sets = man["instances"] if isinstance(man["instances"], list) else [man["instances"]]
            pipe.detector = []
            for files in sets:
                group = []
                for f in files if isinstance(files, list) else [files]:
                    det = detector.build(coco=False, anchor=man.get("instance_anchor", 32))
                    det.load_state_dict(load_state(weights / f))
                    group.append(det.to(dev).eval())
                pipe.detector.append(group)
            if man.get("detections"):
                cfg = dict(man["detections"])
                pipe.det = (FrozenHurdle.load(weights / cfg.pop("scorer")), cfg)
        return pipe

    def dense(self, flat_u8: np.ndarray, disk: preprocess.Disk, stem: str) -> np.ndarray:
        """The 8-bit filament map at the model scale (the spine map of a network that has one is not used)."""
        t0 = time.perf_counter()
        fr = Frame(stem, disk, self.scale, downsample(flat_u8, self.scale), coords=self.coords)
        pm = predict_frame(self.models, fr, self.dev, n_flips=self.flips, amp_dtype=inference_dtype(self.dev),
                           label_dys=self.label_dys, tiles=self.tiles, weights=self.weights)
        self.timings["dense"] += time.perf_counter() - t0
        return to_u8(pm[0])

    def detect(self, flat_u8: np.ndarray) -> list | None:
        """Each instance model's detections of a frame, set by set, or None without one."""
        if self.detector is None:
            return None
        t0 = time.perf_counter()
        dets = [[detector.detect(m, flat_u8, self.dev, inference_dtype(self.dev)) for m in group]
                for group in self.detector]
        self.timings["detect"] += time.perf_counter() - t0
        return dets

    def instances(self, img: np.ndarray, stem: str) -> list[str]:
        """Submission RLE strings of the filaments selected in one frame."""
        flat_u8, disk, sec = flat_stage(img)
        self.timings["flat"] += sec
        p_u8 = self.dense(flat_u8, disk, stem)
        rles, secs = decision_stage(flat_u8, p_u8, disk, self.scale, self.scorer, self.tau, self.splits, self.alpha,
                                    self.detect(flat_u8), self.det, self.outline)
        for k, v in secs.items():
            self.timings[k] += v
        return rles

    def run(self, paths: list[Path], workers: int = 4, ahead: int = 4) -> dict[str, list[str]]:
        """Submission RLE strings of every image file. With `workers` > 0, worker processes compute the flat fields
        (`ahead` frames in advance) and the decision layer while this process computes the dense maps in order;
        `timings` then add up each stage's own time, which overlaps with the others."""
        paths = [Path(p) for p in paths]
        if workers <= 0:
            return {p.stem: self.instances(read_image(p), p.stem) for p in paths}
        init = (self.scorer, self.scale, self.tau, self.splits, self.alpha, self.det, self.outline)
        with pool(workers, spawn=True, init=_worker_init, initargs=init) as ex:
            flats = [ex.apply_async(_flat_job, (str(p),)) for p in paths[:ahead]]
            decisions = []
            for i, p in enumerate(paths):
                if i + ahead < len(paths):
                    flats.append(ex.apply_async(_flat_job, (str(paths[i + ahead]),)))
                flat_u8, disk, sec = flats[i].get()
                flats[i] = None
                self.timings["flat"] += sec
                p_u8 = self.dense(flat_u8, disk, p.stem)
                dets = self.detect(flat_u8)
                decisions.append((p.stem, ex.apply_async(_decision_job, (flat_u8, p_u8, disk, dets))))
            out = {}
            for stem, job in decisions:
                out[stem], secs = job.get()
                for k, v in secs.items():
                    self.timings[k] += v
        return out


def export_weights(run_dir: Path, scorer_npz: Path, out: Path, tau: float, models: str = "folds",
                   members: list[Path] | None = None, member_weights: list[float] | None = None,
                   alpha: float | None = None,
                   instances: Path | list | None = None, detections: dict | None = None) -> dict:
    """Collect a run's fold models (or, with models="full", its model trained on all frames) and a fitted scorer
    (a `FrozenHurdle` file) into a weights folder for `Pipeline.load`. `members`: runs whose models are averaged
    instead (seeds of one recipe), with the scorer fitted on their blended out-of-fold maps; `member_weights`, one per
    member, their shares of the average as the blend's (`blend_maps.py --weights`). `instances`: the folder
    of the instance model (its `model.pt`), or a list of folders (the fold models, averaged at inference), for a
    scorer fitted with its columns; an entry may be a list of folders, one fold model of each instance model whose
    columns the scorer read side by side, in that order.
    `detections`: the detection candidates' settings with `scorer`, the fitted detection scorer's file."""
    out.mkdir(parents=True, exist_ok=True)
    sources = [(rd, fd) for rd in members or [run_dir]
               for fd in ([rd / "full"] if models == "full" else sorted(rd.glob("fold*")))]
    shares = dict(zip(members or [run_dir], member_weights)) if member_weights else {}
    dense = []
    for rd, fd in sources:
        cfg = json.loads((fd / "cfg.json").read_text())
        name = f"dense_{rd.name}_{fd.name}.pt"
        shutil.copyfile(fd / "model.pt", out / name)
        dense.append(dict(arch=cfg["arch"], encoder=cfg["encoder"], file=name, in_channels=cfg.get("in_channels", 3),
                          classes=cfg.get("classes", 2))
                     | ({"label_dy": cfg["label_dy"]} if cfg.get("label_dy") else {})
                     | ({"inner_scale": cfg["inner_scale"]} if cfg.get("inner_scale", 1.0) != 1.0 else {})
                     | ({"infer_tile": cfg["infer_tile"]} if cfg.get("infer_tile") else {})
                     | ({"hier_em": True} if cfg.get("hier_em") else {})
                     | ({"weight": shares[rd]} if shares else {}))
        scale = cfg["scale"]
    shutil.copyfile(scorer_npz, out / "scorer.npz")
    inst_files, anchor = None, 32
    if instances is not None:
        sets = instances if isinstance(instances, list) else [instances]
        paired = any(isinstance(s, (list, tuple)) for s in sets)
        anchors = set()
        inst_files = []
        for s in sets:
            files = []
            for d in s if isinstance(s, (list, tuple)) else [s]:
                d = Path(d)
                # the instance model's name too when sets pair several, whose fold folders share names
                f = f"instances_{d.parent.name}_{d.name}.pt" if paired else f"instances_{d.name}.pt"
                shutil.copyfile(d / "model.pt", out / f)
                files.append(f)
                cfg_file = d / "cfg.json"
                anchors.add(json.loads(cfg_file.read_text()).get("anchor", 32) if cfg_file.exists() else 32)
            inst_files.append(files if isinstance(s, (list, tuple)) else files[0])
        assert len(anchors) == 1, f"instance models with different anchors: {anchors}"
        anchor = anchors.pop()
        inst_files = inst_files if isinstance(instances, list) else inst_files[0]
    if detections is not None:
        shutil.copyfile(detections["scorer"], out / "det_scorer.npz")
        detections = dict(detections, scorer="det_scorer.npz")
    man = dict(scale=scale, flips=4, tau=tau, splits=True, dense=dense, scorer="scorer.npz") | (
        {"alpha": alpha} if alpha else {}) | ({"instances": inst_files} if instances is not None else {}) | (
        {"instance_anchor": anchor} if anchor != 32 else {}) | (
        {"detections": detections} if detections is not None else {})
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man

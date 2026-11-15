"""End-to-end pipeline on synthetic frames with a tiny random model: loads a weights folder (with and without the
coordinate planes and the spine map, with the alpha rule, and with an untrained instance model whose detections
the scorer reads, alone, as fold models, or paired with a second one), returns valid, pixel-disjoint, one-piece,
hole-free instances, and the overlapped run returns exactly what the frame-by-frame path does; the export names
paired instance models apart."""
import json

import cv2
import numpy as np
import pytest
import torch

from solarfil import detcands, rle
from solarfil import instances as detector
from solarfil.bank import FEATURES
from solarfil.data import read_image
from solarfil.model import build_model
from solarfil.pipeline import Pipeline, export_weights
from solarfil.scorer import FrozenHurdle, Hurdle
from test_preprocess import synthetic_frame


def _pipeline(folder, lean: bool = False, alpha: float | None = None, inst: bool = False,
              dets: bool = False, pair: bool = False) -> Pipeline:
    """`lean`: one input plane (no coordinate planes) and one output map (no spine map), like the final recipe.
    `alpha`: the manifest's level rule (`tree.relevel`). `inst`: an untrained instance model in the folder.
    `dets`: its detections are candidates too, rated by a random detection scorer. `inst=2`: two fold models whose
    utilities are averaged. `pair`: each set holds two instance models whose columns the scorer reads side by
    side."""
    torch.manual_seed(0)
    io = dict(in_channels=1, classes=1) if lean else dict(in_channels=3, classes=2)
    m = build_model("unet", "resnet18", weights=None, **io)
    # An untrained network puts p near 0.5 everywhere, and the tree of such a map (frame-sized components) is slow.
    # Scaled up and shifted down, its map is sparse like a trained network's; the coordinate planes (when present)
    # are read but do not move it.
    with torch.no_grad():
        m.encoder.conv1.weight[:, 1:] = 0
        m.segmentation_head[0].weight *= 20
        m.segmentation_head[0].bias.fill_(-6.0)
    torch.save(m.state_dict(), folder / "dense_0.pt")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES) + (len(detector.COLUMNS) * (2 if pair else 1) if inst else 0)))
    h = rng.random(400)
    scorer = Hurdle(dict(max_iter=20)).fit(X, h, 0.5 + 0.5 * rng.random(400), np.ones(400))
    FrozenHurdle.from_fitted(scorer).save(folder / "scorer.npz")
    inst_files = None
    if inst:
        inst_files = [[f"instances_a_{k}.pt", f"instances_b_{k}.pt"] if pair else f"instances_{k}.pt"
                      for k in range(int(inst))]
        for f in (f for s in inst_files for f in (s if pair else [s])):
            torch.save(detector.build(coco=False).state_dict(), folder / f)
        inst_files = inst_files if int(inst) > 1 or pair else inst_files[0]
    if dets:
        Xd = rng.normal(size=(400, len(detcands.COLUMNS)))
        hd = Hurdle(dict(max_iter=20)).fit(Xd, rng.random(400), 0.5 + 0.5 * rng.random(400), np.ones(400))
        FrozenHurdle.from_fitted(hd).save(folder / "det_scorer.npz")
    man = dict(scale=4, flips=2, tau=0.0, splits=True, scorer="scorer.npz",
               dense=[dict(arch="unet", encoder="resnet18", file="dense_0.pt", **io)]) | (
                   {"alpha": alpha} if alpha else {}) | ({"instances": inst_files} if inst else {}) | (
                   {"detections": dict(scorer="det_scorer.npz", margin=0.0, min_score=0.0, overlap=0.2, gated=True,
                                       candidate_min_score=0.5)} if dets else {})
    (folder / "manifest.json").write_text(json.dumps(man))
    return Pipeline.load(folder, dev=torch.device("cpu"))


@pytest.mark.parametrize("lean, alpha, inst, dets, pair", [
    (False, None, False, False, False), (True, None, False, False, False), (True, 0.5, False, False, False),
    (True, 0.5, True, False, False), (True, 0.5, True, True, False), (True, 0.5, 2, False, False),
    (True, 0.5, 1, False, True)])
def test_pipeline_runs_end_to_end(tmp_path, lean, alpha, inst, dets, pair):
    pipe = _pipeline(tmp_path, lean, alpha, inst, dets, pair)
    assert pipe.coords is not lean and pipe.alpha == alpha and len(pipe.detector or []) == int(inst)
    assert all(len(group) == (2 if pair else 1) for group in pipe.detector or [])
    assert (pipe.det is not None) == dets
    img, _ = synthetic_frame()
    rles = pipe.instances(img, "20140609195854Bh")
    assert rles
    acc = np.zeros((2048, 2048), np.uint8)
    for c in rles:
        m = rle.decode(c).astype(np.uint8)
        acc += m
        # one 8-connected piece without holes, as the annotation protocol draws a filament
        assert cv2.connectedComponents(m, connectivity=8)[0] == 2
        assert cv2.connectedComponents(1 - np.pad(m, 1), connectivity=4)[0] == 2
    assert acc.max() <= 1
    assert all(v >= 0 for v in pipe.timings.values())


@pytest.mark.parametrize("alpha, inst, dets, pair", [(None, False, False, False), (0.5, False, False, False),
                                                    (0.5, True, False, False), (0.5, True, True, False),
                                                    (0.5, 2, False, False), (0.5, 1, False, True)])
def test_overlapped_run_matches_frame_by_frame(tmp_path, alpha, inst, dets, pair):
    pipe = _pipeline(tmp_path, alpha=alpha, inst=inst, dets=dets, pair=pair)
    paths = []
    for seed, stem in enumerate(("20140609195854Bh", "20140610195854Bh")):
        img, _ = synthetic_frame(seed=seed)
        paths.append(tmp_path / f"{stem}.jpeg")
        cv2.imwrite(str(paths[-1]), img)
    one_by_one = {p.stem: pipe.instances(read_image(p), p.stem) for p in paths}
    assert any(one_by_one.values())
    assert pipe.run(paths, workers=2, ahead=1) == one_by_one


def test_export_names_paired_instance_models_apart(tmp_path):
    dense = tmp_path / "runs" / "dense_run" / "fold0"
    dense.mkdir(parents=True)
    (dense / "model.pt").write_bytes(b"d")
    (dense / "cfg.json").write_text(json.dumps(dict(arch="unet", encoder="resnet18", scale=1, in_channels=1,
                                                    classes=1)))
    folders = {}
    for name in ("inst_a", "inst_b"):
        for k in range(2):
            d = tmp_path / "runs" / name / f"fold{k}"
            d.mkdir(parents=True)
            (d / "model.pt").write_bytes(f"{name}{k}".encode())
            (d / "cfg.json").write_text(json.dumps(dict(anchor=32)))
            folders[name, k] = d
    (tmp_path / "scorer.npz").write_bytes(b"s")
    man = export_weights(dense.parent, tmp_path / "scorer.npz", tmp_path / "w1", 0.26,
                         instances=[[folders["inst_a", k], folders["inst_b", k]] for k in range(2)])
    assert man["instances"] == [[f"instances_inst_a_fold{k}.pt", f"instances_inst_b_fold{k}.pt"] for k in range(2)]
    assert (tmp_path / "w1" / "instances_inst_b_fold1.pt").read_bytes() == b"inst_b1"
    man = export_weights(dense.parent, tmp_path / "scorer.npz", tmp_path / "w2", 0.26,
                         instances=[folders["inst_a", k] for k in range(2)])
    assert man["instances"] == ["instances_fold0.pt", "instances_fold1.pt"]  # the released folders' names


def test_outline_step_grows_each_mask_and_keeps_it_whole_and_disjoint_in_both_paths(tmp_path):
    pipe = _pipeline(tmp_path, alpha=None, inst=True)
    img, _ = synthetic_frame()
    plain = pipe.instances(img, "20140609195854Bh")
    pipe.outline = {}  # morph.grow_thin's defaults (t_low 0.1), the optional step's default level
    grown = pipe.instances(img, "20140609195854Bh")
    assert len(grown) == len(plain) > 0
    acc = np.zeros((2048, 2048), np.uint8)
    for a, b in zip(plain, grown):
        ma, mb = rle.decode(a).astype(bool), rle.decode(b).astype(np.uint8)
        assert (mb[ma] == 1).all()  # each filament keeps its pixels
        assert cv2.connectedComponents(mb, connectivity=8)[0] == 2
        assert cv2.connectedComponents(1 - np.pad(mb, 1), connectivity=4)[0] == 2
        acc += mb
    assert acc.max() <= 1
    paths = []
    for seed, stem in enumerate(("20140609195854Bh", "20140610195854Bh")):
        frame, _ = synthetic_frame(seed=seed)
        paths.append(tmp_path / f"{stem}.jpeg")
        cv2.imwrite(str(paths[-1]), frame)
    assert pipe.run(paths, workers=2, ahead=1) == {p.stem: pipe.instances(read_image(p), p.stem) for p in paths}

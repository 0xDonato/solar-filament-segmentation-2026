"""Explain one compact CPU prediction with real candidate/selection values.

The example is the reference image with most selected masks, breaking ties by
image identifier. Inference uses one JPEG. An optional complete CPU cache avoids
neural replay; native output must reproduce the scored reference in either case.
This illustrates decisions without supplying test labels or physical validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solarfil import explain, instances
from solarfil.bank import describe
from solarfil.data import read_image
from solarfil.morphology import crop_box, disk_stretch
from solarfil.pipeline import Pipeline, flat_stage
from solarfil.postprocess import labels_to_rles
from solarfil.preprocess import Disk, from_uint8
from solarfil.scorer import FrozenHurdle
from solarfil.tree import build_tree, select


def cached_frame(path):
    """The optional native arrays produced by the completed CPU trace."""
    with np.load(path, allow_pickle=False) as z:
        flat, probability, geometry, groups = (z[k] for k in ("flat", "probability", "disk", "groups"))
        if (flat.shape != (2048, 2048) or probability.shape != flat.shape
                or flat.dtype != np.uint8 or probability.dtype != np.uint8
                or geometry.shape != (3,) or not np.isfinite(geometry).all()
                or not np.array_equal(groups, [2])):
            raise ValueError("Expected the compact native CPU input schema")
        detections = [[tuple(z[f"0_{j}_{key}"] for key in ("scores", "masks", "maskness")) for j in range(2)]]
    return flat, probability, geometry, detections


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights"))
    ap.add_argument("--reference", type=Path, default=Path("release/submission_cpu_tau0.30.csv"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="Optional completed compact CPU input cache")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Use a fresh explanatory output")
    rows = defaultdict(list)
    for row in csv.DictReader(args.reference.open(newline="")):
        rows[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    stem = min(rows, key=lambda s: (-len(rows[s]), s))
    image = read_image(args.data / "test/test_images" / f"{stem}.jpeg")
    torch.set_num_threads(4)
    manifest = json.loads((args.weights / "manifest.json").read_text())
    if (manifest["scale"] != 1 or manifest["flips"] != 4 or len(manifest["dense"]) != 2
            or manifest.get("alpha") or manifest.get("detections")
            or [len(group) for group in manifest["instances"]] != [2]):
        raise ValueError("This explanation requires the compact four-network tree pipeline")
    if args.cache:
        receipt = json.loads((args.cache / "result.json").read_text())
        if not receipt["complete_cpu_reference_exact"] or not receipt["cached_replay_exact"]:
            raise ValueError("Require the complete checked CPU cache")
        flat, probability, geometry, detections = cached_frame(args.cache / f"frames/{stem}.npz")
        disk = Disk(*geometry)
    else:
        pipe = Pipeline.load(args.weights, dev=torch.device("cpu"))
        flat, disk, _ = flat_stage(image)
        probability = pipe.dense(flat, disk, stem)
        detections = pipe.detect(flat)
    probability = probability.astype(np.float32) / 255.
    tree = build_tree(probability, splits=manifest["splits"])
    base = describe(tree, probability, from_uint8(flat), disk)
    columns = [instances.describe(tree, scores, instances.unpack(packed), maskness)
               for scores, packed, maskness in detections[0]]
    design = np.concatenate([base, *columns], axis=1)
    utility, presence, overlap = FrozenHurdle.load(args.weights / manifest["scorer"]).predict(design)
    tau = .30
    selected = select(tree, utility, tau)
    painted = tree.labels(selected)
    if Counter(labels_to_rles(painted)) != Counter(rows[stem]):
        raise ValueError("The explained native masks differ from the scored CPU reference")
    check = explain.check(tree, utility, tau, selected)
    if not check["optimal"]:
        raise ValueError("Independent integer-program/enumeration checks disagree")
    _, milp_nodes = explain.optimum_milp(tree, utility, tau)
    table, children = explain.dp_table(tree, utility, tau), tree.children()

    def parts(v):
        chosen, stack = [], list(children[v])
        while stack:
            node = stack.pop()
            if table["best"][node] <= 0:
                continue
            if table["take"][node]:
                chosen.append(node)
            else:
                stack.extend(children[node])
        np.testing.assert_allclose(sum(utility[n] - tau for n in chosen), table["kids"][v], atol=1e-9, rtol=0)
        return sorted(chosen)

    whole = max(selected, key=lambda n: (len(children[n]) >= 2, table["kids"][n], tree.area[n]))
    split = [v for v in range(tree.n) if len(children[v]) >= 2 and not table["take"][v]
             and not set(explain.ancestors(tree, v)) & set(selected) and len(parts(v)) >= 2]
    other = min(split, key=lambda v: (tree.area[v], v)) if split else min(range(tree.n), key=lambda v: utility[v])
    cases = [("Keep whole candidate", whole), ("Keep parts" if split else "Reject candidate", other)]
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=(1, 1.3))
    lo, hi = disk_stretch(image, disk.cx, disk.cy, disk.r)
    top = [(image, "Original JPEG", "gray", lo, hi), (flat, "Flattened image", "gray", 0, 255),
           (probability, "Mean filament map", "magma", 0, 1),
           (image, f"{len(selected)} selected masks", "gray", lo, hi)]
    for column, (value, title, cmap, vmin, vmax) in enumerate(top):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
        if column == 3:
            for node in selected:
                ax.contour(tree.node_mask(node), [.5], colors=["#ff9c20"], linewidths=.7)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    detailed = []
    for column, (title, node) in enumerate(cases):
        ax = fig.add_subplot(grid[1, 2 * column:2 * column + 2])
        mask = tree.node_mask(node)
        x0, y0, x1, y1 = crop_box(mask, mask.shape, min_side=120, margin=1.4)
        crop = np.s_[y0:y1, x0:x1]
        ax.imshow(image[crop], cmap="gray", vmin=lo, vmax=hi)
        ax.contour(mask[crop], [.5], colors=["#ff9c20"], linewidths=1.1)
        alternatives = parts(node)
        for part in alternatives:
            ax.contour(tree.node_mask(part)[crop], [.5], colors=["#00caff"], linewidths=1)
        ax.set_title(f"{title}: node {node}\nu={utility[node]:.3f}, u−0.30={table['gain'][node]:+.3f}; "
                     f"best parts={table['kids'][node]:+.3f}", fontsize=11)
        ax.set_axis_off()
        detailed.append(dict(decision=title, node=int(node), chosen=node in selected,
                             utility=float(utility[node]), presence=float(presence[node]),
                             conditional_iou=float(overlap[node]), gain=float(table["gain"][node]),
                             best_parts=float(table["kids"][node]), alternative_nodes=alternatives,
                             alternatives=[dict(node=int(v), utility=float(utility[v]),
                                                gain=float(table["gain"][v])) for v in alternatives],
                             area=int(tree.area[node]), map_level=float(tree.levels[tree.level[node]])))
    fig.suptitle(f"Compact CPU selection on one JPEG: {stem}\nOrange: whole candidate; cyan: best alternative parts. "
                 "Test labels are unavailable.", fontsize=12)
    args.out.mkdir(parents=True)
    fig.savefig(args.out / "selection_example.png", dpi=160)
    plt.close(fig)
    result = dict(stem=stem, tau=tau, candidates=tree.n, selected=len(selected), columns=design.shape[1],
                  scored_CPU_reference_RLE_sets_exact=True, additive_optimum_check=check, cases=detailed,
                  selected_nodes=[int(v) for v in selected], milp_optimal_nodes=milp_nodes,
                  selection_rule="reference image with most masks;lexical image tie-break",
                  source="compact four-network CPU reference", test_labels_read=0,
                  new_fits=0, new_neural_forwards=0 if args.cache else None,
                  new_neural_images=int(args.cache is None),
                  scope=__doc__.strip())
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

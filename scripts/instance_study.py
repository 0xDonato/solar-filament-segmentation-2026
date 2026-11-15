"""Instance descriptors in the scorer: does an independent instance model's evidence raise the decision layer?

`--build` gives every node of the averaged map's component tree six columns from the out-of-fold detections of the
instance model (`train_instances.py`, Mask R-CNN; `solarfil.instances.describe`): the best IoU with a detection,
its score, the best score x IoU, the best score-weighted share of the node a detection covers, the number of
detections overlapping it, and the best score x maskness x IoU. The evaluation is `agree_study.evaluate`: the
hurdle scorer cross-fitted with and without the columns on the same partitions (random ones, and partitions of the
annotation groups), and a paired bootstrap over frames: the release rule of `solarfil.rule`. Version 4 of the public
notebook tushhaaarrrr/solar-filament-segmentation-2026-0-40 measured +0.0034 from RF-DETR's evidence.

Usage: instance_study.py --run ens_lean_2seeds --inst inst_maskrcnn [--build] [--workers 3] [--partial]
       --inst inst_maskrcnn,inst_maskrcnn_union,inst_maskrcnn_consensus  (several detectors: their columns side by
       side, each detector's columns built first with --build on its own)
       --base inst_maskrcnn  (the base arm has that detector's columns too: --inst inst_maskrcnn_a16 tests a
       replacement of the released columns, --inst inst_maskrcnn,inst_maskrcnn_a16 an addition to them)
       --partitions 5 --group-partitions 5  (the release rule's partitions since 10-18: five random and five of the
       annotation groups; studies before then read one group partition, the fixed group folds)
(--partial: only the frames whose fold already has detections, a noisy first look)
       --tag vs_set1  (suffixes the combined column folder, the scores files and the study JSON, so that a repeat of
       a study already run writes only new files: <dets>_split_<tag>/, scores_<dets>_split_<tag>_<scheme>.npz and
       <dets>_study[_vs_<base>]_<tag>.json; it refuses to run when any of them exists, and does not combine with
       --build, whose per-detector folders it does not rename)
"""
import argparse
import json

import numpy as np

from agree_study import evaluate
from solarfil.infer import read_map_png
from solarfil.instances import COLUMNS, describe, load_instances
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.tree import build_tree


def one(args):
    """Write one frame's instance columns (`describe`) for every node of its split tree; returns the node count."""
    blend, inst, out = args
    p, _ = read_map_png(blend)
    tree = build_tree(p, splits=True)
    np.savez_compressed(out, A=describe(tree, *load_instances(inst)))
    return tree.n


def main():
    """Build the detectors' columns if asked, combine several side by side, and run the paired study on them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inst", default="inst_maskrcnn")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--group-partitions", type=int, default=1,
                    help="partitions of the annotation groups; the release rule reads 5 since 10-18 (rule.py)")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--base", default=None, help="a detector whose columns the base arm has as well")
    ap.add_argument("--tag", default="", help="suffix of the combined column folder, scores files and study JSON")
    a = ap.parse_args()
    if a.tag and a.build:
        ap.error("--tag does not rename the --build folders: build first, then run the study with --tag")
    run = RUNS / a.run
    insts = a.inst.split(",")
    suffix = f"_{a.tag}" if a.tag else ""
    scores_stem = "+".join(insts) + "_split" + suffix
    study_json = run / (f"{'+'.join(insts)}_study{f'_vs_{a.base}' if a.base else ''}{'_partial' if a.partial else ''}"
                        f"{suffix}.json")
    if a.tag:
        new = [run / f"scores_{scores_stem}_{s}.npz" for s in ("random", "group")] + [study_json]
        if len(insts) > 1:
            new.append(run / scores_stem)
        clash = [str(q) for q in new if q.exists()]
        assert not clash, f"--tag {a.tag}: {clash} exist already"
    for name in insts if a.build else []:
        extra = f"{name}_split"
        (run / extra).mkdir(exist_ok=True)
        inst = {q.stem: q for q in (RUNS / name).glob("fold*/oof/*.npz")}
        maps = sorted(run.glob("fold*/oof/*.png"))
        missing = [q.stem for q in maps if q.stem not in inst]
        assert a.partial or not missing, f"{name}: {len(missing)} frames without detections, e.g. {missing[:3]}"
        maps = [q for q in maps if q.stem in inst]
        with pool(a.workers) as workers:
            sizes = workers.map(one, [(str(q), str(inst[q.stem]), str(run / extra / f"{q.stem}.npz")) for q in maps],
                                chunksize=2)
        print(f"{name}: {len(maps)} frames, {sum(sizes)} nodes")
    extra = f"{insts[0]}_split"
    if len(insts) > 1:  # the detectors' columns side by side, in a folder of their own
        extra = "+".join(insts) + "_split" + suffix
        (run / extra).mkdir(exist_ok=True)
        common = set.intersection(*({q.stem for q in (run / f"{n}_split").glob("*.npz")} for n in insts))
        for stem in common:
            A = np.concatenate([np.load(run / f"{n}_split" / f"{stem}.npz")["A"] for n in insts], 1)
            np.savez_compressed(run / extra / f"{stem}.npz", A=A)
    columns = [f"{n}:{c}" for n in insts for c in COLUMNS]
    stems = {q.stem for q in (run / extra).glob("*.npz")} if a.partial else None
    result = evaluate(a, run, extra, columns, stems, base_dir=f"{a.base}_split" if a.base else None,
                      scores_stem=scores_stem)
    print(json.dumps(result, indent=1))
    study_json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

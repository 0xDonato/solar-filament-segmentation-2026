"""Agreement descriptors: does the scorer gain from how far the ensemble's members agree on each candidate?

The bank describes every node of the averaged map's component tree. Two members that agree on a filament give
the same averaged map as two members of which only one draws it at twice the strength, and the DSB 2018 4th place
scored its instances with features of the agreement between its models' masks (write-up 55118). For every node
`--build` adds six columns from the members' own maps: the mean absolute difference of their maps over the node,
the weaker member's mean and the gap between the members' means there, and, at the node's own threshold, each
member's component that overlaps the node most: the smaller and the larger of the two IoUs with the node, and the
smaller share of the node that a member covers.

`--eval` cross-fits the hurdle scorer with and without the six columns on the same random partitions of frames (and
on the annotation groups), selects at each tau, and compares the pooled PQ with a paired bootstrap over frames.

Usage: agree_study.py --run ens_lean_2seeds --members e3_effv2s_n_long_lean e3_effv2s_n_long_lean_s1 [--build]
       [--tag r34] (columns in agree_r34_split/: e.g. --members ens_lean_2seeds e1_r34, another model's evidence)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from decide import crossfit, sweep
from solarfil.bank import load_banks
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.postprocess import fill_holes
from solarfil.rule import TAUS, partitions, release_rule
from solarfil.tree import build_tree

COLUMNS = ["a_dmean", "a_pmin", "a_pgap", "a_iou_min", "a_iou_max", "a_cover_min"]


def agreement(tree, members: list[np.ndarray]) -> np.ndarray:
    """(n, 6) agreement columns of every node of `tree` (built on the members' mean map)."""
    n = tree.n
    A = np.zeros((n, len(COLUMNS)), np.float32)
    if n == 0:
        return A
    y0, _, x0, _ = tree.box
    ww, W = tree.crop_shape[1], tree.shape[1]
    sy, sx = np.divmod(tree.support, ww)
    flat_idx = (sy + y0) * W + (sx + x0)  # full-frame index of every support pixel
    vals = [m.ravel()[flat_idx] for m in members]
    a = tree.area.astype(np.float64)
    d_sum = np.zeros(n)
    m_sum = np.zeros((len(members), n))
    iou = np.zeros((len(members), n))
    cover = np.zeros((len(members), n))
    for k, t in enumerate(tree.levels):
        ids = tree.node_of[k]
        new = (ids >= 0) & (tree.level[np.maximum(ids, 0)] == k)
        if not new.any():
            continue
        v = ids[new].astype(np.int64)
        d_sum += np.bincount(v, weights=np.abs(vals[0][new] - vals[-1][new]), minlength=n)
        for j, m in enumerate(members):
            m_sum[j] += np.bincount(v, weights=vals[j][new], minlength=n)
            nc, lab = cv2.connectedComponents(fill_holes(m > t).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
            comp_area = np.bincount(lab.ravel(), minlength=nc)
            c = lab.ravel()[flat_idx[new]].astype(np.int64)
            on = c > 0
            cover[j] += np.bincount(v[on], minlength=n)
            key, inter = np.unique(v[on] * nc + c[on], return_counts=True)
            kv, kc = np.divmod(key, nc)
            ratio = inter / (a[kv] + comp_area[kc] - inter)
            np.maximum.at(iou[j], kv, ratio)
    means = m_sum / a
    A[:, 0] = d_sum / a
    A[:, 1] = means.min(0)
    A[:, 2] = means.max(0) - means.min(0)
    A[:, 3] = iou.min(0)
    A[:, 4] = iou.max(0)
    A[:, 5] = (cover / a).min(0)
    return A


def one(args):
    """Worker: build the split tree of one blended map, write its members' agreement columns to `out` (array A) and
    return the node count."""
    blend, members, out = args
    p, _ = read_map_png(blend)
    tree = build_tree(p, splits=True)
    A = agreement(tree, [read_map_png(m)[0] for m in members])
    np.savez_compressed(out, A=A)
    return tree.n


def _extra(a) -> str:
    """The folder of the agreement columns inside the run: `agree_<tag>_split`, or `agree_split` without a tag."""
    return f"agree_{a.tag}_split" if a.tag else "agree_split"


def build(a, run: Path) -> None:
    """Write the agreement columns of every out-of-fold frame of `run` to `<run>/<extra>/<stem>.npz`, from the maps of
    the members in `a.members`."""
    out = run / _extra(a)
    out.mkdir(exist_ok=True)
    member_maps = [{q.stem: q for q in (RUNS / m).glob("fold*/oof/*.png")} for m in a.members]
    jobs = [(str(q), [str(mm[q.stem]) for mm in member_maps], str(out / f"{q.stem}.npz"))
            for q in sorted(run.glob("fold*/oof/*.png"))]
    with pool(a.workers) as workers:
        sizes = workers.map(one, jobs, chunksize=2)
    print(f"{len(jobs)} frames, {sum(sizes)} nodes")


def evaluate(a, run: Path, extra_dir: str = "agree_split", columns: list[str] = COLUMNS, stems=None,
             base_dir: str | None = None, scores_stem: str | None = None) -> dict:
    """The scorer cross-fitted with and without the columns in `<run>/<extra_dir>/<stem>.npz` (array A, one row
    per node of the bank's tree), on the same partitions, and the paired bootstrap of the PQ difference. `stems`
    limits both arms to those frames (a first look before every fold of a model is in). `base_dir`: columns the base
    arm has as well (the released instance columns, say), so the difference is against them rather than against the
    descriptors alone. The columns' scores of partition 0 go to `<run>/scores_<scores_stem>_<scheme>.npz`
    (`scores_stem` defaults to `extra_dir`)."""
    banks = load_banks(run / "bank_split")
    if stems is not None:
        banks = [b for b in banks if b.stem in stems]
    plain = [b.X for b in banks]
    base = plain
    if base_dir is not None:
        base = []
        for b in banks:
            A = np.load(run / base_dir / f"{b.stem}.npz")["A"]
            assert len(A) == b.tree.n, (b.stem, len(A), b.tree.n)
            base.append(np.concatenate([b.X, A], 1))
    extra = []
    for b in banks:
        A = np.load(run / extra_dir / f"{b.stem}.npz")["A"]
        assert len(A) == b.tree.n, (b.stem, len(A), b.tree.n)
        extra.append(np.concatenate([b.X, A], 1))
    rows = {}  # (scheme, rep, arm) -> {tau: (frames, 4) totals}
    for scheme, rep, folds in partitions([b.stem for b in banks], a.partitions, getattr(a, "group_partitions", 1)):
        for arm, X in (("base", base), ("agree", extra)):
            for b, x in zip(banks, X):
                b.X = x
            scores = crossfit(banks, folds, "hurdle")
            res = sweep(banks, scores, taus=TAUS)
            rows[scheme, rep, arm] = {t: v[1] for t, v in res.items()}
            if rep == 0 and arm != "base":  # the columns' scores
                np.savez_compressed(run / f"scores_{scores_stem or extra_dir}_{scheme}.npz",
                                    **{b.stem: u for b, u in zip(banks, scores)})
    for b, x in zip(banks, plain):
        b.X = x
    out = dict(run=a.run, extra=extra_dir, columns=columns, frames=len(banks), base_extra=base_dir)
    out.update(release_rule(rows, len(banks), TAUS))
    return out


def main():
    """Parse the arguments, build the columns if asked, and print and write `<run>/agree[_<tag>]_study.json`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--group-partitions", type=int, default=1, help="partitions of the annotation groups (rule.py)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = RUNS / a.run
    if a.build:
        build(a, run)
    result = evaluate(a, run, _extra(a))
    print(json.dumps(result, indent=1))
    (run / f"{_extra(a)[:-6]}_study.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

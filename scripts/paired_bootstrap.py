"""Paired comparison of two out-of-fold selections on the same training frames (host-exact PQ).

Each file is a selection in the submission format (`oof_report.py --csv`). Both are scored record by record, and
the frames are resampled with replacement to give the standard error of the PQ difference B - A, its 90%
interval, the share of resamples where B does not beat A, the difference per fold, and the spread of the
difference over random 180-frame subsets (the size of the test set).

Usage: paired_bootstrap.py A.csv B.csv [--scheme obs] [--boot 4000]
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from solarfil import metric
from solarfil.data import Dataset, H, W
from solarfil.folds import load_folds


def read_selection(path: str) -> dict[str, list[dict]]:
    """A selection file's instances by frame stem, as COCO RLE dicts at 2048 x 2048."""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["filament_id"].rsplit("_", 1)[0]
            out[stem].append({"size": [H, W], "counts": row["segmentation_rle"].encode()})
    return out


def per_frame(ds: Dataset, sel: dict[str, list[dict]], stems: list[str]) -> np.ndarray:
    """The selection's host counts on each frame of `stems`, one [TP, FP, FN, IoU sum] row per frame."""
    rep = metric.evaluate(ds.observations, sel, stems)
    return np.stack([rep.per_image[s].as_array() for s in stems])


def pq(rows: np.ndarray) -> np.ndarray:
    """PQ from [TP, FP, FN, IoU sum] totals on the last axis."""
    return rows[..., 3] / (rows[..., 0] + 0.5 * rows[..., 1] + 0.5 * rows[..., 2])


def main():
    """Score both selections on the folds' frames and print the paired comparison as JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--scheme", default="obs")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = Dataset()
    folds = load_folds(args.scheme)
    stems = sorted(folds)
    a = per_frame(ds, read_selection(args.a), stems)
    b = per_frame(ds, read_selection(args.b), stems)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(stems), size=(args.boot, len(stems)))
    d = pq(b[idx].sum(axis=1)) - pq(a[idx].sum(axis=1))
    sub = np.stack([rng.choice(len(stems), 180, replace=False) for _ in range(args.boot)])
    d180 = pq(b[sub].sum(axis=1)) - pq(a[sub].sum(axis=1))
    fold_of = np.array([folds[s] for s in stems])
    by_fold = {int(k): round(float(pq(b[fold_of == k].sum(0)) - pq(a[fold_of == k].sum(0))), 4)
               for k in sorted(set(fold_of))}
    print(json.dumps(dict(
        a=args.a, b=args.b, frames=len(stems), pq_a=round(float(pq(a.sum(0))), 5), pq_b=round(float(pq(b.sum(0))), 5),
        diff=round(float(pq(b.sum(0)) - pq(a.sum(0))), 5), se=round(float(d.std()), 5),
        ci90=[round(float(np.quantile(d, 0.05)), 5), round(float(np.quantile(d, 0.95)), 5)],
        p_b_not_better=round(float((d <= 0).mean()), 4), by_fold=by_fold,
        sd_180=round(float(d180.std()), 5), p_b_not_better_180=round(float((d180 <= 0).mean()), 4))))


if __name__ == "__main__":
    main()

"""Recompute the report tables from native per-observation sufficient statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("pq", "mean_pair_iou", "mean_pair_dice", "uncovered_references",
           "uncovered_predictions", "splits", "merges")


def pq(rows):
    denominator = rows[..., 0] + .5 * rows[..., 1] + .5 * rows[..., 2]
    return np.divide(rows[..., 3], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def metrics(rows):
    denominator = np.stack([rows[..., 0] + .5 * rows[..., 1] + .5 * rows[..., 2],
                            rows[..., 4], rows[..., 4], *[rows[..., 7]] * 4], axis=-1)
    numerator = rows[..., [3, 5, 6, 8, 9, 10, 11]]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def group_draws(rows, groups, seed):
    grouped = np.stack([rows[groups == g].sum(0) for g in np.unique(groups)])
    draw = np.random.default_rng(seed).integers(0, len(grouped), (2000, len(grouped)))
    return grouped[draw].sum(1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a fresh output file")
    with np.load(args.data / "component_rows.npz", allow_pickle=False) as z:
        components = {key: z[key] for key in z.files}
    with np.load(args.data / "native_rows.npz", allow_pickle=False) as z:
        native = {key: z[key] for key in z.files}
    np.testing.assert_array_equal(components["stems"], native["stems"])
    np.testing.assert_array_equal(components["groups"], native["groups"])
    np.testing.assert_array_equal(components["full|0.3"], native["group|0.3"][:, :4])
    groups = native["groups"]
    if len(groups) != 707 or len(np.unique(groups)) != 19 or int(native["multi"].sum()) != 296:
        raise ValueError("Incomplete native comparison cohort")
    for collection in (components, native):
        for value in collection.values():
            if value.dtype.kind in "fiu" and (not np.isfinite(value).all() or (value < 0).any()):
                raise ValueError("Invalid native sufficient statistics")
    thresholds = [key for key in components if key.startswith("threshold|")]
    best = max(thresholds, key=lambda key: pq(components[key].sum(0)))
    keys = ["full|0.3", "without_record|0.3", "without_union|0.3", "without_detectors|0.3", best]
    full = components[keys[0]]
    ref_draws = pq(group_draws(full, groups, 0))
    table1 = []
    for key in keys:
        rows = components[key]
        delta = float(pq(rows.sum(0)) - pq(full.sum(0)))
        interval = np.quantile(pq(group_draws(rows, groups, 0)) - ref_draws, [.05, .95]).tolist()
        table1.append(dict(system=key, pq=float(pq(rows.sum(0))), delta=delta, paired_group90=interval))
    keep = native["multi"]
    if len(np.unique(groups[keep])) != 12:
        raise ValueError("Unexpected matched annotation-group cohort")
    model = native["group|0.3"][keep].copy()
    model *= (model[:, 7] - 1)[:, None]
    human = native["annotator"][keep]
    if model[:, 7].sum() != 1196 or human[:, 7].sum() != 1196:
        raise ValueError("The two arms require identical target-record weights")
    table2 = dict(model=dict(zip(METRICS, metrics(model.sum(0)).tolist(), strict=True)),
                  second_annotator=dict(zip(METRICS, metrics(human.sum(0)).tolist(), strict=True)))
    full_interval = np.quantile(pq(group_draws(full, groups, 20261102)), [.05, .95]).tolist()
    result = dict(table1=table1, table2=table2, full_pq_group90=full_interval,
                  frames=707, groups=19, matched_frames=296, matched_groups=12,
                  comparisons_per_arm=1196, bootstrap_draws=2000, interval_percent=90,
                  new_fits=0, neural_forwards=0,
                  scope="Conditional on stored native predictions, fitted models, selected recipes and observed groups")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

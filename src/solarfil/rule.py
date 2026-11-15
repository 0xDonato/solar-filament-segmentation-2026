"""The release rule: a change ships when its paired out-of-fold gain is positive at every tau from 0.24 to 0.30 in both
cross-fitting schemes, and the two schemes' mean gain at tau 0.26 has its 90% interval above zero, the frames
resampled jointly for both schemes since their cells correlate.

The scorer is cross-fitted on several partitions in each scheme: random partitions of the frames, and partitions that
hold out whole annotation groups (`folds.group_partition`). Each arm's per-frame totals are averaged over a scheme's
partitions before the frame bootstrap, so neither cell rests on one scorer fit.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from solarfil.folds import group_partition

TAUS = (0.24, 0.26, 0.28, 0.30)
DECIDING_TAU = 0.26


def partitions(stems: list[str], n_random: int, n_group: int = 1, ds=None, first: int = 0,
               first_group: int = 0) -> Iterator[tuple[str, int, dict]]:
    """(scheme, rep, folds) of each partition: `n_random` random partitions of `stems` into five folds (seeds
    `first`, `first + 1`, ...), then `n_group` partitions of the annotation groups (`first_group`, ...; partition 0 is
    the fixed group folds, later ones re-drawn)."""
    for rep in range(first, first + n_random):
        rng = np.random.default_rng(rep)
        yield "random", rep, {s: int(k) for s, k in zip(stems, rng.permutation(len(stems)) % 5)}
    for rep in range(first_group, first_group + n_group):
        yield "group", rep, group_partition(rep, ds)


def pq(tot) -> float:
    """PQ from summed (TP, FP, FN, sum of the TPs' IoU)."""
    return tot[3] / (tot[0] + 0.5 * tot[1] + 0.5 * tot[2])


def release_rule(rows: dict, n_frames: int, taus=TAUS, n_boot: int = 2000) -> dict:
    """The arms' PQ, the gains and the rule from `rows`, {(scheme, rep, arm): {tau: (frames, 4) totals}} with the arms
    "base" and "agree" on the same frames in the same order. Per scheme and tau: each arm's mean PQ over the
    partitions, the gain of each partition, and the 90% interval of the gain of the partitions' mean totals over
    `n_boot` frame resamples."""
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n_frames, n_frames) for _ in range(n_boot)]
    out, gain, boot_gain = {}, {}, {}
    for scheme in ("random", "group"):
        reps = sorted({r for s, r, _ in rows if s == scheme})
        for tau in taus:
            base = np.mean([rows[scheme, r, "base"][tau] for r in reps], 0)  # the partitions' mean totals per frame
            agree = np.mean([rows[scheme, r, "agree"][tau] for r in reps], 0)
            diff = np.array([pq(agree[i].sum(0)) - pq(base[i].sum(0)) for i in boots])
            per_partition = [float(pq(rows[scheme, r, "agree"][tau].sum(0)) - pq(rows[scheme, r, "base"][tau].sum(0)))
                             for r in reps]
            gain[scheme, tau], boot_gain[scheme, tau] = float(np.mean(per_partition)), diff
            out[f"{scheme}_{tau}"] = dict(
                base=round(float(np.mean([pq(rows[scheme, r, "base"][tau].sum(0)) for r in reps])), 5),
                agree=round(float(np.mean([pq(rows[scheme, r, "agree"][tau].sum(0)) for r in reps])), 5),
                per_partition=[round(g, 5) for g in per_partition],
                gain_ci90=[round(float(x), 5) for x in np.percentile(diff, [5, 95])])
    mean = {tau: 0.5 * (boot_gain["random", tau] + boot_gain["group", tau]) for tau in taus}
    out["release_rule"] = dict(
        schemes_mean={str(tau): dict(gain=round(0.5 * (gain["random", tau] + gain["group", tau]), 5),
                                     ci90=[round(float(x), 5) for x in np.percentile(mean[tau], [5, 95])])
                      for tau in taus},
        every_cell_positive=all(g > 0 for g in gain.values()),
        passes=all(g > 0 for g in gain.values()) and float(np.percentile(mean[DECIDING_TAU], 5)) > 0)
    return out

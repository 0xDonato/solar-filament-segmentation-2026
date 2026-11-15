"""What each component of the released system does to the judged measures, not to PQ alone (forum 724503: "the
contribution of each component to the final results").

Every row is a variant of the release (`ens_v5_inc5`: seven dense networks at 5:5:1:1:1:1:1, the three detector sets'
columns, the hurdle scorer and the exact selection on the component tree) built from banks and maps that exist,
with no training:

- the release without one detector set's six columns (three rows), and the release with the 41 descriptors alone, the
  scorer refitted on what is left;
- the five-recipe block removed: the {seed 0, W} blend (`ens_s0_win`, seed 0 and W) with the same three sets;
- W's whole recipe (its 0.24 px label shift, rotation and zoom, its seed) replaced by seed 1 of the plain recipe:
  the two-seed blend (`ens_lean_2seeds`) against the {seed 0, W} blend (`ens_s0_win`), both with the two sets that are
  complete on both banks. No existing run removes the shift alone;
- the component tree and its scorer replaced by plain threshold and connected components on the release's blended maps,
  at the (threshold, minimum area) cell of `oof_eval.py`'s grid with the best pooled PQ on all 707 frames (chosen in
  sample, which favours the baseline). It has no tau, so the same masks stand against the release at both picks.

The scorer is cross-fitted as in the increment test (`member_rule.py`): random partitions 15-19 and group
partitions 10-14, never random 25-29 or group 20-24. Each partition's selection at tau 0.26 and 0.30 is turned back
into masks (`tree.build_tree` on the run's out-of-fold maps, as `oof_select.py` does) and read with the host's
float32 counts (`metric.iou_dice`, `metric.Report`, `reference.frame_row` and `reference.legend_counts`, as
`judges_reference.py --frames all` reads a selection): PQ, the IoU of the overlapping pairs (mean and the lower 10%
quantile), their Dice mean, and per comparison (one per annotator record) the splits (1:n, n >= 2), merges (n:1,
n >= 2), missed filaments (1:0) and spurious masks (0:1). Chirality legibility at the 8 px fixed before the test is read
with
`chirality_legibility.py`'s rows (`read_frame`, `same_rows`) and `chirality.leave_group_out`: the legible share,
accuracy, and accuracy and log loss beyond the latitude prior, on the multiply annotated frames.

Each value is the mean over a scheme's five partitions; a difference is the row minus its reference, partition by
partition, then averaged. Intervals are 90% frame-bootstrap intervals (2,000 resamples drawn from seed 0 as
`rule.release_rule` draws them) of the partitions' mean per-frame totals, paired on the same frames; the 10%
quantile's interval averages the partitions' resampled quantiles.

Output: `ablation_metrics.csv` in long form (row, tau, scheme, measure, value, ci_low, ci_high, source), where a
measure `d_<m>` is the paired difference from the row's reference, `ablation_metrics.md` with the tables and the
sanity checks, and a cache of scores and per-frame readings under `ablation_cache/` for reruns.

Usage: ablation_metrics.py [--out-dir report] [--workers 3] [--threads 3]
       [--boot 2000] [--frames N]   (--frames: a smoke run on the first N frames, cached apart)
       ablation_metrics.py --profile compact --out-dir <new-report-folder> [--cache <existing-cache>]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import chirality_legibility as cl
import oof_eval
from decide import crossfit, sweep
from member_rule import banks_with_columns
from solarfil import chirality, metric, reference
from solarfil.data import Dataset, H, W
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import REPO, RUNS
from solarfil.postprocess import threshold_labels
from solarfil.rule import partitions, pq
from solarfil.tree import build_tree, select

SETS = ("inst_maskrcnn", "inst_maskrcnn_grp", "inst_maskrcnn_union")
SET_COLUMNS = 6  # columns per detector set (instance_study.COLUMNS)
N_DESCRIPTORS = 41
TAUS = (0.26, 0.30)
FIRST_RANDOM, FIRST_GROUP, N_PARTITIONS = 15, 10, 5  # the increment test's read partitions
BARB = chirality.BRANCH  # 8 px, fixed before the test

#: scorer arms: the run whose bank and maps they use, the folder of detector columns, and a set to drop from it
ARMS = {
    "v6": ("ens_v5_inc5", "+".join(SETS) + "_split", None),
    **{f"v6-{s}": ("ens_v5_inc5", "+".join(SETS) + "_split", i) for i, s in enumerate(SETS)},
    "v6-detectors": ("ens_v5_inc5", None, None),
    "v5": ("ens_s0_win", "+".join(SETS) + "_split", None),
    "v5-inst_maskrcnn": ("ens_s0_win", "+".join(SETS) + "_split", 0),
    "v4": ("ens_s0_win", "inst_maskrcnn+inst_maskrcnn_grp_split", None),
    "v3": ("ens_lean_2seeds", "inst_maskrcnn+inst_maskrcnn_grp_split", None),
    # release v7 is v5 without the per-record frame-fold set (arm v5-inst_maskrcnn); its own ablations:
    "v7-inst_maskrcnn_grp": ("ens_s0_win", "+".join(SETS) + "_split", (0, 1)),
    "v7-inst_maskrcnn_union": ("ens_s0_win", "+".join(SETS) + "_split", (0, 2)),
    "v7-detectors": ("ens_s0_win", None, None),
    # release v9 is v7 plus the native RTMDet-Ins-x sets M and U (65 columns)
    "v9": ("ens_s0_win", "inst_maskrcnn_grp+inst_maskrcnn_union+rtmdet_x_M+rtmdet_x_U_split", None),
    # release v10 is v9 with the 2x network as a third dense member (ens_x2_add's maps, bank and the same 65 columns)
    "v10": ("ens_x2_add", "inst_maskrcnn_grp+inst_maskrcnn_union+rtmdet_x_M+rtmdet_x_U_split", None),
}
THRESHOLD = "threshold"  # the arm of plain threshold and components on the release's maps
THRESHOLD_V7 = "threshold_v7"  # the same on release v7's two-network maps
THRESHOLDS = {THRESHOLD: "ens_v5_inc5", THRESHOLD_V7: "ens_s0_win"}  # each threshold arm's run

#: report rows: (row, arm, reference arm, what it removes)
ROWS = [
    ("v6", "v6", None, "release v6 (ens_v5_inc5, 59 columns)"),
    ("v6 without inst_maskrcnn", "v6-inst_maskrcnn", "v6", "the per-record detector set's six columns"),
    ("v6 without inst_maskrcnn_grp", "v6-inst_maskrcnn_grp", "v6", "the group-trained detector set's six columns"),
    ("v6 without inst_maskrcnn_union", "v6-inst_maskrcnn_union", "v6", "the union detector set's six columns"),
    ("v6 without detectors", "v6-detectors", "v6", "all 18 detector columns (descriptors only)"),
    ("v6 without the five recipes (v5)", "v5", "v6", "the five-recipe block (release v5's blend, three sets)"),
    ("compact: v5 without inst_maskrcnn", "v5-inst_maskrcnn", "v5", "v5's per-record frame-fold detector set"),
    ("compact against v6", "v5-inst_maskrcnn", "v6", "the five recipes and the per-record frame-fold set"),
    ("W replaced by seed 1 (v3 vs v4)", "v3", "v4", "W's recipe: label shift, rotation and zoom, seed; two sets"),
    ("v6 tree -> threshold+components", THRESHOLD, "v6", "the component tree and scorer (thr+CC on v6's maps)"),
    ("v7 without inst_maskrcnn_grp", "v7-inst_maskrcnn_grp", "v5-inst_maskrcnn", "v7's group-fold detector columns"),
    ("v7 without inst_maskrcnn_union", "v7-inst_maskrcnn_union", "v5-inst_maskrcnn", "v7's union detector columns"),
    ("v7 without detectors", "v7-detectors", "v5-inst_maskrcnn", "both of v7's detector sets (descriptors only)"),
    ("v7 tree -> threshold+components", THRESHOLD_V7, "v5-inst_maskrcnn",
     "the component tree and scorer (thr+CC on v7's maps)"),
    ("v9 against v7", "v9", "v5-inst_maskrcnn", "RTMDet sets M and U added to v7 (65 columns)"),
    ("v10 against v9", "v10", "v9", "the 2x network added to v9's two-network blend (65 columns)"),
]
HOST = ("pq", "pair_iou_mean", "pair_iou_p10", "dice_mean", "splits", "merges", "missed_1_0", "spurious_0_1")
CHIR = ("legible_share", "chirality_accuracy", "accuracy_beyond_latitude", "log_loss_beyond_latitude")
_G: dict = {}


# --- scores ----------------------------------------------------------------------------------------------------------

def arm_banks(arm: str, n_frames: int | None, arm_config: tuple | None = None):
    """The arm's banks with its detector columns (the six of each set the arm drops removed), first `n_frames`."""
    run, sets_dir, drop = ARMS[arm] if arm_config is None else arm_config
    banks = banks_with_columns(RUNS / run, sets_dir)[:n_frames]
    if drop is not None:
        cols = np.concatenate([N_DESCRIPTORS + d * SET_COLUMNS + np.arange(SET_COLUMNS) for d in np.atleast_1d(drop)])
        for b in banks:
            n_sets = len(sets_dir.removesuffix("_split").split("+"))
            assert b.X.shape[1] == N_DESCRIPTORS + n_sets * SET_COLUMNS, (arm, b.stem, b.X.shape)
            b.X = np.delete(b.X, cols, 1)
    return banks


def arm_scores(arm: str, cache: Path, n_frames: int | None, threads: int,
               arm_config: tuple | None = None) -> dict:
    """{(scheme, rep): {stem: out-of-fold utilities}} on the read partitions, and the bank's own per-frame totals
    {(scheme, rep, tau): (frames, 4)}; cross-fitted once and cached in `cache`."""
    from threadpoolctl import threadpool_limits
    banks = arm_banks(arm, n_frames, arm_config)
    stems = [b.stem for b in banks]
    out, totals = {}, {}
    for scheme, rep, folds in partitions(stems, N_PARTITIONS, N_PARTITIONS, first=FIRST_RANDOM,
                                         first_group=FIRST_GROUP):
        f = cache / f"scores_{arm}_{scheme}_{rep}.npz"
        if not f.exists():
            with threadpool_limits(threads):
                s = crossfit(banks, folds, "hurdle")
            np.savez_compressed(f, **{st: x for st, x in zip(stems, s)})
            print(f"scored {arm} {scheme} {rep}", flush=True)
        z = np.load(f)
        out[scheme, rep] = {st: z[st] for st in stems}
        for tau, (_, rows) in sweep(banks, [z[st] for st in stems], taus=TAUS).items():
            totals[scheme, rep, tau] = rows
    return out, totals


def threshold_cell(stems: list[str], cache: Path, workers: int,
                   run: str = THRESHOLDS[THRESHOLD]) -> tuple[float, int]:
    """The (threshold, minimum area) cell of `oof_eval.py`'s default grid with the best pooled PQ on the run's
    maps (by default release v6's)."""
    f = cache / ("threshold_cell.json" if run == THRESHOLDS[THRESHOLD] else f"threshold_cell_{run}.json")
    if f.exists():
        return tuple(json.loads(f.read_text())["cell"])
    grid = [(t, m) for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6) for m in (0, 100, 200, 300, 500)]
    jobs = [(str(RUNS / run), s, grid, True, 0.0, 0.0, False) for s in stems]
    with pool(workers, spawn=True) as w:
        res = w.map(oof_eval.score_one, jobs, chunksize=4)
    tot = {g: sum(r[g] for _, r, _ in res) for g in grid}
    best = max(grid, key=lambda g: pq(tot[g]))
    f.write_text(json.dumps(dict(cell=best, pq=round(float(pq(tot[best])), 5),
                                 grid={f"{t}_{m}": round(float(pq(v)), 5) for (t, m), v in tot.items()})))
    return best


# --- masks, one frame at a time --------------------------------------------------------------------------------------

def node_pixels(tree, v: int) -> np.ndarray:
    """Full-frame flat pixel indices of node v."""
    c = tree.node_pixels(v)
    y0, _, x0, x1 = tree.box
    return (y0 + c // (x1 - x0)) * W + x0 + c % (x1 - x0)


def label_pixels(lab: np.ndarray, n: int) -> list[np.ndarray]:
    """Flat pixel indices of labels 1..n of a label image."""
    if n == 0:
        return []
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    order = np.argsort(flat[idx], kind="stable")
    return np.split(idx[order], np.cumsum(np.bincount(flat[idx], minlength=n + 1)[1:])[:-1])


def unit_counts(units: list[np.ndarray], gt: metric.GTIndex) -> list[np.ndarray]:
    """Per record, the intersection counts (filaments, units) of every unit with every filament."""
    buf = np.zeros(H * W, np.uint8)
    out = [np.zeros((len(rec), len(units)), np.int64) for rec in gt.records]
    for j, pix in enumerate(units):
        buf[pix] = 1
        for k, rec in enumerate(gt.records):
            for i, idx in enumerate(rec):
                out[k][i, j] = int(buf[idx].sum())
        buf[pix] = 0
    return out


def selection_reading(stem: str, gt: metric.GTIndex, counts: list[np.ndarray], areas: np.ndarray,
                      cols: list[int]) -> tuple[np.ndarray, np.ndarray, list]:
    """One selection's host reading on the frame: [`reference.frame_row`, comparisons, 1:0, 0:1, splits, merges],
    the IoU of its overlapping pairs, and its (record, filament, mask) matches at IoU > 0.5."""
    cols = np.asarray(cols, int)
    rep, pairs = metric.Report(), []
    for k, rec in enumerate(gt.records):
        a_gt = np.array([len(i) for i in rec], np.int64)
        iou, dice = metric.iou_dice(counts[k][:, cols], a_gt, areas[cols])
        rep.add_matrices(stem, iou, dice)
        pairs += [(k, int(i), int(j)) for i, j in zip(*np.nonzero(iou > 0.5))]
    c = reference.legend_counts(rep)
    row = np.concatenate([reference.frame_row(rep), [len(rep.ious), c["1:0"], c["0:1"], c["splits"], c["merges"]]])
    return row, rep.iou_values, pairs


def unit_barbs(pix: np.ndarray) -> list:
    """`chirality.barbs` of one unit's mask at the length fixed before the test."""
    m = np.zeros(H * W, bool)
    m[pix] = True
    return chirality.barbs(m.reshape(H, W), BARB)


def frame_pass(job) -> tuple[str, dict, dict | None]:
    """One frame of one run: every arm's selections at each tau (and the threshold arm's components when asked),
    read as `selection_reading` does, with each selected mask's barbs on multiply annotated frames, and the frame's
    annotator chirality readings (`chirality_legibility.read_frame`) when asked."""
    stem, png, sel_scores, cell, want_gt = job
    ds = _G.setdefault("ds", Dataset())
    obs = ds.observations[stem]
    gt = metric.GTIndex(obs)
    p, _ = read_map_png(png)
    tree = build_tree(p, splits=True)
    picks = {}
    for key, s in sel_scores.items():
        assert tree.n == len(s), (stem, key, tree.n, len(s))
        for tau in TAUS:
            picks[(*key, tau)] = select(tree, s, tau) if tree.n else []
    nodes = sorted(set().union(*picks.values())) if picks else []
    units = [node_pixels(tree, v) for v in nodes]
    col = {v: j for j, v in enumerate(nodes)}
    sel = {key: [col[v] for v in vs] for key, vs in picks.items()}
    if cell is not None:  # (threshold arm, threshold, minimum area)
        lab, n = threshold_labels(p, cell[1], cell[2])
        sel[(cell[0], "all", 0, 0.0)] = list(range(len(units), len(units) + n))
        units += label_pixels(lab, n)
    counts = unit_counts(units, gt)
    areas = np.array([len(u) for u in units], np.int64)
    multi = obs.n_records > 1
    barbs = {j: unit_barbs(units[j]) for j in sorted(set().union(*sel.values()))} if multi and sel else {}
    out = {}
    for key, cols in sel.items():
        row, ious, pairs = selection_reading(stem, gt, counts, areas, cols)
        out[key] = (row, ious, dict(barbs=[barbs[j] for j in cols], pairs=pairs) if multi else None)
    gtc = cl.read_frame((stem, {}, BARB)) if want_gt and multi else None
    return stem, out, gtc


def run_pass(run: str, keys: dict, stems: list[str], cell, want_gt: bool, workers: int):
    """`frame_pass` over every frame of a run: {key: {stem: reading}} and the annotators' chirality frames."""
    maps = {q.stem: q for q in (RUNS / run).glob("fold*/oof/*.png")}
    jobs = [(s, str(maps[s]), {k: v[s] for k, v in keys.items()}, cell, want_gt) for s in stems]
    readings, gtc = {}, {}
    with pool(workers, spawn=True) as w:
        for i, (stem, out, g) in enumerate(w.imap_unordered(frame_pass, jobs, chunksize=2)):
            for key, r in out.items():
                readings.setdefault(key, {})[stem] = r
            if g is not None:
                gtc[stem] = g
            if (i + 1) % 100 == 0:
                print(f"{run}: {i + 1}/{len(jobs)} frames", flush=True)
    return readings, gtc


# --- per-frame arrays and chirality rows ---------------------------------------------------------------------------

def chirality_rows(per_stem: dict, gtc: dict, stems: list[str]) -> np.ndarray:
    """The reader's rows of the legibility test (`chirality_legibility.same_rows` at 8 px) as an array
    [frame index, group, y, h, hx, m, ls, cons]."""
    groups = sorted({g["group"] for g in gtc.values()})
    frames = [dict(gtc[s], readers={"r": per_stem[s][2]}) for s in stems if s in gtc]
    rows, _ = cl.same_rows(frames, "r", BARB)
    fidx = {s: i for i, s in enumerate(stems)}
    return np.array([[fidx[r["f"]], groups.index(r["g"]), r["y"], *(r[c] for c in chirality.FULL)] for r in rows],
                    float).reshape(-1, 3 + len(chirality.FULL))


def chirality_frame_sums(rows: np.ndarray, n_frames: int) -> np.ndarray:
    """Per frame [rows, legible, right with the bearing, accuracy gain over the latitude prior, log-loss change],
    from leave-one-group-out fits of the prior and the prior plus the bearing (`chirality.beyond_latitude`'s
    models)."""
    out = np.zeros((n_frames, 5))
    if len(rows) == 0:
        return out
    y, g, f = rows[:, 2], rows[:, 1], rows[:, 0].astype(int)
    p0 = chirality.leave_group_out(rows[:, 3:3 + len(chirality.PRIOR)], y, g)
    p1 = chirality.leave_group_out(rows[:, 3:], y, g)
    q0, q1 = np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)
    dl = -(y * np.log(q1) + (1 - y) * np.log(1 - q1)) + (y * np.log(q0) + (1 - y) * np.log(1 - q0))
    a0, a1 = ((p0 > 0.5) == (y == 1)).astype(float), ((p1 > 0.5) == (y == 1)).astype(float)
    m = rows[:, 3 + chirality.FULL.index("m")] != 0
    for j, v in enumerate((np.ones(len(y)), m, a1, a1 - a0, dl)):
        np.add.at(out[:, j], f, v)
    return out


def frame_arrays(per_stem: dict, gtc: dict, stems: list[str]) -> dict:
    """One selection's per-frame arrays: host rows (frames, 12), pair IoUs with their frame index, chirality rows."""
    host = np.stack([per_stem[s][0] for s in stems])
    ious = [per_stem[s][1].astype(np.float64) for s in stems]
    return dict(host=host, iou=np.concatenate(ious), iou_frame=np.repeat(np.arange(len(stems)), [len(v) for v in ious]),
                chir=chirality_rows(per_stem, gtc, stems))


def save_arrays(path: Path, arrays: dict) -> None:
    """{key: frame_arrays} to one npz, keys joined with '|'."""
    np.savez_compressed(path, **{f"{'|'.join(map(str, k))}|{name}": v for k, d in arrays.items()
                                 for name, v in d.items()})


def load_arrays(path: Path) -> dict:
    """The inverse of `save_arrays` (tau and rep restored as numbers)."""
    out: dict = {}
    with np.load(path) as z:
        for name in z.files:
            arm, scheme, rep, tau, field = name.split("|")
            out.setdefault((arm, scheme, int(rep), float(tau)), {})[field] = z[name]
    return out


# --- measures and the paired bootstrap ------------------------------------------------------------------------------

def host_measures(s: np.ndarray) -> dict:
    """The host's measures from summed host rows (last axis 12)."""
    comps = s[..., 7]
    return dict(pq=pq(np.moveaxis(s[..., :4], -1, 0)), pair_iou_mean=s[..., 5] / s[..., 4],
                dice_mean=s[..., 6] / s[..., 4], splits=s[..., 10] / comps, merges=s[..., 11] / comps,
                missed_1_0=s[..., 8] / comps, spurious_0_1=s[..., 9] / comps)


def chir_measures(s: np.ndarray) -> dict:
    """The legibility measures from summed chirality frame sums (last axis 5)."""
    n = s[..., 0]
    return dict(legible_share=s[..., 1] / n, chirality_accuracy=s[..., 2] / n,
                accuracy_beyond_latitude=s[..., 3] / n, log_loss_beyond_latitude=s[..., 4] / n)


def p10(values: np.ndarray, frames: np.ndarray, weights: np.ndarray, q: float = 0.1) -> np.ndarray:
    """The lower q quantile (inverted CDF) of `values` under per-frame weights, one per row of `weights`."""
    order = np.argsort(values, kind="stable")
    v, f = values[order], frames[order]
    out = []
    for chunk in np.array_split(weights, max(1, len(weights) // 250)):
        cum = np.cumsum(chunk[:, f], axis=1)
        out.append(v[np.argmax(cum >= q * cum[:, -1:], axis=1)])
    return np.concatenate(out)


def per_rep(arrays: dict, arm: str, scheme: str, tau: float) -> list[dict]:
    """The arm's frame arrays for each partition of the scheme (the threshold arm's one selection for all)."""
    if arm in THRESHOLDS:
        return [arrays[arm, "all", 0, 0.0]]
    first = FIRST_RANDOM if scheme == "random" else FIRST_GROUP
    return [arrays[arm, scheme, r, tau] for r in range(first, first + N_PARTITIONS)]


def readings(reps: list[dict], C: np.ndarray, n_frames: int) -> dict:
    """Every measure of one arm: the partitions' values (mean over partitions) and, per bootstrap count row of C,
    the measure of the partitions' mean per-frame totals (the 10% quantile: the mean of the partitions')."""
    one = np.ones((1, n_frames))
    sums = [chirality_frame_sums(r["chir"], n_frames) for r in reps]
    host, chir = np.mean([r["host"] for r in reps], 0), np.mean(sums, 0)
    point = {m: [] for m in HOST + CHIR}
    for r, c in zip(reps, sums):
        for m, v in {**host_measures(r["host"].sum(0)), **chir_measures(c.sum(0))}.items():
            point[m].append(float(v))
        point["pair_iou_p10"].append(float(p10(r["iou"], r["iou_frame"], one)[0]))
    boot = {**host_measures(C @ host), **chir_measures(C @ chir)}
    boot["pair_iou_p10"] = np.mean([p10(r["iou"], r["iou_frame"], C) for r in reps], 0)
    return dict(point={m: np.array(v) for m, v in point.items()}, boot=boot)


def bootstrap_counts(n_frames: int, n_boot: int) -> np.ndarray:
    """(n_boot, frames) resample counts, drawn as `rule.release_rule` draws its resamples."""
    rng = np.random.default_rng(0)
    C = np.zeros((n_boot, n_frames))
    for b in range(n_boot):
        np.add.at(C[b], rng.integers(0, n_frames, n_frames), 1)
    return C


def ci(x: np.ndarray) -> tuple[float, float]:
    """The 90% interval of bootstrap values."""
    lo, hi = np.percentile(x, [5, 95])
    return float(lo), float(hi)


def table(arrays: dict, n_frames: int, n_boot: int, sources: dict,
          row_defs: list | None = None) -> list[dict]:
    """The long-form rows of the CSV: each row's level and its paired difference from its reference."""
    row_defs = ROWS if row_defs is None else row_defs
    C = bootstrap_counts(n_frames, n_boot)
    out = []
    for tau in TAUS:
        for scheme in ("random", "group"):
            got = {}
            for arm in {a for _, a, ref, _ in row_defs} | {ref for _, _, ref, _ in row_defs if ref}:
                got[arm] = readings(per_rep(arrays, arm, scheme, tau), C, n_frames)
            for row, arm, ref, _ in row_defs:
                a = got[arm]
                for m in HOST + CHIR:
                    out.append(dict(row=row, tau=tau, scheme=scheme, measure=m, value=a["point"][m].mean(),
                                    ci=ci(a["boot"][m]), source=sources[arm]))
                    if ref:
                        r = got[ref]
                        d = (a["point"][m].mean() - r["point"][m].mean() if len(a["point"][m]) == 1
                             else (a["point"][m] - r["point"][m]).mean())
                        out.append(dict(row=row, tau=tau, scheme=scheme, measure=f"d_{m}", value=d,
                                        ci=ci(a["boot"][m] - r["boot"][m]), source=f"{sources[arm]}; minus {ref}"))
    return out


# --- outputs ----------------------------------------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    """The long-form table."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["row", "tau", "scheme", "measure", "value", "ci_low", "ci_high", "source"])
        for r in rows:
            w.writerow([r["row"], r["tau"], r["scheme"], r["measure"], f"{r['value']:.6f}", f"{r['ci'][0]:.6f}",
                        f"{r['ci'][1]:.6f}", r["source"]])


def write_md(path: Path, rows: list[dict], checks: dict, cell,
             row_defs: list | None = None, n_frames: int = 707) -> None:
    """The summary: each row's paired differences at each tau in both schemes, then the sanity checks."""
    row_defs = ROWS if row_defs is None else row_defs
    lines = ["# Component comparisons on the host measures", "",
             "Paired differences (row minus its reference) with 90% frame-bootstrap intervals; scorer cross-fitted on "
             "random partitions 15-19 and group partitions 10-14, each value the mean over a scheme's five partitions. "
             f"Host measures on {n_frames} frames; splits, merges, 1:0 and 0:1 per comparison (annotator record); "
             "chirality at 8 px on the multiply annotated frames. Script: `scripts/ablation_metrics.py`. "
             f"Threshold+components cell: threshold {cell[0]}, minimum area {cell[1]} px.", ""]
    look = {(r["row"], r["tau"], r["scheme"], r["measure"]): r for r in rows}
    for tau in TAUS:
        for scheme in ("random", "group"):
            lines += [f"## tau {tau}, {scheme} partitions", "",
                      "| row | " + " | ".join(HOST + CHIR) + " |", "|---" * (1 + len(HOST + CHIR)) + "|"]
            for row, _, ref, _ in row_defs:
                key = "" if ref is None else "d_"
                cells = []
                for m in HOST + CHIR:
                    r = look[row, tau, scheme, key + m]
                    cells.append(f"{r['value']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" if ref
                                 else f"{r['value']:.4f}")
                lines.append(f"| {row}{'' if ref is None else f' (vs {ref})'} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## Sanity checks", "", "```", json.dumps(checks, indent=1), "```", ""]
    path.write_text("\n".join(lines))


def sanity(arrays: dict, totals: dict) -> dict:
    """PQ of the release and of the {seed 0, W} map at tau 0.26 from the masks and from the banks, per scheme (mean of
    the partitions)."""
    out = {}
    for arm in ("v6", "v5", "v4", "v3"):
        for scheme, first in (("random", FIRST_RANDOM), ("group", FIRST_GROUP)):
            for tau in TAUS:
                reps = range(first, first + N_PARTITIONS)
                masks = np.mean([pq(arrays[arm, scheme, r, tau]["host"][:, :4].sum(0)) for r in reps])
                banks = np.mean([pq(totals[arm][scheme, r, tau].sum(0)) for r in reps])
                out[f"{arm} {scheme} {tau}"] = dict(masks=round(float(masks), 5), banks=round(float(banks), 5))
    return out


def compact_spec() -> tuple[dict, list]:
    """The four-network system and its own component removals, using only its two detector sets."""
    run, columns = "ens_s0_win", "inst_maskrcnn_grp+inst_maskrcnn_union_split"
    arms = {
        "v5-inst_maskrcnn": (run, columns, None),
        "v7-inst_maskrcnn_grp": (run, columns, 0),
        "v7-inst_maskrcnn_union": (run, columns, 1),
        "v7-detectors": (run, None, None),
    }
    rows = [("compact pipeline", "v5-inst_maskrcnn", None, "two maps and two detector sets")]
    rows += [r for r in ROWS if r[1] in {*arms, THRESHOLD_V7} and r[2] == "v5-inst_maskrcnn"]
    assert len(rows) == 5
    return arms, rows


def compact_main(a) -> None:
    """Rebuild only compact component rows, retaining the shared native metric and resampling code."""
    tag = f"_n{a.frames}" if a.frames else ""
    csv_path = a.out_dir / f"ablation_metrics{tag}.csv"
    md_path = a.out_dir / f"ablation_metrics{tag}.md"
    for target in (csv_path, md_path):
        if target.exists():
            raise FileExistsError(f"Choose a new output directory: {target} exists")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cache = a.cache or a.out_dir / f"ablation_cache{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    arms, row_defs = compact_spec()
    scores, totals = {}, {}
    for arm, config in arms.items():
        scores[arm], totals[arm] = arm_scores(arm, cache, a.frames, a.threads, config)
    stems = sorted(scores["v5-inst_maskrcnn"]["random", FIRST_RANDOM])
    for arm in arms:
        assert sorted(scores[arm]["random", FIRST_RANDOM]) == stems, arm
    run = "ens_s0_win"
    arrays_file = cache / f"arrays_{run}.npz"
    arrays = load_arrays(arrays_file) if arrays_file.exists() else {}
    gt_file = cache / "gt_chirality.json"
    gtc = json.loads(gt_file.read_text()) if gt_file.exists() else None
    if not arrays_file.exists():
        keys = {(arm, scheme, rep): values for arm in arms
                for (scheme, rep), values in scores[arm].items()}
        got, g = run_pass(run, keys, stems, None, gtc is None, a.workers)
        if gtc is None:
            gtc = g
            gt_file.write_text(json.dumps(gtc))
        save_arrays(arrays_file, {key: frame_arrays(per, gtc, stems) for key, per in got.items()})
        arrays.update(load_arrays(arrays_file))
    missing = [arm for arm in arms if not any(key[0] == arm for key in arrays)]
    for arm in missing:
        target = cache / f"arrays_{run}+{arm}.npz"
        if not target.exists():
            keys = {(arm, scheme, rep): values for (scheme, rep), values in scores[arm].items()}
            got, g = run_pass(run, keys, stems, None, gtc is None, a.workers)
            if gtc is None:
                gtc = g
                gt_file.write_text(json.dumps(gtc))
            save_arrays(target, {key: frame_arrays(per, gtc, stems) for key, per in got.items()})
        arrays.update(load_arrays(target))
    cell = threshold_cell(stems, cache, a.workers, run)
    target = cache / f"arrays_{run}+{THRESHOLD_V7}.npz"
    if not target.exists():
        got, g = run_pass(run, {}, stems, (THRESHOLD_V7, *cell), gtc is None, a.workers)
        if gtc is None:
            gtc = g
            gt_file.write_text(json.dumps(gtc))
        save_arrays(target, {key: frame_arrays(per, gtc, stems) for key, per in got.items()})
    arrays.update(load_arrays(target))
    selected = set(arms) | {THRESHOLD_V7}
    arrays = {key: value for key, value in arrays.items() if key[0] in selected}
    assert all(len(value["host"]) == len(stems) for value in arrays.values())
    sources = {arm: f"runs/{config[0]} bank_split + {config[1] or 'descriptors only'}"
               + (f" without detector index {config[2]}" if config[2] is not None else "")
               for arm, config in arms.items()}
    sources[THRESHOLD_V7] = f"runs/{run} out-of-fold maps, threshold {cell[0]}, min area {cell[1]} px"
    checks = {}
    for arm in arms:
        for scheme, first in (("random", FIRST_RANDOM), ("group", FIRST_GROUP)):
            for tau in TAUS:
                reps = range(first, first + N_PARTITIONS)
                native = np.mean([pq(arrays[arm, scheme, rep, tau]["host"][:, :4].sum(0)) for rep in reps])
                bank = np.mean([pq(totals[arm][scheme, rep, tau].sum(0)) for rep in reps])
                checks[f"{arm} {scheme} {tau}"] = dict(masks=float(native), banks=float(bank))
    rows = table(arrays, len(stems), a.boot, sources, row_defs)
    write_csv(csv_path, rows)
    write_md(md_path, rows, checks, cell, row_defs, len(stems))
    print(json.dumps(checks, indent=1))


def main():
    """Score every arm, read every selection's masks, and write the table, the summary and the cache."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3, help="OpenMP threads of the scorer fits (one process)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=None, help="smoke run on the first N frames")
    ap.add_argument("--profile", choices=("all", "compact"), default="all",
                    help="compact: only the four-network system and its component rows; all: every study arm")
    ap.add_argument("--cache", type=Path, default=None,
                    help="cache directory to reuse or populate (default: under --out-dir)")
    a = ap.parse_args()
    if a.profile == "compact":
        compact_main(a)
        return
    tag = f"_n{a.frames}" if a.frames else ""
    cache = a.cache or a.out_dir / f"ablation_cache{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    scores, totals = {}, {}
    for arm in ARMS:
        scores[arm], totals[arm] = arm_scores(arm, cache, a.frames, a.threads)
    stems = sorted(scores["v6"]["random", FIRST_RANDOM])
    for arm in ARMS:
        assert sorted(scores[arm]["random", FIRST_RANDOM]) == stems, arm
    cell = threshold_cell(stems, cache, a.workers)
    gt_file, arrays = cache / "gt_chirality.json", {}
    gtc = json.loads(gt_file.read_text()) if gt_file.exists() else None
    for run in dict.fromkeys(r for r, _, _ in ARMS.values()):
        f = cache / f"arrays_{run}.npz"
        if f.exists():
            arrays.update(load_arrays(f))
        # arms added after the run's cache was written are read apart, each into its own file
        later = [arm for arm, (rr, _, _) in ARMS.items() if rr == run and f.exists()
                 and not any(k[0] == arm for k in arrays)]
        for arm, g_ in [(None, f)] if not f.exists() else [(arm, cache / f"arrays_{run}+{arm}.npz") for arm in later]:
            if not g_.exists():
                keys = {(x, s, r): v for x, (rr, _, _) in ARMS.items() if rr == run and arm in (None, x)
                        for (s, r), v in scores[x].items()}
                thr = (THRESHOLD, *cell) if run == ARMS["v6"][0] and arm is None else None
                got, g = run_pass(run, keys, stems, thr, gtc is None, a.workers)
                if gtc is None:
                    gtc = g
                    gt_file.write_text(json.dumps(gtc))
                save_arrays(g_, {key: frame_arrays(per, gtc, stems) for key, per in got.items()})
            arrays.update(load_arrays(g_))
    cell7 = threshold_cell(stems, cache, a.workers, THRESHOLDS[THRESHOLD_V7])
    f = cache / f"arrays_{THRESHOLDS[THRESHOLD_V7]}+{THRESHOLD_V7}.npz"
    if not f.exists():
        got, _ = run_pass(THRESHOLDS[THRESHOLD_V7], {}, stems, (THRESHOLD_V7, *cell7), False, a.workers)
        save_arrays(f, {key: frame_arrays(per, gtc, stems) for key, per in got.items()})
    arrays.update(load_arrays(f))
    sources = {arm: f"runs/{run} bank_split + {sets or 'no detector columns'}"
               + (" without " + ", ".join(SETS[d] for d in np.atleast_1d(drop)) if drop is not None else "")
               for arm, (run, sets, drop) in ARMS.items()}
    for arm, c in ((THRESHOLD, cell), (THRESHOLD_V7, cell7)):
        sources[arm] = f"runs/{THRESHOLDS[arm]} out-of-fold maps, threshold {c[0]}, min area {c[1]} px"
    rows = table(arrays, len(stems), a.boot, sources)
    checks = sanity(arrays, totals)
    write_csv(a.out_dir / f"ablation_metrics{tag}.csv", rows)
    write_md(a.out_dir / f"ablation_metrics{tag}.md", rows, checks, cell, n_frames=len(stems))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()

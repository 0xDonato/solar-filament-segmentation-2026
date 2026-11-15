"""An interval on every number the report states for the released system.

The released system is `ens_v5_inc5`. The scorer's out-of-fold utilities come from `ablation_metrics.py`'s cache
(cross-fitted on random partitions 15-19 and group partitions 10-14, never random 25-29 or group 20-24); each
partition's exact selection on the component tree is turned back into masks at every tau of 0.22-0.36 (step 0.02) and
read with the host's float32 counts, as `ablation_metrics.frame_pass` reads them. On the frames two or three annotators
drew, every annotator is read against every other with `reference.against_annotators`, as `judges_reference.py` reads
them.

- `report_intervals.csv`: the release at tau 0.26 and 0.30 in both schemes, on all 707 frames and on the 296 multiply
  annotated frames, beside the second annotator on the same frames and the release minus the annotator: PQ, SQ, RQ,
  matches, the pair IoU mean and lower 10% quantile, the Dice mean, splits (1:n), merges (n:1), 1:0 and 0:1 per
  comparison and as totals, the split and merge support shares and spine continuity; and the bands of the report's Fig.
  2 (the host's three panels: the 50-bin IoU and Dice shares of the overlapping pairs and the relations per
  comparison, mask degree 0, 1, 2, >=3 and filament degree 0, 1, 2, >=3), bin by bin (`fig2_bands.png`).
  Multiscale IoU is not here: `judges_reference.py` does not compute it.
- `threshold_curve.csv` / `.png`: plain threshold and 8-connected components on the release's blended maps across
  `oof_eval.py`'s thresholds, each at its best minimum area (pooled PQ, chosen in sample, which favours the
  baseline), beside the release's tree at both picks, with the paired difference from the tree at tau 0.26.
- `operating_point.csv` / `.png`: PQ, matches, 1:0, 0:1, splits and merges per record over tau 0.22-0.36,
  paired against tau 0.26, both schemes, with the second annotator's level on each count.
- the sentence on what the intervals exclude, in `report_intervals.md` with the summary.

Values are means over a scheme's five partitions. Intervals are 90%: frames resampled with replacement (2,000
draws, seed 0, `ablation_metrics.bootstrap_counts`) of the partitions' mean per-frame totals, and annotation groups
resampled whole (`cluster_low`, `cluster_high`; 2,000 draws, seed 0, as the group-cluster rows of the release rule
draw them). Differences are paired: both sides read on the same resampled frames.

Usage: uv run python scripts/report_intervals.py
       [--out-dir report] [--workers 3] [--boot 2000] [--frames N]
       (reads `ablation_cache/`, caches its own frame pass in `intervals_cache/`; --frames: a smoke run)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import csv
import json
from pathlib import Path

import numpy as np

import ablation_metrics as am
import oof_eval
from decide import sweep
from solarfil import metric, reference
from solarfil.data import Dataset, W
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import REPO, RUNS
from solarfil.rule import pq
from solarfil.tree import build_tree, select

RUN = os.environ.get("REPORT_RELEASE_RUN", "ens_v5_inc5")  # set by configure(); spawned workers inherit it
TAU_GRID = tuple(round(0.22 + 0.02 * i, 2) for i in range(8))
PICKS = (0.26, 0.30)
REF_TAU = 0.26
SCHEMES = {"random": am.FIRST_RANDOM, "group": am.FIRST_GROUP}
THR_GRID = [(t, m) for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6) for m in (0, 100, 200, 300, 500)]
DEG = 4  # degree bins 0, 1, 2, >= 3
FIELDS = {"host": 12, "rel": 4, "spine": 3, "ihist": 50, "dhist": 50, "gdeg": DEG, "pdeg": DEG}
PER_RECORD = ("matches", "missed_1_0", "spurious_0_1", "splits", "merges")
SANITY = {("random", 0.26): 0.4711, ("group", 0.26): 0.4691}
ARM = os.environ.get("REPORT_RELEASE_ARM", "v6")  # the release's arm in `ablation_metrics.py`'s cache (`scores_<ARM>_<scheme>_<rep>.npz`)


def configure(run: str | None = None, arm: str | None = None, sanity: str | None = None) -> None:
    """Read another release than v6 (e.g. v7: run ens_s0_win, arm v5-inst_maskrcnn); the defaults read v6. `sanity`:
    'random,group' PQ records at tau 0.26 for the sanity check."""
    global RUN, ARM, SANITY
    RUN, ARM = run or RUN, arm or ARM
    os.environ["REPORT_RELEASE_RUN"], os.environ["REPORT_RELEASE_ARM"] = RUN, ARM
    if sanity:
        r, g = (float(x) for x in sanity.split(","))
        SANITY = {("random", 0.26): r, ("group", 0.26): g}


def add_release_args(ap) -> None:
    """The options that point a report script at another release (all default to v6)."""
    ap.add_argument("--run", default=None, help="the release's run of out-of-fold maps (default ens_v5_inc5)")
    ap.add_argument("--arm", default=None, help="the release's arm in the ablation cache (default v6)")
    ap.add_argument("--cache", type=Path, default=None, help="the ablation cache to read (default <out-dir>/ablation_cache)")
    ap.add_argument("--sanity", default=None, help="'random,group' PQ records at tau 0.26 (default v6's)")


def apply_release_args(a) -> Path:
    """Configure the release from the parsed options; the ablation cache to read."""
    configure(a.run, a.arm, a.sanity)
    return a.cache or a.out_dir / "ablation_cache"
_G: dict = {}


# --- one frame -------------------------------------------------------------------------------------------------------

def degree_bins(c) -> np.ndarray:
    """A degree Counter as counts at 0, 1, 2 and >= 3."""
    return np.array([c.get(0, 0), c.get(1, 0), c.get(2, 0), sum(v for k, v in c.items() if k >= 3)], np.float64)


def report_fields(rep: metric.Report, rel: np.ndarray, spine: np.ndarray) -> tuple[dict, np.ndarray]:
    """The per-frame fields of one reading (a selection's or the annotators'), and its pair IoUs."""
    c = reference.legend_counts(rep)
    host = np.concatenate([reference.frame_row(rep), [len(rep.ious), c["1:0"], c["0:1"], c["splits"], c["merges"]]])
    iou, dice = rep.iou_values.astype(np.float64), rep.dice_values.astype(np.float64)
    return dict(host=host, rel=rel.astype(np.float64), spine=spine,
                ihist=np.histogram(iou, reference.HOST_BINS)[0].astype(np.float64),
                dhist=np.histogram(dice, reference.HOST_BINS)[0].astype(np.float64),
                gdeg=degree_bins(rep.gt_degree), pdeg=degree_bins(rep.pred_degree)), iou


def spine_samples(frame: reference.Frame) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixels of every annotated spine's samples (`reference.spine_samples`) and each sample's spine index."""
    flat, owner, k = [], [], 0
    for rec in frame.spines:
        for xy in rec:
            if len(xy):
                p = reference.spine_samples(xy, frame.shape)
                flat.append(p[:, 1] * W + p[:, 0])
                owner.append(np.full(len(p), k))
                k += 1
    if not flat:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(flat), np.concatenate(owner)


def spine_sums(hits: np.ndarray, owner: np.ndarray, cols: list[int]) -> np.ndarray:
    """[spines, sum of the share inside any mask, sum of the share inside the one mask holding most] of the
    selection `cols` (disjoint masks; `hits` is units x samples), as `reference.spine_cover` reads them."""
    n_sp = int(owner.max()) + 1 if owner.size else 0
    if n_sp == 0:
        return np.zeros(3)
    n = np.bincount(owner, minlength=n_sp).astype(np.float64)
    if not cols:
        return np.array([n_sp, 0.0, 0.0])
    per = np.stack([np.bincount(owner, weights=hits[j], minlength=n_sp) for j in cols])
    return np.array([n_sp, (per.sum(0) / n).sum(), (per.max(0) / n).sum()])


def selection_fields(stem, gt, counts, areas, cols, hits, owner) -> tuple[dict, np.ndarray]:
    """One selection's fields on a frame: the host's reading (as `ablation_metrics.selection_reading`), the split
    and merge support against the frame's other records (`reference.relations`) and spine continuity."""
    cols = np.asarray(cols, int)
    rep, ious = metric.Report(), []
    for k, rec in enumerate(gt.records):
        a_gt = np.array([len(i) for i in rec], np.int64)
        iou, dice = metric.iou_dice(counts[k][:, cols], a_gt, areas[cols])
        rep.add_matrices(stem, iou, dice)
        ious.append(iou)
    rel = sum((reference.relations(ious[k], [ious[j] for j in range(len(ious)) if j != k])
               for k in range(len(ious))), np.zeros(4, np.int64))
    return report_fields(rep, rel, spine_sums(hits, owner, list(cols)))


def annotator_fields(frame: reference.Frame) -> tuple[dict, np.ndarray]:
    """The annotators' fields on a multiply annotated frame (every ordered pair of records)."""
    c = reference.against_annotators(frame)
    s = c.spines
    return report_fields(c.report, c.relations, np.array([len(s), s["inside_any"].sum(), s["inside_one"].sum()]))


def frame_job(job):
    """One frame: every partition's selection at every tau of the grid, the annotators (multiply annotated frames)
    and plain threshold and components at every cell of the grid."""
    stem, png, sel_scores = job
    ds = _G.setdefault("ds", Dataset())
    obs = ds.observations[stem]
    gt, frame = metric.GTIndex(obs), reference.Frame.of(obs)
    p, _ = read_map_png(png)
    tree = build_tree(p, splits=True)
    picks = {}
    for key, s in sel_scores.items():
        assert tree.n == len(s), (stem, key, tree.n, len(s))
        for tau in TAU_GRID:
            picks[(*key, tau)] = select(tree, s, tau) if tree.n else []
    nodes = sorted(set().union(*picks.values())) if picks else []
    units = [am.node_pixels(tree, v) for v in nodes]
    col = {v: j for j, v in enumerate(nodes)}
    counts = am.unit_counts(units, gt)
    areas = np.array([len(u) for u in units], np.int64)
    flat, owner = spine_samples(frame)
    hits = np.array([np.isin(flat, u) for u in units], np.float64).reshape(len(units), len(flat))
    out = {key: selection_fields(stem, gt, counts, areas, [col[v] for v in vs], hits, owner)
           for key, vs in picks.items()}
    ann = annotator_fields(frame) if obs.n_records > 1 else None
    _, thr, _ = oof_eval.score_one((str(RUNS / RUN), stem, THR_GRID, True, 0.0, 0.0, False))
    return stem, out, ann, np.stack([thr[g] for g in THR_GRID])


# --- per-frame arrays ------------------------------------------------------------------------------------------------

def load_scores(cache: Path) -> dict:
    """{(scheme, rep): {stem: utilities}} of the release from `ablation_metrics.py`'s cache (read partitions only)."""
    out = {}
    for scheme, first in SCHEMES.items():
        for rep in range(first, first + am.N_PARTITIONS):
            with np.load(cache / f"scores_{ARM}_{scheme}_{rep}.npz") as z:
                out[scheme, rep] = {s: z[s] for s in z.files}
    return out


def stack_readings(per_stem: dict, stems: list[str]) -> dict:
    """{field: (frames, k)} plus the pair IoUs and their frame index; frames without a reading get zeros."""
    out = {f: np.zeros((len(stems), k)) for f, k in FIELDS.items()}
    ious, fidx = [], []
    for i, s in enumerate(stems):
        if s in per_stem:
            d, iou = per_stem[s]
            for f in FIELDS:
                out[f][i] = d[f]
            ious.append(iou)
            fidx.append(np.full(len(iou), i))
    out["iou"] = np.concatenate(ious) if ious else np.zeros(0)
    out["iou_frame"] = np.concatenate(fidx) if fidx else np.zeros(0, int)
    return out


def run_pass(stems: list[str], scores: dict, workers: int, cache: Path, tag: str) -> tuple[dict, dict, np.ndarray]:
    """Every frame through `frame_job` (cached in `cache`): {(scheme, rep, tau): arrays}, the annotators' arrays, and
    the threshold grid's per-frame [TP, FP, FN, IoU sum] (frames, cells, 4)."""
    f = cache / f"intervals_pass{tag}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            return z["sel"].item(), z["ann"].item(), z["thr"]
    maps = {q.stem: q for q in (RUNS / RUN).glob("fold*/oof/*.png")}
    jobs = [(s, str(maps[s]), {k: v[s] for k, v in scores.items()}) for s in stems]
    sel, ann, thr = {}, {}, {}
    with pool(workers, spawn=True) as w:
        for i, (stem, out, a, t) in enumerate(w.imap_unordered(frame_job, jobs, chunksize=2)):
            for key, r in out.items():
                sel.setdefault(key, {})[stem] = r
            if a is not None:
                ann[stem] = a
            thr[stem] = t
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} frames", flush=True)
    sel = {k: stack_readings(v, stems) for k, v in sel.items()}
    ann, thr = stack_readings(ann, stems), np.stack([thr[s] for s in stems])
    np.savez(f, sel=np.array(sel, dtype=object), ann=np.array(ann, dtype=object), thr=thr)
    return sel, ann, thr


# --- measures and resampling -----------------------------------------------------------------------------------------

def measures(S: dict) -> dict:
    """Every measure from weighted field sums (leading axis: draws)."""
    h = S["host"]
    comps = h[..., 7]
    tp, fp, fn = h[..., 0], h[..., 1], h[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = am.host_measures(h)
        out.update(sq=h[..., 3] / tp, rq=tp / (tp + 0.5 * fp + 0.5 * fn), matches=tp / comps,
                   comparisons=comps, matches_total=tp, missed_1_0_total=h[..., 8], spurious_0_1_total=h[..., 9],
                   splits_total=h[..., 10], merges_total=h[..., 11],
                   splits_supported_share=S["rel"][..., 1] / S["rel"][..., 0],
                   merges_supported_share=S["rel"][..., 3] / S["rel"][..., 2],
                   spine_inside_any=S["spine"][..., 1] / S["spine"][..., 0],
                   spine_inside_one=S["spine"][..., 2] / S["spine"][..., 0])
        for name, f in (("iou", "ihist"), ("dice", "dhist")):
            share = S[f] / S[f].sum(-1, keepdims=True)
            out.update({f"fig2_{name}_bin{b:02d}": share[..., b] for b in range(50)})
        for name, f in (("mask_degree", "pdeg"), ("filament_degree", "gdeg")):
            out.update({f"fig2_{name}_{d}": S[f][..., i] / comps for i, d in enumerate(("0", "1", "2", "ge3"))})
    return out


def weights(stems: list[str], groups: list[str], panel: np.ndarray, n_boot: int) -> dict:
    """Weight matrices over all frames for one panel (a boolean frame mask): the point (1, n), the frame
    bootstrap (draws, n; `ablation_metrics.bootstrap_counts` on the panel's frames) and the group-cluster
    bootstrap (draws, n; groups of the panel drawn whole, seed 0)."""
    n = len(stems)
    idx = np.flatnonzero(panel)
    C = np.zeros((n_boot, n))
    C[:, idx] = am.bootstrap_counts(len(idx), n_boot)
    gs = sorted({groups[i] for i in idx})
    gi = np.array([gs.index(groups[i]) for i in idx])
    draw = np.random.default_rng(0).integers(0, len(gs), (n_boot, len(gs)))
    cnt = np.stack([np.bincount(d, minlength=len(gs)) for d in draw])
    K = np.zeros((n_boot, n))
    K[:, idx] = cnt[:, gi]
    return dict(point=panel[None].astype(np.float64), frame=C, cluster=K)


def reading(reps: list[dict], Wt: dict) -> dict:
    """One series (a scheme's partitions, or the annotators): per-partition point values, and the measures of
    the partitions' mean per-frame totals under each bootstrap (the 10% quantile: the partitions' mean)."""
    point = {}
    for r in reps:
        m = measures({f: Wt["point"] @ r[f] for f in FIELDS})
        m["pair_iou_p10"] = am.p10(r["iou"], r["iou_frame"], Wt["point"])
        for k, v in m.items():
            point.setdefault(k, []).append(float(np.ravel(v)[0]))
    mean = {f: np.mean([r[f] for r in reps], 0) for f in FIELDS}
    out = dict(point={k: np.array(v) for k, v in point.items()})
    for b in ("frame", "cluster"):
        m = measures({f: Wt[b] @ mean[f] for f in FIELDS})
        m["pair_iou_p10"] = np.mean([am.p10(r["iou"], r["iou_frame"], Wt[b]) for r in reps], 0)
        out[b] = m
    return out


def ci(x: np.ndarray) -> tuple[float, float]:
    """The 90% interval of bootstrap values (NaN draws ignored)."""
    lo, hi = np.nanpercentile(x, [5, 95])
    return float(lo), float(hi)


def level_rows(row: str, tau, scheme: str, a: dict, keys) -> list[dict]:
    """Long-form rows of one series' levels."""
    return [dict(row=row, tau=tau, scheme=scheme, measure=k, value=float(np.mean(a["point"][k])),
                 ci=ci(a["frame"][k]), cl=ci(a["cluster"][k])) for k in keys]


def diff_rows(row: str, tau, scheme: str, a: dict, b: dict, keys) -> list[dict]:
    """Long-form rows of a paired difference a - b (partition by partition when both have partitions)."""
    out = []
    for k in keys:
        pa, pb = a["point"][k], b["point"][k]
        v = float((pa - pb).mean()) if len(pa) == len(pb) else float(pa.mean() - pb.mean())
        out.append(dict(row=row, tau=tau, scheme=scheme, measure=f"d_{k}", value=v,
                        ci=ci(a["frame"][k] - b["frame"][k]), cl=ci(a["cluster"][k] - b["cluster"][k])))
    return out


# --- the three tables -------------------------------------------------------------------------------------------------

def series(sel: dict, scheme: str, tau: float) -> list[dict]:
    """The scheme's five partitions' arrays at tau."""
    return [sel[scheme, r, tau] for r in range(SCHEMES[scheme], SCHEMES[scheme] + am.N_PARTITIONS)]


def intervals_table(sel: dict, ann: dict, W_all: dict, W_multi: dict) -> tuple[list[dict], dict]:
    """The release's levels at both picks on both panels, the annotator's, and the release minus the annotator."""
    rows, got = [], {}
    keys = None
    human = reading([ann], W_multi)
    for tau in PICKS:
        for scheme in SCHEMES:
            for panel, Wt in (("all frames", W_all), ("multiply annotated frames", W_multi)):
                a = reading(series(sel, scheme, tau), Wt)
                keys = keys or list(a["point"])
                got[tau, scheme, panel] = a
                rows += level_rows(f"v6 ({panel})", tau, scheme, a, keys)
            comparable = [k for k in keys if not k.endswith("_total") and k != "comparisons"]
            rows += diff_rows("v6 minus second annotator (multiply annotated frames)", tau, scheme,
                              got[tau, scheme, "multiply annotated frames"], human, comparable)
    rows += level_rows("second annotator (multiply annotated frames)", "", "", human, keys)
    got["human"] = human
    return rows, got


def operating_table(sel: dict, human: dict, W_all: dict, W_multi: dict) -> list[dict]:
    """The per-record counts over the tau grid, paired against tau 0.26, on both panels, with the
    annotator's level and the release minus the annotator on the multiply annotated frames."""
    keys = ("pq",) + PER_RECORD
    rows = level_rows("second annotator (multiply annotated frames)", "", "", human, keys)
    for scheme in SCHEMES:
        for panel, Wt in (("all frames", W_all), ("multiply annotated frames", W_multi)):
            got = {tau: reading(series(sel, scheme, tau), Wt) for tau in TAU_GRID}
            for tau in TAU_GRID:
                rows += level_rows(f"v6 ({panel})", tau, scheme, got[tau], keys)
                if tau != REF_TAU:
                    rows += diff_rows(f"v6 ({panel}) minus tau {REF_TAU}", tau, scheme, got[tau], got[REF_TAU], keys)
                if panel != "all frames":
                    rows += diff_rows(f"v6 minus second annotator ({panel})", tau, scheme, got[tau], human, keys)
    return rows


def threshold_table(thr: np.ndarray, sel: dict, W_all: dict) -> tuple[list[dict], list]:
    """PQ of threshold and components at each threshold's best minimum area (in sample), and of the release's tree
    at both picks, with the paired difference from the tree at tau 0.26 in each scheme."""
    def pq_reading(per_frame: list[np.ndarray]) -> dict:
        """PQ of per-frame [TP, FP, FN, IoU sum] (one array per partition) with both bootstraps."""
        mean = np.mean(per_frame, 0)
        return dict(point=dict(pq=np.array([pq(x.sum(0)) for x in per_frame])),
                    frame=dict(pq=pq(np.moveaxis(W_all["frame"] @ mean, -1, 0))),
                    cluster=dict(pq=pq(np.moveaxis(W_all["cluster"] @ mean, -1, 0))))

    rows, best = [], []
    tree = {(s, tau): pq_reading([x["host"][:, :4] for x in series(sel, s, tau)]) for s in SCHEMES for tau in PICKS}
    for (s, tau), r in tree.items():
        rows += level_rows("v6 tree", tau, s, r, ["pq"])
    for t in sorted({g[0] for g in THR_GRID}):
        cells = [i for i, g in enumerate(THR_GRID) if g[0] == t]
        i = max(cells, key=lambda c: pq(thr[:, c].sum(0)))
        r = pq_reading([thr[:, i]])
        name = f"threshold+components (min area {THR_GRID[i][1]} px)"
        best.append((t, THR_GRID[i][1], r))
        rows += level_rows(name, t, "in-sample cell", r, ["pq"])
        for s in SCHEMES:
            rows += diff_rows(f"{name} minus v6 tree tau {REF_TAU}", t, s, r, tree[s, REF_TAU], ["pq"])
    return rows, (best, tree)


# --- outputs ----------------------------------------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    """Long form: row, tau, scheme, measure, value, ci_low, ci_high, cluster_low, cluster_high."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["row", "tau", "scheme", "measure", "value", "ci_low", "ci_high", "cluster_low", "cluster_high"])
        for r in rows:
            w.writerow([r["row"], r["tau"], r["scheme"], r["measure"], *(f"{x:.6f}" for x in
                                                                         (r["value"], *r["ci"], *r["cl"]))])


def threshold_figure(best: list, tree: dict, path: Path) -> None:
    """PQ against the threshold for threshold and components, with the release's tree at both picks as horizontal
    bands."""
    from solarfil.plots import COLUMN_WIDTH, MODEL_STYLES, PAPER, plt
    t = np.array([b[0] for b in best])
    v = np.array([b[2]["point"]["pq"][0] for b in best])
    lo, hi = np.array([ci(b[2]["frame"]["pq"]) for b in best]).T
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8), layout="constrained")
        ax.errorbar(t, v, yerr=[v - lo, hi - v], fmt="o-", ms=3, lw=0.9, capsize=1.5, color="#555555",
                    label="threshold + components (best min. area)")
        for (s, tau), style in zip(tree, MODEL_STYLES):
            r = tree[s, tau]
            c0, c1 = ci(r["frame"]["pq"])
            ax.axhline(r["point"]["pq"].mean(), color=style["color"], ls=style["ls"], lw=0.8,
                       label=f"tree, tau {tau:.2f}, {s}")
            ax.axhspan(c0, c1, color=style["color"], alpha=0.08, lw=0)
        ax.set(xlabel="threshold", ylabel="PQ (out of fold, 707 frames)")
        ax.grid(axis="y")
        fig.legend(frameon=False, fontsize=5.5, loc="outside lower center", ncol=2)
        fig.savefig(path)
    plt.close(fig)


def operating_figure(rows: list[dict], path: Path) -> None:
    """Six panels over tau: PQ and matches on all frames, the four counts per record on the multiply annotated
    frames beside the second annotator; both schemes, both picks marked."""
    from solarfil.plots import ANNOTATOR_STYLE, FULL_WIDTH, MODEL_STYLES, PAPER, plt
    look = {(r["row"], r["tau"], r["scheme"], r["measure"]): r for r in rows}
    panels = [("pq", "all frames"), ("matches", "all frames")] + [(m, "multiply annotated frames")
                                                                  for m in PER_RECORD[1:]]
    with plt.rc_context(PAPER):
        fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.6), layout="constrained", sharex=True)
        for ax, (m, panel) in zip(axes.ravel(), panels):
            for s, style in zip(SCHEMES, MODEL_STYLES):
                rr = [look[f"v6 ({panel})", t, s, m] for t in TAU_GRID]
                v = np.array([r["value"] for r in rr])
                ax.plot(TAU_GRID, v, color=style["color"], ls=style["ls"], marker=style["marker"], ms=2.5, lw=0.9,
                        label=s)
                ax.fill_between(TAU_GRID, [r["ci"][0] for r in rr], [r["ci"][1] for r in rr],
                                color=style["color"], alpha=0.12, lw=0)
            if panel != "all frames":
                h = look["second annotator (multiply annotated frames)", "", "", m]
                ax.axhline(h["value"], color=ANNOTATOR_STYLE["color"], ls=":", lw=0.9, label="second annotator")
                ax.axhspan(*h["ci"], color=ANNOTATOR_STYLE["color"], alpha=0.08, lw=0)
            for tau in PICKS:
                ax.axvline(tau, color="#999999", lw=0.5, ls="--")
            unit = "" if m == "pq" else " per record"
            ax.set_title(f"{m.replace('_', ' ')}{unit} ({'707' if panel == 'all frames' else '296'} frames)")
            ax.grid(axis="y")
        axes[0, 0].legend(frameon=False)
        axes[0, 2].legend(frameon=False)
        for ax in axes[1]:
            ax.set_xlabel("tau")
        fig.savefig(path)
    plt.close(fig)


def fig2_bands(got: dict, path: Path) -> None:
    """The report's Fig. 2 panels on the multiply annotated frames with 90% frame-bootstrap bands, bin by bin:
    the release at both picks (random partitions) and the second annotator."""
    from solarfil.plots import ANNOTATOR_STYLE, DEGREE_TICKS, FULL_WIDTH, MODEL_STYLES, PAPER, plt
    panel = "multiply annotated frames"
    ser = [(f"v6, tau {t:.2f}", got[t, "random", panel], st) for t, st in zip(PICKS, MODEL_STYLES)]
    ser.append(("second annotator", got["human"], ANNOTATOR_STYLE))
    x = (reference.HOST_BINS[:-1] + reference.HOST_BINS[1:]) / 2
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.3), layout="constrained")
        for a, name in ((ax[0], "iou"), (ax[1], "dice")):
            for label, r, st in ser:
                k = [f"fig2_{name}_bin{b:02d}" for b in range(50)]
                v = np.array([np.mean(r["point"][q]) for q in k])
                lo, hi = np.array([ci(r["frame"][q]) for q in k]).T
                a.step(x, v, where="mid", color=st["color"], ls=st["ls"], lw=0.8, label=label)
                a.fill_between(x, lo, hi, step="mid", color=st["color"], alpha=0.15, lw=0)
            a.set(xlabel="IoU" if name == "iou" else "Dice", ylabel="share of pairs", xlim=(0, 1))
            a.grid(axis="y")
        ax[0].legend(frameon=False)
        keys = ([f"fig2_mask_degree_{d}" for d in ("ge3", "2", "1", "0")]
                + [f"fig2_filament_degree_{d}" for d in ("0", "1", "2", "ge3")])
        pos = np.arange(len(keys))
        for i, (_, r, st) in enumerate(ser):
            v = np.array([np.mean(r["point"][q]) for q in keys])
            lo, hi = np.array([ci(r["frame"][q]) for q in keys]).T
            ax[2].bar(pos + (i - 1) * 0.27, v, 0.27, yerr=[v - lo, hi - v], color=st["color"], capsize=1,
                      error_kw=dict(lw=0.6))
        ax[2].set_xticks(pos, DEGREE_TICKS, rotation=45)
        ax[2].set(yscale="log", ylabel="per comparison")
        ax[2].grid(axis="y")
        fig.savefig(path)
    plt.close(fig)


def sanity(sel: dict, totals: dict, thr: np.ndarray, cache: Path, n_frames: int | None) -> dict:
    """The release's PQ from the masks against the banks at every tau, the record's 0.4711 / 0.4691 at tau 0.26, the
    masks against `ablation_metrics.py`'s cached readings at both picks, and the threshold grid against its cell
    file."""
    out = {"masks_vs_banks": {}, "record": {}}
    for scheme, first in SCHEMES.items():
        for tau in TAU_GRID:
            reps = range(first, first + am.N_PARTITIONS)
            m = np.mean([pq(sel[scheme, r, tau]["host"][:, :4].sum(0)) for r in reps])
            b = np.mean([pq(totals[scheme, r][tau].sum(0)) for r in reps])
            out["masks_vs_banks"][f"{scheme} {tau}"] = [round(float(m), 5), round(float(b), 5)]
            if (scheme, tau) in SANITY:  # the record is the banks' reading; masks may sit 0.0001 above
                out["record"][f"{scheme} {tau}"] = dict(masks=round(float(m), 5), banks=round(float(b), 5),
                                                        record=SANITY[scheme, tau],
                                                        ok=bool(round(float(b), 4) == round(SANITY[scheme, tau], 4)
                                                                and abs(m - b) < 1e-4))
    if n_frames is None:
        named = cache / (f"arrays_{RUN}.npz" if ARM == "v6" else f"arrays_{RUN}+{ARM}.npz")
        # A single-arm export uses arrays_<run>.npz. Its internal arm keys
        # below still have to match; a differently labelled cache cannot pass.
        reference = named if named.exists() else cache / f"arrays_{RUN}.npz"
        arr = am.load_arrays(reference)
        out["ablation_reference_arrays"] = dict(path=str(reference),
            sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
        diff = max(float(np.abs(arr[ARM, s, r, tau]["host"] - sel[s, r, tau]["host"]).max())
                   for s, f in SCHEMES.items() for r in range(f, f + am.N_PARTITIONS) for tau in PICKS)
        out["max_abs_diff_vs_ablation_cache_host_rows"] = diff
        cell = cache / ("threshold_cell.json" if ARM == "v6" else f"threshold_cell_{RUN}.json")
        grid = json.loads(cell.read_text())["grid"]
        out["max_abs_diff_vs_threshold_cell_grid"] = max(
            abs(round(float(pq(thr[:, i].sum(0))), 5) - grid[f"{t}_{m}"]) for i, (t, m) in enumerate(THR_GRID))
    return out


def write_md(path: Path, rows: list[dict], op: list[dict], thr_rows: list[dict], checks: dict) -> None:
    """The summary with the drafted sentences."""
    look = {(r["row"], r["tau"], r["scheme"], r["measure"]): r for r in rows + op + thr_rows}

    def f(r, d=4):
        """value [frame CI] {cluster CI}."""
        return (f"{r['value']:.{d}f} [{r['ci'][0]:.{d}f}, {r['ci'][1]:.{d}f}] "
                f"{{{r['cl'][0]:.{d}f}, {r['cl'][1]:.{d}f}}}")

    keys = ("pq", "sq", "rq", "matches", "pair_iou_mean", "pair_iou_p10", "dice_mean", "splits", "merges",
            "missed_1_0", "spurious_0_1", "matches_total", "splits_total", "merges_total", "missed_1_0_total",
            "spurious_0_1_total", "splits_supported_share", "merges_supported_share", "spine_inside_any",
            "spine_inside_one")
    L = ["# Report intervals for release v6 (items 4.3, 4.6, 4.7, 4.15)", "",
         "Value [90% frame-bootstrap interval] {90% annotation-group cluster interval}; 2,000 draws, seed 0. "
         "Scorer cross-fitted on random partitions 15-19 and group partitions 10-14; values are the mean of a "
         "scheme's five partitions. Script: `scripts/report_intervals.py`.", ""]
    for panel, ann in (("all frames", False), ("multiply annotated frames", True)):
        L += [f"## (4.3) v6 on {panel}", "", "| measure | " + " | ".join(
            f"tau {t} {s}" for t in PICKS for s in SCHEMES) + (" | second annotator |" if ann else " |"),
              "|---" * (5 + ann) + "|"]
        for k in keys:
            cells = [f(look[f"v6 ({panel})", t, s, k]) for t in PICKS for s in SCHEMES]
            if ann:
                cells.append(f(look["second annotator (multiply annotated frames)", "", "", k]))
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Multiscale IoU: not computed by `judges_reference.py`, so no interval here. Fig. 2 bands bin by bin: "
          "`report_intervals.csv`, measures `fig2_*`, and `fig2_bands.png`.", "",
          "## (4.7) operating points, per record (all frames; counts also on the multiply annotated frames)", "",
          "| tau | scheme | PQ | matches | 1:0 | 0:1 | splits | merges |", "|---" * 8 + "|"]
    for s in SCHEMES:
        for t in TAU_GRID:
            L.append(f"| {t} | {s} | " + " | ".join(f"{look['v6 (all frames)', t, s, k]['value']:.4f}"
                                                   for k in ("pq",) + PER_RECORD) + " |")
    L.append("| human | multi | | " + " | ".join(
        f"{look['second annotator (multiply annotated frames)', '', '', k]['value']:.4f}" for k in PER_RECORD) + " |")
    L += ["", "Paired differences from tau 0.26 and against the annotator: `operating_point.csv`.", ""]
    s = []
    for sch in SCHEMES:
        h = look["second annotator (multiply annotated frames)", "", "", "missed_1_0"]
        m = look["v6 (multiply annotated frames)", 0.3, sch, "missed_1_0"]
        d = look["v6 minus second annotator (multiply annotated frames)", 0.3, sch, "d_missed_1_0"]
        s.append((sch, h, m, d))
    sch, h, m, d = s[0]
    g = s[1][3]
    sure = [n for n, x in (("random-partition frame", d["ci"]), ("group-partition frame", g["ci"]),
                           ("random-partition cluster", d["cl"]), ("group-partition cluster", g["cl"])) if x[0] > 0]
    verdict = f"trails a second annotator on missed filaments (above zero in the {', '.join(sure) or 'no'} " \
              f"interval{'s' if len(sure) != 1 else ''} of four)"
    L += ["## Drafted sentences", "",
          f"- (4.7) On the 296 multiply annotated frames, tau 0.30 {verdict}: "
          f"{m['value']:.3f} 1:0 per record against {h['value']:.3f} per ordered pair, "
          f"{d['value']:+.3f} (90% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}; group clusters {d['cl'][0]:+.3f} to "
          f"{d['cl'][1]:+.3f}) with random partitions, {s[1][3]['value']:+.3f} ({s[1][3]['ci'][0]:+.3f} to "
          f"{s[1][3]['ci'][1]:+.3f}) with group partitions.",
          "- (4.15) These intervals resample the frames (and, in braces, whole annotation groups) of one fitted "
          "pipeline: they are conditional on the trained networks, detectors, blend weights and scorer fits, and "
          "include neither retraining variance nor the search over configurations that chose them, so they "
          "understate the spread a fresh rebuild would show.", "",
          "## (4.6) threshold curve", "", "| threshold | min area | PQ | minus tree tau 0.26 random | group |",
          "|---" * 5 + "|"]
    for r in thr_rows:
        if r["row"].startswith("threshold+components") and "minus" not in r["row"]:
            a = r["row"].split("min area ")[1].rstrip(")")
            dr = look[f"{r['row']} minus v6 tree tau {REF_TAU}", r["tau"], "random", "d_pq"]
            dg = look[f"{r['row']} minus v6 tree tau {REF_TAU}", r["tau"], "group", "d_pq"]
            L.append(f"| {r['tau']} | {a} | {f(r)} | {f(dr)} | {f(dg)} |")
    L += ["", "v6 tree: " + "; ".join(f"tau {t} {s} {f(look['v6 tree', t, s, 'pq'])}"
                                      for t in PICKS for s in SCHEMES), "",
          "## Sanity checks", "", "```", json.dumps(checks, indent=1), "```", ""]
    path.write_text("\n".join(L))


def main():
    """Read every selection and the annotators, then write the three tables, the figures and the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=None, help="smoke run on the first N frames")
    add_release_args(ap)
    a = ap.parse_args()
    tag = f"_n{a.frames}" if a.frames else ""
    cache = apply_release_args(a)
    scores = load_scores(cache)
    stems = sorted(scores["random", am.FIRST_RANDOM])[:a.frames]
    banks = {b.stem: b for b in am.arm_banks(ARM, None)}
    totals = {k: {tau: rows for tau, (_, rows) in sweep([banks[s] for s in stems], [v[s] for s in stems],
                                                        taus=TAU_GRID).items()} for k, v in scores.items()}
    own = a.out_dir / "intervals_cache"
    own.mkdir(parents=True, exist_ok=True)
    sel, ann, thr = run_pass(stems, scores, a.workers, own, tag)
    obs = Dataset().observations
    groups = [obs[s].group for s in stems]
    multi = np.array([obs[s].n_records > 1 for s in stems])
    W_all = weights(stems, groups, np.ones(len(stems), bool), a.boot)
    W_multi = weights(stems, groups, multi, a.boot)
    rows, got = intervals_table(sel, ann, W_all, W_multi)
    op = operating_table(sel, got["human"], W_all, W_multi)
    thr_rows, (best, tree) = threshold_table(thr, sel, W_all)
    checks = sanity(sel, totals, thr, cache, a.frames)
    out = a.out_dir
    write_csv(out / f"report_intervals{tag}.csv", rows)
    write_csv(out / f"operating_point{tag}.csv", op)
    write_csv(out / f"threshold_curve{tag}.csv", thr_rows)
    threshold_figure(best, tree, out / f"threshold_curve{tag}.png")
    operating_figure(op, out / f"operating_point{tag}.png")
    fig2_bands(got, out / f"fig2_bands{tag}.png")
    write_md(out / f"report_intervals{tag}.md", rows, op, thr_rows, checks)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()

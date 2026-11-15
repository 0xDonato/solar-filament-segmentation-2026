"""The released system against the annotators on common objects, and by GONG station
and by epoch.

The released system is `ens_v5_inc5`. Its out-of-fold masks are the exact tree selections of the hurdle scorer
cross-fitted on random partitions 15-19 and group partitions 10-14 (`ablation_metrics.py`'s cache; never random 25-29 or
group 20-24) at tau 0.26 and 0.30. Values are means over a scheme's five partitions. Intervals are 90%: frames resampled
with replacement and, in braces, annotation groups resampled whole (`report_intervals.weights`, 2,000 draws, seed 0),
both on the partitions' mean per-frame sums; differences are paired on the same draws.

`common_objects.csv` (one row per object and held-out annotator) and `common_objects_summary.csv`. On the
training frames with three annotator records, two filaments of different records are *the same object* when each is
the other's best match by IoU and that IoU is above 0.2 (a link test, not a quality test). For a held-out annotator h
the two other records p and q define the objects: every mutually linked pair (p, q) is a *two-annotator object*, and
it is an *all-three object* when h's best match of p's filament and of q's filament is one filament linked to both.
Each object is read against two references that h did not draw: each of p's and q's outlines on its own (pairwise,
the host's unit: one record against another; two comparisons per object) and their consensus, the pixels both drew
(one comparison). The candidates are the release's selected masks and h's own filaments; each candidate's mask with the
best IoU against the reference stands for it. *Detected* is IoU > 0.5 (the host's match threshold); *outline given
detection* is the mean IoU and Dice over the detected comparisons, and *outline when both detect* reads the two
candidates on the same comparisons, so neither side's outline is flattered by its own detections. Every all-three
object is a two-annotator object; on the all-three set h has drawn the object by construction (IoU > 0.2), on the
two-annotator set h may have missed it.

`station_epoch.csv`. The station is the stem's 15th character (`Observation.station`: B, C, L, M, T, U) and
the time its first 14 digits (`Observation.time`); epochs are four blocks of about equal frame count (2011-2012,
2013-2014, 2015-2016, 2017-2022). Per cell, out of fold on all 707 frames: frames, records, PQ, matches, 1:0 and 0:1
per record (`report_intervals.py`'s cached frame pass), the cell minus the other frames, and the cell minus its
annotation-group expectation (each frame replaced by the mean per-frame counts of its group's frames outside the
cell), beside the cell's group and record-count mix.

Outputs under --out-dir: `common_objects.csv`, `common_objects_summary.csv`, `station_epoch.csv`,
`report_tables.md` (both tables summarised, the sanity checks and one drafted sentence for each item), and the frame
pass of the common-object table in `objects_cache/`.

Usage: uv run python scripts/report_objects_epochs.py
       [--out-dir report] [--workers 3] [--boot 2000] [--frames N]
       (--frames: a smoke run of the common-object table on the first N three-record frames, written with a suffix)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ablation_metrics as am
import report_intervals as ri
from solarfil import metric
from solarfil.data import Dataset
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import REPO, RUNS
from solarfil.rule import pq
from solarfil.tree import build_tree, select

RUN = ri.RUN
PICKS = ri.PICKS
SCHEMES = ri.SCHEMES
LINK = 0.2  # IoU above which two annotators' mutually best filaments are one object
MATCH = 0.5  # the host's match threshold: detected
EPOCHS = ((2011, 2012), (2013, 2014), (2015, 2016), (2017, 2022))
STATIONS = "BCLMTU"
REFS = ("pairwise", "consensus")
_G: dict = {}


# --- one frame --------------------------------------------------------------------------------------------------

def keys() -> list[tuple]:
    """The release's selections read: (scheme, partition, tau)."""
    return [(s, r, t) for t in PICKS for s, f in SCHEMES.items() for r in range(f, f + am.N_PARTITIONS)]


def pair_iou(inter: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice from intersections (rows x cols) and the two sides' areas."""
    a, b = a[:, None].astype(np.float64), b[None, :].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(inter / (a + b - inter)), np.nan_to_num(2 * inter / (a + b))


def mutual_links(iou: np.ndarray) -> list[tuple[int, int]]:
    """(i, j) where i and j are each other's best match and the IoU is above LINK."""
    if iou.size == 0:
        return []
    return [(i, j) for i, j in enumerate(iou.argmax(1)) if iou[i, j] > LINK and iou[:, j].argmax() == i]


def best(iou: np.ndarray, dice: np.ndarray, cols: list[int]) -> tuple[float, float]:
    """IoU and Dice of the candidate column with the best IoU (0, 0 when there is none)."""
    if not cols:
        return 0.0, 0.0
    j = cols[int(np.argmax(iou[cols]))]
    return float(iou[j]), float(dice[j])


def objects(recs: list[list[np.ndarray]]) -> list[dict]:
    """The frame's objects per held-out record h: the (p, q) links, whether h drew it too, and the three references
    (p's outline, q's outline, the pixels both drew) as flat pixel indices."""
    fil = [f for rec in recs for f in rec]
    off = np.cumsum([0] + [len(r) for r in recs])
    inter = am.unit_counts(fil, SimpleNamespace(records=[fil]))[0]
    area = np.array([len(f) for f in fil])
    iou_all, _ = pair_iou(inter, area, area)

    def block(p: int, q: int) -> np.ndarray:
        """IoU of record p's filaments against record q's."""
        return iou_all[off[p]:off[p + 1], off[q]:off[q + 1]]

    out = []
    for h in range(3):
        p, q = (r for r in range(3) if r != h)
        hp, hq = dict(mutual_links(block(p, h))), dict(mutual_links(block(q, h)))
        for i, j in mutual_links(block(p, q)):
            k = hp.get(i)
            three = k is not None and hq.get(j) == k
            ref = [recs[p][i], recs[q][j], np.intersect1d(recs[p][i], recs[q][j], assume_unique=True)]
            out.append(dict(h=h, p=p, q=q, i=i, j=j, three=three, refs=ref, h_cols=list(range(off[h], off[h + 1]))))
    return out


def frame_objects(job) -> tuple[str, list[dict], dict]:
    """One three-record frame: every object with, per reference, h's best IoU and Dice and each release selection's;
    and each selection's host row on the frame (checked against `report_intervals.py`'s cache)."""
    stem, png, sel_scores = job
    ds = _G.setdefault("ds", Dataset())
    obs = ds.observations[stem]
    gt = metric.GTIndex(obs)
    tree = build_tree(read_map_png(png)[0], splits=True)
    picks = {}
    for (scheme, rep), s in sel_scores.items():
        assert tree.n == len(s), (stem, scheme, rep, tree.n, len(s))
        for tau in PICKS:
            picks[scheme, rep, tau] = select(tree, s, tau) if tree.n else []
    nodes = sorted(set().union(*picks.values()))
    units = [am.node_pixels(tree, v) for v in nodes]
    col = {v: j for j, v in enumerate(nodes)}
    sel = {k: [col[v] for v in vs] for k, vs in picks.items()}
    counts = am.unit_counts(units, gt)
    areas = np.array([len(u) for u in units], np.int64)
    host = {k: am.selection_reading(stem, gt, counts, areas, c)[0] for k, c in sel.items()}
    objs = objects(gt.records)
    fil = [f for rec in gt.records for f in rec]
    refs = [r for o in objs for r in o["refs"]]
    cands = units + fil
    inter = am.unit_counts(cands, SimpleNamespace(records=[refs]))[0] if refs else np.zeros((0, len(cands)))
    iou, dice = pair_iou(inter, np.array([len(r) for r in refs]), np.array([len(c) for c in cands]))
    rows = []
    for n, o in enumerate(objs):
        vals = np.zeros((3, 1 + len(sel), 2))
        for r in range(3):
            row = 3 * n + r
            vals[r, 0] = best(iou[row], dice[row], [len(units) + c for c in o["h_cols"]])
            for m, k in enumerate(keys()):
                vals[r, 1 + m] = best(iou[row], dice[row], sel[k])
        rows.append(dict(h=o["h"], annotator=obs.records[o["h"]].annotator, three=o["three"],
                         cons_area=len(o["refs"][2]), vals=vals))
    return stem, rows, host


def objects_pass(stems: list[str], scores: dict, workers: int, cache: Path, tag: str) -> tuple[dict, dict]:
    """`frame_objects` over the three-record frames, cached: {stem: rows}, {stem: {key: host row}}."""
    f = cache / f"objects_pass{tag}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            return z["rows"].item(), z["host"].item()
    maps = {q.stem: q for q in (RUNS / RUN).glob("fold*/oof/*.png")}
    jobs = [(s, str(maps[s]), {k: v[s] for k, v in scores.items()}) for s in stems]
    rows, host = {}, {}
    with pool(workers, spawn=True) as w:
        for i, (stem, r, h) in enumerate(w.imap_unordered(frame_objects, jobs, chunksize=1)):
            rows[stem], host[stem] = r, h
            if (i + 1) % 25 == 0:
                print(f"objects: {i + 1}/{len(jobs)} frames", flush=True)
    np.savez(f, rows=np.array(rows, dtype=object), host=np.array(host, dtype=object))
    return rows, host


# --- the table --------------------------------------------------------------------------------------------------

def comparisons(rows: dict, stems: list[str], three_only: bool, ref: str) -> dict:
    """Flat comparisons of one set and reference: frame index, h's (IoU, Dice) and each selection's (n, keys, 2)."""
    rset = [0, 1] if ref == "pairwise" else [2]
    f, hum, v6 = [], [], []
    for i, s in enumerate(stems):
        for o in rows[s]:
            if three_only and not o["three"]:
                continue
            for r in rset:
                f.append(i)
                hum.append(o["vals"][r, 0])
                v6.append(o["vals"][r, 1:])
    return dict(f=np.array(f, int), human=np.array(hum).reshape(-1, 2), v6=np.array(v6).reshape(-1, len(keys()), 2))


def frame_sums(f: np.ndarray, own: np.ndarray, other: np.ndarray, n: int) -> np.ndarray:
    """Per frame [comparisons, detected, IoU sum detected, Dice sum detected, both detected, own IoU sum there,
    own Dice sum there] of one candidate (`own`, (m, 2)) beside the other candidate (`other`)."""
    d, b = own[:, 0] > MATCH, (own[:, 0] > MATCH) & (other[:, 0] > MATCH)
    cols = (np.ones(len(f)), d, d * own[:, 0], d * own[:, 1], b, b * own[:, 0], b * own[:, 1])
    return np.stack([np.bincount(f, weights=c, minlength=n) for c in cols], 1)


def object_measures(S: np.ndarray) -> dict:
    """Detection, outline given detection and outline when both detect, from summed `frame_sums` (last axis 7)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(comparisons=S[..., 0], detected=S[..., 1] / S[..., 0], iou_given_detected=S[..., 2] / S[..., 1],
                    dice_given_detected=S[..., 3] / S[..., 1], both_detected=S[..., 4] / S[..., 0],
                    iou_both_detected=S[..., 5] / S[..., 4], dice_both_detected=S[..., 6] / S[..., 4])


def series_reading(sums: list[np.ndarray], Wt: dict) -> dict:
    """Per-partition point values and both bootstraps of the partitions' mean per-frame sums."""
    point = {}
    for s in sums:
        for k, v in object_measures(Wt["point"] @ s).items():
            point.setdefault(k, []).append(float(np.ravel(v)[0]))
    mean = np.mean(sums, 0)
    return dict(point={k: np.array(v) for k, v in point.items()},
                frame=object_measures(Wt["frame"] @ mean), cluster=object_measures(Wt["cluster"] @ mean))


def common_table(rows: dict, stems: list[str], groups: list[str], n_boot: int) -> list[dict]:
    """The common-object table in long form: per set, reference, scheme and tau, the release's and the held-out
    annotator's levels and the release minus the annotator."""
    Wt = ri.weights(stems, groups, np.ones(len(stems), bool), n_boot)
    out, ks = [], keys()
    for set_name, three in (("all three drew", True), ("two others drew", False)):
        for ref in REFS:
            c = comparisons(rows, stems, three, ref)
            for tau in PICKS:
                for scheme in SCHEMES:
                    idx = [m for m, k in enumerate(ks) if k[0] == scheme and k[2] == tau]
                    v6 = series_reading([frame_sums(c["f"], c["v6"][:, m], c["human"], len(stems)) for m in idx], Wt)
                    hu = series_reading([frame_sums(c["f"], c["human"], c["v6"][:, m], len(stems)) for m in idx], Wt)
                    tag = dict(set=set_name, reference=ref, tau=tau, scheme=scheme, n=len(c["f"]))
                    out += [dict(r, **tag) for r in ri.level_rows("v6", tau, scheme, v6, v6["point"])]
                    out += [dict(r, **tag) for r in ri.level_rows("held-out annotator", tau, scheme, hu, hu["point"])]
                    out += [dict(r, **tag) for r in ri.diff_rows("v6 minus held-out annotator", tau, scheme, v6, hu,
                                                                  [k for k in v6["point"]
                                                                   if k not in ("comparisons", "both_detected")])]
    return out


def object_csv(path: Path, rows: dict, stems: list[str], obs: dict) -> int:
    """One line per object and held-out annotator: the frame, the set, the consensus area and, against the
    consensus, the annotator's IoU and Dice beside the release's (share of the five partitions detecting, mean IoU, mean
    Dice) at each scheme and tau."""
    ks = keys()
    head = ["stem", "station", "year", "group", "held_out_annotator", "all_three", "consensus_area_px",
            "annotator_iou", "annotator_dice", "annotator_detected"]
    sets = [(s, t) for t in PICKS for s in SCHEMES]
    head += [f"v6_{s}_tau{t}_{m}" for s, t in sets for m in ("detected_share", "iou_mean", "dice_mean")]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        for s in stems:
            o_ = obs[s]
            for o in rows[s]:
                c = o["vals"][2]
                line = [s, o_.station, o_.time.year, o_.group, o["annotator"], int(o["three"]), o["cons_area"],
                        f"{c[0, 0]:.4f}", f"{c[0, 1]:.4f}", int(c[0, 0] > MATCH)]
                for sc, t in sets:
                    v = c[1 + np.array([m for m, k in enumerate(ks) if k[0] == sc and k[2] == t])]
                    line += [f"{np.mean(v[:, 0] > MATCH):.2f}", f"{v[:, 0].mean():.4f}", f"{v[:, 1].mean():.4f}"]
                w.writerow(line)
                n += 1
    return n


# --- station and epoch -----------------------------------------------------------------------------------------

def epoch_of(year: int) -> str:
    """The epoch block label of a year."""
    return next(f"{a}-{b}" for a, b in EPOCHS if a <= year <= b)


def host_measures(h: np.ndarray) -> dict:
    """PQ and the per-record counts from summed host rows (last axis 12)."""
    comps = h[..., 7]
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(pq=pq(np.moveaxis(h[..., :4], -1, 0)), matches=h[..., 0] / comps, missed_1_0=h[..., 8] / comps,
                    spurious_0_1=h[..., 9] / comps)


def group_expectation(H: np.ndarray, groups: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Per frame of the cell, the mean host row of its group's frames outside the cell (inside it when the group has
    no frame elsewhere)."""
    E = np.zeros_like(H)
    for g in np.unique(groups[cell]):
        own = (groups == g) & cell
        rest = (groups == g) & ~cell
        E[own] = H[rest].mean(0) if rest.any() else H[own].mean(0)
    return E


def mix(values: list[str]) -> str:
    """Counts as 'value:count' by count, then value."""
    return " ".join(f"{k}:{v}" for k, v in sorted(Counter(values).items(), key=lambda kv: (-kv[1], kv[0])))


def station_epoch_table(sel: dict, stems: list[str], obs: dict, n_boot: int) -> list[dict]:
    """Per cell, scheme and tau, the levels, the cell minus the other frames and the cell minus its group
    expectation, each with both intervals, and the cell's group and record mixes."""
    groups = np.array([obs[s].group for s in stems])
    Wt = ri.weights(stems, list(groups), np.ones(len(stems), bool), n_boot)
    cells = [("station", c, np.array([obs[s].station == c for s in stems])) for c in STATIONS]
    cells += [("epoch", f"{a}-{b}", np.array([a <= obs[s].time.year <= b for s in stems])) for a, b in EPOCHS]
    cells.append(("all", "all", np.ones(len(stems), bool)))
    out = []
    for tau in PICKS:
        for scheme, first in SCHEMES.items():
            reps = [sel[scheme, r, tau]["host"] for r in range(first, first + am.N_PARTITIONS)]
            H = np.mean(reps, 0)
            for kind, name, m in cells:
                E = group_expectation(H, groups, m)

                def stat(w, rows, mask):
                    """Measures of the masked frames' weighted sums."""
                    return host_measures(w[:, mask] @ rows[mask])

                point = [host_measures(r[m].sum(0)) for r in reps]
                rest_point = [host_measures(r[~m].sum(0)) for r in reps] if (~m).any() else None
                exp_point = host_measures(E[m].sum(0))
                fb, cb = stat(Wt["frame"], H, m), stat(Wt["cluster"], H, m)
                for k in point[0]:
                    v = float(np.mean([p[k] for p in point]))
                    row = dict(kind=kind, cell=name, tau=tau, scheme=scheme, measure=k, value=v,
                               ci=ri.ci(fb[k]), cl=ri.ci(cb[k]), frames=int(m.sum()),
                               records=int(sum(obs[s].n_records for s in np.array(stems)[m])),
                               group_mix=mix(list(groups[m])),
                               records_mix=mix([str(obs[s].n_records) for s in np.array(stems)[m]]))
                    if rest_point is not None:
                        fr, cr = stat(Wt["frame"], H, ~m), stat(Wt["cluster"], H, ~m)
                        row.update(minus_rest=v - float(np.mean([p[k] for p in rest_point])),
                                   minus_rest_ci=ri.ci(fb[k] - fr[k]), minus_rest_cl=ri.ci(cb[k] - cr[k]))
                    if kind == "all":  # the group expectation of all frames is the frames themselves
                        out.append(row)
                        continue
                    fe, ce = stat(Wt["frame"], E, m), stat(Wt["cluster"], E, m)
                    row.update(expected=float(exp_point[k]), minus_expected=v - float(exp_point[k]),
                               minus_expected_ci=ri.ci(fb[k] - fe[k]), minus_expected_cl=ri.ci(cb[k] - ce[k]))
                    out.append(row)
    return out


# --- outputs ----------------------------------------------------------------------------------------------------------

def fmt(x) -> str:
    """A number to six places, or empty."""
    return "" if x is None else f"{x:.6f}"


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """The common-object table in long form."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["set", "reference", "comparisons", "who", "tau", "scheme", "measure", "value", "ci_low",
                    "ci_high", "cluster_low", "cluster_high"])
        for r in rows:
            w.writerow([r["set"], r["reference"], r["n"], r["row"], r["tau"], r["scheme"], r["measure"],
                        *(fmt(x) for x in (r["value"], *r["ci"], *r["cl"]))])


def write_station_csv(path: Path, rows: list[dict]) -> None:
    """The station and epoch table in long form."""
    cols = ["kind", "cell", "frames", "records", "tau", "scheme", "measure", "value", "ci_low", "ci_high",
            "cluster_low", "cluster_high", "minus_rest", "minus_rest_ci_low", "minus_rest_ci_high",
            "minus_rest_cluster_low", "minus_rest_cluster_high", "group_expected", "minus_expected",
            "minus_expected_ci_low", "minus_expected_ci_high", "minus_expected_cluster_low",
            "minus_expected_cluster_high", "group_mix", "records_mix"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            none = (None, None)
            w.writerow([r["kind"], r["cell"], r["frames"], r["records"], r["tau"], r["scheme"], r["measure"],
                        *(fmt(x) for x in (r["value"], *r["ci"], *r["cl"], r.get("minus_rest"),
                                           *r.get("minus_rest_ci", none), *r.get("minus_rest_cl", none),
                                           r.get("expected"), r.get("minus_expected"),
                                           *r.get("minus_expected_ci", none), *r.get("minus_expected_cl", none))),
                        r["group_mix"], r["records_mix"]])


def sanity(host: dict, sel: dict, all_stems: list[str], rows: dict, station: list[dict]) -> dict:
    """The pass's host rows against `report_intervals.py`'s cache; the earlier reading of the same measure on
    this pass; the station table's all-frames PQ against the record 0.4711 / 0.4691 at tau 0.26, and the station cells
    summing to all."""
    pos = {s: i for i, s in enumerate(all_stems)}
    diff = max(float(np.abs(h - sel[k]["host"][pos[s]]).max()) for s, d in host.items() for k, h in d.items())
    out = {"max_abs_diff_host_rows_vs_intervals_cache": diff}
    pw = [o["vals"][r] for d in rows.values() for o in d if o["three"] for r in (0, 1)]
    if pw:
        pw = np.array(pw)
        out["all_three_pairwise_annotator_detected"] = round(float(np.mean(pw[:, 0, 0] > MATCH)), 4)
        out["all_three_pairwise_v6_random15_tau0.26_detected"] = round(float(np.mean(pw[:, 1, 0] > MATCH)), 4)
        out["m4_round23_for_comparison"] = "another annotator 0.666, v5 tau 0.26 0.789 (filament rows, IoU > 0.2 links)"
    look = {(r["cell"], r["tau"], r["scheme"], r["measure"]): r for r in station}
    for scheme in SCHEMES:
        rec = ri.SANITY[scheme, 0.26]
        v = look["all", 0.26, scheme, "pq"]["value"]
        out[f"all_frames_pq_{scheme}_0.26"] = [round(v, 5), rec, bool(abs(round(v, 4) - rec) < 1.5e-4)]
    out["station_frames_sum"] = sum(look[c, 0.26, "random", "pq"]["frames"] for c in STATIONS)
    out["epoch_frames_sum"] = sum(look[f"{a}-{b}", 0.26, "random", "pq"]["frames"] for a, b in EPOCHS)
    return out


def write_md(path: Path, common: list[dict], station: list[dict], checks: dict, n_frames: int, n_obj: dict) -> None:
    """The summary of both tables, the definitions, the drafted sentences and the checks."""
    def f(r, k="", d=3):
        """value [frame CI] {cluster CI} of a row, or of its `k` difference."""
        if k:
            return (f"{r[k]:+.{d}f} [{r[k + '_ci'][0]:+.{d}f}, {r[k + '_ci'][1]:+.{d}f}] "
                    f"{{{r[k + '_cl'][0]:+.{d}f}, {r[k + '_cl'][1]:+.{d}f}}}")
        sign = "+" if r["row"].startswith("v6 minus") else ""
        return (f"{r['value']:{sign}.{d}f} [{r['ci'][0]:{sign}.{d}f}, {r['ci'][1]:{sign}.{d}f}] "
                f"{{{r['cl'][0]:{sign}.{d}f}, {r['cl'][1]:{sign}.{d}f}}}")

    look = {(r["set"], r["reference"], r["row"], r["tau"], r["scheme"], r["measure"]): r for r in common}
    meas = ("detected", "iou_given_detected", "dice_given_detected", "both_detected", "iou_both_detected",
            "dice_both_detected")
    L = ["# Common objects and station/epoch readings for release v6 (items 4.5, 4.13)", "",
         "Value [90% frame-bootstrap interval] {90% annotation-group cluster interval}; 2,000 draws, seed 0; v6 is "
         "the mean of a scheme's five scorer partitions (random 15-19, group 10-14). Script: "
         "`scripts/report_objects_epochs.py`.", "",
         "## (4.5) Common objects", "",
         f"Frames with three annotator records: {n_frames} (annotation groups: {checks['groups_three']}). Objects "
         f"per held-out annotator: {n_obj['three']} all-three objects and {n_obj['two']} two-annotator objects "
         "(summed over the three held-out annotators). Two filaments of different records are one object when each "
         f"is the other's best match by IoU and the IoU is above {LINK}. For held-out annotator h, the objects are "
         "the linked pairs of the other two records p and q; all-three objects are those h also drew (h's best match "
         "of both is one filament linked to both). References h did not draw: p's and q's outlines each "
         "(pairwise; two comparisons per object) and the pixels both drew (consensus; one comparison). Each "
         f"candidate is its mask with the best IoU against the reference. Detected: IoU > {MATCH}; outline given "
         "detection: mean IoU and Dice over the detected comparisons; both detected: the comparisons where v6 and "
         "h both detect, with each side's outline read on them. Denominators: comparisons for the detection "
         "shares, detected comparisons for the outline given detection, jointly detected comparisons for the "
         "paired outline. The group-cluster interval draws from only "
         f"{len(checks['groups_three'].split())} groups here and is coarse.", ""]
    for set_name in ("all three drew", "two others drew"):
        for ref in REFS:
            n = look[set_name, ref, "v6", 0.26, "random", "detected"]["n"]
            L += [f"### {set_name}, {ref} reference ({n} comparisons)", "",
                  "| tau | scheme | who | " + " | ".join(meas) + " |", "|---" * (3 + len(meas)) + "|"]
            for t in PICKS:
                for s in SCHEMES:
                    for who, pre in (("v6", ""), ("held-out annotator", ""), ("v6 minus held-out annotator", "d_")):
                        cells = [f(look[set_name, ref, who, t, s, pre + m])
                                 if (set_name, ref, who, t, s, pre + m) in look else "" for m in meas]
                        L.append(f"| {t} | {s} | {who} | " + " | ".join(cells) + " |")
            L.append("")
    sl = {(r["cell"], r["tau"], r["scheme"], r["measure"]): r for r in station}
    L += ["## (4.13) By GONG station and epoch (out of fold, all 707 frames)", "",
          "Station: the stem's 15th character (B Big Bear, the one letter the data page names; C, L, M, T, U by the "
          "GONG site convention: Cerro Tololo, Learmonth, Mauna Loa, El Teide, Udaipur). Epoch: the stem's year in "
          "blocks of about equal frame count. 1:0 and 0:1 are per record (comparison). 'minus rest' is the cell "
          "minus the other frames; 'minus group exp.' is the cell minus its annotation-group expectation (each "
          "frame replaced by the mean per-frame counts of its group's frames outside the cell).", ""]
    for tau in PICKS:
        for scheme in SCHEMES:
            L += [f"### tau {tau}, {scheme} partitions", "",
                  "| cell | frames | records | PQ | minus rest | minus group exp. | 1:0 / record | 0:1 / record | "
                  "records mix |", "|---" * 9 + "|"]
            for c in list(STATIONS) + [f"{a}-{b}" for a, b in EPOCHS] + ["all"]:
                r = sl[c, tau, scheme, "pq"]
                L.append(f"| {c} | {r['frames']} | {r['records']} | {f(r) if 'row' in r else _lv(r)} | "
                         + (f(r, 'minus_rest') if "minus_rest" in r else "") + " | "
                         + (f(r, 'minus_expected') if "minus_expected" in r else "") + " | "
                         + f"{_lv(sl[c, tau, scheme, 'missed_1_0'])} | {_lv(sl[c, tau, scheme, 'spurious_0_1'])} | "
                         + f"{r['records_mix']} |")
            L.append("")
    L += ["Group mix per cell (frames per annotation group): `station_epoch.csv`, column `group_mix`.", "",
          "## Drafted sentences", "", *checks.pop("sentences"), "", "## Sanity checks", "", "```",
          json.dumps(checks, indent=1), "```", ""]
    path.write_text("\n".join(L))


def _lv(r, d=3) -> str:
    """A level with both intervals."""
    return f"{r['value']:.{d}f} [{r['ci'][0]:.{d}f}, {r['ci'][1]:.{d}f}] {{{r['cl'][0]:.{d}f}, {r['cl'][1]:.{d}f}}}"


def sentences(common: list[dict], station: list[dict]) -> list[str]:
    """One drafted report sentence for each item, filled from the tables."""
    look = {(r["set"], r["reference"], r["row"], r["tau"], r["scheme"], r["measure"]): r for r in common}
    k = ("all three drew", "consensus")
    v, h = (look[(*k, w, 0.26, "random", "detected")] for w in ("v6", "held-out annotator"))
    d = look[(*k, "v6 minus held-out annotator", 0.26, "random", "d_detected")]
    dg = look[(*k, "v6 minus held-out annotator", 0.26, "group", "d_detected")]
    o = look[(*k, "v6 minus held-out annotator", 0.26, "random", "d_iou_both_detected")]
    og = look[(*k, "v6 minus held-out annotator", 0.26, "group", "d_iou_both_detected")]
    vi, hi = (look[(*k, w, 0.26, "random", "iou_both_detected")] for w in ("v6", "held-out annotator"))
    s1 = (f"- (4.5) On the {v['n']} filaments all three annotators drew, read against the pixels the other two "
          f"annotators both drew, v6 at tau 0.26 matches (IoU > 0.5) {v['value']:.1%} where the held-out annotator "
          f"matches {h['value']:.1%} ({d['value']:+.1%}, 90% CI {d['ci'][0]:+.1%} to {d['ci'][1]:+.1%}, group "
          f"clusters {d['cl'][0]:+.1%} to {d['cl'][1]:+.1%}; group-held-out scorer {dg['value']:+.1%}, "
          f"{dg['ci'][0]:+.1%} to {dg['ci'][1]:+.1%}), and on the objects both match its outline's IoU is "
          f"{vi['value']:.3f} against {hi['value']:.3f} ({o['value']:+.3f}, {o['ci'][0]:+.3f} to {o['ci'][1]:+.3f}; "
          f"group-held-out {og['value']:+.3f}, {og['ci'][0]:+.3f} to {og['ci'][1]:+.3f}).")
    sl = {(r["cell"], r["tau"], r["scheme"], r["measure"]): r for r in station}
    cells = list(STATIONS) + [f"{a}-{b}" for a, b in EPOCHS]
    raw, adj = [], []
    for c in cells:
        for t in PICKS:
            for s in SCHEMES:
                r = sl[c, t, s, "pq"]
                if r["minus_rest_ci"][0] > 0 or r["minus_rest_ci"][1] < 0:
                    raw.append(f"{c} ({t}, {s}, {r['minus_rest']:+.3f})")
                if r["minus_expected_ci"][0] > 0 or r["minus_expected_ci"][1] < 0:
                    adj.append(f"{c} ({t}, {s}, {r['minus_expected']:+.3f})")
    miss = []
    for c in cells:
        for m in ("missed_1_0", "spurious_0_1"):
            r = sl[c, 0.26, "random", m]
            if r["minus_rest_ci"][0] > 0 or r["minus_rest_ci"][1] < 0:
                e = r["minus_expected_ci"]
                miss.append(f"{m.split('_')[0]} {c} {r['value']:.2f} ({r['minus_rest']:+.2f}; against its group "
                            f"expectation {r['minus_expected']:+.2f}, {'beyond' if e[0] > 0 or e[1] < 0 else 'within'}"
                            " its interval)")
    pqs = [sl[c, 0.26, "random", "pq"]["value"] for c in STATIONS]
    nfr = [sl[c, 0.26, "random", "pq"]["frames"] for c in STATIONS]
    eps = [sl[f"{a}-{b}", 0.26, "random", "pq"]["value"] for a, b in EPOCHS]
    s2 = (f"- (4.13) Out of fold at tau 0.26, v6's PQ by GONG station runs from {min(pqs):.3f} to {max(pqs):.3f} "
          f"({min(nfr)}-{max(nfr)} frames each) and by epoch from {min(eps):.3f} to {max(eps):.3f}; "
          f"of the 40 cell readings (ten cells, two taus, two schemes) the cell-minus-rest frame interval "
          f"excludes zero in "
          f"{len(raw)}{': ' + '; '.join(raw) if raw else ''}, and after each cell's annotation-group mix is "
          f"accounted for in {len(adj)}{': ' + '; '.join(adj) if adj else ''} (about 4 expected by chance at 90%). "
          f"Per record, 1:0 and 0:1 (tau 0.26, random) differ from the other frames beyond the interval in: "
          f"{'; '.join(miss) or 'none'}.")
    return [s1, s2]


def main():
    """Run the common-object frame pass, read the station and epoch cells from the cached intervals pass, and write the
    tables and the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=None, help="smoke run of the common-object table on the first N "
                    "three-record frames")
    ri.add_release_args(ap)
    a = ap.parse_args()
    global RUN
    cache_in = ri.apply_release_args(a)
    RUN = ri.RUN
    tag = f"_n{a.frames}" if a.frames else ""
    scores = ri.load_scores(cache_in)
    all_stems = sorted(scores["random", am.FIRST_RANDOM])
    obs = Dataset().observations
    three = [s for s in all_stems if obs[s].n_records == 3][:a.frames]
    cache = a.out_dir / "objects_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows, host = objects_pass(three, scores, a.workers, cache, tag)
    sel, _, _ = ri.run_pass(all_stems, scores, a.workers, a.out_dir / "intervals_cache", "")
    common = common_table(rows, three, [obs[s].group for s in three], a.boot)
    n_obj = object_csv(a.out_dir / f"common_objects{tag}.csv", rows, three, obs)
    counts = dict(three=sum(o["three"] for d in rows.values() for o in d), two=n_obj)
    station = station_epoch_table(sel, all_stems, obs, a.boot)
    checks = sanity(host, sel, all_stems, rows, station)
    checks["groups_three"] = mix([obs[s].group for s in three])
    checks["sentences"] = sentences(common, station)
    write_summary_csv(a.out_dir / f"common_objects_summary{tag}.csv", common)
    write_station_csv(a.out_dir / f"station_epoch{tag}.csv", station)
    write_md(a.out_dir / f"report_tables{tag}.md", common, station, checks, len(three), counts)
    print(json.dumps({k: v for k, v in checks.items()}, indent=1))


if __name__ == "__main__":
    main()

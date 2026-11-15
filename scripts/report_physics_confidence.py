"""The physics of the released system's masks, and a per-mask confidence a scientist can
use.

The released system is `ens_v5_inc5`. Its out-of-fold masks are the exact tree selections of the hurdle scorer
cross-fitted on random partitions 15-19 and group partitions 10-14 (`ablation_metrics.py`'s cache of utilities; never
random 25-29 or group 20-24) at tau 0.26 and 0.30, the masks rebuilt from the run's out-of-fold maps as
`report_objects_epochs.py` rebuilds them. Its test masks are read only from our two submission files, decoded with
`solarfil.rle.decode`.

Every measure is a projected image measure, without a B0 or P correction: north up, east left, as MAGFiLO's own
hemisphere rule reads the frames. Per filament (a mask, or an annotator's polygon): area (pixels), length (the
longest geodesic path of the largest skeleton component plus one pixel), tilt (the
principal axis of the pixels from the image x axis with y up, degrees in (-90, 90]; negative in the north and
positive in the south means the eastern tip is poleward), latitude (the arcsine of the centroid's height above the
disk centre over the disk radius, `chirality.latitude`'s formula), projected longitude (the arcsine of the centroid's
offset from the central meridian over the radius times the cosine of the latitude, west positive), and the bearing
call of `chirality` (`barbs` at the 8 px fixed before the test, `score`, `call`; Right, Left or illegible), a
morphological proxy for chirality, not magnetic chirality. The disk is `cache/v1/disks.json` (`cache.load_disks`),
fitted on every training and test frame.

- `physics.csv`: per source (the annotators' polygons, every record of the 707 training frames; the release out of fold
  at each tau and scheme; the release's test masks at each tau), hemisphere and |latitude| band (all, 0-20, 20-40, 40-90
  degrees): the objects, mean tilt, the share with the eastern tip poleward, mean and median area and length, the
  legible share and the Right share among legible bearing calls, the annotators' own Right share among their Left/Right
  labels, and over both hemispheres the share of legible calls that follow the hemispheric rule (Right in the north,
  Left in the south; MAGFiLO's Technical Validation and Fig. 10 read the annotators' labels the same way against Hazra
  et al. and Pevtsov et al.). Out of fold, the release minus the annotators is paired on the same frames.
- `physics_bias.csv`: on matched objects (a release mask and an annotator's filament with the host's IoU > 0.5, per
  record), the release minus the annotator for length, area and tilt (tilt differences wrapped to (-90, 90]; also on
  filaments the annotator drew at least 100 px long): bias (mean difference), error (mean absolute difference), relative
  bias (ratio of sums minus one) and the median absolute error, on all frames and on the multiply annotated frames
  beside a second annotator minus the first on the same frames (every ordered pair of records; its bias is zero by
  symmetry, its error is the human reference).
- `test_filaments_v6_tau0.26.csv` and `_tau0.30.csv`: one row per test mask with the measures above and, when the
  test utilities are given (`--test-utilities`, written by `--test-pass`), the scorer's raw utility and its
  calibrated value.

The scorer's value of a mask is u = h m (`solarfil.scorer`): the expected matched IoU of the mask against a
random annotator record. Realized, it is the mask's matched IoU summed over the frame's records (0 for a record it
does not match) over the record count: the same quantity, read per record. It is never a probability of being a
filament. The value is recalibrated by isotonic regression (weights: the frame's records, as the scorer is fitted)
cross-fitted on the same partitions as the scorer: for each partition, the masks of the frames in one fold are mapped
by a fit on the selected masks of the other four folds.

- `confidence_calibration.csv`: reliability bins (value 0-0.1, ..., 0.9-1) of the raw and the calibrated value on the
  selected masks, per tau and scheme: masks, mean value, realized mean with its 90% frame interval; summary rows
  (mean value minus realized, the optimism; expected calibration error); and the calibrator for the test masks
  (isotonic on every selected out-of-fold mask of the random partitions at that tau), as its knots.
- `reliability.png`: the bins at both taus, random partitions, with their 90% bands.
- `withheld_curve.csv` / `.png`: as the masks whose calibrated value is below a cut are withheld, the share of masks
  withheld, the matched filaments lost and the spurious masks removed per record (comparison), and PQ, each with a
  90% frame interval, at both taus in both schemes.

Values are means over a scheme's five partitions (each partition's masks weighted one fifth). Intervals are 90%:
frames resampled with replacement, 2,000 draws from seed 0 (`report_intervals.weights`); on test, the 180 test frames
resampled the same way. Differences are paired on the same draws.

Outputs under --out-dir: the files above, `report_physics_confidence.md` (summary, sanity checks and one drafted
report sentence for each item) and the frame pass in `physics_cache/`.

Usage: uv run python scripts/report_physics_confidence.py
       [--out-dir report] [--workers 3] [--boot 2000] [--frames N]
       [--test-utilities FILE.npz]
       (--frames: a smoke run on the first N training frames, written with a suffix)
The test utilities need the dense networks and detectors on the 180 test frames (a GPU job):
       uv run python scripts/report_physics_confidence.py --test-pass FILE.npz
       [--weights weights/effv2s_s0_win_inc5_inst5grp5uni5] [--device cuda] [--limit N]
It re-runs the release pipeline's decision stage keeping the utilities, reports how many of its selections give the
submission's masks byte for byte, and saves each submission mask's utility in file order, taken from the re-run mask
with the best IoU (above 0.5) and saved with that IoU: on another device the masks can differ (a one-frame CPU run
gave the submission's two masks at IoU 0.83 and above, not byte for byte).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize
from sklearn.isotonic import IsotonicRegression

import ablation_metrics as am
import report_intervals as ri
from solarfil import chirality, metric, rle
from solarfil.cache import DEFAULT_CACHE, load_disks
from solarfil.data import Dataset, W
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import REPO, RUNS
from solarfil.rule import partitions
from solarfil.tree import build_tree, select

RUN = ri.RUN
PICKS = ri.PICKS
SCHEMES = ri.SCHEMES
SUB = REPO / "submissions"
TEST_CSV = {t: SUB / f"s0win_inc5_inst5grp5uni5_tau{t:.2f}" / f"ens_v5_inc5_inst5grp5uni5_tau{t:.2f}.csv"
            for t in PICKS}
WEIGHTS = REPO / "weights" / "effv2s_s0_win_inc5_inst5grp5uni5"
STATS = ("area", "length", "tilt", "lat", "lon", "rho", "cx", "cy", "barbs", "S", "call")
COL = {k: i for i, k in enumerate(STATS)}
BANDS = {"all": (0, 91), "0-20": (0, 20), "20-40": (20, 40), "40-90": (40, 91)}
MATCH = 0.5
LONG = 100  # px: the tilt comparison restricted to filaments the annotator drew at least this long
BINS = np.linspace(0, 1, 11)
CUTS = tuple(round(0.05 * i, 2) for i in range(17))  # 0 to 0.8
CALL_NAME = {0: "illegible", chirality.LEFT: "Left", chirality.RIGHT: "Right"}
_G: dict = {}


# --- per-filament measures --------------------------------------------------------------------------------------------

def skeleton_length(m: np.ndarray) -> float:
    """The longest geodesic path of the largest 8-connected skeleton component of a mask crop, plus one pixel;
    1 for a mask without a skeleton edge."""
    p, _, _ = chirality.crop(m)
    if p.size == 0:
        return 0.0
    ys, _, g = chirality.skeleton_graph(skeletonize(p))
    if g is None or len(ys) < 2:
        return 1.0
    _, lab = connected_components(g, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    g = g[keep][:, keep]
    d0 = dijkstra(g, indices=0)
    far = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1 = dijkstra(g, indices=far)
    return float(np.max(np.where(np.isfinite(d1), d1, 0))) + 1.0


def shape_stats(pix: np.ndarray, disk) -> np.ndarray:
    """The `STATS` of one filament given as full-frame flat pixel indices, on its frame's disk."""
    ys, xs = np.divmod(pix.astype(np.int64), W)
    y0, x0 = ys.min(), xs.min()
    m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), bool)
    m[ys - y0, xs - x0] = True
    cy, cx = ys.mean(), xs.mean()
    xx, yy = xs - cx, -(ys - cy)  # x right (west), y up (north)
    tilt = 0.5 * np.degrees(np.arctan2(2 * (xx * yy).mean(), (xx * xx).mean() - (yy * yy).mean()))
    lat = np.degrees(np.arcsin(np.clip((disk.cy - cy) / disk.r, -1, 1)))
    cos = max(np.cos(np.radians(lat)), 1e-6)
    lon = np.degrees(np.arcsin(np.clip((cx - disk.cx) / (disk.r * cos), -1, 1)))
    sc = chirality.score(chirality.barbs(m))
    return np.array([len(pix), skeleton_length(m), tilt, lat, lon, np.hypot(cx - disk.cx, cy - disk.cy) / disk.r,
                     cx, cy, sc["barbs"], sc["S"], chirality.call(sc)], np.float64)


def stats_of(units: list[np.ndarray], disk) -> np.ndarray:
    """`shape_stats` of several filaments as an (n, len(STATS)) array."""
    return np.array([shape_stats(u, disk) for u in units]).reshape(len(units), len(STATS))


# --- one training frame -----------------------------------------------------------------------------------------------

def match_matrix(counts: list[np.ndarray], gt: metric.GTIndex, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per record and unit, the filament the unit matches at the host's IoU > 0.5 (-1 for none) and that IoU."""
    R, n = len(gt.records), len(areas)
    idx, iou = -np.ones((R, n), np.int64), np.zeros((R, n), np.float64)
    for k, rec in enumerate(gt.records):
        if not len(rec) or not n:
            continue
        a_gt = np.array([len(i) for i in rec], np.int64)
        m = metric.iou_dice(counts[k], a_gt, areas)[0].astype(np.float64)
        best = m.argmax(0)
        ok = m[best, np.arange(n)] > MATCH
        idx[k, ok], iou[k, ok] = best[ok], m[best, np.arange(n)][ok]
    return idx, iou


def annotator_pairs(gt: metric.GTIndex) -> np.ndarray:
    """Every ordered pair of records' filaments matched at IoU > 0.5: rows [k, i, l, j, IoU] (record k's filament i
    is the reference, record l's filament j the second annotator's)."""
    fil = [f for rec in gt.records for f in rec]
    off = np.cumsum([0] + [len(r) for r in gt.records])
    if len(gt.records) < 2 or not fil:
        return np.zeros((0, 5))
    counts = am.unit_counts(fil, gt)
    areas = np.array([len(f) for f in fil], np.int64)
    out = []
    for k, rec in enumerate(gt.records):
        a_gt = np.array([len(i) for i in rec], np.int64)
        m = metric.iou_dice(counts[k], a_gt, areas)[0]
        for i, c in zip(*np.nonzero(m > MATCH)):
            lrec = int(np.searchsorted(off, c, side="right") - 1)
            if lrec != k:
                out.append([k, i, lrec, c - off[lrec], float(m[i, c])])
    return np.array(out, np.float64).reshape(-1, 5)


def frame_job(job) -> tuple[str, dict]:
    """One training frame: every partition's selection at both picks as unit indices with their utilities, the
    units' measures and matches per record, the annotators' filaments with their measures and labels, and the
    annotators' matched pairs."""
    stem, png, sel_scores = job
    ds = _G.setdefault("ds", Dataset())
    disk = _G.setdefault("disks", load_disks(DEFAULT_CACHE))[stem]
    obs = ds.observations[stem]
    gt = metric.GTIndex(obs)
    tree = build_tree(read_map_png(png)[0], splits=True)
    picks = {}
    for (scheme, rep), s in sel_scores.items():
        assert tree.n == len(s), (stem, scheme, rep, tree.n, len(s))
        for tau in PICKS:
            v = select(tree, s, tau) if tree.n else []
            picks[scheme, rep, tau] = (np.asarray(v, np.int64), np.asarray(s, np.float64)[np.asarray(v, np.int64)])
    nodes = sorted(set().union(*[set(v.tolist()) for v, _ in picks.values()]))
    col = {v: j for j, v in enumerate(nodes)}
    units = [am.node_pixels(tree, v) for v in nodes]
    areas = np.array([len(u) for u in units], np.int64)
    idx, iou = match_matrix(am.unit_counts(units, gt), gt, areas)
    gt_stats = [stats_of(rec, disk) for rec in gt.records]
    cats = [np.array([f.category for f in r.filaments], np.int64) for r in obs.records]
    return stem, dict(sel={k: (np.array([col[x] for x in v.tolist()], np.int64), u) for k, (v, u) in picks.items()},
                      units=stats_of(units, disk), idx=idx, iou=iou, gt=gt_stats, cat=cats,
                      n_gt=np.array([len(r) for r in gt.records]), hp=annotator_pairs(gt), group=obs.group)


def frame_pass(stems: list[str], scores: dict, workers: int, cache: Path, tag: str) -> dict:
    """`frame_job` over the training frames, cached in `cache`: {stem: reading}."""
    f = cache / f"frames{tag}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            return z["frames"].item()
    maps = {q.stem: q for q in (RUNS / RUN).glob("fold*/oof/*.png")}
    jobs = [(s, str(maps[s]), {k: v[s] for k, v in scores.items()}) for s in stems]
    out = {}
    with pool(workers, spawn=True) as w:
        for i, (stem, r) in enumerate(w.imap_unordered(frame_job, jobs, chunksize=2)):
            out[stem] = r
            if (i + 1) % 100 == 0:
                print(f"frames: {i + 1}/{len(jobs)}", flush=True)
    np.savez(f, frames=np.array(out, dtype=object))
    return out


# --- test masks -------------------------------------------------------------------------------------------------------

def read_submission(path: Path) -> dict[str, list[tuple[str, str]]]:
    """{stem: [(filament_id, RLE counts), ...]} of a submission file, in file order."""
    out: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["filament_id"].rsplit("_", 1)[0], []).append((row["filament_id"],
                                                                             row["segmentation_rle"]))
    return out


def test_job(job) -> tuple[str, np.ndarray]:
    """The measures of one test frame's masks, decoded from the submission's RLE counts."""
    stem, counts = job
    disk = _G.setdefault("disks", load_disks(DEFAULT_CACHE))[stem]
    return stem, stats_of([np.flatnonzero(rle.decode(c)) for c in counts], disk)


def test_pass(workers: int, cache: Path) -> dict:
    """{tau: (filament ids, stems, frame index, measures)} of both submission files, cached."""
    f = cache / "test.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            return z["test"].item()
    out = {}
    stems = Dataset().test_stems
    for tau, path in TEST_CSV.items():
        sub = read_submission(path)
        assert set(sub) <= set(stems), path
        with pool(workers, spawn=True) as w:
            got = dict(w.map(test_job, [(s, [c for _, c in sub[s]]) for s in sorted(sub)], chunksize=2))
        ids = [fid for s in sorted(sub) for fid, _ in sub[s]]
        fr = np.concatenate([np.full(len(sub[s]), stems.index(s)) for s in sorted(sub)])
        out[tau] = dict(ids=ids, f=fr, X=np.concatenate([got[s] for s in sorted(sub)]))
    np.savez(f, test=np.array(out, dtype=object))
    return out


# --- object tables ----------------------------------------------------------------------------------------------------

def annotator_table(frames: dict, stems: list[str]) -> dict:
    """Every annotator polygon of every record: frame index, weight 1, measures and label."""
    f, X, cat = [], [], []
    for i, s in enumerate(stems):
        for st, c in zip(frames[s]["gt"], frames[s]["cat"]):
            f.append(np.full(len(st), i))
            X.append(st)
            cat.append(c)
    return dict(f=np.concatenate(f), w=np.ones(sum(len(x) for x in X)), X=np.concatenate(X), cat=np.concatenate(cat))


def mask_table(frames: dict, stems: list[str], scheme: str, tau: float) -> dict:
    """The selected masks of a scheme's five partitions at tau, each weighted one fifth: frame index, partition,
    measures, utility, records matched, matched IoU sum and the frame's records."""
    first = SCHEMES[scheme]
    cols = {k: [] for k in ("f", "rep", "X", "u", "tp", "iou", "R")}
    for i, s in enumerate(stems):
        d = frames[s]
        for rep in range(first, first + am.N_PARTITIONS):
            j, u = d["sel"][scheme, rep, tau]
            cols["f"].append(np.full(len(j), i))
            cols["rep"].append(np.full(len(j), rep))
            cols["X"].append(d["units"][j].reshape(len(j), len(STATS)))
            cols["u"].append(u)
            cols["tp"].append((d["idx"][:, j] >= 0).sum(0))
            cols["iou"].append(d["iou"][:, j].sum(0))
            cols["R"].append(np.full(len(j), len(d["n_gt"])))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["w"] = np.full(len(out["f"]), 1.0 / am.N_PARTITIONS)
    return out


# --- ratio estimators with frame intervals ----------------------------------------------------------------------------

def ratio(f, w, num, den, n: int, Wt: dict) -> tuple[float, np.ndarray]:
    """Point value and bootstrap draws of sum(w num) / sum(w den) with per-frame sums resampled by `Wt`."""
    a, b = np.bincount(f, w * num, n), np.bincount(f, w * den, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(a.sum() / b.sum()) if b.sum() else float("nan"), (Wt["frame"] @ a) / (Wt["frame"] @ b)


def physics_cells(T: dict, n: int, Wt: dict, labels: bool) -> dict:
    """{(hemisphere, band, measure): (point, draws)} of one object table; medians and counts carry no draws."""
    lat, tilt, call = T["X"][:, COL["lat"]], T["X"][:, COL["tilt"]], T["X"][:, COL["call"]]
    one, out = np.ones(len(lat)), {}
    for hemi, sgn in (("N", 1), ("S", -1)):
        for band, (lo, hi) in BANDS.items():
            m = (np.sign(lat) == sgn) & (np.abs(lat) >= lo) & (np.abs(lat) < hi)
            f, w = T["f"][m], T["w"][m]
            leg = call[m] != 0

            def r(num, den=None, f=f, w=w, m=m):
                """A ratio on this cell's objects."""
                return ratio(f, w, num, one[m] if den is None else den, n, Wt)
            out[hemi, band, "objects"] = (float(w.sum()), None)
            out[hemi, band, "tilt_mean"] = r(tilt[m])
            out[hemi, band, "east_tip_poleward_share"] = r((sgn * tilt[m] < 0).astype(float))
            for q in ("area", "length"):
                out[hemi, band, f"{q}_mean"] = r(T["X"][m, COL[q]])
                out[hemi, band, f"{q}_median"] = (float(np.median(T["X"][m, COL[q]])) if m.any() else np.nan, None)
            out[hemi, band, "legible_share"] = r(leg.astype(float))
            out[hemi, band, "right_share_legible"] = r((call[m] == chirality.RIGHT).astype(float), leg.astype(float))
            if labels:
                lr = np.isin(T["cat"][m], (chirality.LEFT, chirality.RIGHT)).astype(float)
                out[hemi, band, "label_right_share"] = r((T["cat"][m] == chirality.RIGHT).astype(float), lr)
    north = lat > 0
    follow = ((north & (call == chirality.RIGHT)) | (~north & (call == chirality.LEFT))).astype(float)
    out["both", "all", "objects"] = (float(T["w"].sum()), None)
    out["both", "all", "north_share"] = ratio(T["f"], T["w"], north.astype(float), one, n, Wt)
    out["both", "all", "abs_latitude_median"] = (float(np.median(np.abs(lat))), None)
    out["both", "all", "hemispheric_rule_share_legible"] = ratio(T["f"], T["w"], follow, (call != 0).astype(float),
                                                                 n, Wt)
    if labels:
        lr = np.isin(T["cat"], (chirality.LEFT, chirality.RIGHT))
        lf = ((north & (T["cat"] == chirality.RIGHT)) | (~north & (T["cat"] == chirality.LEFT))).astype(float)
        out["both", "all", "label_hemispheric_rule_share"] = ratio(T["f"], T["w"], lf, lr.astype(float), n, Wt)
    return out


def physics_rows(frames: dict, stems: list[str], test: dict, n_boot: int) -> tuple[list[dict], dict]:
    """`physics.csv`'s rows: every source's cells, and the release out of fold minus the annotators on the same
    frames."""
    n = len(stems)
    Wt = ri.weights(stems, [frames[s]["group"] for s in stems], np.ones(n, bool), n_boot)
    ann = physics_cells(annotator_table(frames, stems), n, Wt, True)
    rows, got = [], {"annotators": ann}
    rows += cell_rows("annotators (every record, training frames)", "", "", ann)
    for tau in PICKS:
        for scheme in SCHEMES:
            c = physics_cells(mask_table(frames, stems, scheme, tau), n, Wt, False)
            got[f"oof_{scheme}_{tau}"] = c
            rows += cell_rows("v6 out of fold", tau, scheme, c)
            rows += cell_rows("v6 out of fold minus annotators", tau, scheme, c, ann)
    nt = len(Dataset().test_stems)
    Wtest = dict(frame=am.bootstrap_counts(nt, n_boot))
    for tau in PICKS:
        T = dict(f=test[tau]["f"], w=np.ones(len(test[tau]["f"])), X=test[tau]["X"])
        c = physics_cells(T, nt, Wtest, False)
        got[f"test_{tau}"] = c
        rows += cell_rows("v6 test (180 frames)", tau, "", c)
    return rows, got


def cell_rows(source: str, tau, scheme: str, c: dict, ref: dict | None = None) -> list[dict]:
    """Long-form rows of one source's cells, or of its paired difference from `ref` where both have draws."""
    out = []
    for (hemi, band, m), (v, d) in c.items():
        if ref is not None:
            if d is None or (hemi, band, m) not in ref:
                continue
            rv, rd = ref[hemi, band, m]
            v, d = v - rv, d - rd
        lo, hi = ri.ci(d) if d is not None and np.isfinite(d).any() else (np.nan, np.nan)
        out.append(dict(source=source, tau=tau, scheme=scheme, hemisphere=hemi, band=band, measure=m, value=v,
                        ci_low=lo, ci_high=hi))
    return out


# --- object-level bias and error --------------------------------------------------------------------------------

def wrap(d: np.ndarray) -> np.ndarray:
    """Angle differences in degrees wrapped to (-90, 90]."""
    return -((-(d + 90.0)) % 180.0) + 90.0


def matched_pairs(frames: dict, stems: list[str], scheme: str, tau: float) -> dict:
    """The release's matched (mask, annotator filament) pairs of a scheme's five partitions at tau, each weighted one
    fifth."""
    f, A, B, multi = [], [], [], []
    for i, s in enumerate(stems):
        d = frames[s]
        for rep in range(SCHEMES[scheme], SCHEMES[scheme] + am.N_PARTITIONS):
            j, _ = d["sel"][scheme, rep, tau]
            for k in range(len(d["n_gt"])):
                hit = d["idx"][k, j] >= 0
                A.append(d["units"][j[hit]].reshape(-1, len(STATS)))
                B.append(d["gt"][k][d["idx"][k, j[hit]]].reshape(-1, len(STATS)))
                f.append(np.full(hit.sum(), i))
                multi.append(np.full(hit.sum(), len(d["n_gt"]) > 1))
    return dict(f=np.concatenate(f), a=np.concatenate(A), b=np.concatenate(B), multi=np.concatenate(multi),
                w=np.full(sum(len(x) for x in f), 1.0 / am.N_PARTITIONS))


def human_pairs(frames: dict, stems: list[str]) -> dict:
    """Every ordered pair of annotators' matched filaments on the multiply annotated frames."""
    f, A, B = [], [], []
    for i, s in enumerate(stems):
        d = frames[s]
        for k, a, lrec, b, _ in d["hp"]:
            A.append(d["gt"][int(lrec)][int(b)])
            B.append(d["gt"][int(k)][int(a)])
            f.append(i)
    return dict(f=np.array(f, int), a=np.array(A).reshape(-1, len(STATS)), b=np.array(B).reshape(-1, len(STATS)),
                multi=np.ones(len(f), bool), w=np.ones(len(f)))


def bias_rows(P: dict, who: str, tau, scheme: str, panel: str, n: int, Wt: dict, mask: np.ndarray) -> list[dict]:
    """Bias, error, relative bias and median absolute error of length, area and tilt on the pairs in `mask`."""
    out = []
    f, w, a, b = P["f"][mask], P["w"][mask], P["a"][mask], P["b"][mask]
    for q in ("length", "area", "tilt", "tilt_long"):
        c = COL["tilt" if q == "tilt_long" else q]
        keep = b[:, COL["length"]] >= LONG if q == "tilt_long" else np.ones(len(f), bool)
        d = a[keep, c] - b[keep, c]
        d = wrap(d) if q.startswith("tilt") else d
        ff, ww, one = f[keep], w[keep], np.ones(keep.sum())
        meas = {"bias": ratio(ff, ww, d, one, n, Wt), "error": ratio(ff, ww, np.abs(d), one, n, Wt)}
        if not q.startswith("tilt"):
            rel = ratio(ff, ww, a[keep, c], b[keep, c], n, Wt)
            meas["relative_bias"] = (rel[0] - 1, rel[1] - 1)
        meas["median_abs_error"] = (float(np.median(np.abs(d))) if len(d) else np.nan, None)
        meas["pairs"] = (float(ww.sum()), None)
        for m, (v, dr) in meas.items():
            lo, hi = ri.ci(dr) if dr is not None else (np.nan, np.nan)
            out.append(dict(who=who, tau=tau, scheme=scheme, panel=panel, quantity=q, measure=m, value=v,
                            ci_low=lo, ci_high=hi))
    return out


def bias_table(frames: dict, stems: list[str], n_boot: int) -> list[dict]:
    """`physics_bias.csv`: the release minus the annotator on matched objects (all frames and multiply annotated frames)
    and a second annotator minus the first on the multiply annotated frames."""
    n = len(stems)
    groups = [frames[s]["group"] for s in stems]
    multi = np.array([len(frames[s]["n_gt"]) > 1 for s in stems])
    W_all, W_multi = (ri.weights(stems, groups, p, n_boot) for p in (np.ones(n, bool), multi))
    rows = []
    for tau in PICKS:
        for scheme in SCHEMES:
            P = matched_pairs(frames, stems, scheme, tau)
            rows += bias_rows(P, "v6 minus annotator", tau, scheme, "all frames", n, W_all, np.ones(len(P["f"]), bool))
            rows += bias_rows(P, "v6 minus annotator", tau, scheme, "multiply annotated frames", n, W_multi,
                              P["multi"])
    H = human_pairs(frames, stems)
    rows += bias_rows(H, "second annotator minus annotator", "", "", "multiply annotated frames", n, W_multi,
                      np.ones(len(H["f"]), bool))
    return rows


# --- recalibration ----------------------------------------------------------------------------------------------

def isotonic(u: np.ndarray, y: np.ndarray, w: np.ndarray) -> IsotonicRegression:
    """Isotonic map from the scorer's value to the realized matched IoU per record, clipped to [0, 1]."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(u, y, sample_weight=w)


def cross_fit(T: dict, stems: list[str], folds: dict) -> np.ndarray:
    """Each mask's calibrated value: the isotonic map fitted on the other folds' masks of its own partition."""
    out = np.zeros(len(T["u"]))
    y = T["iou"] / T["R"]
    for rep in np.unique(T["rep"]):
        on = T["rep"] == rep
        fold = np.array([folds[rep][stems[i]] for i in T["f"]])
        for k in range(5):
            te, tr = on & (fold == k), on & (fold != k)
            out[te] = isotonic(T["u"][tr], y[tr], T["R"][tr]).predict(T["u"][te])
    return out


def reliability(T: dict, value: np.ndarray, n: int, Wt: dict) -> list[dict]:
    """Bins of `value`: masks, mean value and realized mean per record with draws; weights: records x partition
    share."""
    w, y = T["w"] * T["R"], T["iou"] / T["R"]
    b = np.clip(np.digitize(value, BINS) - 1, 0, len(BINS) - 2)
    out = []
    for i in range(len(BINS) - 1):
        m = b == i
        if not m.any():
            continue
        real = ratio(T["f"][m], w[m], y[m], np.ones(m.sum()), n, Wt)
        out.append(dict(lo=BINS[i], hi=BINS[i + 1], masks=float(T["w"][m].sum()),
                        value=float(np.average(value[m], weights=w[m])), realized=real[0], draws=real[1],
                        weight=float(w[m].sum())))
    return out


def ece(T: dict, value: np.ndarray, n: int, Wt: dict) -> tuple[float, np.ndarray]:
    """Expected calibration error over `BINS` (weights as `reliability`), with draws."""
    w, y = T["w"] * T["R"], T["iou"] / T["R"]
    b = np.clip(np.digitize(value, BINS) - 1, 0, len(BINS) - 2)
    V = np.stack([np.bincount(T["f"][b == i], (w * value)[b == i], n) for i in range(len(BINS) - 1)])
    Y = np.stack([np.bincount(T["f"][b == i], (w * y)[b == i], n) for i in range(len(BINS) - 1)])
    N = np.stack([np.bincount(T["f"][b == i], w[b == i], n) for i in range(len(BINS) - 1)])
    point = float(np.abs(V.sum(1) - Y.sum(1)).sum() / N.sum())
    draws = np.abs(Wt["frame"] @ V.T - Wt["frame"] @ Y.T).sum(1) / (Wt["frame"] @ N.T).sum(1)
    return point, draws


def pq_rows(v: np.ndarray, tp, fp, iou, n_gt) -> np.ndarray:
    """PQ from per-frame TP, FP, matched IoU sums and filament counts under weight rows v (draws x frames)."""
    return (v @ iou) / (v @ tp + 0.5 * (v @ fp) + 0.5 * (v @ (n_gt - tp)))


def withheld(T: dict, value: np.ndarray, frames: dict, stems: list[str], n: int, Wt: dict) -> list[dict]:
    """The withheld-mask curve over `CUTS`: masks withheld, matched filaments lost and spurious masks removed per
    record, and PQ, each with draws (the partitions' mean per-frame totals)."""
    comps = np.array([len(frames[s]["n_gt"]) for s in stems], float)
    n_gt = np.array([frames[s]["n_gt"].sum() for s in stems], float)
    w = T["w"]
    fp = T["R"] - T["tp"]
    base = [np.bincount(T["f"], w * x, n) for x in (T["tp"], fp, T["iou"], np.ones(len(w)))]
    out = []
    for c in CUTS:
        keep = value >= c
        tp, fpk, iou, cnt = (np.bincount(T["f"][keep], (w * x)[keep], n) for x in (T["tp"], fp, T["iou"],
                                                                                  np.ones(len(w))))

        one = np.ones((1, n))
        F = Wt["frame"]
        rows = {"masks_withheld_share": (1 - cnt.sum() / base[3].sum(), 1 - (F @ cnt) / (F @ base[3])),
                "matched_lost_per_record": (((base[0] - tp).sum()) / comps.sum(),
                                            (Wt["frame"] @ (base[0] - tp)) / (Wt["frame"] @ comps)),
                "spurious_removed_per_record": (((base[1] - fpk).sum()) / comps.sum(),
                                                (Wt["frame"] @ (base[1] - fpk)) / (Wt["frame"] @ comps)),
                "pq": (float(pq_rows(one, tp, fpk, iou, n_gt)[0]), pq_rows(F, tp, fpk, iou, n_gt))}
        out.append(dict(cut=c, **{k: v for k, v in rows.items()}))
    return out


def confidence(frames: dict, stems: list[str], n_boot: int, ds: Dataset, all_stems: list[str]) -> dict:
    """The recalibration at both picks in both schemes on `stems`: calibrated values, reliability bins, ECE, optimism
    and the withheld curve; and the test calibrators. The partitions' folds are drawn on `all_stems` (every training
    frame), as the scorer's were."""
    n = len(stems)
    Wt = ri.weights(stems, [frames[s]["group"] for s in stems], np.ones(n, bool), n_boot)
    folds = {rep: f for _, rep, f in partitions(all_stems, am.N_PARTITIONS, am.N_PARTITIONS, ds=ds,
                                                first=am.FIRST_RANDOM, first_group=am.FIRST_GROUP)}
    out = {}
    for tau in PICKS:
        for scheme in SCHEMES:
            T = mask_table(frames, stems, scheme, tau)
            cal = cross_fit(T, stems, folds)
            w, y = T["w"] * T["R"], T["iou"] / T["R"]
            kinds = (("raw", T["u"]), ("calibrated", cal))
            opt = {k: ratio(T["f"], w, v - y, np.ones(len(y)), n, Wt) for k, v in kinds}
            out[tau, scheme] = dict(T=T, cal=cal, bins={k: reliability(T, v, n, Wt) for k, v in
                                                        (("raw", T["u"]), ("calibrated", cal))},
                                    ece={k: ece(T, v, n, Wt) for k, v in (("raw", T["u"]), ("calibrated", cal))},
                                    optimism=opt, curve=withheld(T, cal, frames, stems, n, Wt))
        T = out[tau, "random"]["T"]
        out[tau, "calibrator"] = isotonic(T["u"], T["iou"] / T["R"], T["w"] * T["R"])
    return out


# --- the test pass (GPU) ----------------------------------------------------------------------------------------------

def run_test_pass(out: Path, weights: Path, dev: str, limit: int | None) -> None:
    """Re-run the release pipeline on the test frames keeping each node's utility; check the selection at each pick
    against the submission's masks; save {"<stem>|<tau>": utilities of the submission's masks in file order} and
    {"<stem>|<tau>|iou": each submission mask's IoU with the re-run mask it takes its utility from}."""
    import torch

    from solarfil import instances as detector
    from solarfil.bank import describe
    from solarfil.data import read_image
    from solarfil.infer import to_native
    from solarfil.pipeline import Pipeline, flat_stage
    from solarfil.postprocess import labels_to_rles
    from solarfil.preprocess import from_uint8

    pipe = Pipeline.load(weights, torch.device(dev))
    subs = {t: read_submission(p) for t, p in TEST_CSV.items()}
    ds, res, exact, worst = Dataset(), {}, 0, 1.0
    for stem in ds.test_stems[:limit]:
        flat_u8, disk, _ = flat_stage(read_image(ds.image_path(stem)))
        p = to_native(pipe.dense(flat_u8, disk, stem).astype(np.float32) / 255.0, pipe.scale)
        tree = build_tree(p, splits=pipe.splits)
        u = np.zeros(0)
        if tree.n:
            X = describe(tree, p, from_uint8(flat_u8), disk)
            us = []
            for group in pipe.detect(flat_u8):
                cols = [detector.describe(tree, s, detector.unpack(pk), mk) for s, pk, mk in group]
                us.append(pipe.scorer.predict(np.concatenate([X, *cols], 1))[0])
            u = np.mean(us, 0)
        for tau in PICKS:
            nodes = select(tree, u, tau) if tree.n else []
            theirs = [c for _, c in subs[tau].get(stem, [])]
            mine = labels_to_rles(tree.labels(nodes)) if len(nodes) else []
            exact += mine == theirs
            got, iou = submission_utilities(theirs, mine, np.asarray(u, np.float64)[np.asarray(nodes, np.int64)])
            res[f"{stem}|{tau}"], res[f"{stem}|{tau}|iou"] = got, iou
            worst = min([worst, *iou.tolist()])
        print(f"{stem}: {tree.n} nodes", flush=True)
    np.savez(out, **res)
    n = len(ds.test_stems[:limit])
    print(json.dumps(dict(frames=n, selections_byte_identical=f"{exact} of {n * len(PICKS)}",
                          lowest_iou_to_submission_mask=round(worst, 4))))


def submission_utilities(theirs: list[str], mine: list[str], u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each submission mask (in file order), the utility of the re-run's selected mask with the best IoU and
    that IoU; NaN where no re-run mask overlaps it above 0.5 (the re-run's masks may differ on another device)."""
    if not theirs:
        return np.zeros(0), np.zeros(0)
    if not mine:
        return np.full(len(theirs), np.nan), np.zeros(len(theirs))
    inter, a, b = metric.overlap_counts([rle.to_coco(c) for c in theirs], [rle.to_coco(c) for c in mine])
    iou = metric.iou_dice(inter, a, b)[0].astype(np.float64)
    best = iou.argmax(1)
    top = iou[np.arange(len(theirs)), best]
    return np.where(top > MATCH, u[best], np.nan), top


def load_test_utilities(path: Path | None, test: dict) -> dict:
    """{tau: utility per test mask in `test`'s order} from a `--test-pass` file, or NaN without one."""
    out = {}
    for tau in PICKS:
        out[tau] = np.full(len(test[tau]["ids"]), np.nan)
        if path is None:
            continue
        with np.load(path) as z:
            vals = []
            for fid in test[tau]["ids"]:
                stem, k = fid.rsplit("_", 1)
                key = f"{stem}|{tau}"
                vals.append(z[key][int(k)] if key in z.files else np.nan)
            out[tau] = np.array(vals, np.float64)
    return out


# --- outputs ----------------------------------------------------------------------------------------------------------

def fmt(x) -> str:
    """A number to six places, empty for NaN or None."""
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.6f}"


def write_rows(path: Path, rows: list[dict], cols: list[str]) -> None:
    """A long-form CSV of dict rows."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r[c]) if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])


def write_test_tables(out_dir: Path, test: dict, util: dict, conf: dict, tag: str) -> None:
    """One CSV per pick: every test mask's measures, bearing call, raw utility and calibrated value."""
    stems = Dataset().test_stems
    for tau in PICKS:
        X, u = test[tau]["X"], util[tau]
        cal = np.where(np.isfinite(u), conf[tau, "calibrator"].predict(np.nan_to_num(u)), np.nan)
        rows = []
        for i, fid in enumerate(test[tau]["ids"]):
            x = X[i]
            rows.append(dict(frame=stems[test[tau]["f"][i]], filament_id=fid, centroid_x_px=x[COL["cx"]],
                             centroid_y_px=x[COL["cy"]], latitude_deg=x[COL["lat"]], longitude_deg=x[COL["lon"]],
                             r_over_R=x[COL["rho"]], area_px=int(x[COL["area"]]), length_px=x[COL["length"]],
                             tilt_deg=x[COL["tilt"]], hemisphere="N" if x[COL["lat"]] > 0 else "S",
                             barbs_8px=int(x[COL["barbs"]]), bearing_score=x[COL["S"]],
                             bearing_call=CALL_NAME[int(x[COL["call"]])], scorer_utility=u[i],
                             expected_matched_iou_per_record=cal[i]))
        write_rows(out_dir / f"test_filaments_v6_tau{tau:.2f}{tag}.csv", rows, list(rows[0]))


def calibration_rows(conf: dict) -> list[dict]:
    """`confidence_calibration.csv`: bins, summaries and the test calibrators' knots."""
    rows = []
    for tau in PICKS:
        for scheme in SCHEMES:
            c = conf[tau, scheme]
            for kind, bins in c["bins"].items():
                for b in bins:
                    lo, hi = ri.ci(b["draws"])
                    rows.append(dict(row="bin", tau=tau, scheme=scheme, value_kind=kind, bin_low=b["lo"],
                                     bin_high=b["hi"], masks=b["masks"], value=b["value"], realized=b["realized"],
                                     ci_low=lo, ci_high=hi))
                for m, (v, d) in (("ece", c["ece"][kind]), ("optimism", c["optimism"][kind])):
                    lo, hi = ri.ci(d)
                    rows.append(dict(row=m, tau=tau, scheme=scheme, value_kind=kind, bin_low="", bin_high="",
                                     masks=float(c["T"]["w"].sum()), value=v, realized="", ci_low=lo, ci_high=hi))
        iso = conf[tau, "calibrator"]
        for x, y in zip(iso.X_thresholds_, iso.y_thresholds_):
            rows.append(dict(row="test_calibrator_knot", tau=tau, scheme="random", value_kind="raw -> calibrated",
                             bin_low="", bin_high="", masks="", value=float(x), realized=float(y), ci_low="",
                             ci_high=""))
    return rows


def curve_rows(conf: dict) -> list[dict]:
    """`withheld_curve.csv`: the curve's measures with their intervals."""
    rows = []
    for tau in PICKS:
        for scheme in SCHEMES:
            for r in conf[tau, scheme]["curve"]:
                for m in ("masks_withheld_share", "matched_lost_per_record", "spurious_removed_per_record", "pq"):
                    v, d = r[m]
                    lo, hi = ri.ci(d)
                    rows.append(dict(tau=tau, scheme=scheme, cut=r["cut"], measure=m, value=v, ci_low=lo,
                                     ci_high=hi))
    return rows


def plots(conf: dict, out_dir: Path, tag: str) -> None:
    """`reliability.png` and `withheld_curve.png` (random partitions, both picks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, raw_c, cal_c = "#3d3d3a", "#8a8a85", "#2a6fb0"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, tau in zip(axes, PICKS):
        for kind, color in (("raw", raw_c), ("calibrated", cal_c)):
            bins = conf[tau, "random"]["bins"][kind]
            x = [b["value"] for b in bins]
            y = [b["realized"] for b in bins]
            ci = np.array([ri.ci(b["draws"]) for b in bins])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.2, lw=0)
            ax.plot(x, y, "o-", color=color, lw=2, ms=5, label=f"{kind} value")
        ax.plot([0, 1], [0, 1], color=ink, lw=1, ls=":")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"tau {tau}", xlabel="mask's value (expected matched IoU per record)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("realized matched IoU per record")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("v6 selected masks out of fold (random partitions 15-19), 90% frame bands", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability{tag}.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for tau, color in zip(PICKS, (cal_c, "#c26a1b")):
        cur = conf[tau, "random"]["curve"]
        sh = [r["masks_withheld_share"][0] for r in cur]
        pq = [r["pq"][0] for r in cur]
        ci = np.array([ri.ci(r["pq"][1]) for r in cur])
        axes[0].fill_between(sh, ci[:, 0], ci[:, 1], color=color, alpha=0.2, lw=0)
        axes[0].plot(sh, pq, "o-", color=color, lw=2, ms=4, label=f"tau {tau}")
        axes[1].plot([r["spurious_removed_per_record"][0] for r in cur],
                     [r["matched_lost_per_record"][0] for r in cur], "o-", color=color, lw=2, ms=4, label=f"tau {tau}")
    axes[0].set(xlabel="share of masks withheld", ylabel="PQ (out of fold)")
    axes[1].set(xlabel="spurious masks removed per record", ylabel="matched filaments lost per record")
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], color=ink, lw=1, ls=":")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Withholding masks below a calibrated-value cut (0 to 0.8 by 0.05), random partitions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"withheld_curve{tag}.png", dpi=150)
    plt.close(fig)


def sentences(phys: dict, bias: list[dict], conf: dict) -> list[str]:
    """One drafted report sentence for each item, filled from the tables."""
    def v(src, hemi, band, m):
        """Point and interval of a physics cell."""
        p, d = phys[src][hemi, band, m]
        return p, ri.ci(d)
    t = "test_0.26"
    tn, ts = v(t, "N", "all", "tilt_mean")[0], v(t, "S", "all", "tilt_mean")[0]
    an, as_ = v("annotators", "N", "all", "tilt_mean")[0], v("annotators", "S", "all", "tilt_mean")[0]
    hr, hci = v(t, "both", "all", "hemispheric_rule_share_legible")
    ah = v("annotators", "both", "all", "hemispheric_rule_share_legible")[0]
    al = v("annotators", "both", "all", "label_hemispheric_rule_share")[0]
    b = {(r["who"], r["tau"], r["scheme"], r["panel"], r["quantity"], r["measure"]): r for r in bias}
    k = ("v6 minus annotator", 0.26, "random", "all frames")
    h = ("second annotator minus annotator", "", "", "multiply annotated frames")
    km = ("v6 minus annotator", 0.26, "random", "multiply annotated frames")
    L, A = (b[(*k, q, "relative_bias")] for q in ("length", "area"))
    te, he, me = b[(*k, "tilt_long", "error")], b[(*h, "tilt_long", "error")], b[(*km, "tilt_long", "error")]
    s1 = (f"- (4.4) Read as projected image measures without a B0 correction, v6's 180 test masks at tau 0.26 keep "
          f"the annotators' tilt pattern (mean tilt {tn:+.1f} deg in the north and {ts:+.1f} deg in the south, "
          f"against {an:+.1f} and {as_:+.1f} deg on the annotators' polygons: the eastern tip poleward), and "
          f"{hr:.0%} of their legible bearing calls follow the hemispheric rule (90% CI {hci[0]:.0%} to "
          f"{hci[1]:.0%}; the same rule on the annotators' polygons {ah:.0%}, their own labels {al:.0%}); on "
          f"filaments both drew (IoU > 0.5) our masks are {L['value']:+.1%} in length (90% CI {L['ci_low']:+.1%} to "
          f"{L['ci_high']:+.1%}) and {A['value']:+.1%} in area ({A['ci_low']:+.1%} to {A['ci_high']:+.1%}) against "
          f"the annotator, and their tilt differs by {te['value']:.1f} deg on average on filaments at least {LONG} px "
          f"long, against {he['value']:.1f} deg between two annotators on the multiply annotated frames "
          f"(ours {me['value']:.1f} there).")
    c = conf[0.26, "random"]
    o_raw, o_cal = c["optimism"]["raw"], c["optimism"]["calibrated"]
    e_raw, e_cal = c["ece"]["raw"][0], c["ece"]["calibrated"][0]
    cur = {r["cut"]: r for r in c["curve"]}
    q = cur[0.3]
    s2 = (f"- (4.8) Each test mask carries the scorer's value recalibrated, cross-fitted on the scorer's own "
          f"partitions, to the expected matched IoU per annotator record (not a probability of being a filament): on "
          f"the selected out-of-fold masks at tau 0.26 the raw value is optimistic by {o_raw[0]:+.3f} (90% CI "
          f"{ri.ci(o_raw[1])[0]:+.3f} to {ri.ci(o_raw[1])[1]:+.3f}) and the calibrated one by {o_cal[0]:+.3f} "
          f"(expected calibration error {e_raw:.3f} to {e_cal:.3f}); withholding the masks below 0.3 removes "
          f"{q['masks_withheld_share'][0]:.0%} of them, {q['spurious_removed_per_record'][0]:.2f} spurious masks "
          f"per record for {q['matched_lost_per_record'][0]:.2f} matched filaments lost, and moves PQ from "
          f"{cur[0.0]['pq'][0]:.4f} to {q['pq'][0]:.4f}.")
    return [s1, s2]


def write_md(path: Path, phys: dict, bias: list[dict], conf: dict, checks: dict, sent: list[str]) -> None:
    """The summary of both items, the drafted sentences and the sanity checks."""
    def cell(src, hemi, band, m, d=1):
        """A physics cell as value [interval]."""
        p, dr = phys[src][hemi, band, m]
        if dr is None:
            return f"{p:.{d}f}"
        lo, hi = ri.ci(dr)
        return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
    srcs = [("annotators", "annotators (training)")]
    srcs += [(f"oof_{s}_{t}", f"v6 OOF {s} {t}") for t in PICKS for s in SCHEMES]
    srcs += [(f"test_{t}", f"v6 test {t}") for t in PICKS]
    L = ["# Physics of v6's masks and a usable confidence (items 4.4, 4.8)", "",
         "Projected image measures, no B0 or P correction (north up, east left). Values [90% frame interval]; out of "
         "fold the mean of a scheme's five scorer partitions (random 15-19, group 10-14). Script: "
         "`scripts/report_physics_confidence.py`.", "", "## (4.4) Tilt by hemisphere and |latitude| band (deg)", "",
         "| source | N 0-20 | N 20-40 | N 40-90 | S 0-20 | S 20-40 | S 40-90 |", "|---" * 7 + "|"]
    for s, name in srcs:
        L.append(f"| {name} | " + " | ".join(cell(s, h, b, "tilt_mean") for h in "NS"
                                             for b in ("0-20", "20-40", "40-90")) + " |")
    L += ["", "## (4.4) Hemispheric preference, size and position", "",
          "| source | objects | north share | legible share N / S | Right share of legible N / S | follows rule | "
          "median area / length (px) |", "|---" * 7 + "|"]
    for s, name in srcs:
        L.append(f"| {name} | {phys[s]['both', 'all', 'objects'][0]:.0f} | "
                 f"{cell(s, 'both', 'all', 'north_share', 3)} | {cell(s, 'N', 'all', 'legible_share', 3)} / "
                 f"{cell(s, 'S', 'all', 'legible_share', 3)} | {cell(s, 'N', 'all', 'right_share_legible', 3)} / "
                 f"{cell(s, 'S', 'all', 'right_share_legible', 3)} | "
                 f"{cell(s, 'both', 'all', 'hemispheric_rule_share_legible', 3)} | "
                 f"{cell(s, 'N', 'all', 'area_median', 0)}/"
                 f"{cell(s, 'S', 'all', 'area_median', 0)} N/S, {cell(s, 'N', 'all', 'length_median', 0)}/"
                 f"{cell(s, 'S', 'all', 'length_median', 0)} N/S |")
    a = phys["annotators"]
    L += ["", f"Annotators' own labels: Right share N {cell('annotators', 'N', 'all', 'label_right_share', 3)}, "
          f"S {cell('annotators', 'S', 'all', 'label_right_share', 3)}; follow the hemispheric rule "
          f"{cell('annotators', 'both', 'all', 'label_hemispheric_rule_share', 3)} "
          f"({a['both', 'all', 'objects'][0]:.0f} polygons).", "",
          "## (4.4) Matched objects: bias and error (`physics_bias.csv`)", "",
          "| who | tau | scheme | panel | quantity | bias | error (mean abs.) | relative bias | median abs. | pairs |",
          "|---" * 10 + "|"]
    look = {(r["who"], r["tau"], r["scheme"], r["panel"], r["quantity"], r["measure"]): r for r in bias}
    for key in dict.fromkeys((r["who"], r["tau"], r["scheme"], r["panel"], r["quantity"]) for r in bias):
        def g(m, d=2, key=key):
            """A bias row's value [interval]."""
            r = look.get((*key, m))
            if r is None:
                return ""
            return (f"{r['value']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]" if np.isfinite(r["ci_low"])
                    else f"{r['value']:.{d}f}")
        L.append(f"| {' | '.join(map(str, key))} | {g('bias')} | {g('error')} | {g('relative_bias', 3)} | "
                 f"{g('median_abs_error')} | {look[(*key, 'pairs')]['value']:.0f} |")
    L += ["", "## (4.8) Calibration on the selected masks", "",
          "| tau | scheme | masks | optimism raw | optimism calibrated | ECE raw | ECE calibrated |", "|---" * 7 + "|"]
    for tau in PICKS:
        for scheme in SCHEMES:
            c = conf[tau, scheme]
            o = {k: f"{v[0]:+.4f} [{ri.ci(v[1])[0]:+.4f}, {ri.ci(v[1])[1]:+.4f}]" for k, v in c["optimism"].items()}
            e = {k: f"{v[0]:.4f} [{ri.ci(v[1])[0]:.4f}, {ri.ci(v[1])[1]:.4f}]" for k, v in c["ece"].items()}
            L.append(f"| {tau} | {scheme} | {c['T']['w'].sum():.0f} | {o['raw']} | {o['calibrated']} | {e['raw']} | "
                     f"{e['calibrated']} |")
    L += ["", "## (4.8) Withheld-mask curve (random partitions)", "",
          "| tau | cut | masks withheld | matched lost / record | spurious removed / record | PQ |", "|---" * 6 + "|"]
    for tau in PICKS:
        for r in conf[tau, "random"]["curve"]:
            if r["cut"] in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                cs = {m: f"{r[m][0]:.3f} [{ri.ci(r[m][1])[0]:.3f}, {ri.ci(r[m][1])[1]:.3f}]"
                      for m in ("masks_withheld_share", "matched_lost_per_record", "spurious_removed_per_record")}
                L.append(f"| {tau} | {r['cut']} | {cs['masks_withheld_share']} | {cs['matched_lost_per_record']} | "
                         f"{cs['spurious_removed_per_record']} | {r['pq'][0]:.4f} [{ri.ci(r['pq'][1])[0]:.4f}, "
                         f"{ri.ci(r['pq'][1])[1]:.4f}] |")
    L += ["", "## Drafted sentences", "", *sent, "", "## Sanity checks", "", "```", json.dumps(checks, indent=1),
          "```", ""]
    path.write_text("\n".join(L))


def sanity(frames: dict, stems: list[str], conf: dict, test: dict, util: dict) -> dict:
    """PQ of the rebuilt masks against the record (0.4711 / 0.4691 at tau 0.26), mask counts against the test
    files, the annotators' polygons against their earlier count, and the test utilities' coverage."""
    out = {}
    for scheme in SCHEMES:
        rec = ri.SANITY[scheme, 0.26]
        v = conf[0.26, scheme]["curve"][0]["pq"][0]
        out[f"pq_cut0_{scheme}_0.26"] = [round(v, 5), rec, bool(abs(round(v, 4) - rec) < 1.5e-4)]
        out[f"pq_cut0_{scheme}_0.30"] = round(conf[0.30, scheme]["curve"][0]["pq"][0], 5)
    out["annotator_polygons"] = int(sum(len(x) for s in stems for x in frames[s]["gt"]))
    out["r2_round26_annotator_polygons"] = 8199
    for tau in PICKS:
        out[f"test_masks_{tau}"] = len(test[tau]["ids"])
        out[f"test_frames_with_masks_{tau}"] = int(len(np.unique(test[tau]["f"])))
        out[f"test_utilities_present_{tau}"] = int(np.isfinite(util[tau]).sum())
        T = conf[tau, "random"]["T"]
        out[f"oof_masks_per_partition_random_{tau}"] = round(float(T["w"].sum()), 1)
    return out


def main():
    """Run the frame passes, compute both items and write every output (or, with --test-pass, the test utilities)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=None, help="smoke run on the first N training frames")
    ap.add_argument("--test-utilities", type=Path, default=None, help="a --test-pass file")
    ap.add_argument("--test-pass", type=Path, default=None, help="write the test utilities here and stop (GPU)")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="--test-pass on the first N test frames")
    ap.add_argument("--test-csv", nargs=2, type=Path, default=None,
                    help="the release's test submissions at tau 0.26 and 0.30 (default v6's)")
    ri.add_release_args(ap)
    a = ap.parse_args()
    global RUN
    cache_in = ri.apply_release_args(a)
    RUN = ri.RUN
    if a.test_csv:
        TEST_CSV.update(zip(PICKS, a.test_csv))
    if a.test_pass:
        run_test_pass(a.test_pass, a.weights, a.device, a.limit)
        return
    tag = f"_n{a.frames}" if a.frames else ""
    cache = a.out_dir / "physics_cache"
    cache.mkdir(parents=True, exist_ok=True)
    scores = ri.load_scores(cache_in)
    stems = sorted(scores["random", am.FIRST_RANDOM])
    ds = Dataset()
    use = stems[:a.frames]
    frames = frame_pass(use, scores, a.workers, cache, tag)
    test = test_pass(a.workers, cache)
    util = load_test_utilities(a.test_utilities, test)
    phys_rows, phys = physics_rows(frames, use, test, a.boot)
    bias = bias_table(frames, use, a.boot)
    conf = confidence(frames, use, a.boot, ds, stems)
    write_rows(a.out_dir / f"physics{tag}.csv", phys_rows,
               ["source", "tau", "scheme", "hemisphere", "band", "measure", "value", "ci_low", "ci_high"])
    write_rows(a.out_dir / f"physics_bias{tag}.csv", bias,
               ["who", "tau", "scheme", "panel", "quantity", "measure", "value", "ci_low", "ci_high"])
    write_rows(a.out_dir / f"confidence_calibration{tag}.csv", calibration_rows(conf),
               ["row", "tau", "scheme", "value_kind", "bin_low", "bin_high", "masks", "value", "realized", "ci_low",
                "ci_high"])
    write_rows(a.out_dir / f"withheld_curve{tag}.csv", curve_rows(conf),
               ["tau", "scheme", "cut", "measure", "value", "ci_low", "ci_high"])
    write_test_tables(a.out_dir, test, util, conf, tag)
    plots(conf, a.out_dir, tag)
    checks = sanity(frames, use, conf, test, util)
    sent = sentences(phys, bias, conf)
    write_md(a.out_dir / f"report_physics_confidence{tag}.md", phys, bias, conf, checks, sent)
    print(json.dumps(checks, indent=1))
    print("\n".join(sent))


if __name__ == "__main__":
    main()

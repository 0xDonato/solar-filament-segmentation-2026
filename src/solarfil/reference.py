"""A selection beside its annotators: the model scored the way a second annotator of the same frame is scored.

The host scores a submission against every annotator record of a frame and pools the counts (`solarfil.metric`).
On a frame that several annotators drew, the same code can score each annotator against every other one, in every
ordered pair of records (record k as the ground truth, record j in place of the predictions). That gives each
number of the evaluation a human reference: how far two annotators of the same frame disagree under the same
rules. Every comparison below is made for a selection (its masks against every record of the frame,
`against_masks`) and for the annotators (every ordered pair of records, `against_annotators`):

* the host's distributions (`metric.Report`): the IoU and Dice of every overlapping pair (IoU > 0), and the number
  of overlapping partners of each filament (1:n; 1:0 is a filament no mask overlaps) and of each mask (n:1; 0:1 is a
  mask that overlaps no filament of the record);
* relation support (`relations`): how many splits (a filament that two or more masks overlap) and merges (a mask
  that overlaps two or more filaments) copy a choice another annotator of the frame made;
* spine continuity (`spine_cover`): how much of the spine an annotator drew along each filament lies inside one
  mask, and across how many masks it runs;
* for a selection, the anatomy of the 0:1 and 1:0 bars: whether another annotator of the frame drew the masks no
  record's filament overlaps, and how large they and the missed filaments are.

`*_summary` pool the frames' comparisons into the numbers the report quotes, with 90% frame-bootstrap intervals.
`mask_shape` counts the pieces and holes of a mask: the annotation protocol draws a filament as one piece without
holes, and the pipeline's masks are built that way.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
from pycocotools import mask as mask_util

from . import metric
from .data import H, Observation, W

#: one row per (annotated spine, set of masks it is read against)
SPINE = np.dtype([("inside_any", "f8"), ("inside_one", "f8"), ("pieces", "i8"), ("samples", "i8")])
#: one row per mask of a selection: area, long side of its bounding box, the records of its frame it overlaps
MASK = np.dtype([("area", "i8"), ("length", "f8"), ("records_hit", "i8"), ("records", "i8")])
#: one row per annotated filament: area, long side, overlapped by no mask, other records that drew it too
FILAMENT = np.dtype([("area", "i8"), ("length", "f8"), ("missed", "?"), ("others_drew", "i8"), ("others", "i8")])

PIECE_SAMPLES = 3  # spine samples a mask must hold to count as a piece the spine runs through
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
HOST_BINS = np.linspace(0, 1, 51)  # the host's 50-bin histograms on [0, 1]


@dataclass(frozen=True)
class Overlap:
    """Host-exact float32 IoU and Dice of a record's filaments (rows) with a set of masks (columns)."""

    iou: np.ndarray
    dice: np.ndarray

    @classmethod
    def of(cls, filaments: list[dict], masks: list[dict]) -> Overlap:
        """From COCO RLEs of the filaments and the masks."""
        return cls(*metric.iou_dice(*metric.overlap_counts(filaments, masks)))


@dataclass
class Frame:
    """One frame's annotator records: each record's filament masks (COCO RLEs) and spines ((n, 2) x, y polylines)."""

    stem: str
    records: list[list[dict]]
    spines: list[list[np.ndarray]]
    shape: tuple[int, int] = (H, W)

    def __post_init__(self):
        assert [len(r) for r in self.records] == [len(s) for s in self.spines], self.stem

    @classmethod
    def of(cls, obs: Observation) -> Frame:
        """From a training observation: each record's GT RLEs and spines."""
        return cls(obs.stem, [metric.gt_rles_cached(r) for r in obs.records],
                   [[f.spine_xy() for f in r.filaments] for r in obs.records])

    @cached_property
    def pairs(self) -> dict[tuple[int, int], Overlap]:
        """Record k's filaments against record j's, for every ordered pair k != j."""
        n = len(self.records)
        return {(k, j): Overlap.of(self.records[k], self.records[j]) for k in range(n) for j in range(n) if k != j}

    @cached_property
    def labels(self) -> list[np.ndarray]:
        """Label image of each record's filaments."""
        return [label_image(r, self.shape) for r in self.records]

    def drawn_by_others(self, k: int) -> np.ndarray:
        """For each filament of record k, how many other records drew a filament that overlaps it."""
        out = np.zeros(len(self.records[k]), np.int64)
        for j in range(len(self.records)):
            if j != k:
                out += (self.pairs[k, j].iou > 0).any(axis=1)
        return out


@dataclass
class Comparison:
    """One frame's masks (a selection's, or each annotator's) compared with the frame's records."""

    stem: str
    records: int  # annotator records of the frame
    report: metric.Report  # the host's counts, overlapping pairs and degrees, pooled over the comparisons
    relations: np.ndarray  # [splits, supported splits, merges, supported merges]
    spines: np.ndarray  # SPINE rows
    masks: np.ndarray | None = None  # MASK rows (selections only)
    filaments: np.ndarray | None = None  # FILAMENT rows (selections only)

    @property
    def comparisons(self) -> int:
        """Records scored: one per record for a selection, one per ordered pair of records for the annotators."""
        return len(self.report.ious)


def against_masks(frame: Frame, masks: list[dict]) -> Comparison:
    """A selection's masks scored against every record of the frame, as the host scores a submission."""
    overlaps = [Overlap.of(gt, masks) for gt in frame.records]
    report = metric.Report()
    rel = np.zeros(4, np.int64)
    for k, o in enumerate(overlaps):
        report.add_matrices(frame.stem, o.iou, o.dice)
        rel += relations(o.iou, [p.iou for j, p in enumerate(overlaps) if j != k])
    labels = label_image(masks, frame.shape)
    spines = _concat([spine_rows(labels, s) for s in frame.spines], SPINE)
    return Comparison(frame.stem, len(frame.records), report, rel, spines, _mask_rows(masks, overlaps),
                      _filament_rows(frame, overlaps))


def _mask_rows(masks: list[dict], overlaps: list[Overlap]) -> np.ndarray:
    """MASK rows: each mask's area and length, and how many of the frame's records it overlaps."""
    n = len(overlaps)
    hit = np.array([(o.iou > 0).any(axis=0) for o in overlaps]).reshape(n, len(masks))
    return _table(MASK, *_area_length(masks), hit.sum(axis=0), np.full(len(masks), n))


def _filament_rows(frame: Frame, overlaps: list[Overlap]) -> np.ndarray:
    """FILAMENT rows: each annotated filament's area and length, whether no mask overlaps it, and how many of the
    frame's other records drew it too."""
    n = len(frame.records)
    return _concat([_table(FILAMENT, *_area_length(gt), ~(o.iou > 0).any(axis=1), frame.drawn_by_others(k),
                           np.full(len(gt), n - 1))
                    for k, (gt, o) in enumerate(zip(frame.records, overlaps))], FILAMENT)


def against_annotators(frame: Frame) -> Comparison:
    """Every record scored as a second annotator: record j's filaments in place of the masks against record k, for
    every ordered pair k != j. A split or merge between two records is supported by a third record."""
    n = len(frame.records)
    report = metric.Report()
    rel = np.zeros(4, np.int64)
    spines = []
    for (k, j), o in frame.pairs.items():
        report.add_matrices(frame.stem, o.iou, o.dice)
        rel += relations(o.iou, [frame.pairs[t, j].iou for t in range(n) if t not in (k, j)])
        spines.append(spine_rows(frame.labels[j], frame.spines[k]))
    return Comparison(frame.stem, n, report, rel, _concat(spines, SPINE))


def relations(iou: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """Splits and merges of a set of masks against one record, and how many another record supports.

    `iou` is the record's IoU matrix (filaments x masks) and `others` holds every other record's against the same
    masks. A split, a filament that two or more masks overlap, is supported when at least two of those masks each
    match (IoU > 0.5) a filament of one other record: that annotator drew the structure in pieces, as the masks do.
    A merge, a mask that overlaps two or more filaments, is supported when it matches a filament of another record:
    that annotator drew the structure as one piece. Returns [splits, supported splits, merges, supported merges].
    """
    hit = iou > 0
    split = hit.sum(axis=1) >= 2
    merge = hit.sum(axis=0) >= 2
    matched = np.array([(o > metric.IOU_THRESHOLD).any(axis=0) for o in others]).reshape(len(others), iou.shape[1])
    split_supported = ((hit[split].astype(np.int64) @ matched.T) >= 2).any(axis=1)
    merge_supported = matched[:, merge].any(axis=0)
    return np.array([split.sum(), split_supported.sum(), merge.sum(), merge_supported.sum()], np.int64)


def label_image(rles: list[dict], shape: tuple[int, int] = (H, W)) -> np.ndarray:
    """int32 image of masks 1..n (0 = background); where masks overlap, the later one wins."""
    labels = np.zeros(shape, np.int32)
    if rles:
        stack = mask_util.decode(rles)
        for k in range(stack.shape[2]):
            labels[stack[:, :, k] > 0] = k + 1
    return labels


def spine_samples(xy: np.ndarray, shape: tuple[int, int] = (H, W)) -> np.ndarray:
    """Integer (x, y) pixels along a polyline: each segment from end to end in steps of at most one pixel (so a
    vertex is sampled once for each segment it ends), clipped to the frame."""
    points = np.concatenate([_segment(a, b) for a, b in zip(xy[:-1], xy[1:])]) if len(xy) > 1 else xy
    p = np.round(points).astype(np.int64).reshape(-1, 2)
    p[:, 0] = np.clip(p[:, 0], 0, shape[1] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, shape[0] - 1)
    return p


def _segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Points from a to b, both ends included, at most one pixel apart."""
    n = max(2, int(np.ceil(np.hypot(*(b - a)))) + 1)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


def spine_cover(labels: np.ndarray, xy: np.ndarray) -> tuple[float, float, int, int]:
    """How a spine lies on a label image of masks: the share of its samples inside any mask, the share inside the
    one mask that holds most of them, the masks holding at least `PIECE_SAMPLES` samples (the pieces it runs
    through; 0 means untouched), and the number of samples."""
    p = spine_samples(xy, labels.shape)
    v = labels[p[:, 1], p[:, 0]]
    _, counts = np.unique(v[v > 0], return_counts=True)
    inside_one = counts.max() / len(v) if counts.size else 0.0
    return float((v > 0).mean()), float(inside_one), int((counts >= PIECE_SAMPLES).sum()), len(v)


def spine_rows(labels: np.ndarray, spines: list[np.ndarray]) -> np.ndarray:
    """SPINE rows of a record's spines on a label image of the masks they are read against."""
    return np.array([spine_cover(labels, xy) for xy in spines if len(xy)], dtype=SPINE)


def mask_shape(mask: np.ndarray) -> tuple[int, int]:
    """Pieces (8-connected) and holes (4-connected background regions the mask encloses) of a boolean mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0
    crop = np.pad(mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1], 1).astype(np.uint8)
    pieces = cv2.connectedComponents(crop, connectivity=8)[0] - 1
    holes = cv2.connectedComponents(1 - crop, connectivity=4)[0] - 2  # minus the mask and the outside
    return pieces, holes


# --- pooling the frames ------------------------------------------------------------------------------------------

def merge_reports(reports: Iterable[metric.Report]) -> metric.Report:
    """One report pooling several: the counts, the overlapping pairs and the degrees."""
    out = metric.Report()
    for r in reports:
        out.total = out.total + r.total
        for stem, pq in r.per_image.items():
            out.per_image[stem] = out.per_image.get(stem, metric.PQ()) + pq
        out.ious += r.ious
        out.dices += r.dices
        out.gt_degree.update(r.gt_degree)
        out.pred_degree.update(r.pred_degree)
    return out


def frame_row(report: metric.Report) -> np.ndarray:
    """[TP, FP, FN, IoU sum of the TPs, overlapping pairs, their IoU sum, their Dice sum] of one frame's report."""
    iou, dice = report.iou_values.astype(np.float64), report.dice_values.astype(np.float64)
    return np.concatenate([report.total.as_array(), [iou.size, iou.sum(), dice.sum()]])


def frame_intervals(rows: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    """90% intervals of pooled PQ and of the mean IoU and Dice of the overlapping pairs from `frame_row` rows, the
    frames resampled with replacement (the PQ interval is `metric.bootstrap_pq`'s for the same seed)."""
    rng = np.random.default_rng(seed)
    s = rows[rng.integers(0, len(rows), size=(n, len(rows)))].sum(axis=1)
    stats = dict(pq=s[:, 3] / (s[:, 0] + 0.5 * s[:, 1] + 0.5 * s[:, 2]), iou=s[:, 5] / s[:, 4], dice=s[:, 6] / s[:, 4])
    return {k: [_r(np.quantile(v, 0.05), 5), _r(np.quantile(v, 0.95), 5)] for k, v in stats.items()}


def distribution(values: np.ndarray) -> dict:
    """Mean, quantiles and the host's 50-bin histogram counts of IoU or Dice values."""
    v = values.astype(np.float64)
    return dict(mean=_r(v.mean()), **{f"p{round(q * 100)}": _r(np.quantile(v, q)) for q in QUANTILES},
                hist50=np.histogram(v, HOST_BINS)[0].tolist())


def legend_counts(report: metric.Report) -> dict[str, int]:
    """The four counts `plots.reference_figure` prints for a series: filaments no mask overlaps (1:0), masks that
    overlap no filament (0:1), splits (1:n, n >= 2) and merges (n:1, n >= 2). Anything with the report's two degree
    counters will do (`plots.ReferenceSeries`)."""
    g, m = report.gt_degree, report.pred_degree
    return {"1:0": g[0], "0:1": m[0], "splits": _at_least(g, 2), "merges": _at_least(m, 2)}


def host_summary(parts: list[Comparison], n_boot: int = 2000, seed: int = 0) -> dict:
    """The host's numbers pooled over the frames: PQ, SQ, RQ and counts; the IoU and Dice of the overlapping pairs;
    the filament (1:n) and mask (n:1) degrees, also per comparison; 90% frame-bootstrap intervals on PQ and on the
    two means."""
    parts = [p for p in parts if p.comparisons]
    report = merge_reports(p.report for p in parts)
    ci = frame_intervals(np.stack([frame_row(p.report) for p in parts]), n_boot, seed)
    n = len(report.ious)
    counts = legend_counts(report)
    return dict(frames=len(parts), comparisons=n, **report.total.summary(), pq_ci90=ci["pq"],
                pairs=int(report.iou_values.size), iou=distribution(report.iou_values) | {"mean_ci90": ci["iou"]},
                dice=distribution(report.dice_values) | {"mean_ci90": ci["dice"]},
                filament_degree=_degrees(report.gt_degree), mask_degree=_degrees(report.pred_degree),
                counts=counts, per_comparison={k: _r(v / n) for k, v in counts.items()})


def relation_summary(parts: list[Comparison]) -> dict:
    """Splits and merges and their support, on the frames another annotator drew too (two or more records)."""
    multi = [p for p in parts if p.records > 1 and p.comparisons]
    splits, split_sup, merges, merge_sup = (int(v) for v in sum((p.relations for p in multi), np.zeros(4, np.int64)))
    n = sum(p.comparisons for p in multi)
    return dict(frames=len(multi), comparisons=n, splits=splits, splits_supported=split_sup, merges=merges,
                merges_supported=merge_sup, splits_per_comparison=_r(splits / max(n, 1)),
                merges_per_comparison=_r(merges / max(n, 1)), splits_supported_share=_r(split_sup / max(splits, 1)),
                merges_supported_share=_r(merge_sup / max(merges, 1)))


def spine_summary(parts: list[Comparison]) -> dict:
    """Spine continuity: the mean share of a spine inside any mask and inside one mask, the shares of spines at
    least 90% or 70% inside one mask, untouched (no mask holds `PIECE_SAMPLES` of its samples), and among the
    touched ones, inside one mask and running through two or more masks."""
    s = _concat([p.spines for p in parts], SPINE)
    one, touched = s["inside_one"], s["pieces"] >= 1
    return dict(spines=len(s), inside_any=_r(s["inside_any"].mean()), inside_one=_r(one.mean()),
                inside_one_ge_0_9=_r((one >= 0.9).mean()), inside_one_ge_0_7=_r((one >= 0.7).mean()),
                untouched=_r((~touched).mean()),
                touched=dict(spines=int(touched.sum()), inside_one=_r(one[touched].mean()),
                             inside_one_ge_0_9=_r((one[touched] >= 0.9).mean()),
                             in_pieces=_r((s["pieces"][touched] >= 2).mean())))


def miss_summary(parts: list[Comparison]) -> dict:
    """What makes a selection's 0:1 and 1:0 bars.

    The 0:1 bar counts a mask once for every record of its frame whose filaments it does not overlap. Its share
    from masks that another record's filament does overlap is what another annotator drew; masks no record overlaps
    were drawn by no annotator, and their areas and lengths are set against the annotated filaments' (quantiles).
    The 1:0 bar counts the filaments no mask overlaps; on frames with several records, whether another record drew
    an overlapping filament tells a filament every annotator drew from one only its own annotator did.
    """
    m = _concat([p.masks for p in parts], MASK)
    f = _concat([p.filaments for p in parts], FILAMENT)
    a5, a10, a50 = np.quantile(f["area"], [0.05, 0.1, 0.5])
    l5, l10, l50 = np.quantile(f["length"], [0.05, 0.1, 0.5])
    missing = m["records"] - m["records_hit"]
    nobody, somebody = m["records_hit"] == 0, (m["records_hit"] > 0) & (missing > 0)
    out = dict(filament_area_p5_p10_p50=[round(a5), round(a10), round(a50)],
               filament_length_p5_p10_p50=[_r(l5, 1), _r(l10, 1), _r(l50, 1)],
               masks=len(m), bar_0_1=int(missing.sum()), masks_no_record_overlaps=int(nobody.sum()),
               masks_some_records_overlap=int(somebody.sum()),
               bar_0_1_drawn_by_another_annotator=_r(missing[somebody].sum() / max(missing.sum(), 1)))
    for name, keep in (("masks_no_record_overlaps", nobody), ("masks", np.ones(len(m), bool))):
        area, length = m["area"][keep], m["length"][keep]
        if area.size:
            out[f"{name}_area_p10_p50_p90"] = [round(x) for x in np.quantile(area, [0.1, 0.5, 0.9])]
            out[f"{name}_length_p10_p50_p90"] = [_r(x, 1) for x in np.quantile(length, [0.1, 0.5, 0.9])]
            out[f"{name}_below_filament_area_p5"] = _r((area < a5).mean())
            out[f"{name}_below_filament_area_p10"] = _r((area < a10).mean())
            out[f"{name}_below_filament_length_p10"] = _r((length < l10).mean())
    missed, multi, alone = f["missed"], f["others"] > 0, f["others_drew"] == 0
    out.update(bar_1_0=int(missed.sum()), bar_1_0_multi=int((missed & multi).sum()),
               bar_1_0_multi_drawn_by_no_other_record=int((missed & multi & alone).sum()),
               hit_multi=int((~missed & multi).sum()),
               hit_multi_drawn_by_no_other_record=int((~missed & multi & alone).sum()))
    if missed.any():
        area = f["area"][missed]
        out["bar_1_0_area_p10_p50_p90"] = [round(x) for x in np.quantile(area, [0.1, 0.5, 0.9])]
        out["bar_1_0_below_filament_area_p10"] = _r((area < a10).mean())
    return out


def summarize(parts: list[Comparison], n_boot: int = 2000, seed: int = 0) -> dict:
    """Every summary of this module for one set of comparisons (the miss anatomy for a selection only)."""
    out = dict(host=host_summary(parts, n_boot, seed), relations=relation_summary(parts),
               spines=spine_summary(parts))
    if parts and parts[0].masks is not None:
        out["misses"] = miss_summary(parts)
    return out


def _area_length(rles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel area and the long side of the bounding box of each mask."""
    if not rles:
        return np.zeros(0, np.int64), np.zeros(0)
    return mask_util.area(rles).astype(np.int64), mask_util.toBbox(rles)[:, 2:].max(axis=1)


def _table(dtype: np.dtype, *columns) -> np.ndarray:
    out = np.zeros(len(columns[0]), dtype)
    for name, col in zip(dtype.names, columns):
        out[name] = col
    return out


def _concat(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype)


def _degrees(c: Counter) -> dict[str, int]:
    return {str(d): int(v) for d, v in sorted(c.items())}


def _at_least(c: Counter, d: int) -> int:
    return int(sum(v for k, v in c.items() if k >= d))


def _r(x, digits: int = 4) -> float:
    return round(float(x), digits)

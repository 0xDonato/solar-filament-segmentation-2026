"""Host-exact pooled Panoptic Quality and the judges' distributions.

This reproduces the host's self-evaluation notebook (Kaggle `azimahmadzadeh/self-evaluation-notebook`):

* every annotator record of an image is compared with *all* predictions of that image;
* a (GT, prediction) pair is a true positive when IoU > 0.5 (strict); a prediction that hits no GT filament of
  the record is a false positive, a GT filament hit by no prediction a false negative;
* TP, FP, FN and the matched IoUs are pooled over all records:
  ``PQ = sum(IoU of TPs) / (TP + 0.5 FP + 0.5 FN)``.

IoUs are computed on RLEs with pycocotools and the threshold is applied to the float32 ratio of the integer
pixel counts, as the host's torch code does. The judges' plots use the same matrices: the IoU and Dice
distributions of every overlapping pair (value > 0), and the number of overlapping partners of each GT filament
(1:n) and of each prediction (n:1).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


def overlap_counts(gt: list[dict], pred: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersection counts (n_gt, n_pred) and the areas of the GT and predicted masks (COCO RLE dicts)."""
    a_gt = mask_util.area(gt).astype(np.int64) if gt else np.zeros(0, np.int64)
    a_pr = mask_util.area(pred).astype(np.int64) if pred else np.zeros(0, np.int64)
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), np.int64), a_gt, a_pr
    iou = np.asarray(mask_util.iou(gt, pred, [0] * len(pred)), dtype=np.float64).reshape(len(gt), len(pred))
    s = a_gt[:, None] + a_pr[None, :]
    inter = np.rint(iou * s / (1.0 + iou)).astype(np.int64)  # IoU = I / (s - I)  =>  I = IoU s / (1 + IoU)
    return inter, a_gt, a_pr


def iou_dice(inter: np.ndarray, a_gt: np.ndarray, a_pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 IoU and Dice matrices, as the host computes them from the counts."""
    i32 = inter.astype(np.float32)
    s32 = (a_gt[:, None] + a_pr[None, :]).astype(np.float32)
    union = s32 - i32
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union == 0, np.float32(0), i32 / union).astype(np.float32)
        dice = np.where(union == 0, np.float32(0), 2 * i32 / s32).astype(np.float32)
    return iou, dice


@dataclass
class PQ:
    """Pooled counts; add records with `add_record` or merge with `+`."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    def add_matrix(self, iou: np.ndarray) -> PQ:
        """Add one record's counts from its (n_gt, n_pred) IoU matrix: pairs with IoU > 0.5 are true positives,
        unmatched predictions false positives and unmatched GT filaments false negatives. Returns self."""
        n_gt, n_pred = iou.shape
        if n_gt == 0:
            self.fp += n_pred
        elif n_pred == 0:
            self.fn += n_gt
        else:
            hit = iou > IOU_THRESHOLD
            self.tp += int(hit.sum())
            self.iou_sum += float(sum(iou[hit].tolist()))
            self.fp += int((hit.sum(axis=0) == 0).sum())
            self.fn += int((hit.sum(axis=1) == 0).sum())
        return self

    def __add__(self, other: PQ) -> PQ:
        return PQ(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.iou_sum + other.iou_sum)

    @property
    def pq(self) -> float:
        """Panoptic Quality, IoU sum / (TP + 0.5 FP + 0.5 FN); 0 when there is nothing to count."""
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        """Segmentation quality, the mean IoU of the true positives."""
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        """Recognition quality, TP / (TP + 0.5 FP + 0.5 FN)."""
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0

    def as_array(self) -> np.ndarray:
        """[TP, FP, FN, IoU sum] as float64."""
        return np.array([self.tp, self.fp, self.fn, self.iou_sum], dtype=np.float64)

    def summary(self) -> dict:
        """PQ, SQ and RQ (rounded to five decimals) and the counts, as a dict."""
        return dict(pq=round(self.pq, 5), sq=round(self.sq, 5), rq=round(self.rq, 5),
                    tp=self.tp, fp=self.fp, fn=self.fn)


@dataclass
class Report:
    """Pooled PQ plus everything the judges plot, accumulated record by record."""

    total: PQ = field(default_factory=PQ)
    per_image: dict[str, PQ] = field(default_factory=dict)
    ious: list[np.ndarray] = field(default_factory=list)  # overlapping pairs (IoU > 0)
    dices: list[np.ndarray] = field(default_factory=list)
    gt_degree: Counter = field(default_factory=Counter)  # partners (IoU > 0) per GT filament: 1:n
    pred_degree: Counter = field(default_factory=Counter)  # partners (IoU > 0) per prediction: n:1

    def add_record(self, stem: str, gt: list[dict], pred: list[dict]) -> None:
        """Score one record's GT RLEs against the frame's predicted RLEs."""
        inter, a_gt, a_pr = overlap_counts(gt, pred)
        self.add_matrices(stem, *iou_dice(inter, a_gt, a_pr))

    def add_matrices(self, stem: str, iou: np.ndarray, dice: np.ndarray) -> None:
        """Add one record from its IoU and Dice matrices: the pooled and per-image counts, the overlapping pairs' values
        and each GT filament's and prediction's number of overlapping partners."""
        part = PQ().add_matrix(iou)
        self.total = self.total + part
        self.per_image[stem] = self.per_image.get(stem, PQ()) + part
        self.ious.append(iou[iou > 0])
        self.dices.append(dice[dice > 0])
        n_gt, n_pred = iou.shape
        if n_gt and n_pred:
            hit = iou > 0
            self.gt_degree.update(hit.sum(axis=1).tolist())
            self.pred_degree.update(hit.sum(axis=0).tolist())
        else:
            self.gt_degree.update([0] * n_gt)
            self.pred_degree.update([0] * n_pred)

    @property
    def iou_values(self) -> np.ndarray:
        """IoU of every overlapping pair so far."""
        return np.concatenate(self.ious) if self.ious else np.zeros(0, np.float32)

    @property
    def dice_values(self) -> np.ndarray:
        """Dice of every overlapping pair so far."""
        return np.concatenate(self.dices) if self.dices else np.zeros(0, np.float32)

    def summary(self) -> dict:
        """The pooled PQ summary plus the record count, the 1:n and n:1 degree counts, and the number of broken GT
        filaments (degree >= 2) and merged predictions (degree >= 2)."""
        out = self.total.summary()
        out["records"] = int(sum(1 for _ in self.ious))
        out["gt_1:n"] = dict(sorted(self.gt_degree.items()))
        out["pred_n:1"] = dict(sorted(self.pred_degree.items()))
        out["broken_gt"] = int(sum(c for d, c in self.gt_degree.items() if d >= 2))  # GT overlapped by >= 2 preds
        out["merged_pred"] = int(sum(c for d, c in self.pred_degree.items() if d >= 2))  # pred overlapping >= 2 GT
        return out


def evaluate(observations, predictions: dict[str, list[dict]], stems=None) -> Report:
    """Score predictions (stem -> list of COCO RLE dicts) against the records of the given observations.

    `observations` maps stem -> `data.Observation`; a stem without predictions scores all its filaments as FN,
    as the host's loop over GT records does.
    """
    rep = Report()
    for stem in stems if stems is not None else observations:
        pred = predictions.get(stem, [])
        for rec in observations[stem].records:
            rep.add_record(stem, gt_rles_cached(rec), pred)
    return rep


_GT_CACHE: dict[str, list[dict]] = {}


def gt_rles_cached(record) -> list[dict]:
    """GT RLEs of a record, rasterised once per process."""
    rles = _GT_CACHE.get(record.record_id)
    if rles is None:
        rles = _GT_CACHE[record.record_id] = record.rles()
    return rles


def bootstrap_pq(per_image: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """90% bootstrap interval of pooled PQ from per-image [TP, FP, FN, IoU sum] rows (resampling images)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_image), size=(n, len(per_image)))
    s = per_image[idx].sum(axis=1)
    q = s[:, 3] / (s[:, 0] + 0.5 * s[:, 1] + 0.5 * s[:, 2])
    return float(np.quantile(q, 0.05)), float(np.quantile(q, 0.95))


# --- fast scoring of pixel-disjoint predictions given as a label image -------------------------------------------

class GTIndex:
    """Flat pixel indices of every GT filament of every record, built once per observation.

    For pixel-disjoint predictions stored as a label image (0 = background, 1..n = instances), intersections are
    ``bincount(labels[gt_pixels])``: the same integer counts as the host's mask products, far faster.
    """

    def __init__(self, obs):
        self.stem = obs.stem
        self.records: list[list[np.ndarray]] = []
        for rec in obs.records:
            idx = []
            for rle in gt_rles_cached(rec):
                m = mask_util.decode(rle)
                idx.append(np.flatnonzero(m.ravel()).astype(np.int32))
            self.records.append(idx)

    def score(self, labels: np.ndarray, n: int | None = None, report: Report | None = None) -> PQ:
        """Pooled counts of a label image over this observation's records."""
        flat = labels.ravel()
        n = int(flat.max()) if n is None else n
        a_pr = np.bincount(flat, minlength=n + 1)[1:].astype(np.int64)
        present = np.flatnonzero(a_pr)  # instance ids may have gaps
        a_pr = a_pr[present]
        total = PQ()
        for idx in self.records:
            a_gt = np.array([len(i) for i in idx], np.int64)
            inter = np.zeros((len(idx), len(present)), np.int64)
            for g, i in enumerate(idx):
                inter[g] = np.bincount(flat[i], minlength=n + 1)[1:][present]
            iou, dice = iou_dice(inter, a_gt, a_pr)
            total = total + PQ().add_matrix(iou)
            if report is not None:
                report.add_matrices(self.stem, iou, dice)
        return total


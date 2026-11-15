"""Tiles of the report's morphology panel (Fig. 3): fixed rules pick real structures from out-of-fold selections.

Every rule is fixed before drawing. It ranks the candidates of every frame, and the panel takes the best candidate
whose frame no earlier tile uses: one tile per frame, ties broken by stem order. A filament is matched when one of
our masks has IoU > 0.5 with it (the host's float32 rule), and its annotated polygon is the record filament of best
IoU. The rules (`RULES`), on the selections at tau 0.26 unless stated:
- (a) barbs: among our matched masks whose annotated polygon is at least the median annotated area, the one whose
  polygon is least convex (lowest solidity, area over convex hull area);
- (b) long filament in one piece: the matched mask with the longest annotated spine (polyline length);
- (c) near neighbours kept apart: two of our masks matching two different filaments of the same record, with the
  smallest gap (background pixels between their nearest pixels, at most `MAX_GAP`);
- (d) compact dark structure rejected: the tree candidate with the lowest flat-field minimum (`f_min`) among those
  with elongation < 2 and area >= 200 px that no record's filament overlaps and no selected mask overlaps; equally
  dark candidates go by area, largest first, then by stem (the flat field's 8-bit floor makes the darkest umbrae tie
  at -0.5);
- (e) annotator disagreement: the largest filament that one record drew as one polygon and another record of the
  same frame as two or more pieces (each piece at least half inside it and covering at least a tenth of it);
- (f) genuine failure: the largest filament that every record of a three-record frame drew (IoU > 0.5 between the
  records) and that none of our masks matches in any record;
- (g) operating-point change: the largest mask selected at tau 0.26 that no mask selected at tau 0.30 overlaps;
- (h) prediction no annotator drew: the largest mask that no record's filament overlaps;
- (i), (j) test frames: two test frames drawn with `numpy.random.default_rng(0)` over the sorted test stems (the
  first two of the permutation with a selected mask), cropped around the frame's largest selected mask.

Each tile is a square crop at native resolution around its structure (`crop_box`). The frame is stretched linearly
between the 1st and 99th percentiles of the solar disk (`disk_stretch`).
"""
from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_util
from skimage.morphology import convex_hull_image

from . import metric

RULES = {
    "a": ("barbs", "the matched mask whose annotated polygon (at least the median area) is least convex"),
    "b": ("long filament in one piece", "the matched mask with the longest annotated spine"),
    "c": ("near neighbours kept apart", "two masks matching two filaments of one record, with the smallest gap"),
    "d": ("compact dark structure rejected", "the darkest unselected compact candidate that no annotator drew; equally "
                                             "dark candidates by area, largest first, then by stem"),
    "e": ("annotators disagree", "the largest filament one record drew whole and another in two or more pieces"),
    "f": ("genuine failure", "the largest filament every record of a three-record frame drew and we missed"),
    "g": ("operating point", "the largest mask selected at tau 0.26 that no mask at tau 0.30 overlaps"),
    "h": ("drawn by no annotator", "the largest mask that no record's filament overlaps"),
    "i": ("test frame", "a seeded test frame, around its largest selected mask"),
    "j": ("test frame", "a second seeded test frame, around its largest selected mask"),
}
LABELS = {"a": "barbs", "b": "long, one piece", "c": "neighbours apart", "d": "dark spot rejected",
          "e": "annotators disagree", "f": "genuine failure", "g": "τ 0.26, not 0.30", "h": "no annotator drew it",
          "i": "test frame", "j": "test frame"}  # short labels printed inside the tiles; the caption states the rules
MAX_GAP = 6.0  # px between two masks for rule (c)
COMPACT_ELONGATION, COMPACT_AREA = 2.0, 200  # rule (d)
PIECE_INSIDE, PIECE_COVER = 0.5, 0.1  # rule (e): a piece at least half inside the filament, covering a tenth of it


def solidity(mask: np.ndarray) -> float:
    """The mask's pixels over the pixels of its convex hull (scikit-image's definition: 1 for a convex mask). The
    area inside the contours through the boundary pixels' centres, which `zoom_figure.py` divides by, falls short of
    the pixel count by about half the perimeter, so thin filaments would score above 1 there."""
    y0, y1, x0, x1 = bbox(mask)
    crop = mask[y0:y1, x0:x1]
    return float(crop.sum()) / float(convex_hull_image(crop).sum())


def spine_length(xy: np.ndarray) -> float:
    """Length of an (n, 2) polyline."""
    return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of a non-empty mask, end exclusive."""
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def gap(a: np.ndarray, b: np.ndarray, limit: float = MAX_GAP) -> float:
    """Background pixels between two disjoint masks: the distance between their nearest pixel centres minus one (0
    for 4-adjacent masks). `inf` when it exceeds `limit` (their boxes already lie farther apart)."""
    (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = bbox(a), bbox(b)
    if max(by0 - ay1, ay0 - by1, bx0 - ax1, ax0 - bx1) > limit + 1:
        return np.inf
    m = int(np.ceil(limit)) + 2
    y0, y1 = max(min(ay0, by0) - m, 0), min(max(ay1, by1) + m, a.shape[0])
    x0, x1 = max(min(ax0, bx0) - m, 0), min(max(ax1, bx1) + m, a.shape[1])
    d = cv2.distanceTransform((~a[y0:y1, x0:x1]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = float(d[b[y0:y1, x0:x1]].min()) - 1.0
    return g if g <= limit else np.inf


def pieces(inter: np.ndarray, area_filament: float, area_others: np.ndarray) -> np.ndarray:
    """Indices of another record's filaments that are pieces of one filament: at least `PIECE_INSIDE` of the piece
    inside it, and covering at least `PIECE_COVER` of it (`inter`: the pixels each shares with it)."""
    return np.flatnonzero((inter >= PIECE_INSIDE * area_others) & (inter >= PIECE_COVER * area_filament))


def crop_box(mask: np.ndarray, shape: tuple[int, int], min_side: int = 160, margin: float = 1.3
             ) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a square crop centred on a mask's box, `margin` times its long side and at least
    `min_side`, shifted inside the frame."""
    y0, y1, x0, x1 = bbox(mask)
    side = int(min(max(min_side, np.ceil(margin * max(y1 - y0, x1 - x0))), min(shape)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, shape[0] - side))
    left = int(np.clip(cx - side // 2, 0, shape[1] - side))
    return left, top, left + side, top + side


def disk_stretch(img: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, float]:
    """The 1st and 99th percentiles of the frame inside the solar disk."""
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    lo, hi = np.percentile(img[inside], [1, 99])
    return float(lo), float(hi)


def best_matches(ious: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every mask (columns of each record's IoU matrix): its best IoU over all records' filaments, and that
    filament's record and index (-1 without overlap)."""
    n = ious[0].shape[1] if ious else 0
    best, rec, fil = np.zeros(n, np.float32), np.full(n, -1), np.full(n, -1)
    for r, iou in enumerate(ious):
        if iou.size == 0:
            continue
        f = iou.argmax(axis=0)
        v = iou[f, np.arange(n)]
        better = v > best
        best[better], rec[better], fil[better] = v[better], r, f[better]
    return best, rec, fil


def frame_candidates(stem: str, gt: list[list[dict]], spines: list[list[np.ndarray]], sel: list[dict],
                     sel30: list[dict], median_area: float) -> list[dict]:
    """The candidates of rules (a), (b), (c), (e), (f), (g) and (h) on one frame (rule (d) reads the bank:
    `compact_candidates`). `gt` holds each record's filament RLEs and `spines` their polylines; `sel` and `sel30` the
    masks selected at tau 0.26 and 0.30. Each candidate gives its rule, stem, sort key (smaller first), the masks to
    draw and the values for the caption."""
    out = []
    n_rec = len(gt)
    inter = [metric.overlap_counts(g, sel) for g in gt]
    ious = [metric.iou_dice(*x)[0] for x in inter]
    best, rec, fil = best_matches(ious)
    gt_area = [rle_areas(g) for g in gt]
    sel_area = rle_areas(sel)
    record_ious = [[float(iou[:, k].max()) if iou.size else 0.0 for iou in ious] for k in range(len(sel))]
    for k in np.flatnonzero(best > metric.IOU_THRESHOLD):
        r, f = int(rec[k]), int(fil[k])
        base = dict(stem=stem, masks=[int(k)], filament=[r, f], ious=record_ious[k])
        if gt_area[r][f] >= median_area:
            s = solidity(_decode(gt[r][f]))
            out.append(dict(base, rule="a", key=(s,), values=dict(solidity=round(s, 4), area=int(gt_area[r][f]),
                                                                    iou=round(float(best[k]), 4))))
        length = spine_length(spines[r][f])
        out.append(dict(base, rule="b", key=(-length,), values=dict(spine_length=round(length, 1),
                                                                      iou=round(float(best[k]), 4))))
    out += _neighbours(stem, gt, sel, ious, record_ious)
    out += _disagreements(stem, gt, gt_area, sel, ious, record_ious)
    if n_rec == 3:
        out += _failures(stem, gt, gt_area, ious)
    if sel:
        none30 = (metric.overlap_counts(sel, sel30)[0] == 0).all(axis=1) if sel30 else np.ones(len(sel), bool)
        no_record = np.all([(x[0] == 0).all(axis=0) for x in inter], axis=0) if gt else np.ones(len(sel), bool)
        for k in np.flatnonzero(none30):
            out.append(dict(stem=stem, rule="g", key=(-int(sel_area[k]),), masks=[int(k)], ious=record_ious[k],
                            values=dict(area=int(sel_area[k]))))
        for k in np.flatnonzero(no_record):
            out.append(dict(stem=stem, rule="h", key=(-int(sel_area[k]),), masks=[int(k)], ious=record_ious[k],
                            values=dict(area=int(sel_area[k]))))
    return out


def _neighbours(stem, gt, sel, ious, record_ious) -> list[dict]:
    """Rule (c): pairs of masks matching two different filaments of the same record, within `MAX_GAP`."""
    pairs = set()
    for iou in ious:
        hit = np.argwhere(iou > metric.IOU_THRESHOLD)  # (filament, mask)
        for i in range(len(hit)):
            for j in range(i + 1, len(hit)):
                if hit[i, 0] != hit[j, 0] and hit[i, 1] != hit[j, 1]:
                    pairs.add(tuple(sorted((int(hit[i, 1]), int(hit[j, 1])))))
    out = []
    decoded = {}
    for k1, k2 in sorted(pairs):
        a = decoded.setdefault(k1, _decode(sel[k1]))
        b = decoded.setdefault(k2, _decode(sel[k2]))
        g = gap(a, b)
        if np.isfinite(g):
            out.append(dict(stem=stem, rule="c", key=(round(g, 3),), masks=[k1, k2],
                            ious=[record_ious[k1], record_ious[k2]], values=dict(gap_px=round(g, 3))))
    return out


def _disagreements(stem, gt, gt_area, sel, ious, record_ious) -> list[dict]:
    """Rule (e): a filament one record drew whole and another record of the frame in two or more pieces."""
    out = []
    for r, g in enumerate(gt):
        for r2, g2 in enumerate(gt):
            if r2 == r or not g or not g2:
                continue
            inter = metric.overlap_counts(g, g2)[0]
            for f in range(len(g)):
                p = pieces(inter[f], gt_area[r][f], gt_area[r2])
                if len(p) >= 2:
                    ours = np.flatnonzero(ious[r][f] > 0).tolist() if sel else []
                    out.append(dict(stem=stem, rule="e", key=(-int(gt_area[r][f]),), filament=[r, f],
                                    pieces=[r2, p.tolist()], masks=ours,
                                    ious=[record_ious[k] for k in ours],
                                    values=dict(area=int(gt_area[r][f]), pieces=len(p))))
    return out


def _failures(stem, gt, gt_area, ious) -> list[dict]:
    """Rule (f): a filament every record of the frame drew (IoU > 0.5 in each other record) that none of our masks
    matches in any record."""
    out = []
    pair = {(r, r2): metric.iou_dice(*metric.overlap_counts(gt[r], gt[r2]))[0]
            for r in range(len(gt)) for r2 in range(len(gt)) if r != r2}
    for r, g in enumerate(gt):
        for f in range(len(g)):
            versions = [(r, f)]
            for r2 in range(len(gt)):
                if r2 != r and pair[r, r2].size and pair[r, r2][f].max() > metric.IOU_THRESHOLD:
                    versions.append((r2, int(pair[r, r2][f].argmax())))
            if len(versions) < len(gt):
                continue
            best = [float(ious[q][v].max()) if ious[q].size else 0.0 for q, v in versions]
            if max(best) <= metric.IOU_THRESHOLD:
                out.append(dict(stem=stem, rule="f", key=(-int(gt_area[r][f]),), filament=[r, f],
                                versions=versions, masks=[], ious=best, values=dict(area=int(gt_area[r][f]))))
    return out


def compact_candidates(stem: str, X: np.ndarray, B: np.ndarray, features: list[str]) -> np.ndarray:
    """Rule (d)'s first filter on a frame's bank: the nodes with elongation < 2 and area >= 200 px that no record's
    filament overlaps (best IoU 0 with every record). The selection check needs the nodes' pixels
    (`compact_pick`)."""
    col = {n: i for i, n in enumerate(features)}
    area = np.exp(X[:, col["log_area"]])
    ok = (X[:, col["elong"]] < COMPACT_ELONGATION) & (np.rint(area) >= COMPACT_AREA)
    if B is not None and B.size:
        ok &= (B == 0).all(axis=1)
    return np.flatnonzero(ok)


def compact_pick(stem: str, tree, nodes: np.ndarray, selected: np.ndarray, X: np.ndarray, B: np.ndarray,
                 features: list[str]) -> list[dict]:
    """Rule (d)'s candidates: the nodes of `compact_candidates` that no selected mask overlaps (`selected`: the
    union of the frame's selected masks), darkest first. Equally dark nodes go by area, largest first, then by stem:
    a node and its ancestors share the darkest pixel, and the flat field's 8-bit floor makes the darkest umbrae of
    several frames tie at -0.5. `B` is the bank's best IoU of every node with each record."""
    col = {n: i for i, n in enumerate(features)}
    out = []
    for v in nodes:
        if selected[tree.node_mask(int(v))].any():
            continue
        area = int(tree.area[v])
        f_min = float(X[v, col["f_min"]])
        out.append(dict(stem=stem, rule="d", key=(round(f_min, 6), -area), tiebreak=(int(v),), node=int(v), masks=[],
                        ious=[round(float(x), 4) for x in B[v]],
                        values=dict(f_min=round(f_min, 4), elongation=round(float(X[v, col["elong"]]), 3),
                                    area=area, p_mean=round(float(X[v, col["p_mean"]]), 4))))
    return out


def pick(candidates: list[dict], n: int, used: set[str] | None = None) -> list[dict]:
    """The first `n` candidates in order of (key, stem, then the order within a frame) whose frames differ from each
    other and from `used`."""
    used = set() if used is None else used
    out = []
    for c in sorted(candidates, key=lambda c: (c["key"], c["stem"], c.get("tiebreak", ()), c.get("masks", []),
                                               c.get("filament", []))):
        if c["stem"] in used:
            continue
        out.append(c)
        used.add(c["stem"])
        if len(out) == n:
            break
    return out


def test_frames(stems: list[str], has_masks: set[str], n: int = 2, seed: int = 0) -> list[str]:
    """The first `n` test stems with a selected mask in the order `numpy.random.default_rng(seed)` permutes the
    sorted stems."""
    stems = sorted(stems)
    order = np.random.default_rng(seed).permutation(len(stems))
    return [stems[i] for i in order if stems[i] in has_masks][:n]


def rle_areas(rles: list[dict]) -> np.ndarray:
    """Pixel areas of COCO RLEs (an empty list gives an empty array)."""
    return mask_util.area(rles).astype(np.int64) if rles else np.zeros(0, np.int64)


def _decode(r: dict) -> np.ndarray:
    return mask_util.decode(r).astype(bool)

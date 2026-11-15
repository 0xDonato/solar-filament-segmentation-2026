"""Detection candidates: the instance model's own masks as candidates beside the component tree's nodes.

The tree offers only threshold components of the dense map, so a filament that the map merges with a neighbour, or
splits at every level, has no candidate. On fold 0 the detector's masks cover 11.6% of the ground-truth filaments
that no node covers (score >= 0.05). Each detection gets descriptors of its own (the
detector's score and maskness, the map and flat field inside and around it, its shape and disk position, its best IoU
with a node and with a stronger detection), its matched IoU against every annotator record for training, and its
pixel overlaps with the nodes and the other detections, from which the selection's conflicts follow.

With `gated`, each detection also offers a mask cut from our own 2048 px map inside the detection's grown support, at
0.5 of the map's local peak there, joined through the detection's mask into one piece without holes: the detector
decides the grouping, the map the outline, as a detect-then-refine cascade refines each box with a segmenter. The
detector's masks at 1024 px follow outlines at about SQ 0.64; the map's at 0.69.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .instances import SCALE
from .postprocess import fill_holes
from .preprocess import Disk
from .tree import Tree

COLUMNS = ["d_score", "d_maskness", "d_log_area", "d_elong", "d_p_mean", "d_p_max", "d_p_q10", "d_frac_p05",
           "d_frac_p25", "d_frac_p50", "d_p_ring", "d_p_contrast", "d_f_mean", "d_f_contrast", "d_rho", "d_abs_lat",
           "d_abs_lon", "d_best_node_iou", "d_best_node_t", "d_n_overlapping", "d_iou_stronger", "img_n_det_03",
           "d_gated"]


def full_pixels(mask_1024: np.ndarray) -> np.ndarray:
    """Sorted flat indices (2048 x 2048) of a detection mask given at 1024 px."""
    ys, xs = np.nonzero(mask_1024)
    dy, dx = np.divmod(np.arange(SCALE * SCALE), SCALE)
    yy = (ys[:, None] * SCALE + dy[None]).ravel()
    xx = (xs[:, None] * SCALE + dx[None]).ravel()
    return np.sort(yy * 2048 + xx).astype(np.int64)


def clean_pixels(pix: np.ndarray) -> np.ndarray:
    """The largest 8-connected piece of a pixel set, holes filled: a filament mask is one piece without holes."""
    if len(pix) == 0:
        return pix
    ys, xs = np.divmod(pix, 2048)
    y0, x0 = ys.min(), xs.min()
    m = np.zeros((ys.max() - y0 + 3, xs.max() - x0 + 3), np.uint8)
    m[ys - y0 + 1, xs - x0 + 1] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    m = fill_holes(m > 0)
    yy, xx = np.nonzero(m)
    return np.sort((yy + y0 - 1) * 2048 + (xx + x0 - 1)).astype(np.int64)


def node_overlaps(tree: Tree, pix: np.ndarray) -> np.ndarray:
    """Pixels each node of `tree` shares with a set of full-frame flat indices."""
    n = tree.n
    inter = np.zeros(n)
    if n == 0 or len(pix) == 0:
        return inter
    y0, y1, x0, x1 = tree.box
    ww, W = tree.crop_shape[1], tree.shape[1]
    py, px = np.divmod(pix, W)
    inside = (py >= y0) & (py < y1) & (px >= x0) & (px < x1)
    pos = np.full(tree.crop_shape[0] * ww, -1, np.int64)
    pos[tree.support] = np.arange(len(tree.support))
    q = pos[(py[inside] - y0) * ww + (px[inside] - x0)]
    q = q[q >= 0]
    for k in range(len(tree.levels)):
        ids = tree.node_of[k][q]
        ids = ids[ids >= 0]
        ids = ids[tree.level[ids] == k]
        if len(ids):
            inter += np.bincount(ids, minlength=n)
    return inter


def gated_pixels(p: np.ndarray, det: np.ndarray, alpha: float = 0.5, grow: int = 7) -> np.ndarray:
    """Flat indices of the map's own mask inside a detection (`det`, sorted flat indices at 2048 px): the pixels of
    its grown support where p exceeds alpha times the 99th percentile of p there, joined through the detection, as
    one 8-connected piece without holes (empty when the map shows nothing there)."""
    ys, xs = np.divmod(det, 2048)
    y0, y1 = max(ys.min() - grow, 0), min(ys.max() + grow + 1, 2048)
    x0, x1 = max(xs.min() - grow, 0), min(xs.max() + grow + 1, 2048)
    d = np.zeros((y1 - y0, x1 - x0), np.uint8)
    d[ys - y0, xs - x0] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    support = cv2.dilate(d, k) > 0
    pc = p[y0:y1, x0:x1]
    peak = np.percentile(pc[support], 99)
    if peak < 0.05:
        return np.zeros(0, np.int64)
    on = support & (pc > alpha * peak)
    if not on.any():
        return np.zeros(0, np.int64)
    m = fill_holes(on | ((d > 0) & (pc > 0.02)))  # the detection bridges the map's faint gaps
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(0, np.int64)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    yy, xx = np.nonzero(lab == best)
    return np.sort((yy + y0) * 2048 + (xx + x0)).astype(np.int64)


def candidates(tree: Tree, p: np.ndarray, flat: np.ndarray, disk: Disk, scores: np.ndarray, masks: np.ndarray,
               maskness: np.ndarray, min_score: float = 0.05, gated: bool = False) -> dict:
    """Descriptors, pixel sets and overlaps of a frame's detections with score >= `min_score` (and, with `gated`,
    of each one's map-gated mask; `keep` maps every candidate to its detection)."""
    keep = np.flatnonzero(scores >= min_score)
    keep = keep[[masks[i].any() for i in keep]] if len(keep) else keep
    pix = [clean_pixels(full_pixels(masks[i])) for i in keep]
    kind = [0] * len(keep)
    if gated:
        extra = [(i, gated_pixels(p, q)) for i, q in zip(keep, pix)]
        extra = [(i, g) for (i, g), q in zip(extra, pix) if len(g) >= 20 and not np.array_equal(g, q)]
        keep = np.concatenate([keep, np.array([i for i, _ in extra], np.int64)]) if extra else keep
        pix += [g for _, g in extra]
        kind += [1] * len(extra)
    X, inter_nodes, inter_dets, area = describe_sets(tree, p, flat, disk, pix, scores[keep], maskness[keep], kind,
                                                     float((scores >= 0.3).sum()))
    return dict(X=X, pix=pix, keep=keep, inter_nodes=inter_nodes, inter_dets=inter_dets, area=area)


def describe_sets(tree: Tree, p: np.ndarray, flat: np.ndarray, disk: Disk, pix: list[np.ndarray], score: np.ndarray,
                  maskness: np.ndarray, kind, n_det_03: float) -> tuple[np.ndarray, ...]:
    """`COLUMNS` of candidate pixel sets (full-frame flat indices, each with its own score, maskness and kind), their
    pixel overlaps with the tree's nodes and with each other, and their areas. Any candidate kind that offers whole
    masks beside the tree (detections, crop masks, watershed parts) is described this way."""
    n = len(pix)
    X = np.zeros((n, len(COLUMNS)), np.float32)
    inter_nodes = np.zeros((n, tree.n), np.float32)
    inter_dets = np.zeros((n, n), np.float32)
    if n == 0:
        return X, inter_nodes, inter_dets, np.zeros(0, np.float64)
    pf, ff = p.ravel(), flat.ravel() - 1.0
    a_node = tree.area.astype(np.float64)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for j, q in enumerate(pix):
        inter_nodes[j] = node_overlaps(tree, q)
        vals, fvals = pf[q], ff[q]
        ys, xs = np.divmod(q, 2048)
        y0, y1, x0, x1 = max(ys.min() - 4, 0), min(ys.max() + 5, 2048), max(xs.min() - 4, 0), min(xs.max() + 5, 2048)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m[ys - y0, xs - x0] = 1
        ring = (cv2.dilate(m, k5) > 0) & (m == 0)
        pr, fr = p[y0:y1, x0:x1][ring], (flat[y0:y1, x0:x1] - 1.0)[ring]
        cy, cx = ys.mean(), xs.mean()
        cov = np.cov(np.stack([xs, ys]).astype(np.float64)) if len(q) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        xr, yr = (cx - disk.cx) / disk.r, (cy - disk.cy) / disk.r
        lat = np.arcsin(np.clip(yr, -1, 1))
        lon = np.arcsin(np.clip(xr / max(np.cos(lat), 1e-3), -1, 1))
        best = inter_nodes[j] / (a_node + len(q) - inter_nodes[j]) if tree.n else np.zeros(0)
        bv = int(np.argmax(best)) if tree.n else -1
        X[j, :18] = [score[j], maskness[j], np.log(len(q)), np.sqrt(ev[1] / max(ev[0], 1.0)), vals.mean(), vals.max(),
                     np.percentile(vals, 10), (vals > 0.05).mean(), (vals > 0.25).mean(), (vals > 0.5).mean(),
                     pr.mean() if len(pr) else 0.0, vals.mean() - (pr.mean() if len(pr) else 0.0), fvals.mean(),
                     fvals.mean() - (fr.mean() if len(fr) else 0.0), np.hypot(xr, yr), np.degrees(abs(lat)),
                     np.degrees(abs(lon)), best[bv] if bv >= 0 else 0.0]
        X[j, 18] = tree.levels[tree.level[bv]] if bv >= 0 and best[bv] > 0 else 0.0
        X[j, 21] = n_det_03
        X[j, 22] = kind[j]
    for j in range(n):
        for k in range(j + 1, n):
            c = len(np.intersect1d(pix[j], pix[k], assume_unique=True))
            inter_dets[j, k] = inter_dets[k, j] = c
    areas = np.array([len(q) for q in pix], np.float64)
    iou_dd = inter_dets / (areas[:, None] + areas[None, :] - inter_dets)
    np.fill_diagonal(iou_dd, 0.0)
    X[:, 19] = (iou_dd > 0.3).sum(1)
    stronger = np.asarray(score)[None, :] > np.asarray(score)[:, None]
    X[:, 20] = np.where(stronger, iou_dd, 0.0).max(1)
    return X, inter_nodes, inter_dets, areas


def match(pix: list[np.ndarray], gt) -> np.ndarray:
    """(n, R) matched IoU (0 when IoU <= 0.5) of every detection against every record of `gt` (a GTIndex)."""
    T = np.zeros((len(pix), len(gt.records)), np.float32)
    for j, q in enumerate(pix):
        for r, rec in enumerate(gt.records):
            best = 0.0
            for g in rec:
                c = len(np.intersect1d(q, g, assume_unique=True))
                if c:
                    best = max(best, c / (len(q) + len(g) - c))
            T[j, r] = best if best > 0.5 else 0.0
    return T


def select_mixed(tree: Tree, u_nodes: np.ndarray, c: dict, u_dets: np.ndarray, tau: float, overlap: float = 0.2,
                 margin: float = 0.0, min_score: float = 0.0) -> tuple[list[int], list[int]]:
    """Exact maximum-weight independent set over the nodes and detection candidates worth keeping: maximise
    sum(u - tau) (a detection's weight lowered by `margin`, and only detections whose detector score is at least
    `min_score`) with no node beside one of its ancestors and no two candidates sharing more than `overlap` of the
    smaller. Without a detection worth keeping it is the tree's own exact selection."""
    from .tree import select

    nodes = np.flatnonzero(u_nodes > tau)
    dsel = np.flatnonzero((u_dets > tau + margin) & (c["X"][:, 0] >= min_score)) if len(u_dets) else np.zeros(0, int)
    if len(dsel) == 0:
        return list(select(tree, u_nodes, tau)), []
    ids = [("n", v) for v in nodes] + [("d", j) for j in dsel]
    w = np.array([u_nodes[v] - tau for v in nodes] + [u_dets[j] - tau - margin for j in dsel])
    pos = {key: i for i, key in enumerate(ids)}
    rows = []
    for v in nodes:  # a node and any of its ancestors
        a = tree.parent[v]
        while a >= 0:
            if ("n", a) in pos:
                rows.append((pos["n", v], pos["n", a]))
            a = tree.parent[a]
    a_node = tree.area.astype(np.float64)
    area = c["area"]
    for j in dsel:
        for v in nodes:
            if c["inter_nodes"][j, v] > overlap * min(a_node[v], area[j]):
                rows.append((pos["d", j], pos["n", v]))
        for k in dsel:
            if k > j and c["inter_dets"][j, k] > overlap * min(area[j], area[k]):
                rows.append((pos["d", j], pos["d", k]))
    cons = []
    if rows:
        A = np.zeros((len(rows), len(ids)))
        for r, (i, k) in enumerate(rows):
            A[r, i] = A[r, k] = 1
        cons = [LinearConstraint(A, -np.inf, 1)]
    res = milp(-w, constraints=cons, integrality=np.ones(len(ids)), bounds=Bounds(0, 1))
    x = np.round(res.x).astype(bool) if res.x is not None else np.zeros(len(ids), bool)
    return [v for (t, v), on in zip(ids, x) if on and t == "n"], [j for (t, j), on in zip(ids, x) if on and t == "d"]


def paint(pixel_sets: list[np.ndarray], utilities: list[float], shape=(2048, 2048)) -> np.ndarray:
    """Label image of chosen masks (full-frame flat indices) painted from the highest utility down, each taking only
    free pixels and kept as one 8-connected piece without holes, labelled 1..n in that order."""
    lab = np.zeros(shape, np.int32)
    flat = lab.ravel()
    n = 0
    for i in np.argsort(-np.asarray(utilities, np.float64), kind="stable"):
        q = pixel_sets[i]
        q = q[flat[q] == 0]
        q = clean_pixels(q)
        q = q[flat[q] == 0]  # filled holes never take another filament's pixels
        if len(q) == 0:
            continue
        n += 1
        flat[q] = n
    return lab


def join(a: dict, b: dict) -> dict:
    """One candidate set from two (detections, crop masks, watershed parts) over the same tree: their rows stacked,
    with the pixel overlaps between the two sets filled in, so `select_mixed` keeps the union pixel-disjoint."""
    na, nb = len(a["pix"]), len(b["pix"])
    inter = np.zeros((na + nb, na + nb), np.float32)
    inter[:na, :na], inter[na:, na:] = a["inter_dets"], b["inter_dets"]
    for j in range(na):
        for k in range(nb):
            c = len(np.intersect1d(a["pix"][j], b["pix"][k], assume_unique=True))
            inter[j, na + k] = inter[na + k, j] = c
    return dict(X=np.concatenate([a["X"], b["X"]]), pix=list(a["pix"]) + list(b["pix"]),
                keep=np.concatenate([np.asarray(a["keep"]), np.asarray(b["keep"])]),
                inter_nodes=np.concatenate([a["inter_nodes"], b["inter_nodes"]]), inter_dets=inter,
                area=np.concatenate([a["area"], b["area"]]))

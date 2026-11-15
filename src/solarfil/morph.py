"""Outline changes of a segmentation: every instance grown or shrunk by whole pixels, or grown by thin protrusions.

`adjust_width` changes every outline by whole pixels (a +-1 px change moves PQ by about 0.02). Growth
claims only unlabelled pixels, so instances stay pixel-disjoint; shrinking keeps each instance's largest piece.
`grow_thin` adds to each instance only the thin protrusions of the filament map that touch it (the optional outline
step). In practice these extend thin filament ends: a thin branch leaving a long straight side keeps its
root inside the opening, so it stays out.
"""
from __future__ import annotations

import cv2
import numpy as np

from .postprocess import fill_holes

CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def adjust_width(labels: np.ndarray, px: int) -> np.ndarray:
    """Label image with every instance dilated (px > 0) or eroded (px < 0) by |px| pixels (4-connected steps)."""
    lab = labels.astype(np.int32).copy()
    for _ in range(abs(px)):
        if px > 0:
            grown = cv2.dilate(lab.astype(np.float32), CROSS).astype(np.int32)  # max label in the cross
            lab = np.where(lab == 0, grown, lab)
        else:
            ero = cv2.erode(lab.astype(np.float32), CROSS).astype(np.int32)
            lab = np.where(ero == lab, lab, 0)
    if px < 0:  # keep the largest piece of every instance
        out = np.zeros_like(lab)
        for i in np.unique(lab[lab > 0]):
            n, cc, stats, _ = cv2.connectedComponentsWithStats((lab == i).astype(np.uint8), connectivity=8)
            if n > 1:
                out[cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = i
        lab = out
    return lab


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def grow_thin(labels: np.ndarray, p: np.ndarray, t_low: float = 0.1, band: int = 8, r: int = 2) -> np.ndarray:
    """Label image with every instance grown by thin protrusions of the map `p` (same shape): pixels within `band`
    px of the instance with p >= `t_low`, connected to it, that an opening by a disk of radius `r` removes from the
    grown shape, so only branches narrower than about 2r + 1 px are added and the body is not dilated. Growth never
    enters another instance or the 1 px ring around it, nor a pixel an earlier instance's growth took; each instance
    keeps its pixels, stays one 8-connected piece and gets its holes filled."""
    lab = labels.astype(np.int32)
    out = lab.copy()
    h, w = lab.shape
    pad = band + r + 2
    for k in np.unique(lab[lab > 0]):
        ys, xs = np.nonzero(lab == k)
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
        lc = lab[y0:y1, x0:x1]
        m = (lc == k).astype(np.uint8)
        forbid = cv2.dilate(((lc != 0) & (lc != k)).astype(np.uint8), _disk(1)) > 0
        region = (cv2.dilate(m, _disk(band)) > 0) & (p[y0:y1, x0:x1] >= t_low) & ~forbid
        grown = _piece_with(m, (m > 0) | region)
        thin = grown & ~(cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, _disk(r)) > 0)
        new = _piece_with(m, (m > 0) | (thin & region))
        new = (fill_holes(new) & ~forbid) | (m > 0)
        crop = out[y0:y1, x0:x1]
        crop[new & ((crop == 0) | (lc == k))] = k  # unclaimed pixels only, so instances stay disjoint
    return out


def _piece_with(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The 8-connected pieces of `mask` that touch the instance `m`."""
    n, cl = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.unique(cl[m > 0])
    return np.isin(cl, keep[keep > 0])

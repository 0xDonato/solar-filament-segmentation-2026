"""From a probability map to pixel-disjoint instances: thresholding and connected components.

This is the baseline decoder; the pipeline's decision layer (component tree + learned scorer) replaces it.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import rle


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background regions (4-connected) that do not touch the frame border."""
    inv = (~binary).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    hole = np.ones(n, bool)
    hole[border] = False
    hole[0] = False  # label 0 is the foreground of `inv`
    return binary | hole[lab]


def threshold_labels(p: np.ndarray, t: float, min_area: int = 0, holes: bool = True) -> tuple[np.ndarray, int]:
    """Label image (int32, 0 = background) of the 8-connected components of ``p > t`` with area >= min_area."""
    b = p > t
    if holes:
        b = fill_holes(b)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    keep = np.zeros(n, np.int32)
    ok = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(min_area, 1))
    ok = ok[ok > 0]
    keep[ok] = np.arange(1, len(ok) + 1, dtype=np.int32)
    return keep[lab], len(ok)


def labels_to_rles(labels: np.ndarray) -> list[str]:
    """Submission RLE strings of the instances of a label image, in label order."""
    out = []
    n = int(labels.max())
    if n == 0:
        return out
    for i in range(1, n + 1):
        m = labels == i
        if m.any():
            out.append(rle.encode(m))
    return out

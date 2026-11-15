"""The preprocessed frame cache: flat fields for every frame and the dense targets of the training frames.

Layout under ``<cache>/``, every array a 2048 x 2048 uint8 PNG:

* ``flat/<stem>.png``: the flat field ``F`` on [0.5, 1.5] (`preprocess.to_uint8`);
* ``p/<stem>.png``: 255 x the soft mean over the image's records of each record's filament mask (the probability
  that a random annotator marks the pixel);
* ``spine/<stem>.png``: 255 x the soft mean over records of the spine polylines drawn 3 px wide;
* ``disks.json``: the fitted disk (cx, cy, r) of every frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from . import paths, preprocess
from .data import Dataset, H, Observation, W, read_image

DEFAULT_CACHE = paths.CACHE
SPINE_WIDTH = 3


def record_union(rec) -> np.ndarray:
    """Union of a record's filament masks (bool, 2048 x 2048)."""
    rles = rec.rles()
    return mask_util.decode(mask_util.merge(rles, intersect=False)).astype(bool) if rles else np.zeros((H, W), bool)


def record_spines(rec, width: int = SPINE_WIDTH) -> np.ndarray:
    """Union of a record's spines drawn as `width` px polylines at sub-pixel precision (bool, 2048 x 2048)."""
    m = np.zeros((H, W), np.uint8)
    for f in rec.filaments:
        pts = np.rint(f.spine_xy() * 16).astype(np.int32)  # 4 fractional bits
        if len(pts) >= 2:
            cv2.polylines(m, [pts], False, 1, thickness=width, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def soft_targets(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    """uint8 soft-mean filament and spine maps of one observation."""
    p = np.zeros((H, W), np.float32)
    s = np.zeros((H, W), np.float32)
    for rec in obs.records:
        p += record_union(rec)
        s += record_spines(rec)
    n = max(obs.n_records, 1)
    return np.rint(255 * p / n).astype(np.uint8), np.rint(255 * s / n).astype(np.uint8)


CONTACT_PX = 3


def record_contact(rec, px: int = CONTACT_PX) -> np.ndarray:
    """A record's contact borders (bool, 2048 x 2048): the pixels of its filaments that lie within `px` of another of
    its filaments (each filament mask dilated by a disc of radius `px`; pixels covered by two or more dilations,
    inside the record's union). DSB 2018 1st's border target (Kaggle topic 54741)."""
    count = np.zeros((H, W), np.uint8)
    union = np.zeros((H, W), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    for f in rec.filaments:
        xy = f.polygon_xy()
        if not len(xy):
            continue
        x0, y0 = (max(0, int(np.floor(v)) - px - 2) for v in xy.min(0))
        x1, y1 = (min(n, int(np.ceil(v)) + px + 3) for v, n in zip(xy.max(0), (W, H)))
        sub = f.mask()[y0:y1, x0:x1].astype(np.uint8)
        count[y0:y1, x0:x1] += cv2.dilate(sub, k).astype(bool)
        union[y0:y1, x0:x1] |= sub.astype(bool)
    return (count >= 2) & union


def soft_contact(obs: Observation, px: int = CONTACT_PX) -> np.ndarray:
    """uint8 soft mean over the observation's records of each record's contact borders (`record_contact`)."""
    c = np.zeros((H, W), np.float32)
    for rec in obs.records:
        c += record_contact(rec, px)
    return np.rint(255 * c / max(obs.n_records, 1)).astype(np.uint8)


def record_planes(obs: Observation) -> np.ndarray:
    """(n_records, 2048, 2048) uint8: 255 x each record's union mask, for losses averaged over the records."""
    return np.stack([record_union(r).astype(np.uint8) * 255 for r in obs.records])


_DATASETS: dict[str, Dataset] = {}


def build_one(args) -> tuple[str, dict]:
    """Cache one frame (a pool task `(root, cache, stem, with_targets)`): writes the 8-bit flat field to `flat/` and,
    with targets, the soft filament and spine maps to `p/` and `spine/`; returns the stem and the fitted disk."""
    root, cache, stem, with_targets = args
    ds = _DATASETS.setdefault(str(root), Dataset(root))  # parse the annotation file once per process
    cache = Path(cache)
    img = read_image(ds.image_path(stem))
    flat, disk = preprocess.flat_field(img)
    cv2.imwrite(str(cache / "flat" / f"{stem}.png"), preprocess.to_uint8(flat))
    if with_targets:
        p, s = soft_targets(ds.observations[stem])
        cv2.imwrite(str(cache / "p" / f"{stem}.png"), p)
        cv2.imwrite(str(cache / "spine" / f"{stem}.png"), s)
    return stem, dict(cx=disk.cx, cy=disk.cy, r=disk.r)


def load_png(path: Path) -> np.ndarray:
    """A cached PNG as stored (any depth or channel count); raises FileNotFoundError when it cannot be read."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return a


def load_disks(cache: Path = DEFAULT_CACHE) -> dict[str, preprocess.Disk]:
    """The fitted solar disk of every cached frame, from the cache's `disks.json`."""
    d = json.loads((Path(cache) / "disks.json").read_text())
    return {k: preprocess.Disk(**v) for k, v in d.items()}

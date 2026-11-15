"""The submission's mask encoding: COCO compressed RLE counts of a 2048 x 2048 mask, without the size field."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from .data import H, W


def encode(mask: np.ndarray) -> str:
    """Counts string of a boolean (H, W) mask, as `pycocotools.mask.encode` of the Fortran-order uint8 array."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def to_coco(counts: str, h: int = H, w: int = W) -> dict:
    """The COCO RLE dict of a counts string at size (h, w)."""
    return {"size": [h, w], "counts": counts.encode("ascii")}


def decode(counts: str, h: int = H, w: int = W) -> np.ndarray:
    """The bool (h, w) mask of a counts string."""
    return mask_util.decode(to_coco(counts, h, w)).astype(bool)

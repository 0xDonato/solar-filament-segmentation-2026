"""The leaderboard CSV: one row per filament, ``filament_id = <stem>_<k>`` and the RLE counts of its mask."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import rle
from .data import H, W


def write(instances: dict[str, list[str]], path: Path | str) -> int:
    """Write {stem: [RLE counts, ...]} and return the number of rows."""
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(instances):
            for k, counts in enumerate(instances[stem]):
                w.writerow([f"{stem}_{k}", counts])
                n += 1
    return n


def validate(path: Path | str, stems: list[str]) -> dict:
    """Check what the scorer checks or implies: header, unique ids, known stems, non-empty decodable masks of the
    right size, pixel-disjoint instances within each image, and no quoting of the RLE field."""
    rows = list(csv.reader(open(path, newline="")))
    assert rows[0] == ["filament_id", "segmentation_rle"], rows[0]
    ids = [r[0] for r in rows[1:]]
    assert len(ids) == len(set(ids)), "duplicate filament_id"
    known = set(stems)
    per_image: dict[str, list[str]] = {}
    for fid, counts in rows[1:]:
        stem = fid.split("_", 1)[0]
        assert stem in known, fid
        per_image.setdefault(stem, []).append(counts)
    raw = Path(path).read_text()
    assert '"' not in raw, "quoted field"
    for stem, rles in per_image.items():
        acc = np.zeros((H, W), np.uint8)
        for c in rles:
            m = rle.decode(c)
            assert m.shape == (H, W) and m.any(), stem
            acc += m
        assert acc.max() <= 1, f"overlapping masks in {stem}"
    return dict(rows=len(ids), images_with_instances=len(per_image), images=len(stems))

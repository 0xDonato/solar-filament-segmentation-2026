"""Strong-teacher pseudo-labels on unlabelled GONG frames, for pretraining a student.

- Frames: the archive frames (fetch script not included; none within +-1 h of any competition frame) and the
  training frames' neighbours (`data/gong_neighbors/<train stem>/<stem>.jpg`). The fold-k student sees no frame
  within +-1 h of a fold-k frame, nor of a test frame; the full-data student none within +-1 h of a test frame.
- Teacher of fold k: v6's seven members, each network the one of fold k (the full-data networks for the full-data
  student), at v6's shares 5:5:1:1:1:1:1, eight dihedral views, as soft maps in the annotators' frame
  (`infer.write_map_png`: R filament, G spine).
- The student's pretraining crops come from a pool of frames held in memory and refreshed from disk as it trains
  (`PoolSampler`), since several thousand frames do not fit in memory at once.
"""
from __future__ import annotations

import bisect
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .cache import load_png
from .frames import CropSampler, Frame, downsample, shift_rows
from .preprocess import Disk

V6_MEMBERS = ["e3_effv2s_n_long_lean", "e3_effv2s_n_long_lean_win", "e3_effv2s_win_x15_s3", "e3_maxvit_t_win_s3",
              "e3_convnext_b_stem_win_s3", "e3_effv2s_win_dml1_s3", "e3_convnext_t_stem_win_s3"]
V6_WEIGHTS = [5, 5, 1, 1, 1, 1, 1]
HOUR = timedelta(hours=1)
SCIENCE_NAME = re.compile(r"[0-9]{14}[BCLMTU]h")


def stem_time(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def science_stem(stem: str, archive: bool = False) -> bool:
    """Only the six science-grade site codes; archive years are restricted to 2011-2022."""
    if SCIENCE_NAME.fullmatch(stem) is None:
        return False
    try:
        stamp = stem_time(stem)
    except ValueError:
        return False
    return not archive or 2011 <= stamp.year <= 2022


def require_archive(stems, frames: dict, archive: Path, minimum: int = 5000) -> int:
    """Production needs at least5000accepted archive frames, separately from neighbours."""
    n = sum(1 for s in stems if archive in frames[s].parents)
    if n < minimum:
        raise ValueError(f"Production needs >={minimum} eligible archive frames after filtering; have {n}")
    return n


def _near(t: datetime, times: list) -> bool:
    i = bisect.bisect_left(times, t - HOUR)
    return i < len(times) and times[i] <= t + HOUR


def eligible(fold: int, folds: dict, test_stems: list, archive: Path, neighbors: Path | None) -> dict[str, Path]:
    """The unlabelled frames the fold's student may see, stem -> JPEG: none within +-1 h of a frame of fold `fold`
    (any fold < 0: none) or of a test frame; archive frames also none within +-1 h of any training frame."""
    test_t = sorted(stem_time(s) for s in test_stems)
    all_t = sorted(test_t + [stem_time(s) for s in folds])
    held_t = sorted(test_t + [stem_time(s) for s, k in folds.items() if k == fold])
    out, last_site = {}, {}
    for jpg in sorted((archive / "jpg").glob("*.jpg")):
        if not science_stem(jpg.stem, archive=True):
            continue
        stamp, site = stem_time(jpg.stem), jpg.stem[14]
        if _near(stamp, all_t) or (site in last_site and stamp - last_site[site] < 2 * HOUR):
            continue
        out[jpg.stem], last_site[site] = jpg, stamp
    if neighbors is not None and neighbors.exists():
        for jpg in sorted(neighbors.glob("*/*.jpg")):
            if science_stem(jpg.stem) and jpg.stem not in folds and not _near(stem_time(jpg.stem), held_t):
                out.setdefault(jpg.stem, jpg)
    return out


def flat_one(args) -> tuple[str, dict] | None:
    """Pool worker: the flat field of one JPEG into `<dst>/<stem>.png`; None when the disk fit is off the
    competition's fixed geometry (radius or centre more than 5 px away) or the file does not decode."""
    import cv2

    from .data import read_image
    from .preprocess import flat_field, to_uint8

    stem, jpg, dst = args
    try:
        img = read_image(jpg)
        flat, d = flat_field(img)
    except Exception:
        return None
    if abs(d.r - 904.0) > 5 or np.hypot(d.cx - 1023.0, d.cy - 1024.1) > 5:
        return None
    cv2.imwrite(str(Path(dst) / f"{stem}.png"), to_uint8(flat))
    return stem, dict(cx=d.cx, cy=d.cy, r=d.r)


def write_disks(path: Path, disks: dict) -> None:
    path.write_text(json.dumps(dict(sorted(disks.items())), indent=0))


def load_disks_json(path: Path) -> dict[str, Disk]:
    return {k: Disk(**v) for k, v in json.loads(Path(path).read_text()).items()}


class PseudoLoader:
    """Loads one pseudo-labelled frame at the recipe's scale: the flat field, and the teacher's filament and spine
    maps as targets, moved down by the recipe's `label_dy` as the records' targets are."""

    def __init__(self, flat_dir: Path, map_dir: Path, disks: dict, cfg: dict):
        self.flat_dir, self.map_dir, self.disks, self.cfg = Path(flat_dir), Path(map_dir), disks, cfg

    def __call__(self, stem: str) -> Frame:
        sc = self.cfg["scale"]
        flat = downsample(load_png(self.flat_dir / f"{stem}.png"), sc)
        bgr = load_png(self.map_dir / f"{stem}.png")
        fr = Frame(stem, self.disks[stem], sc, flat, p=downsample(bgr[:, :, 2], sc), spine=downsample(bgr[:, :, 1], sc),
                   coords=self.cfg["coords"])
        if self.cfg["label_dy"]:
            fr.p, fr.spine = (shift_rows(a, self.cfg["label_dy"] / sc) for a in (fr.p, fr.spine))
        return fr


FIL_LEVEL = 127  # a teacher pixel counts as filament for crop centring above this (p > 0.5)


class PoolSampler(CropSampler):
    """CropSampler over a pool of `pool` frames held in memory; every `swap_every` crops one pool frame is replaced
    by a random frame of `stems` loaded from disk."""

    def __init__(self, loader: PseudoLoader, stems: list, crop: int, seed: int, pool: int = 96, swap_every: int = 4,
                 **aug):
        self.loader, self.stems, self.swap_every, self.k = loader, list(stems), swap_every, 0
        self.prng = np.random.default_rng(seed + 555)
        pick = self.prng.choice(len(self.stems), min(pool, len(self.stems)), replace=False)
        frames = [loader(self.stems[i]) for i in pick]
        super().__init__(frames, crop, seed=seed, fil_px=[self._fil(fr) for fr in frames], **aug)

    @staticmethod
    def _fil(fr: Frame) -> np.ndarray:
        return np.argwhere(fr.p > FIL_LEVEL).astype(np.int32)

    def _sample(self):
        self.k += 1
        if self.k % self.swap_every == 0:
            j = int(self.prng.integers(len(self.frames)))
            fr = self.loader(self.stems[int(self.prng.integers(len(self.stems)))])
            self.frames[j], self.fil_px[j] = fr, self._fil(fr)
        return super()._sample()

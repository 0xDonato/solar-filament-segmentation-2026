"""Training data: the MAGFiLO Kaggle-2026 COCO file grouped by observation and annotator record.

An *observation* is one GONG H-alpha frame (file stem such as ``20140609195854Bh``). A *record* is one
annotator's complete annotation of that frame; the host scores a submission against every record of an
image and pools the counts. Record ids are ``<annotator>-<stem>`` where ``<annotator>`` has six digits and
its first four identify the annotation group.

Ground-truth masks are rasterised with pycocotools, the library that defines the competition's RLE format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from . import paths

H = W = 2048
DEFAULT_ROOT = paths.DATA
ANNOTATION_FILE = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def _num_list(value) -> list[float]:
    """The JSON stores polygons, spines and boxes as strings of JSON lists."""
    return json.loads(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class Filament:
    """One annotated filament: its outline polygon, its spine and its chirality category."""
    polygon: tuple[float, ...]  # flat x0, y0, x1, y1, ... in pixel coordinates
    spine: tuple[float, ...]  # flat polyline, same convention
    category: int  # 1 Left, 2 Right, 3 Unidentifiable (chirality)
    area: float

    def rle(self) -> dict:
        """COCO RLE of the polygon at 2048 x 2048 (pycocotools rasterisation)."""
        return mask_util.merge(mask_util.frPyObjects([list(self.polygon)], H, W))

    def mask(self) -> np.ndarray:
        """The polygon's bool mask at 2048 x 2048."""
        return mask_util.decode(self.rle()).astype(bool)

    def spine_xy(self) -> np.ndarray:
        """The spine as an (n, 2) array of x, y points."""
        return np.asarray(self.spine, dtype=np.float64).reshape(-1, 2)

    def polygon_xy(self) -> np.ndarray:
        """The polygon as an (n, 2) array of x, y vertices."""
        return np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)


@dataclass
class Record:
    """One annotator's drawing of a frame: the filaments that annotator outlined."""
    record_id: str  # "<annotator>-<stem>"
    filaments: list[Filament] = field(default_factory=list)

    @property
    def annotator(self) -> str:
        """The annotator id, the part of the record id before the first dash."""
        return self.record_id.split("-", 1)[0]

    @property
    def group(self) -> str:
        """The annotator's 4-digit annotation group."""
        return self.annotator[:4]

    def rles(self) -> list[dict]:
        """COCO RLEs of the record's filaments, in annotation order."""
        return [f.rle() for f in self.filaments]


@dataclass
class Observation:
    """One training frame, named by its stem, with every annotator record drawn on it."""
    stem: str
    records: list[Record] = field(default_factory=list)

    @property
    def station(self) -> str:
        """The GONG station letter, the stem's 15th character."""
        return self.stem[14]

    @property
    def time(self) -> datetime:
        """The observation time parsed from the stem's first 14 digits."""
        return datetime.strptime(self.stem[:14], "%Y%m%d%H%M%S")

    @property
    def n_records(self) -> int:
        """Number of annotator records of the frame."""
        return len(self.records)

    @property
    def group(self) -> str:
        """The annotation group of the frame; all its records share one (asserted)."""
        groups = {r.group for r in self.records}
        assert len(groups) == 1, (self.stem, groups)
        return groups.pop()


class Dataset:
    """The training observations and the test stems of one data root."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    @cached_property
    def observations(self) -> dict[str, Observation]:
        """Every training observation by stem (sorted), parsed once from the COCO file: each image is one record, each
        annotation one filament, records sorted by id."""
        coco = json.loads((self.root / ANNOTATION_FILE).read_text())
        obs: dict[str, Observation] = {}
        records: dict[str, Record] = {}
        for im in coco["images"]:
            stem = Path(im["file_name"]).stem
            rec = Record(im["id"])
            assert rec.record_id.split("-", 1)[1] == stem, im
            records[rec.record_id] = rec
            obs.setdefault(stem, Observation(stem)).records.append(rec)
        for a in coco["annotations"]:
            seg = _num_list(a["segmentation"])
            assert len(seg) == 1, a["id"]
            records[a["image_id"]].filaments.append(
                Filament(
                    polygon=tuple(seg[0]),
                    spine=tuple(_num_list(a["spine"])),
                    category=int(a["category_id"]),
                    area=float(a["area"]),
                )
            )
        for o in obs.values():
            o.records.sort(key=lambda r: r.record_id)
        return dict(sorted(obs.items()))

    @property
    def train_stems(self) -> list[str]:
        """The training stems, sorted."""
        return list(self.observations)

    @cached_property
    def test_stems(self) -> list[str]:
        """The stems of the test images present under the data root, sorted."""
        return sorted(p.stem for p in (self.root / "test/test_images").glob("*.jpeg"))

    def image_path(self, stem: str) -> Path:
        """Path of a stem's JPEG in the training or test image folder."""
        split = "train/train_images" if stem in self.observations else "test/test_images"
        return self.root / split / f"{stem}.jpeg"


def read_image(path: Path | str) -> np.ndarray:
    """The 2048 x 2048 uint8 grayscale frame."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    assert img.shape == (H, W), (path, img.shape)
    return img

"""Frames at a model scale, network inputs, and the random training crops.

A frame is held as uint8 arrays at ``2048 / scale`` pixels: the flat field and, for training frames, the soft
filament and spine targets. Network input channels are ``5 (F - 1)`` and the disk coordinates ``x / R``, ``y / R``
of every pixel; the coordinate planes travel with the pixels under augmentation, so each pixel always carries its
true position on the disk (the annotation protocol depends on it: filaments beyond about 70 degrees from the
central meridian were not drawn).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cache import DEFAULT_CACHE, load_disks, load_png
from .data import H
from .preprocess import Disk

F_GAIN = 5.0


@dataclass
class Frame:
    """One frame prepared for training at a model scale: the flat-fielded image, its disk, and on training frames
    the soft target and spine planes."""
    stem: str
    disk: Disk
    scale: int
    flat: np.ndarray  # uint8, (S, S)
    p: np.ndarray | None = None  # uint8 soft target
    spine: np.ndarray | None = None
    n_records: int = 0
    recs: np.ndarray | None = None  # uint8 (n_records, S, S): each record's own mask, for per-record losses
    coords: bool = True  # with the disk-coordinate input planes x/R, y/R

    @property
    def size(self) -> int:
        """Side of the frame at its model scale, in px."""
        return self.flat.shape[0]


def downsample(a: np.ndarray, scale: int) -> np.ndarray:
    """Area-average downsampling by an integer factor (identity for 1)."""
    if scale == 1:
        return a
    s = a.shape[0] // scale
    return cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA)


def shift_rows(a: np.ndarray, d: float) -> np.ndarray:
    """The content of an (..., H, W) array moved down by `d` pixels (up if negative, |d| < 1), interpolating linearly
    between neighbouring rows; a uint8 array stays uint8. The annotators' masks sit about 0.24 px above the image
    features (`scripts/label_offset.py`), so recipes with `label_dy` move their targets down onto the image and
    their predicted maps back up (`infer.predict_frame`)."""
    if not d:
        return a
    assert abs(d) < 1, d
    x = a.astype(np.float32)
    zero = np.zeros_like(x[..., :1, :])
    if d > 0:  # out(r) = (1 - d) x(r) + d x(r - 1)
        out = (1 - d) * x + d * np.concatenate([zero, x[..., :-1, :]], axis=-2)
    else:  # out(r) = (1 + d) x(r) - d x(r + 1)
        out = (1 + d) * x - d * np.concatenate([x[..., 1:, :], zero], axis=-2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8) if a.dtype == np.uint8 else out.astype(a.dtype)


def load_frame(stem: str, scale: int, cache: Path = DEFAULT_CACHE, disks: dict | None = None,
               targets: bool = True, n_records: int = 0) -> Frame:
    """A frame at the model scale, with its soft filament and spine targets when `targets`."""
    cache = Path(cache)
    disks = disks if disks is not None else load_disks(cache)
    flat = downsample(load_png(cache / "flat" / f"{stem}.png"), scale)
    fr = Frame(stem, disks[stem], scale, flat, n_records=n_records)
    if targets:
        fr.p = downsample(load_png(cache / "p" / f"{stem}.png"), scale)
        fr.spine = downsample(load_png(cache / "spine" / f"{stem}.png"), scale)
    return fr


def coord_planes(disk: Disk, scale: int, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """x/R and y/R of the model-scale pixels [y0, y0 + h) x [x0, x0 + w)."""
    ys = ((np.arange(y0, y0 + h, dtype=np.float32) + 0.5) * scale - 0.5 - disk.cy) / disk.r
    xs = ((np.arange(x0, x0 + w, dtype=np.float32) + 0.5) * scale - 0.5 - disk.cx) / disk.r
    return np.stack([np.broadcast_to(xs[None, :], (h, w)), np.broadcast_to(ys[:, None], (h, w))])


def input_planes(fr: Frame, y0: int = 0, x0: int = 0, h: int | None = None, w: int | None = None,
                 contrast: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """(3, h, w) float32 network input of a window of the frame; (1, h, w) without the coordinate planes."""
    h = h or fr.size
    w = w or fr.size
    f = fr.flat[y0: y0 + h, x0: x0 + w].astype(np.float32) / 255.0 - 0.5  # F - 1
    f = (f * contrast + offset) * F_GAIN
    if not fr.coords:
        return f[None].astype(np.float32)
    return np.concatenate([f[None], coord_planes(fr.disk, fr.scale, y0, x0, h, w)]).astype(np.float32)


class CropSampler:
    """Random training crops with D4 and photometric augmentation."""

    def __init__(self, frames: list[Frame], crop: int, seed: int = 0, p_on_filament: float = 0.5,
                 fil_px: list | None = None, affine: float = 0.0, max_zoom: float = 1.2, copy_paste: float = 0.0,
                 elastic: float = 0.0, grid: float = 0.0, n_rec: int = 0):
        """A crop is centred near a random filament pixel with probability `p_on_filament`, else anywhere on the
        disk. `fil_px`: the filament-pixel index of another sampler over the same frames, to share it. `affine`:
        probability that a crop is resampled with a random rotation and a zoom between 1 / max_zoom and max_zoom
        (0 leaves the random stream of a recipe without it unchanged). `copy_paste`: probability that one filament
        of another training frame is pasted onto empty disk in the crop; `elastic` and `grid`: probabilities of an
        elastic and a grid distortion of the crop; `n_rec` > 0 appends each record's own mask after the two target
        planes, padded to `n_rec` planes with -1 (absent records). All default off, leaving older recipes' crops
        unchanged."""
        self.frames = frames
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.p_on = p_on_filament
        self.affine = affine
        self.max_zoom = max_zoom
        self.copy_paste, self.elastic, self.grid, self.n_rec = copy_paste, elastic, grid, n_rec
        self.fil_px = fil_px if fil_px is not None else [np.argwhere(fr.p > 0).astype(np.int32) for fr in frames]

    def _origin(self, i: int) -> tuple[int, int]:
        fr, c, rng = self.frames[i], self.crop, self.rng
        s = fr.size
        if len(self.fil_px[i]) and rng.random() < self.p_on:
            cy, cx = self.fil_px[i][rng.integers(len(self.fil_px[i]))]
            cy += int(rng.integers(-c // 3, c // 3 + 1))
            cx += int(rng.integers(-c // 3, c // 3 + 1))
        else:
            r = fr.disk.r / fr.scale
            ccx, ccy = (fr.disk.cx + 0.5) / fr.scale - 0.5, (fr.disk.cy + 0.5) / fr.scale - 0.5
            cy, cx = int(ccy + rng.uniform(-r, r)), int(ccx + rng.uniform(-r, r))
        y0 = int(np.clip(cy - c // 2, 0, s - c))
        x0 = int(np.clip(cx - c // 2, 0, s - c))
        return y0, x0

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """One augmented crop: its input planes and its filament and spine targets."""
        x, y, _ = self._sample()
        return x, y

    def _sample(self) -> tuple[np.ndarray, np.ndarray, int]:
        """A crop's input planes and targets, and the index of its frame."""
        rng, c = self.rng, self.crop
        i = int(rng.integers(len(self.frames)))
        fr = self.frames[i]
        y0, x0 = self._origin(i)
        contrast, offset = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))), float(rng.uniform(-0.02, 0.02))
        rad = None
        if self.affine > 0 and rng.random() < self.affine:
            x, y, rad = self._warped(fr, y0, x0, contrast, offset)
        else:
            x = input_planes(fr, y0, x0, c, c, contrast=contrast, offset=offset)
            y = np.stack([a[y0: y0 + c, x0: x0 + c] for a in self._planes(fr)]).astype(np.float32) / 255.0
            if self.copy_paste > 0:
                xr, yr = coord_planes(fr.disk, fr.scale, y0, x0, c, c)
                rad = np.hypot(xr, yr)
        if self.copy_paste > 0 and rng.random() < self.copy_paste:
            self._paste(x, y, rad, i, contrast, offset)
        if self.elastic > 0 and rng.random() < self.elastic:
            x, y = self._remap(x, y, *self._elastic_field())
        if self.grid > 0 and rng.random() < self.grid:
            x, y = self._remap(x, y, *self._grid_field())
        if self.n_rec:
            y = np.concatenate([y, np.full((2 + self.n_rec - len(y), c, c), -1, np.float32)])
        if rng.random() < 0.25:  # seeing
            x[0] = cv2.GaussianBlur(x[0], (0, 0), float(rng.uniform(0.4, 1.2 / fr.scale + 0.4)))
        if rng.random() < 0.25:
            x[0] += rng.normal(0, rng.uniform(0, 0.05), x[0].shape).astype(np.float32)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y), i

    def _warped(self, fr: Frame, y0: int, x0: int, contrast: float, offset: float) -> tuple:
        """The crop around the same centre, rotated by a random angle and zoomed: inputs and targets are resampled
        bilinearly, and the coordinate planes (if any) are computed at each output pixel's source position."""
        c, rng = self.crop, self.rng
        zoom = float(np.exp(rng.uniform(-np.log(self.max_zoom), np.log(self.max_zoom))))
        th = rng.uniform(0, 2 * np.pi)
        a = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) / zoom  # output -> source pixels
        mid = c / 2 - 0.5
        t = np.array([x0 + mid, y0 + mid]) - a @ np.array([mid, mid])
        m = np.hstack([a, t[:, None]]).astype(np.float32)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP

        def warp(img: np.ndarray, border: int) -> np.ndarray:
            """`img` resampled through the crop's affine map, with the given border mode."""
            return cv2.warpAffine(img, m, (c, c), flags=flags, borderMode=border, borderValue=0)

        f = warp(fr.flat, cv2.BORDER_REPLICATE).astype(np.float32) / 255.0 - 0.5
        f = (f * contrast + offset) * F_GAIN
        y = np.stack([warp(a, cv2.BORDER_CONSTANT) for a in self._planes(fr)]).astype(np.float32) / 255.0
        if not fr.coords and not self.copy_paste:
            return f[None].astype(np.float32), y, None
        u = np.arange(c, dtype=np.float32)
        sx = m[0, 0] * u[None, :] + m[0, 1] * u[:, None] + m[0, 2]  # source column of every output pixel
        sy = m[1, 0] * u[None, :] + m[1, 1] * u[:, None] + m[1, 2]
        s = fr.scale
        xr = ((sx + 0.5) * s - 0.5 - fr.disk.cx) / fr.disk.r
        yr = ((sy + 0.5) * s - 0.5 - fr.disk.cy) / fr.disk.r
        rad = np.hypot(xr, yr)
        if not fr.coords:
            return f[None].astype(np.float32), y, rad
        return np.stack([f, xr, yr]).astype(np.float32), y, rad

    def _planes(self, fr: Frame) -> list:
        """The target planes of a frame: the soft filament map, the second plane (spine or contact) and, with
        `n_rec`, each record's own mask."""
        return [fr.p, fr.spine] + (list(fr.recs) if self.n_rec else [])

    def _paste(self, x: np.ndarray, y: np.ndarray, rad: np.ndarray, i: int, contrast: float, offset: float,
               box: int = 256, tries: int = 5) -> None:
        """Copy-paste in place: one whole filament component of another training frame of this
        sampler (so of this fold's training frames), at a random quarter turn and flip, onto a part of the crop
        that is on the disk (r < 0.95 R) and free of any target within 3 px. Image and target planes are blended
        with the component's mask dilated by 3 px (the image with a soft edge)."""
        rng, c = self.rng, x.shape[-1]
        j = int(rng.integers(len(self.frames)))
        src, px = self.frames[j], self.fil_px[j]
        if j == i or not len(px) or c < box:
            return
        cy, cx = px[rng.integers(len(px))]
        s = src.size
        y0, x0 = int(np.clip(cy - box // 2, 0, s - box)), int(np.clip(cx - box // 2, 0, s - box))
        n, lab = cv2.connectedComponents((src.p[y0: y0 + box, x0: x0 + box] > 0).astype(np.uint8), connectivity=8)
        comp = lab == lab[cy - y0, cx - x0]
        if (comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()):
            return  # the filament does not fit in the box whole
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(comp.astype(np.uint8), k).astype(np.float32)
        alpha = cv2.GaussianBlur(mask, (0, 0), 1.0) * mask
        f = (src.flat[y0: y0 + box, x0: x0 + box].astype(np.float32) / 255.0 - 0.5)
        f = (f * contrast + offset) * F_GAIN
        tg = np.stack([a[y0: y0 + box, x0: x0 + box] for a in (src.p, src.spine)]).astype(np.float32) / 255.0
        q, flip = int(rng.integers(4)), rng.random() < 0.5
        tr = lambda a: (np.flip(np.rot90(a, q, axes=(-2, -1)), -1) if flip else np.rot90(a, q, axes=(-2, -1)))  # noqa
        mask, alpha, f, tg = tr(mask), tr(alpha), tr(f), tr(tg)
        on = mask > 0
        for _ in range(tries):
            dy, dx = (int(v) for v in rng.integers(0, c - box + 1, 2))
            win = (slice(dy, dy + box), slice(dx, dx + box))
            if (rad is not None and rad[win][on].max() >= 0.95) or (y[:2][(slice(None),) + win][:, on] > 0).any():
                continue
            x[(0,) + win] = x[(0,) + win] * (1 - alpha) + f * alpha
            for t in range(2):
                y[(t,) + win] = np.where(on, tg[t], y[(t,) + win])
            return

    def _elastic_field(self, alpha: float = 8.0, cells: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """A smooth random displacement field (px), up to about `alpha` px, varying over `cells`-px scales."""
        c, rng = self.crop, self.rng
        g = c // cells + 1
        d = [cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (c, c), interpolation=cv2.INTER_CUBIC) * alpha
             for _ in range(2)]
        return d[0], d[1]

    def _grid_field(self, steps: int = 8, limit: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """Grid distortion: each of `steps` bands along each axis stretched or shrunk by up to `limit`."""
        c, rng = self.crop, self.rng
        u = np.arange(c, dtype=np.float32)

        def axis() -> np.ndarray:
            w = 1 + rng.uniform(-limit, limit, steps)
            edges = np.concatenate([[0], np.cumsum(w) / w.sum() * (c - 1)])
            return np.interp(u, np.linspace(0, c - 1, steps + 1), edges).astype(np.float32) - u

        ax, ay = axis(), axis()
        return np.broadcast_to(ax[None, :], (c, c)).copy(), np.broadcast_to(ay[:, None], (c, c)).copy()

    @staticmethod
    def _remap(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Input and target planes resampled at (u + dx, v + dy), bilinearly; the image reflects at the crop's edge,
        the targets are 0 beyond it."""
        c = x.shape[-1]
        u = np.arange(c, dtype=np.float32)
        mx, my = u[None, :] + dx, u[:, None] + dy
        rx = np.stack([cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for a in x])
        ry = np.stack([cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                       for a in y])
        return rx, ry

    def batch(self, n: int, frames: bool = False) -> tuple:
        """`n` crops, stacked; with `frames`, also the index of each crop's frame (the annotator model reads each
        crop's record count and annotation group)."""
        xs, ys, idx = zip(*(self._sample() for _ in range(n)))
        out = (np.stack(xs), np.stack(ys))
        return out + (np.asarray(idx, np.int64),) if frames else out


def full_size(scale: int) -> int:
    """Side of a 2048 px frame at the model scale `scale`."""
    return H // scale

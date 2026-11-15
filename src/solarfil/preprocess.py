"""Physics normalisation of a GONG H-alpha frame: solar-disk fit and flat field.

The observed intensity is modelled as ``I(x) = I_clv(rho) * B(x) * (1 + c(x))``: the centre-to-limb profile, a
slowly varying background (clouds, gradients) and the chromospheric contrast ``c`` that the model reads
(filaments have ``c < 0``). The flat field is ``F = I / (I_clv * B) = 1 + c``, set to 1 off the disk.

1. Disk: limb points from 360 rays (the steepest intensity drop between 0.85 and 1.12 of the nominal radius),
   then an algebraic least-squares circle fit with two rounds of outlier trimming.
2. ``I_clv``: per-image median intensity in annuli of ``rho = r / R``.
3. ``B``: a normalised Gaussian convolution (sigma 90 px, about 0.1 R) of ``I / I_clv`` in which pixels more
   than 2.5 robust sigmas from 1 (filaments, plage, sunspots) carry no weight; two passes.

All steps run on a 4x downsampled copy except the final division, so one frame takes a fraction of a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .data import H, W

NOMINAL_CENTRE = (1023.1, 1024.1)  # (x, y), measured over all frames by a data audit
NOMINAL_RADIUS = 899.7
RHO_MAX = 0.985  # beyond this the limb line and the sky; the flat field is neutral there
N_ANNULI = 200
BACKGROUND_SIGMA = 90.0  # px at 2048
CLIP_SIGMA = 2.5
DOWN = 4


@dataclass(frozen=True)
class Disk:
    """The solar disk in a frame: centre and radius in 2048-px coordinates."""
    cx: float
    cy: float
    r: float

    def rho(self, shape=(H, W), scale: float = 1.0) -> np.ndarray:
        """r / R on a grid of the given shape, where `scale` maps that grid to 2048-px coordinates."""
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
        xx = (xx + 0.5) * scale - 0.5
        yy = (yy + 0.5) * scale - 0.5
        return np.hypot(xx - self.cx, yy - self.cy) / self.r


def fit_disk(img: np.ndarray, n_rays: int = 360) -> Disk:
    """Fit the limb circle of a 2048 x 2048 frame."""
    sm = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    cx0, cy0 = NOMINAL_CENTRE
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(0.85 * NOMINAL_RADIUS, 1.12 * NOMINAL_RADIUS, 0.5)
    xs = cx0 + np.cos(theta)[:, None] * radii[None, :]
    ys = cy0 + np.sin(theta)[:, None] * radii[None, :]
    prof = ndimage.map_coordinates(sm, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    k = np.argmin(np.gradient(prof, axis=1), axis=1)  # steepest drop along each ray
    px, py = xs[np.arange(n_rays), k], ys[np.arange(n_rays), k]
    keep = np.ones(n_rays, bool)
    for _ in range(3):
        cx, cy, r = _circle_fit(px[keep], py[keep])
        res = np.abs(np.hypot(px - cx, py - cy) - r)
        mad = np.median(res[keep]) + 1e-6
        keep = res < max(3.0, 4.0 * 1.4826 * mad)
    return Disk(float(cx), float(cy), float(r))


def _circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(a, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    return cx, cy, float(np.sqrt(c[2] + cx**2 + cy**2))


def _masked_blur(v: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution; 1 where no weighted pixel is near (off the disk)."""
    num = cv2.GaussianBlur(v * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0).astype(np.float32)


def flat_field(img: np.ndarray, disk: Disk | None = None) -> tuple[np.ndarray, Disk]:
    """The flat field ``F = 1 + c`` (float32, 2048 x 2048) and the disk it was computed with."""
    disk = disk or fit_disk(img)
    f = img.astype(np.float32)
    small = cv2.resize(f, (W // DOWN, H // DOWN), interpolation=cv2.INTER_AREA)
    rho_s = disk.rho(small.shape, scale=DOWN)
    on = rho_s < RHO_MAX
    # centre-to-limb profile from annulus medians
    edges = np.linspace(0, RHO_MAX, N_ANNULI + 1)
    idx = np.clip(np.digitize(rho_s, edges) - 1, 0, N_ANNULI - 1)
    prof = np.array([np.median(small[on & (idx == i)]) if np.any(on & (idx == i)) else np.nan for i in range(N_ANNULI)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(prof)
    prof = np.interp(centres, centres[ok], prof[ok])
    clv_s = np.interp(rho_s, centres, prof).astype(np.float32)
    ratio = np.where(on, small / np.maximum(clv_s, 1.0), 1.0).astype(np.float32)
    # slowly varying background, features masked out
    w = on.astype(np.float32)
    bg = np.ones_like(ratio)
    for _ in range(2):
        r = ratio / bg
        med = np.median(r[on])
        sig = 1.4826 * np.median(np.abs(r[on] - med)) + 1e-6
        w = (on & (np.abs(r - med) < CLIP_SIGMA * sig)).astype(np.float32)
        bg = _masked_blur(ratio, w, BACKGROUND_SIGMA / DOWN)
    model_s = (clv_s * bg).astype(np.float32)
    model = cv2.resize(model_s, (W, H), interpolation=cv2.INTER_LINEAR)
    out = f / np.maximum(model, 1.0)
    rho = disk.rho((H, W))
    out[rho >= RHO_MAX] = 1.0
    return out.astype(np.float32), disk


def global_field(img: np.ndarray, disk: Disk) -> np.ndarray:
    """The frame over its on-disk median, without the centre-to-limb and background corrections of `flat_field`
    (an ablation of it)."""
    f = img.astype(np.float32)
    rho = disk.rho((H, W))
    out = f / max(float(np.median(f[rho < RHO_MAX])), 1.0)
    out[rho >= RHO_MAX] = 1.0
    return out


def to_uint8(flat: np.ndarray) -> np.ndarray:
    """Store F on [0.5, 1.5] in steps of 1/255."""
    return np.clip(np.rint((flat - 0.5) * 255.0), 0, 255).astype(np.uint8)


def from_uint8(u: np.ndarray) -> np.ndarray:
    """The inverse of `to_uint8`: F as float32 on [0.5, 1.5]."""
    return u.astype(np.float32) / 255.0 + 0.5


def coordinate_planes(disk: Disk, shape=(H, W), scale: float = 1.0) -> np.ndarray:
    """(2, h, w) float32: x / R and y / R from the disk centre (solar east-west and north-south)."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    xx = ((xx + 0.5) * scale - 0.5 - disk.cx) / disk.r
    yy = ((yy + 0.5) * scale - 0.5 - disk.cy) / disk.r
    return np.stack([xx, yy]).astype(np.float32)

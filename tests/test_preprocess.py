"""Disk fit and flat field on a synthetic solar disk with limb darkening and a dark filament."""
import cv2
import numpy as np

from solarfil import preprocess
from solarfil.frames import Frame, input_planes


def synthetic_frame(cx=1023.4, cy=1024.2, r=903.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:2048, 0:2048].astype(np.float32)
    rho = np.hypot(xx - cx, yy - cy) / r
    mu = np.sqrt(np.clip(1 - rho**2, 0, 1))
    img = np.where(rho < 1, 180 * (0.65 + 0.35 * mu), 12.0)  # limb darkening, dark sky
    img *= 1 + 0.08 * (xx - 1024) / 1024  # a slow gradient (thin cloud)
    fil = np.zeros_like(img)
    cv2.line(fil, (900, 800), (1150, 860), 1.0, 9)
    img *= 1 - 0.25 * cv2.GaussianBlur(fil, (0, 0), 2)
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), (cx, cy, r)


def test_disk_fit_recovers_geometry():
    img, (cx, cy, r) = synthetic_frame()
    d = preprocess.fit_disk(img)
    assert abs(d.cx - cx) < 1.0 and abs(d.cy - cy) < 1.0 and abs(d.r - r) < 2.0


def test_flat_field_removes_limb_darkening_and_keeps_the_filament():
    img, _ = synthetic_frame()
    f, d = preprocess.flat_field(img)
    rho = d.rho()
    quiet = (rho < 0.9) & (np.abs(np.mgrid[0:2048, 0:2048][0] - 830) > 60)
    assert abs(np.median(f[quiet]) - 1.0) < 0.01  # centre-to-limb profile and gradient divided out
    assert np.percentile(f[quiet], 99) - np.percentile(f[quiet], 1) < 0.15
    assert f[830, 1020] < 0.85  # the filament stays dark
    assert np.all(f[rho >= preprocess.RHO_MAX] == 1.0)  # off-disk is neutral


def test_uint8_storage_round_trip():
    f = np.linspace(0.5, 1.5, 1000, dtype=np.float32)
    assert np.max(np.abs(preprocess.from_uint8(preprocess.to_uint8(f)) - f)) <= 0.5 / 255 + 1e-6


def test_input_planes_carry_disk_coordinates():
    disk = preprocess.Disk(1023.0, 1024.0, 904.0)
    fr = Frame("20140609195854Bh", disk, 2, np.full((1024, 1024), 128, np.uint8))
    x = input_planes(fr, 100, 200, 64, 32)
    assert x.shape == (3, 64, 32)
    # x/R and y/R of the window's first pixel centre (model scale 2: pixel i covers 2i, 2i+1 at 2048)
    assert np.isclose(x[1, 0, 0], ((200 + 0.5) * 2 - 0.5 - disk.cx) / disk.r)
    assert np.isclose(x[2, 0, 0], ((100 + 0.5) * 2 - 0.5 - disk.cy) / disk.r)

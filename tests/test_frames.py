"""Network inputs and training crops, with and without the disk-coordinate planes."""
import numpy as np

from solarfil.frames import CropSampler, Frame, input_planes
from solarfil.preprocess import Disk


def _frame(size=256):
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 256, (size, size), dtype=np.uint8)  # noqa: E731
    return Frame("x", Disk(size / 2 - 3.2, size / 2 + 5.1, size * 0.4), 1, img(), p=img(), spine=img())


def test_without_coordinate_planes_the_input_is_the_flat_field_alone():
    fr, c, y0, x0 = _frame(), 64, 70, 90
    full = input_planes(fr, y0, x0, c, c)
    fr.coords = False
    x = input_planes(fr, y0, x0, c, c)
    assert x.shape == (1, c, c)
    np.testing.assert_array_equal(x[0], full[0])
    xs, ys = CropSampler([fr], c).batch(3)
    assert xs.shape == (3, 1, c, c) and ys.shape == (3, 2, c, c)


def test_coordinate_planes_travel_with_the_pixels_of_a_crop():
    """After the D4 augmentation every crop pixel still carries its true disk position: the crop's x/R and y/R
    planes are a rotation or reflection of the planes of the plain window they were cut from."""
    fr, c = _frame(), 64
    s = CropSampler([fr], c, seed=3)
    for _ in range(8):
        x, _ = s.sample()
        x_min, y_min = x[1].min(), x[2].min()
        x0 = int(round((x_min * fr.disk.r + fr.disk.cx + 0.5) / fr.scale - 0.5))
        y0 = int(round((y_min * fr.disk.r + fr.disk.cy + 0.5) / fr.scale - 0.5))
        plain = input_planes(fr, y0, x0, c, c)[1:]
        views = [np.rot90(v, k, axes=(1, 2)) for v in (plain, plain[:, :, ::-1]) for k in range(4)]
        assert any(np.allclose(v, x[1:], atol=1e-6) for v in views)


class FixedDraws:
    """Stands in for the sampler's generator: `uniform` returns the queued values in order."""

    def __init__(self, *values):
        self.values = list(values)

    def uniform(self, lo, hi):
        return self.values.pop(0)


def _plain(fr, y0, x0, c):
    x = input_planes(fr, y0, x0, c, c)
    y = np.stack([fr.p[y0: y0 + c, x0: x0 + c], fr.spine[y0: y0 + c, x0: x0 + c]]).astype(np.float32) / 255.0
    return x, y


def test_warp_without_rotation_or_zoom_is_the_plain_crop():
    for coords in (True, False):
        fr, c, y0, x0 = _frame(), 64, 70, 90
        fr.coords = coords
        s = CropSampler([fr], c, affine=1.0, max_zoom=1.0)
        s.rng = FixedDraws(0.0, 0.0)  # zoom exponent, angle
        xw, yw, _ = s._warped(fr, y0, x0, 1.0, 0.0)
        x, y = _plain(fr, y0, x0, c)
        assert xw.shape == x.shape
        np.testing.assert_allclose(xw, x, atol=1e-4)
        np.testing.assert_allclose(yw, y, atol=1e-6)


def test_quarter_turn_is_rot90_of_the_plain_crop():
    fr, c, y0, x0 = _frame(), 64, 70, 90
    s = CropSampler([fr], c, affine=1.0, max_zoom=1.0)
    s.rng = FixedDraws(0.0, np.pi / 2)
    xw, yw, _ = s._warped(fr, y0, x0, 1.0, 0.0)
    x, y = _plain(fr, y0, x0, c)
    k = next(k for k in (1, 3) if np.allclose(np.rot90(y, k, axes=(1, 2)), yw, atol=1e-5))
    np.testing.assert_allclose(np.rot90(x, k, axes=(1, 2)), xw, atol=1e-4)


def test_without_affine_the_random_stream_is_unchanged():
    fr = _frame()
    a, b = CropSampler([fr], 64, seed=3), CropSampler([fr], 64, seed=3, affine=0.0, max_zoom=1.5)
    for _ in range(5):
        (xa, ya), (xb, yb) = a.sample(), b.sample()
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ya, yb)


def test_shift_rows_moves_the_content_down_and_back_up():
    from solarfil.frames import shift_rows

    a = np.zeros((40, 30), np.uint8)
    a[18:23, 5:25] = 200
    rows = np.arange(40, dtype=np.float64)[:, None]

    def centre(x):
        return float((x * rows).sum() / x.sum())

    down = shift_rows(a, 0.24)
    assert down.dtype == np.uint8
    assert abs(centre(down.astype(float)) - centre(a.astype(float)) - 0.24) < 0.02
    back = shift_rows(down, -0.24)
    assert abs(centre(back.astype(float)) - centre(a.astype(float))) < 0.02
    f = a.astype(np.float32) / 255
    np.testing.assert_allclose(shift_rows(f, 0.5)[19], 0.5 * f[19] + 0.5 * f[18])

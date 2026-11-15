"""The barb reading on drawn masks of known lean, and the conditional test on data with a planted bearing effect."""
import cv2
import numpy as np

from solarfil import chirality
from solarfil.preprocess import Disk


def ribbon(lean: int = 0, n_barbs: int = 4) -> np.ndarray:
    """A horizontal ribbon 5 px wide with `n_barbs` straight barbs 3 px wide hanging below it: lean +1 runs them
    down and to the right (a backslash), -1 down and to the left (a slash), 0 draws none."""
    m = np.zeros((160, 320), np.uint8)
    m[58:63, 20:300] = 1
    for x in np.linspace(70, 250, n_barbs).astype(int) if lean else []:
        cv2.line(m, (int(x), 60), (int(x) + 22 * lean, 82), 1, 3)
    return m.astype(bool)


def test_backslash_barbs_read_right():
    sc = chirality.score(chirality.barbs(ribbon(+1)))
    assert sc["barbs"] == 4 and sc["Sabs"] > 4 * 20
    assert chirality.call(sc) == chirality.RIGHT


def test_slash_barbs_read_left():
    sc = chirality.score(chirality.barbs(ribbon(-1)))
    assert sc["barbs"] == 4
    assert chirality.call(sc) == chirality.LEFT


def test_bearing_does_not_depend_on_spine_direction_or_side():
    """The mask turned by 180 degrees (barbs above the ribbon, the spine run the other way, the same backslash
    lean) reads the same."""
    above = ribbon(+1)[::-1, ::-1]
    assert chirality.call(chirality.score(chirality.barbs(above))) == chirality.RIGHT


def test_ribbon_without_barbs_gives_no_call():
    sc = chirality.score(chirality.barbs(ribbon(0)))
    assert sc == dict(barbs=0, S=0.0, Sabs=0.0, Sn=0)
    assert chirality.call(sc) == 0
    assert chirality.bearing_features(sc, 20.0) == dict(h=1.0, hx=20.0, m=0.0, ls=0.0, cons=0.0)


def test_branch_threshold_drops_short_barbs():
    lengths = [length for length, _ in chirality.barbs(ribbon(+1), min_length=4)]
    assert chirality.score(chirality.barbs(ribbon(+1)), branch=max(lengths) + 1)["barbs"] == 0


def test_latitude_north_up():
    disk = Disk(cx=160.0, cy=80.0, r=60.0)
    assert chirality.latitude(ribbon(0), disk) > 0  # the ribbon sits above the disk centre


def synthetic_rows(effect: float, n: int = 1600, seed: int = 0) -> list[dict]:
    """Rows of 8 groups and 160 frames: the label follows the hemisphere (70%), and a reader's bearing agrees with
    the label with probability 0.5 + effect / 2 on the half of the rows where it is legible."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n):
        lat = rng.uniform(-40, 40)
        y = int((lat < 0) == (rng.random() < 0.7))
        sc = dict(barbs=0, S=0.0, Sabs=0.0, Sn=0)
        if rng.random() < 0.5:
            right = (rng.random() < 0.5 + effect / 2) == (y == 1)
            S = 30.0 * chirality.C * (1 if right else -1)
            sc = dict(barbs=1, S=S, Sabs=30.0, Sn=int(np.sign(S)))
        rows.append(dict(y=y, g=f"G{q % 8}", f=q % 160, **chirality.bearing_features(sc, lat)))
    return rows


def test_conditional_test_recovers_planted_effect():
    r = chirality.beyond_latitude(synthetic_rows(0.8), n_groups=500, n_frames=500)
    assert r["rows"] == 1600 and r["groups"] == 8 and r["frames"] == 160
    assert 0.45 < r["legible"] < 0.55
    # a legible bearing right 90% of the time overrides the hemisphere where they disagree: accuracy 0.70 -> 0.90 on
    # the legible half, about +0.10 overall
    assert 0.06 < r["accuracy_gain"] < 0.14 and r["accuracy_gain_groups90"][0] > 0
    assert r["log_loss_change"] < -0.05 and r["log_loss_change_groups90"][1] < 0


def test_conditional_test_finds_nothing_without_effect():
    r = chirality.beyond_latitude(synthetic_rows(0.0), n_groups=500, n_frames=500)
    assert abs(r["accuracy_gain"]) < 0.03 and r["log_loss_change"] > -0.01

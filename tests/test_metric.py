"""Parity of `solarfil.metric` with the host's scoring code, and the RLE round trip."""
import numpy as np
import pandas as pd
import pytest

from host_reference import get_overlap_df, get_pq_score
from solarfil import metric, rle

S = 2048


def _blob(rng, n_px=S):
    """A random rotated ellipse, as a stand-in for a filament mask."""
    import cv2

    m = np.zeros((n_px, n_px), np.uint8)
    c = tuple(int(v) for v in rng.integers(100, n_px - 100, 2))
    axes = (int(rng.integers(5, 60)), int(rng.integers(2, 12)))
    cv2.ellipse(m, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return m.astype(bool)


def _jitter(rng, m):
    """A shifted, slightly dilated or eroded copy, so IoUs land around the 0.5 threshold."""
    import cv2

    dy, dx = rng.integers(-4, 5, 2)
    m2 = np.roll(np.roll(m, dy, 0), dx, 1).astype(np.uint8)
    k = int(rng.integers(0, 3))
    if k:
        op = cv2.dilate if rng.random() < 0.5 else cv2.erode
        m2 = op(m2, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    return m2.astype(bool)


def _case(seed):
    rng = np.random.default_rng(seed)
    stems = [f"2014060919585{i}Bh" for i in range(3)]
    gt_rows, pred_rows, gt, pred = [], [], {}, {}
    for stem in stems:
        base = [_blob(rng) for _ in range(int(rng.integers(0, 8)))]
        for annot in ("040301", "040302"):
            masks = [_jitter(rng, m) for m in base if rng.random() < 0.8]
            if not masks:
                continue
            for k, m in enumerate(masks):
                gt_rows.append((f"{annot}-{stem}_{k}", rle.encode(m)))
            gt.setdefault(stem, []).append([rle.to_coco(rle.encode(m)) for m in masks])
        # pixel-disjoint predictions, as the scorer requires
        taken = np.zeros((S, S), bool)
        masks = []
        for m in [_jitter(rng, m) for m in base] + [_blob(rng) for _ in range(int(rng.integers(0, 3)))]:
            m = m & ~taken
            if m.sum():
                masks.append(m)
                taken |= m
        for k, m in enumerate(masks):
            pred_rows.append((f"{stem}_{k}", rle.encode(m)))
        pred[stem] = [rle.to_coco(rle.encode(m)) for m in masks]
    cols = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(gt_rows, columns=cols), pd.DataFrame(pred_rows, columns=cols), gt, pred


@pytest.mark.parametrize("seed", range(6))
def test_pq_matches_host(seed):
    gt_df, pred_df, gt, pred = _case(seed)
    if gt_df.empty:
        pytest.skip("no GT drawn")
    host = get_pq_score(get_overlap_df(gt_df, pred_df))
    rep = metric.Report()
    for stem, records in gt.items():
        for rec in records:
            rep.add_record(stem, rec, pred.get(stem, []))
    assert rep.total.pq == pytest.approx(host, abs=1e-6)


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    m = _blob(rng)
    assert np.array_equal(rle.decode(rle.encode(m)), m)


def test_bootstrap_interval_contains_point():
    rows = np.array([[3, 1, 1, 2.1], [2, 2, 0, 1.3], [5, 0, 2, 3.6], [1, 1, 1, 0.6]], float)
    lo, hi = metric.bootstrap_pq(rows, n=500)
    s = rows.sum(0)
    point = s[3] / (s[0] + 0.5 * s[1] + 0.5 * s[2])
    assert lo <= point <= hi


def test_label_scoring_matches_rle_scoring():
    """`GTIndex.score` on a label image equals the RLE path on real records (another record's masks as predictions)."""
    from solarfil.data import DEFAULT_ROOT, Dataset

    if not (DEFAULT_ROOT / "train").exists():
        pytest.skip("data not present")
    ds = Dataset()
    obs = next(o for o in ds.observations.values() if o.n_records == 3)
    lab = np.zeros((S, S), np.int32)
    for k, f in enumerate(obs.records[0].filaments, start=1):
        m = f.mask() & (lab == 0)
        lab[m] = k
    pred = [rle.to_coco(rle.encode(lab == k)) for k in range(1, lab.max() + 1) if (lab == k).any()]
    rep = metric.Report()
    for rec in obs.records:
        rep.add_record(obs.stem, metric.gt_rles_cached(rec), pred)
    fast = metric.GTIndex(obs).score(lab)
    assert (fast.tp, fast.fp, fast.fn) == (rep.total.tp, rep.total.fp, rep.total.fn)
    assert fast.iou_sum == pytest.approx(rep.total.iou_sum, abs=1e-9)

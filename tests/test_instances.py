"""The instance model's pieces that need no trained network: targets, detection storage, the candidates' columns and
the anchor option."""
import numpy as np
import torch

from solarfil import instances as detector
from solarfil.instances import COLUMNS, SIZE, build, describe, half, load_instances, save_instances, targets
from solarfil.tree import build_tree


def test_targets_give_boxes_of_the_masks_and_allow_empty_records():
    m = np.zeros((SIZE, SIZE), bool)
    m[10:20, 30:90] = True
    t = targets([m, np.zeros_like(m)])
    assert t["boxes"].tolist() == [[30.0, 10.0, 90.0, 20.0]] and t["labels"].tolist() == [1]
    assert t["masks"].shape == (1, SIZE, SIZE)
    e = targets([])
    assert e["boxes"].shape == (0, 4) and e["masks"].shape == (0, SIZE, SIZE)


def test_detections_survive_storage(tmp_path):
    rng = np.random.default_rng(0)
    masks = rng.random((3, SIZE, SIZE)) > 0.9
    save_instances(tmp_path / "d.npz", np.array([0.9, 0.5, 0.1]), masks, np.array([0.8, 0.7, 0.6]))
    scores, back, maskness = load_instances(tmp_path / "d.npz")
    np.testing.assert_allclose(scores, [0.9, 0.5, 0.1])
    np.testing.assert_allclose(maskness, [0.8, 0.7, 0.6])
    assert np.array_equal(back, masks)


def test_a_detection_that_matches_a_candidate_gives_it_iou_one():
    p = np.zeros((2048, 2048), np.float32)
    p[100:120, 100:400] = 0.9  # a filament the detector finds
    p[500:520, 500:700] = 0.6  # one it misses
    tree = build_tree(p, splits=True)
    det = np.zeros((SIZE, SIZE), bool)
    det[50:60, 50:200] = True  # the first filament at 1024 px
    X = describe(tree, np.array([0.8]), det[None], np.array([0.75]))
    found = [v for v in range(tree.n) if tree.area[v] == 20 * 300]
    missed = [v for v in range(tree.n) if tree.area[v] == 20 * 200]
    assert len(found) == 1 and len(missed) == 1
    np.testing.assert_allclose(X[found[0]], [1.0, 0.8, 0.8, 0.8, 1.0, 0.6], atol=1e-6)
    np.testing.assert_allclose(X[missed[0]], np.zeros(len(COLUMNS)))
    assert half(np.repeat(np.repeat(det, 2, 0), 2, 1)).tolist() == det.tolist()


def test_a_frame_without_detections_survives_storage(tmp_path):
    save_instances(tmp_path / "e.npz", np.zeros(0), np.zeros((0, SIZE, SIZE), bool), np.zeros(0))
    scores, masks, maskness = load_instances(tmp_path / "e.npz")
    assert scores.shape == (0,) and masks.shape == (0, SIZE, SIZE) and maskness.shape == (0,)


def test_smaller_anchors_keep_the_proposal_head():
    default, small = build(coco=False), build(coco=False, anchor=16)
    assert [s[0] for s in small.rpn.anchor_generator.sizes] == [16, 32, 64, 128, 256]
    assert [s[0] for s in default.rpn.anchor_generator.sizes] == [32, 64, 128, 256, 512]
    per_location = small.rpn.anchor_generator.num_anchors_per_location()
    assert per_location == default.rpn.anchor_generator.num_anchors_per_location()
    small.load_state_dict(default.state_dict())  # the same weights fit: the COCO proposal head transfers


class Bright(torch.nn.Module):
    """A stand-in detector: one detection, the bright pixels of its input, scored by the share of them in the input's
    left half, so its mask follows the frame under a flip while its score does not."""

    def forward(self, xs):
        on = (xs[0][0] > 0.5).float()
        return [dict(scores=(on[:, :SIZE // 2].sum() / on.sum())[None], masks=on[None, None])]


def test_flipped_views_pool_detections_flipped_back_onto_the_frame():
    flat = np.zeros((2048, 2048), np.uint8)
    flat[100:140, 200:1600] = 255  # a band in the upper left
    one = detector.detect(Bright(), flat, torch.device("cpu"))
    scores, packed, maskness = detector.detect(Bright(), flat, torch.device("cpu"), views=4)
    assert len(one[0]) == 1 and len(scores) == 4 and maskness.tolist() == [1.0] * 4
    band = half(flat == 255)
    assert all(np.array_equal(m, band) for m in detector.unpack(packed))  # every view's mask is back on the band
    np.testing.assert_allclose(scores, [one[0][0], 1 - one[0][0], one[0][0], 1 - one[0][0]], atol=1e-6)
    assert np.array_equal(packed[:1], one[1])

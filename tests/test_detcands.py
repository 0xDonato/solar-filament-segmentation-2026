"""Detection candidates: pixel sets at 2048 px, overlaps with the tree's nodes, and matched IoU against records."""
from types import SimpleNamespace

import numpy as np

from solarfil import detcands
from solarfil.instances import SIZE
from solarfil.preprocess import Disk
from solarfil.tree import build_tree


def test_a_detection_on_a_node_overlaps_it_fully_and_matches_its_record():
    p = np.zeros((2048, 2048), np.float32)
    p[100:120, 100:400] = 0.9
    tree = build_tree(p, splits=True)
    det = np.zeros((1, SIZE, SIZE), bool)
    det[0, 50:60, 50:200] = True  # the same filament at 1024 px
    flat = np.ones((2048, 2048), np.float32)
    c = detcands.candidates(tree, p, flat, Disk(1024.0, 1024.0, 900.0), np.array([0.9]), det, np.array([0.8]))
    assert len(c["keep"]) == 1 and len(c["pix"][0]) == 20 * 300
    v = int(np.argmax(c["inter_nodes"][0]))
    assert c["inter_nodes"][0, v] == tree.area[v] == 20 * 300
    X = dict(zip(detcands.COLUMNS, c["X"][0]))
    assert X["d_best_node_iou"] == 1.0 and X["d_score"] == np.float32(0.9) and X["d_frac_p50"] == 1.0
    yy, xx = np.mgrid[100:120, 100:300]
    gt = SimpleNamespace(records=[[np.sort((yy * 2048 + xx).ravel())], []])  # one record draws two thirds of it
    T = detcands.match(c["pix"], gt)
    np.testing.assert_allclose(T, [[200 / 300, 0.0]], rtol=1e-6)


def test_paint_keeps_masks_disjoint_and_whole():
    a = np.sort(np.array([y * 2048 + x for y in range(10, 20) for x in range(10, 60)]))
    b = np.sort(np.array([y * 2048 + x for y in range(15, 25) for x in range(50, 90)]))  # overlaps a's end
    lab = detcands.paint([a, b], [0.6, 0.9])
    assert lab.max() == 2
    assert (lab == 1).sum() == len(b)  # the higher utility keeps all its pixels and comes first
    assert (lab == 2).sum() == len(a) - 5 * 10  # the other loses the shared 5 x 10 corner


def test_mixed_selection_without_detections_is_the_tree_selection():
    from solarfil.tree import select
    p = np.zeros((2048, 2048), np.float32)
    p[100:120, 100:400] = 0.9
    p[500:520, 500:700] = 0.6
    tree = build_tree(p, splits=True)
    u = np.linspace(0.2, 0.6, tree.n)
    c = dict(X=np.zeros((0, len(detcands.COLUMNS))), inter_nodes=np.zeros((0, tree.n)), inter_dets=np.zeros((0, 0)),
             area=np.zeros(0))
    nodes, dets = detcands.select_mixed(tree, u, c, np.zeros(0), 0.26)
    assert sorted(nodes) == sorted(select(tree, u, 0.26)) and dets == []

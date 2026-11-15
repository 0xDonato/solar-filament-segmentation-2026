"""The comparison of a selection with its annotators (`solarfil.reference`) on tiny synthetic frames: the host's
degrees with splits and merges, their support by another annotator, spine coverage, the anatomy of the 0:1 and 1:0
bars, and the one-piece, no-hole check."""
from collections import Counter

import numpy as np

from solarfil import metric, reference, rle

SHAPE = (24, 40)


def _box(y0, y1, x0, x1):
    m = np.zeros(SHAPE, bool)
    m[y0:y1, x0:x1] = True
    return m


def _rles(masks):
    return [rle.to_coco(rle.encode(m), *SHAPE) for m in masks]


def _frame(*records, spines=None):
    """A frame of the given records (lists of boolean masks); spines default to none drawn."""
    spines = spines or [[np.zeros((0, 2))] * len(r) for r in records]
    return reference.Frame("20140609195854Bh", [_rles(r) for r in records], spines, SHAPE)


SPLIT = _box(2, 5, 2, 30)  # a filament that two masks cover in pieces
PIECES = [_box(2, 5, 2, 16), _box(2, 5, 16, 30)]
PAIR = [_box(10, 13, 2, 12), _box(10, 13, 14, 24)]  # two filaments that one mask covers
JOINED = _box(10, 13, 2, 24)
LONE = _box(18, 21, 2, 10)


def test_degrees_count_misses_splits_and_merges():
    frame = _frame([SPLIT, *PAIR, LONE])
    extra = _box(18, 21, 30, 38)  # overlaps no filament
    c = reference.against_masks(frame, _rles([*PIECES, JOINED, extra]))
    assert c.comparisons == 1
    assert c.report.gt_degree == Counter({0: 1, 1: 2, 2: 1})
    assert c.report.pred_degree == Counter({0: 1, 1: 2, 2: 1})
    assert reference.legend_counts(c.report) == {"1:0": 1, "0:1": 1, "splits": 1, "merges": 1}


def test_annotators_are_scored_in_both_orders():
    """Record B draws A's split filament in pieces and A's pair as one: every ordered pair is a comparison, so the
    annotators' filament and mask degrees are the same counts."""
    c = reference.against_annotators(_frame([SPLIT, *PAIR, LONE], [*PIECES, JOINED, LONE]))
    assert c.comparisons == 2
    assert c.report.gt_degree == c.report.pred_degree == Counter({1: 6, 2: 2})
    assert reference.legend_counts(c.report) == {"1:0": 0, "0:1": 0, "splits": 2, "merges": 2}


def test_relations_are_supported_by_one_record_that_drew_the_same_way():
    iou = np.array([[0.4, 0.3, 0.0],  # filament 0, split by masks 0 and 1
                    [0.0, 0.0, 0.3],  # filaments 1 and 2, merged by mask 2
                    [0.0, 0.0, 0.2]])
    pieces = np.array([[0.6, 0.0, 0.0], [0.0, 0.7, 0.0]])  # another record drew filament 0 as masks 0 and 1
    whole = np.array([[0.0, 0.0, 0.8]])  # another record drew filaments 1 and 2 as mask 2
    assert reference.relations(iou, []).tolist() == [1, 0, 1, 0]
    assert reference.relations(iou, [pieces]).tolist() == [1, 1, 1, 0]
    assert reference.relations(iou, [whole]).tolist() == [1, 0, 1, 1]
    assert reference.relations(iou, [pieces, whole]).tolist() == [1, 1, 1, 1]
    assert reference.relations(iou, [pieces[:1], pieces[1:]]).tolist() == [1, 0, 1, 0]  # the pieces from two records


def test_relation_support_for_a_selection_and_for_the_annotators():
    """A and C draw a filament whole, B in two pieces. The selection's pieces copy B: they split A's and C's filament,
    and B supports both splits. Between annotators, B's pieces split A's and C's filament without support from the
    third record, and A's or C's whole filament merges B's pieces with support from the other."""
    whole_a, whole_c = _box(2, 6, 2, 30), _box(2, 6, 3, 30)
    pieces = [_box(2, 6, 2, 15), _box(2, 6, 17, 30)]
    frame = _frame([whole_a], pieces, [whole_c])
    assert reference.against_masks(frame, _rles(pieces)).relations.tolist() == [2, 2, 0, 0]
    assert reference.against_annotators(frame).relations.tolist() == [2, 0, 2, 2]
    s = reference.relation_summary([reference.against_masks(frame, _rles(pieces))])
    assert (s["comparisons"], s["splits_per_comparison"], s["splits_supported_share"]) == (3, 0.6667, 1.0)


def test_spine_samples_step_one_pixel_and_stay_inside_the_frame():
    p = reference.spine_samples(np.array([[2.0, 5.0], [10.0, 5.0], [10.0, 9.0]]), SHAPE)
    assert len(p) == 9 + 5  # each segment from end to end, the shared vertex twice
    assert p[0].tolist() == [2, 5] and p[-1].tolist() == [10, 9]
    assert np.abs(np.diff(p, axis=0)).max() <= 1
    off = reference.spine_samples(np.array([[-3.0, 1.0], [3.0, 30.0]]), SHAPE)
    assert off[:, 0].min() == 0 and off[:, 1].max() == SHAPE[0] - 1


def test_spine_cover_counts_the_pieces_a_spine_runs_through():
    labels = np.zeros(SHAPE, np.int32)
    labels[5, 0:6], labels[5, 6:20], labels[5, 20:22] = 1, 2, 3  # mask 3 holds two samples: not a piece
    along = np.array([[0.0, 5.0], [23.0, 5.0]])
    assert reference.spine_cover(labels, along) == (22 / 24, 14 / 24, 2, 24)
    assert reference.spine_cover(labels, np.array([[0.0, 15.0], [23.0, 15.0]])) == (0.0, 0.0, 0, 24)


def test_spines_read_on_the_selection_and_on_the_other_annotator():
    """Record A's spine runs along its filament; the selection covers it with one mask, record B in two pieces."""
    spines = [[np.array([[3.0, 3.0], [28.0, 3.0]])], [np.array([[3.0, 3.0], [14.0, 3.0]]),
                                                      np.array([[17.0, 3.0], [28.0, 3.0]])]]
    frame = _frame([_box(2, 5, 2, 30)], PIECES, spines=spines)
    model = reference.against_masks(frame, _rles([_box(1, 6, 1, 31)])).spines
    assert model["inside_one"].tolist() == [1.0, 1.0, 1.0] and model["pieces"].tolist() == [1, 1, 1]
    human = reference.against_annotators(frame).spines  # A's spine on B's pieces, B's spines on A's filament
    assert sorted(human["pieces"].tolist()) == [1, 1, 2]
    s = reference.spine_summary([reference.against_annotators(frame)])
    assert s["spines"] == 3 and s["untouched"] == 0.0 and s["touched"]["in_pieces"] == 0.3333


def test_miss_anatomy_splits_the_0_1_bar_by_who_drew_the_mask():
    """A mask only record B's filament overlaps adds one to the 0:1 bar (for record A), drawn by another annotator;
    a mask nothing overlaps adds one per record. A's filament that no mask overlaps and B did not draw is a 1:0 that
    no other annotator drew."""
    a = [_box(2, 6, 2, 20), _box(12, 16, 2, 20)]
    b = [_box(2, 6, 3, 20), _box(12, 16, 25, 38)]
    masks = _rles([_box(2, 6, 2, 20), _box(12, 16, 25, 38), _box(20, 23, 30, 38)])
    s = reference.miss_summary([reference.against_masks(_frame(a, b), masks)])
    assert (s["masks"], s["bar_0_1"], s["masks_no_record_overlaps"], s["masks_some_records_overlap"]) == (3, 3, 1, 1)
    assert s["bar_0_1_drawn_by_another_annotator"] == 0.3333
    assert (s["bar_1_0"], s["bar_1_0_multi"], s["bar_1_0_multi_drawn_by_no_other_record"]) == (1, 1, 1)
    assert (s["hit_multi"], s["hit_multi_drawn_by_no_other_record"]) == (3, 1)


def test_mask_shape_counts_8_connected_pieces_and_4_connected_holes():
    ring = _box(2, 9, 2, 9) & ~_box(4, 7, 4, 7)
    plus = np.zeros(SHAPE, bool)
    plus[[4, 6, 5, 5], [5, 5, 4, 6]] = True  # four pixels around one: one piece by its corners, the centre a hole
    corner = _box(0, 5, 0, 5) & ~_box(0, 2, 0, 2)  # a notch open to the frame's edge is not a hole
    assert reference.mask_shape(_box(2, 6, 3, 20)) == (1, 0)
    assert reference.mask_shape(ring) == (1, 1)
    assert reference.mask_shape(_box(2, 4, 2, 6) | _box(8, 10, 2, 6)) == (2, 0)
    assert reference.mask_shape(plus) == (1, 1)
    assert reference.mask_shape(corner) == (1, 0)
    assert reference.mask_shape(np.zeros(SHAPE, bool)) == (0, 0)


def test_frame_intervals_share_the_pq_bootstrap_of_the_metric():
    rows = np.array([[3, 1, 1, 2.1, 5, 3.0, 4.0], [2, 2, 0, 1.3, 3, 1.9, 2.6], [5, 0, 2, 3.6, 6, 4.1, 5.0],
                     [1, 1, 1, 0.6, 2, 0.9, 1.3]])
    ci = reference.frame_intervals(rows, n=500, seed=0)
    assert ci["pq"] == [round(x, 5) for x in metric.bootstrap_pq(rows[:, :4], n=500, seed=0)]
    s = rows.sum(axis=0)
    assert ci["iou"][0] <= s[5] / s[4] <= ci["iou"][1] and ci["dice"][0] <= s[6] / s[4] <= ci["dice"][1]

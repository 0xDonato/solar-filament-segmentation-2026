"""The report's figures drawn from stored summaries: `plots.ReferenceSeries` from a `reference.host_summary` dict
holds what the same series holds from the `metric.Report`, so Fig. 2 can be drawn again from the JSON alone."""
import numpy as np

from solarfil import plots, reference
from test_reference import JOINED, LONE, PAIR, PIECES, SPLIT, _box, _frame, _rles


def _parts():
    frame = _frame([SPLIT, *PAIR, LONE], [*PIECES, JOINED, LONE])
    model = reference.against_masks(frame, _rles([*PIECES, JOINED, _box(18, 21, 30, 38)]))
    return [model], [reference.against_annotators(frame)]


def test_series_from_the_summary_match_the_series_from_the_report():
    for parts in _parts():
        rep = reference.merge_reports(p.report for p in parts)
        a = plots.ReferenceSeries.of_report(rep)
        b = plots.ReferenceSeries.of_summary(reference.host_summary(parts, n_boot=50))
        assert a.comparisons == b.comparisons and a.gt_degree == b.gt_degree and a.pred_degree == b.pred_degree
        np.testing.assert_array_equal(a.iou_counts, b.iou_counts)
        np.testing.assert_array_equal(a.dice_counts, b.dice_counts)
        assert abs(a.iou_mean - b.iou_mean) < 1e-4 and abs(a.dice_mean - b.dice_mean) < 1e-4
        assert reference.legend_counts(b) == reference.legend_counts(rep)


def test_figure_two_draws_from_summaries_in_both_formats(tmp_path):
    model, annotators = _parts()
    series = plots.ReferenceSeries.of_summary(reference.host_summary(model, n_boot=50))
    human = plots.ReferenceSeries.of_summary(reference.host_summary(annotators, n_boot=50))
    for name in ("fig.png", "fig.pdf"):
        plots.reference_figure([("model", series)], human, tmp_path / name)
        assert (tmp_path / name).stat().st_size > 1000
    plots.reference_figure([("model", reference.merge_reports(p.report for p in model))],
                           reference.merge_reports(p.report for p in annotators), tmp_path / "from_reports.png")

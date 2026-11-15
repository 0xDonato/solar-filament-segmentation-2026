"""The organizers' measures of out-of-fold selections beside a second annotator (the report's Fig. 2 and Table III).

Each selection file (the submission format, as `oof_select.py` writes it) is compared with the annotators of the
training frames that `--frames` chooses: `multi`, the frames two or three annotators drew, where the human
reference exists (the default), or `all`. `solarfil.reference` scores the selection against every record of a
frame, and every annotator against every other annotator of the same frame in every ordered pair, with the host's
code. The JSON holds, for each selection and for the annotators:
- the host's PQ, SQ, RQ and counts; the IoU and Dice of the overlapping pairs (mean, quantiles and the host's 50-bin
  histogram counts); 90% frame-bootstrap intervals on PQ and on both means;
- the filament (1:n) and mask (n:1) degrees, and the 1:0, 0:1, split and merge counts per comparison (record, or
  ordered pair of records);
- how many splits and merges another annotator of the frame supports, on the frames with several records;
- spine continuity on the annotators' own spines;
- for each selection, the anatomy of the 0:1 and 1:0 bars.
`--figure` draws the host's three panels with each selection beside the second annotator.

`--check-masks` instead counts, in each file (a selection or a test submission), the masks that are not one
8-connected piece and those with holes. `--from-json` draws the figure again from a JSON this script wrote (its
histogram counts, means and degrees), without scoring anything. `--figure` takes one path per format (png, pdf).

Usage: judges_reference.py A.csv [B.csv ...] --out judges_reference.json [--frames multi|all] [--boot 2000]
       [--figure fig2.png [fig2.pdf] --labels "model, tau 0.26" "model, tau 0.30"] [--workers 3]
       judges_reference.py submission.csv [...] --check-masks --out masks.json
       judges_reference.py --from-json judges_reference.json --figure fig2.png fig2.pdf
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paired_bootstrap import read_selection
from solarfil import reference, rle
from solarfil.data import Dataset
from solarfil.parallel import pool
from solarfil.plots import ReferenceSeries, reference_figure

_DS: dict = {}


def compare_frame(args) -> tuple[reference.Comparison, list[reference.Comparison]]:
    """The annotators' comparison on one frame, and each selection's."""
    stem, selections = args
    frame = reference.Frame.of(_DS.setdefault("ds", Dataset()).observations[stem])
    return reference.against_annotators(frame), [reference.against_masks(frame, masks) for masks in selections]


def compare(paths: list[str], labels: list[str], frames: str, n_boot: int, seed: int, workers: int
            ) -> tuple[dict, list[reference.Comparison], list[list[reference.Comparison]]]:
    """The JSON summary, and the annotators' and each selection's comparisons on every chosen frame."""
    obs = Dataset().observations
    stems = [s for s in sorted(obs) if frames == "all" or obs[s].n_records > 1]
    selections = [read_selection(p) for p in paths]
    with pool(workers) as w:
        res = w.map(compare_frame, [(s, [sel.get(s, []) for sel in selections]) for s in stems], chunksize=4)
    annotators = [a for a, _ in res]
    models = [[m[i] for _, m in res] for i in range(len(paths))]
    out = dict(frames=frames, n_frames=len(stems), records=sum(obs[s].n_records for s in stems),
               ordered_pairs=sum(a.comparisons for a in annotators), boot=n_boot, seed=seed,
               annotators=reference.summarize(annotators, n_boot, seed),
               selections=[dict(file=p, label=label, **reference.summarize(parts, n_boot, seed))
                           for p, label, parts in zip(paths, labels, models)])
    return out, annotators, models


def headline(label: str, s: dict) -> str:
    """One line of the numbers the report quotes most."""
    h, r, sp = s["host"], s["relations"], s["spines"]
    return json.dumps(dict(series=label, pq=h["pq"], pq_ci90=h["pq_ci90"], iou=h["iou"]["mean"], dice=h["dice"]["mean"],
                           **h["per_comparison"], splits_supported=r["splits_supported_share"],
                           merges_supported=r["merges_supported_share"], spine_inside_one=sp["inside_one"]))


def pieces_and_holes(counts: str) -> tuple[int, int]:
    """`reference.mask_shape` of one row's mask."""
    return reference.mask_shape(rle.decode(counts))


def check_masks(path: str, workers: int) -> dict:
    """Masks of a selection or submission file that are not one 8-connected piece, and those with holes."""
    with open(path, newline="") as fh:
        counts = [row["segmentation_rle"] for row in csv.DictReader(fh)]
    with pool(workers) as w:
        shapes = np.array(w.map(pieces_and_holes, counts, chunksize=64), np.int64).reshape(-1, 2)
    one_piece, no_holes = shapes[:, 0] == 1, shapes[:, 1] == 0
    return dict(file=path, masks=len(counts), not_one_piece=int((~one_piece).sum()), with_holes=int((~no_holes).sum()),
                one_piece_without_holes=int((one_piece & no_holes).sum()))


def main():
    """Score the selections and the annotators, check masks, or redraw the figure from a JSON, as the flags ask."""
    ap = argparse.ArgumentParser()
    ap.add_argument("selections", nargs="*", help="selection or submission CSV files")
    ap.add_argument("--out", default=None, help="the JSON to write")
    ap.add_argument("--frames", choices=("multi", "all"), default="multi",
                    help="the frames two or three annotators drew, or every training frame")
    ap.add_argument("--boot", type=int, default=2000, help="frame-bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--figure", nargs="+", default=None,
                    help="png and/or pdf files of the host's three panels beside a second annotator")
    ap.add_argument("--labels", nargs="+", default=None, help="one per selection, for the JSON and the figure")
    ap.add_argument("--font-scale", type=float, default=1.0, help="figure type scale at the final placed size")
    ap.add_argument("--check-masks", action="store_true", help="only count masks not in one piece or with holes")
    ap.add_argument("--from-json", default=None, help="draw --figure from this script's JSON, scoring nothing")
    a = ap.parse_args()
    if a.from_json:
        if not a.figure:
            ap.error("--from-json draws a figure: give --figure")
        out = json.loads(Path(a.from_json).read_text())
        labels = a.labels or [s["label"] for s in out["selections"]]
        if len(labels) != len(out["selections"]):
            ap.error("give one label per saved selection")
        models = [(label, ReferenceSeries.of_summary(s["host"]))
                  for label, s in zip(labels, out["selections"])]
        for path in a.figure:
            reference_figure(models, ReferenceSeries.of_summary(out["annotators"]["host"]), path, a.font_scale)
        return
    if not a.selections or not a.out:
        ap.error("give the selection files and --out")
    labels = a.labels or [Path(p).stem for p in a.selections]
    if len(labels) != len(a.selections):
        ap.error("give one label per selection file")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.check_masks:
        out = dict(masks=[check_masks(p, a.workers) for p in a.selections])
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
        return
    out, annotators, models = compare(a.selections, labels, a.frames, a.boot, a.seed, a.workers)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for s in out["selections"]:
        print(headline(s["label"], s))
    print(headline("second annotator", out["annotators"]))
    for path in a.figure or []:
        reference_figure([(label, reference.merge_reports(p.report for p in parts))
                          for label, parts in zip(labels, models)],
                         reference.merge_reports(p.report for p in annotators), path, a.font_scale)


if __name__ == "__main__":
    main()

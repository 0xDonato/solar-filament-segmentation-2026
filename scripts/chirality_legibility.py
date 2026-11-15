"""The report's chirality-legibility row (Table 3): how much the barbs of our masks say about a filament's Left/Right
label beyond the latitude rule, against a second annotator's polygon of the same filaments.

Rows are the annotated Left and Right filaments (frame, record, index) that a reader's mask matches at IoU > 0.5
(the host's float32 rule) and that another record's polygon of the same frame also matches. Each reader is read on
those rows: our out-of-fold selection, one row per filament, and the second annotator, one row per ordered pair of
records (a filament two other records drew counts twice). The label and the latitude come from the labelled polygon,
the barbs from the reader's mask (`solarfil.chirality`). For each reader and branch threshold (4, 8, 12 and 16 px;
8 was fixed before the test) the JSON gives the rows, groups, legible share, held-out accuracy of the latitude prior and
of the prior plus the bearing, the accuracy gain with 90% intervals over groups and frames, and the held-out log-loss
change with its interval over groups. Fits leave one annotation group out; bootstraps draw 4,000 group and 2,000 frame
resamples from seed 0. Each reader and threshold prints one summary line.

Only frames with two or more records can hold a row, so only those are read.

Usage: chirality_legibility.py v6_tau0.26.csv v6_tau0.30.csv --out chirality_legibility.json
       [--names "v6 tau 0.26,v6 tau 0.30"] [--thr 4,8,12,16] [--workers 3]
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

from solarfil import chirality, metric
from solarfil.cache import DEFAULT_CACHE, load_disks
from solarfil.data import Dataset, H, W
from solarfil.parallel import pool

_G: dict = {}


def read_masks(path: str) -> dict[str, list[dict]]:
    """COCO RLEs of a selection file's masks, by stem, in file order."""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rle = {"size": [H, W], "counts": row["segmentation_rle"].encode()}
            out[row["filament_id"].rsplit("_", 1)[0]].append(rle)
    return out


def matches(a: list[dict], b: list[dict]) -> list[tuple[int, int]]:
    """(i, j) of every pair of a's and b's masks with IoU > 0.5."""
    if not a or not b:
        return []
    iou, _ = metric.iou_dice(*metric.overlap_counts(a, b))
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(iou > 0.5))]


def read_frame(args) -> dict:
    """One frame's readings: every record polygon's label, latitude and barbs, every reader mask's barbs, the
    reader-to-record matches and the record-to-record matches (every ordered pair of records)."""
    stem, sels, min_length = args
    if "ds" not in _G:
        _G["ds"], _G["disks"] = Dataset(), load_disks(DEFAULT_CACHE)
    obs, disk = _G["ds"].observations[stem], _G["disks"][stem]
    recs = [metric.gt_rles_cached(r) for r in obs.records]
    gt = []
    for rec, record in zip(recs, obs.records):
        gt.append([])
        for rle, fil in zip(rec, record.filaments):
            m = mask_util.decode(rle).astype(bool)
            gt[-1].append(dict(cat=fil.category, lat=chirality.latitude(m, disk) if m.any() else None,
                               barbs=chirality.barbs(m, min_length)))
    readers = {}
    for name, rles in sels.items():
        masks = [mask_util.decode(r).astype(bool) for r in rles]
        readers[name] = dict(barbs=[chirality.barbs(m, min_length) if m.any() else None for m in masks],
                             pairs=[(k, i, j) for k, rec in enumerate(recs) for i, j in matches(rec, rles)])
    hpairs = [(k, i, j, b) for k in range(len(recs)) for j in range(len(recs)) if k != j
              for i, b in matches(recs[k], recs[j])]
    return dict(stem=stem, group=obs.group, gt=gt, readers=readers, hpairs=hpairs)


def row(f: dict, barb_list: list, thr: int, frame: dict) -> dict:
    """One test row: the labelled polygon f's label and latitude, the reader's barbs at threshold thr."""
    x = chirality.bearing_features(chirality.score(barb_list, thr), f["lat"])
    return dict(y=int(f["cat"] == chirality.RIGHT), g=frame["group"], f=frame["stem"], **x)


def same_rows(frames: list[dict], reader: str, thr: int) -> tuple[list[dict], list[dict]]:
    """The reader's rows and the second annotator's rows on the filaments both read."""
    ours, sec = {}, []
    for fr in frames:
        gt, rd = fr["gt"], fr["readers"][reader]
        for k, i, j in rd["pairs"]:
            f = gt[k][i]
            if f["cat"] in (chirality.LEFT, chirality.RIGHT) and rd["barbs"][j] is not None:
                ours[(fr["stem"], k, i)] = row(f, rd["barbs"][j], thr, fr)
        for k, i, j, b in fr["hpairs"]:
            f, d = gt[k][i], gt[j][b]
            if f["cat"] in (chirality.LEFT, chirality.RIGHT):
                sec.append(((fr["stem"], k, i), row(f, d["barbs"], thr, fr)))
    both = {key for key, _ in sec} & set(ours)
    return [ours[key] for key in sorted(both)], [x for key, x in sec if key in both]


def line(name: str, r: dict) -> str:
    """One result as a printed summary line."""
    def iv(v):
        return f"[{v[0]:+.4f}, {v[1]:+.4f}]"
    return (f"{name:22s} rows {r['rows']} groups {r['groups']} legible {r['legible']:.3f} | accuracy "
            f"{r['accuracy_prior']:.3f} -> {r['accuracy_full']:.3f} ({r['accuracy_gain']:+.4f} groups "
            f"{iv(r['accuracy_gain_groups90'])} frames {iv(r['accuracy_gain_frames90'])}) | log loss "
            f"{r['log_loss_change']:+.4f} groups {iv(r['log_loss_change_groups90'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selections", nargs="+", help="out-of-fold selection CSVs (filament_id, segmentation_rle)")
    ap.add_argument("--names", default="v6 tau 0.26,v6 tau 0.30", help="one reader name per selection, comma-separated")
    ap.add_argument("--thr", default=",".join(map(str, chirality.THRESHOLDS)), help="branch thresholds, px")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    names = args.names.split(",")
    assert len(names) == len(args.selections), "one name per selection file"
    thrs = [int(v) for v in args.thr.split(",")]
    sels = {n: read_masks(p) for n, p in zip(names, args.selections)}
    ds = Dataset()
    stems = [s for s, o in ds.observations.items() if o.n_records >= 2]
    jobs = [(s, {n: sels[n].get(s, []) for n in names}, min(thrs)) for s in stems]
    with pool(args.workers) as p:
        frames = p.map(read_frame, jobs, chunksize=4)
    results = {}
    for thr in thrs:
        print(f"== branch threshold {thr} px")
        results[str(thr)] = {}
        for name in names:
            ours, sec = same_rows(frames, name, thr)
            for label, rows in ((name, ours), (f"second annotator ({name} rows)", sec)):
                results[str(thr)][label] = r = chirality.beyond_latitude(rows)
                print(line(label if rows is ours else "second (both read)", r))
    out = dict(selections=dict(zip(names, args.selections)), frames_read=len(stems), convention_c=chirality.C,
               tangent_nodes=chirality.K, registered_threshold=chirality.BRANCH, thresholds=thrs, results=results)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

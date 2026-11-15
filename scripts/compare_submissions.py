"""How far two submission files of the same frames differ: each file's masks matched one to one at IoU > 0.5.

Usage: compare_submissions.py REFERENCE.csv OTHER.csv [OTHER.csv ...] [--out compare.json]
For each OTHER: masks in both files, matched pairs and their mean IoU, masks of each file without a match, and the
frames whose masks are byte-identical. The pairs are the host metric's own (`pycocotools` IoU, unique above 0.5).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as cocomask

from solarfil.rle import to_coco


def read(path: Path) -> dict[str, list[str]]:
    """The RLE strings of each frame of a submission file."""
    out = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    return out


def compare(ref: dict, other: dict) -> dict:
    """Matched pairs, unmatched masks and identical frames of `other` against `ref`."""
    n_pairs, ious, identical = 0, [], 0
    frames = sorted(set(ref) | set(other))
    for stem in frames:
        a, b = ref.get(stem, []), other.get(stem, [])
        identical += sorted(a) == sorted(b)
        if not a or not b:
            continue
        iou = cocomask.iou([to_coco(x) for x in b], [to_coco(x) for x in a], [0] * len(a))
        hits = np.argwhere(np.asarray(iou) > 0.5)
        n_pairs += len(hits)
        ious += [float(iou[i, j]) for i, j in hits]
    n_ref, n_other = sum(map(len, ref.values())), sum(map(len, other.values()))
    return dict(frames=len(frames), identical_frames=identical, masks_reference=n_ref, masks_other=n_other,
                matched=n_pairs, matched_mean_iou=float(np.mean(ious)) if ious else None,
                reference_unmatched=n_ref - n_pairs, other_unmatched=n_other - n_pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("others", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ref = read(args.reference)
    result = {str(p): compare(ref, read(p)) for p in args.others}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(dict(reference=str(args.reference), others=result), indent=1) + "\n")


if __name__ == "__main__":
    main()

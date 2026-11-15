"""Outlines of a submission's filaments on the H-alpha frames, for checking mask morphology by eye.

Usage: overlay.py --csv <submission.csv> --out <dir> [--stems a,b,...] [--n 6] [--seed 0] [--scale 0.5]
For training frames listed in --stems, the annotators' records are drawn too (white).
"""
import argparse
import csv
import random
from pathlib import Path

from solarfil import rle
from solarfil.data import Dataset, read_image
from solarfil.plots import overlay


def main():
    """Draw the chosen frames' instance outlines (and on training frames the annotators' records) into `--out`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    ds = Dataset()
    per: dict[str, list[str]] = {}
    for fid, counts in list(csv.reader(open(a.csv, newline="")))[1:]:
        per.setdefault(fid.split("_", 1)[0], []).append(counts)
    stems = a.stems.split(",") if a.stems else random.Random(a.seed).sample(sorted(per), a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    obs = ds.observations
    for stem in stems:
        masks = [rle.decode(c) for c in per.get(stem, [])]
        gt = [[f.mask() for f in r.filaments] for r in obs[stem].records] if stem in obs else None
        overlay(read_image(ds.image_path(stem)), masks, out / f"{stem}.png", gt=gt, scale=a.scale)
        print(stem, len(masks), "instances")


if __name__ == "__main__":
    main()

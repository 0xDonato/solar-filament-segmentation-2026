"""Candidate banks of the out-of-fold maps of a run (CPU).

Usage: build_bank.py --run e1_r34 [--splits] [--workers 4]   ->  <runs>/<run>/bank[_split][_test]/<stem>.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np

from solarfil import metric
from solarfil.bank import build_bank, save_bank
from solarfil.cache import DEFAULT_CACHE, load_disks, load_png
from solarfil.data import Dataset
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.preprocess import from_uint8

_STATE: dict = {}


def one(args):
    """Worker: build and save the candidate bank of one map (GT matches only when `with_gt`) and return its node
    count."""
    png, out_dir, cache, with_gt, splits, levels = args
    if "ds" not in _STATE:
        _STATE["ds"] = Dataset()
        _STATE["disks"] = load_disks(Path(cache))
    ds, disks = _STATE["ds"], _STATE["disks"]
    stem = Path(png).stem
    p, _ = read_map_png(png)
    flat = from_uint8(load_png(Path(cache) / "flat" / f"{stem}.png"))
    gt = metric.GTIndex(ds.observations[stem]) if with_gt else None
    bank = build_bank(stem, p, flat, disks[stem], gt, splits=splits, levels=levels)
    save_bank(Path(out_dir) / f"{stem}.npz", bank)
    return bank.tree.n


def main():
    """Parse the arguments, bank the run's out-of-fold (or test) maps, and print the frame and node counts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--test", action="store_true", help="bank the test maps (no GT)")
    ap.add_argument("--splits", action="store_true", help="insert watershed split rows")
    ap.add_argument("--levels", default=None, help="comma-separated thresholds (default: tree.LEVELS)")
    ap.add_argument("--tag", default="", help="suffix of the bank folder")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    levels = tuple(float(x) for x in a.levels.split(",")) if a.levels else None
    run = Path(a.runs) / a.run
    pngs = sorted(run.glob("test/*.png")) if a.test else sorted(run.glob("fold*/oof/*.png"))
    out = run / ("bank" + ("_split" if a.splits else "") + a.tag + ("_test" if a.test else ""))
    out.mkdir(exist_ok=True)
    t0 = time.time()
    with pool(a.workers) as workers:
        sizes = workers.map(one, [(str(p), str(out), a.cache, not a.test, a.splits, levels) for p in pngs], chunksize=2)
    print(f"{len(pngs)} frames, {sum(sizes)} nodes (median {int(np.median(sizes))}/frame), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Average the dense maps of several runs into a new run folder (an ensemble the other scripts can read).

Every out-of-fold map of the blend is the weighted mean of the runs' maps for that frame, each made by the run's
model for the frame's fold, so the blend stays out-of-fold; test maps are blended the same way. `--fusion logit`
averages the maps' logits instead of their probabilities (Vesuvius 2023 1st place, write-up 679238), the 8-bit maps
clipped half a level inside 0 and 1.

Usage: blend_maps.py --out ens_a --runs e3_r34_n e3_effv2s_n [--weights 1 1] [--fusion logit]
"""
import argparse
from pathlib import Path

import numpy as np

from solarfil.infer import read_map_png, write_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS

EPS = 0.5 / 255  # half an 8-bit level: the clip that keeps a stored 0 or 1 finite in logit space


def one(args):
    """Worker: write to `dst` the weighted mean (in probability or logit space, by `fusion`) of the maps in `srcs`, both
    channels."""
    srcs, weights, dst, fusion = args
    acc = None
    for src, w in zip(srcs, weights):
        p, s = read_map_png(src, native=False)
        if fusion == "logit":
            q = np.clip(p, EPS, 1 - EPS)
            p = np.log(q / (1 - q))
        m = np.stack([p, s]) * w
        acc = m if acc is None else acc + m
    acc = acc / sum(weights)
    if fusion == "logit":
        acc[0] = 1 / (1 + np.exp(-acc[0]))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    write_map_png(dst, acc)


def main():
    """Parse the arguments and blend the runs' out-of-fold maps, and their test maps when every run has the same number
    of them, into `<runs>/<out>`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fusion", choices=("mean", "logit"), default="mean")
    a = ap.parse_args()
    w = a.weights or [1.0] * len(a.runs)
    maps = [{p.relative_to(RUNS / r).as_posix(): p for p in (RUNS / r).glob("fold*/oof/*.png")} for r in a.runs]
    tests = [{p.name: p for p in (RUNS / r).glob("test/*.png")} for r in a.runs]
    stems = {Path(k).stem: k for k in maps[0]}
    jobs = []
    for stem, rel in stems.items():
        srcs = [next(str(p) for k, p in m.items() if Path(k).stem == stem) for m in maps]
        jobs.append((srcs, w, str(RUNS / a.out / rel), a.fusion))
    if all(len(t) == len(tests[0]) > 0 for t in tests):
        for name in tests[0]:
            jobs.append(([str(t[name]) for t in tests], w, str(RUNS / a.out / "test" / name), a.fusion))
    with pool(a.workers, spawn=True) as workers:
        workers.map(one, jobs, chunksize=4)
    print(f"blended {len(jobs)} maps from {a.runs} into {a.out}")


if __name__ == "__main__":
    main()

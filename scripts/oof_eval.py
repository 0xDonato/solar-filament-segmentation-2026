"""Out-of-fold host-exact PQ of a dense run under the threshold + connected-components decoder.

Usage: oof_eval.py --run e1_r34 [--scheme obs] [--workers 4] [--dy 0.27 --dx 0] [--frames 300]

Reads ``<runs>/<run>/fold*/oof/<stem>.png`` (model-scale maps), upsamples to 2048, sweeps the threshold and the
minimum area, and prints the pooled PQ grid, the best cell with its 90% bootstrap interval, and the judges'
distribution summaries for that cell. Writes ``<runs>/<run>/oof_eval.json``. ``--dx``/``--dy`` first resample each map
at (x + dx, y + dy), moving its content by (-dx, -dy) pixels (the label-offset check of `label_offset.py`), and
``--frames`` scores a fixed random subset; either writes ``oof_eval_dx<dx>_dy<dy>_n<frames>.json`` instead.

``--csv <file> --thresholds 0.4 --min-areas 300`` scores one cell of the grid and writes its instances in the
submission format to that file instead of any JSON, for paired comparisons with other selections
(`paired_bootstrap.py`); it prints the cell's pooled PQ.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from solarfil import metric, submission
from solarfil.data import Dataset
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.postprocess import labels_to_rles, threshold_labels

_DS: dict = {}


def oof_path(run_dir: Path, stem: str) -> Path:
    """The one out-of-fold map PNG of a stem in the run's fold folders."""
    paths = list(run_dir.glob(f"fold*/oof/{stem}.png"))
    assert len(paths) == 1, (stem, paths)
    return paths[0]


def score_one(args):
    """One frame's [TP, FP, FN, IoU sum] at every (threshold, minimum area) cell, and the last cell's RLEs
    when `with_rles` is set (the caller passes a single cell then)."""
    run_dir, stem, grid, holes, dx, dy, with_rles = args
    ds = _DS.setdefault("ds", Dataset())
    gt = metric.GTIndex(ds.observations[stem])
    p, _ = read_map_png(oof_path(Path(run_dir), stem))
    if dx or dy:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        p = cv2.warpAffine(p, m, p.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    out, rles = {}, []
    for t, a in grid:
        lab, n = threshold_labels(p, t, a, holes)
        out[(t, a)] = gt.score(lab, n).as_array()
        if with_rles:
            rles = labels_to_rles(lab)
    return stem, out, rles


def main():
    """Sweep the decoder grid out of fold, print the grid and best cell, and write the JSON or the one cell's CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--thresholds", default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    ap.add_argument("--min-areas", default="0,100,200,300,500")
    ap.add_argument("--no-holes", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=0, help="score a fixed random subset of this many frames")
    ap.add_argument("--csv", default=None, help="write the one grid cell's instances to this file, and no JSON")
    a = ap.parse_args()
    run_dir = Path(a.runs) / a.run
    stems = sorted(p.stem for p in run_dir.glob("fold*/oof/*.png"))
    if a.frames:
        stems = sorted(np.random.default_rng(0).choice(stems, a.frames, replace=False).tolist())
    thresholds = [float(x) for x in a.thresholds.split(",")]
    grid = list(itertools.product(thresholds, [int(x) for x in a.min_areas.split(",")]))
    if a.csv and len(grid) != 1:
        ap.error("--csv writes one cell: give one threshold and one minimum area")
    if a.csv and Path(a.csv).exists():
        ap.error(f"{a.csv} exists")
    with pool(a.workers) as workers:
        jobs = [(str(run_dir), s, grid, not a.no_holes, a.dx, a.dy, bool(a.csv)) for s in stems]
        done = list(workers.imap_unordered(score_one, jobs, chunksize=4))
    res = {stem: out for stem, out, _ in done}
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        n = submission.write({stem: rles for stem, _, rles in done}, a.csv)
        tot = sum(res[s][grid[0]] for s in stems)
        print(json.dumps(dict(run=a.run, n_frames=len(stems), threshold=grid[0][0], min_area=grid[0][1], csv=a.csv,
                              instances=n, pq=round(tot[3] / (tot[0] + 0.5 * tot[1] + 0.5 * tot[2]), 5),
                              tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]))))
        return
    tot = {g: sum(res[s][g] for s in stems) for g in grid}
    pq = {g: v[3] / (v[0] + 0.5 * v[1] + 0.5 * v[2]) for g, v in tot.items()}
    ts, as_ = sorted({g[0] for g in grid}), sorted({g[1] for g in grid})
    print(f"OOF PQ over {len(stems)} frames (rows: threshold, cols: min area)")
    print("      " + " ".join(f"{x:>7d}" for x in as_))
    for t in ts:
        print(f"{t:5.2f} " + " ".join(f"{pq[(t, x)]:7.4f}" for x in as_))
    best = max(pq, key=pq.get)
    rows = np.array([res[s][best] for s in stems])
    lo, hi = metric.bootstrap_pq(rows)
    v = tot[best]
    summary = dict(run=a.run, n_frames=len(stems), best_threshold=best[0], best_min_area=best[1],
                   pq=round(pq[best], 4), ci90=[round(lo, 4), round(hi, 4)], tp=int(v[0]), fp=int(v[1]),
                   fn=int(v[2]), sq=round(v[3] / v[0], 4), grid={f"{t}_{x}": round(q, 5) for (t, x), q in pq.items()})
    print(json.dumps({k: summary[k] for k in summary if k != "grid"}))
    name = f"oof_eval_dx{a.dx:g}_dy{a.dy:g}_n{a.frames}.json" if (a.dx or a.dy or a.frames) else "oof_eval.json"
    (run_dir / name).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

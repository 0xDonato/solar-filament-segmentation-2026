"""Out-of-fold selection from saved cross-fitted scores, written as submission rows for host-exact scoring.

For every frame of the scores file (`scores_<bank or extra>_<scheme>.npz`, written by `decide.py` or
`agree_study.py`): the component tree of the run's averaged map, the exact selection at tau, the level rule when
`--alpha` is given, and the masks' RLE rows. The rows can be scored with the host's metric (`tests/host_reference.py`).

Usage: oof_select.py --run ens_lean_2seeds --scores scores_inst_maskrcnn_split_random.npz --tau 0.26
       [--alpha 0.5] --out submissions/oof_inst_tau0.26.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

from solarfil import submission
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.postprocess import labels_to_rles
from solarfil.tree import build_tree, relevel, select


def one(args):
    """The stem and selected instance RLEs of one frame at tau, relevelled at alpha when given."""
    blend, u, tau, alpha = args
    p, _ = read_map_png(blend)
    tree = build_tree(p, splits=True)
    assert tree.n == len(u), (Path(blend).stem, tree.n, len(u))
    nodes = select(tree, u, tau) if tree.n else []
    if alpha and len(nodes):
        nodes = relevel(tree, p, nodes, alpha)
    return Path(blend).stem, labels_to_rles(tree.labels(nodes)) if len(nodes) else []


def main():
    """Select every frame of the scores file and write the instances in the submission format."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    run = RUNS / a.run
    z = np.load(run / a.scores)
    blends = {q.stem: q for q in run.glob("fold*/oof/*.png")}
    jobs = [(str(blends[s]), z[s], a.tau, a.alpha) for s in sorted(z.files)]
    with pool(a.workers) as workers:
        res = workers.map(one, jobs, chunksize=2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = submission.write(dict(res), a.out)
    print(json.dumps(dict(frames=len(res), instances=n, tau=a.tau, alpha=a.alpha, scores=a.scores)))


if __name__ == "__main__":
    main()

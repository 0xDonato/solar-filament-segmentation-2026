"""Test submission through the decision layer: tree of each test map, scorer fitted on all out-of-fold nodes,
exact DP at one tau, instance masks, validated CSV.

Usage: decide_test.py --run e1_r34 --tau 0.26 --out submissions/e2.csv [--bank bank] [--splits] [--test-dir test]
(`--test-dir test_full` uses the maps of the run's model trained on all frames.)
       decide_test.py --run ens_lean_2seeds --bank bank_split --extra inst_maskrcnn_split --fit-only
(fits the scorer with the extra columns of `<run>/<extra>/<stem>.npz`, e.g. the instance model's, and saves it as
`scorer_<bank>+<extra>.npz` for `export_weights.py --scorer`; the test frames then go through `predict.py`.
`--drop-set K` leaves out the K-th set's six columns, saved as `scorer_<bank>+<extra>-setK.npz`.)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from solarfil import submission
from solarfil.bank import describe, load_banks
from solarfil.cache import DEFAULT_CACHE, load_disks, load_png
from solarfil.data import Dataset
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.postprocess import labels_to_rles
from solarfil.preprocess import from_uint8
from solarfil.scorer import FrozenHurdle, Hurdle, targets
from solarfil.tree import build_tree, select

_STATE: dict = {}


def one(args):
    """Worker: the instance RLEs of one test map, selected at `tau` by the saved scorer, and their count."""
    png, scorer_path, tau, splits, cache = args
    if "scorer" not in _STATE:
        _STATE["scorer"] = FrozenHurdle.load(scorer_path)
        _STATE["disks"] = load_disks(Path(cache))
    stem = Path(png).stem
    p, _ = read_map_png(png)
    flat = from_uint8(load_png(Path(cache) / "flat" / f"{stem}.png"))
    tree = build_tree(p, splits=splits)
    if tree.n == 0:
        return stem, [], 0
    X = describe(tree, p, flat, _STATE["disks"][stem])
    u = _STATE["scorer"].predict(X)[0]
    sel = select(tree, u, tau)
    lab = tree.labels(sel)
    return stem, labels_to_rles(lab), len(sel)


def main():
    """Parse the arguments, fit and save the scorer on every out-of-fold node, and unless `--fit-only` decide the
    test frames and write and validate the submission CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--bank", default="bank")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default=None,
                    help="folder of extra columns per frame (array A), e.g. inst_maskrcnn_split")
    ap.add_argument("--fit-only", action="store_true", help="fit and save the scorer, decide no test frames")
    ap.add_argument("--drop-set", type=int, default=None,
                    help="leave out the six columns of the K-th detector set of --extra (the compact release)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-dir", default="test")
    a = ap.parse_args()
    run = Path(a.runs) / a.run
    banks = load_banks(run / a.bank)
    if a.extra:
        for b in banks:
            A = np.load(run / a.extra / f"{b.stem}.npz")["A"]
            assert len(A) == b.tree.n, (b.stem, len(A), b.tree.n)
            if a.drop_set is not None:
                A = np.delete(A, np.s_[6 * a.drop_set:6 * a.drop_set + 6], 1)
            b.X = np.concatenate([b.X, A], 1)
    X = np.concatenate([b.X for b in banks])
    h, m, _ = map(np.concatenate, zip(*[targets(b.T) for b in banks]))
    R = np.concatenate([np.full(b.tree.n, b.T.shape[1], float) for b in banks])
    # the fitted scorer as arrays, as the weights folder holds it
    drop = "" if a.drop_set is None else f"-set{a.drop_set}"
    scorer_path = run / (f"scorer_{a.bank}+{a.extra}{drop}.npz" if a.extra else f"scorer_{a.bank}.npz")
    FrozenHurdle.from_fitted(Hurdle().fit(X, h, m, R)).save(scorer_path)
    if a.fit_only:
        print(json.dumps(dict(scorer=str(scorer_path), scorer_nodes=len(X), columns=X.shape[1])))
        return
    assert a.extra is None, "test decisions with extra columns go through predict.py"
    assert a.tau is not None and a.out, "--tau and --out are needed for the test decisions"
    stems = Dataset().test_stems
    jobs = [(str(run / a.test_dir / f"{s}.png"), str(scorer_path), a.tau, a.splits, str(DEFAULT_CACHE)) for s in stems]
    # spawn, not fork: the parent has already run OpenMP (the scorer fit), and forked OpenMP children can deadlock
    with pool(a.workers, spawn=True) as workers:
        res = workers.map(one, jobs)
    inst = {stem: rles for stem, rles, _ in res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = submission.write(inst, a.out)
    check = submission.validate(a.out, stems)
    print(json.dumps(dict(**check, instances_per_image=round(n / len(stems), 2), tau=a.tau,
                          scorer_nodes=len(X))))


if __name__ == "__main__":
    main()

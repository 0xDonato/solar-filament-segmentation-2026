"""The decision layer on the out-of-fold bank of a run: cross-fitted node scorer, exact DP, tau sweep.

Usage: decide.py --run e1_r34 [--scheme obs] [--model hurdle|direct]

Every frame's nodes are scored by a scorer that never saw that frame (folds of `configs/folds.json`), the DP
selects the best antichain for each tau, and the pooled host-exact PQ is computed from the bank's matched IoUs.
Writes `<run>/decide_<scheme>_<model>.json`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from solarfil import metric
from solarfil.bank import FEATURES, load_banks
from solarfil.folds import load_folds
from solarfil.paths import RUNS
from solarfil.scorer import PARAMS, Hurdle, predict_draws, targets
from solarfil.tree import select

TAUS = np.round(np.arange(0.12, 0.42, 0.02), 2)


def crossfit(banks, folds, model: str, params: dict | None = None) -> list[np.ndarray]:
    """Out-of-fold utilities of every bank's nodes, one array per bank: for each fold of `folds`, a hurdle (or a
    direct regressor of u, by `model`) fitted on the other folds' nodes, weighted by their frames' record counts."""
    fold_of = np.array([folds[b.stem] for b in banks])
    X = np.concatenate([b.X for b in banks])
    T_rows = [b.T for b in banks]
    h = np.concatenate([targets(t)[0] for t in T_rows])
    m = np.concatenate([targets(t)[1] for t in T_rows])
    u = np.concatenate([targets(t)[2] for t in T_rows])
    R = np.concatenate([np.full(len(t), t.shape[1], float) for t in T_rows])
    node_fold = np.concatenate([np.full(b.tree.n, fold_of[i]) for i, b in enumerate(banks)])
    pred = np.zeros(len(X))
    for k in sorted(set(fold_of)):
        tr, va = node_fold != k, node_fold == k
        if model == "hurdle":
            pred[va] = Hurdle(params).fit(X[tr], h[tr], m[tr], R[tr]).predict(X[va])[0]
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(**dict(PARAMS, **(params or {}))).fit(X[tr], u[tr], sample_weight=R[tr])
            pred[va] = reg.predict(X[va])
    out, i = [], 0
    for b in banks:
        out.append(pred[i: i + b.tree.n])
        i += b.tree.n
    return out


def crossfit_draws(banks, draws: list[list[np.ndarray]], folds, params: dict | None = None) -> list[np.ndarray]:
    """`crossfit` with draw augmentation: `draws[c][i]` is bank i's design under combination c of the detector draws.
    Each fold's hurdle trains on every combination at once (`Hurdle.fit_draws`) and its utility is the mean over the
    combinations (`predict_draws`)."""
    fold_of = np.array([folds[b.stem] for b in banks])
    T_rows = [b.T for b in banks]
    h = np.concatenate([targets(t)[0] for t in T_rows])
    m = np.concatenate([targets(t)[1] for t in T_rows])
    R = np.concatenate([np.full(len(t), t.shape[1], float) for t in T_rows])
    Xs = [np.concatenate(Xc) for Xc in draws]
    node_fold = np.concatenate([np.full(b.tree.n, fold_of[i]) for i, b in enumerate(banks)])
    pred = np.zeros(len(h))
    for k in sorted(set(fold_of)):
        tr, va = node_fold != k, node_fold == k
        hurdle = Hurdle(params).fit_draws([X[tr] for X in Xs], h[tr], m[tr], R[tr])
        pred[va] = predict_draws(hurdle, [X[va] for X in Xs])
    return np.split(pred, np.cumsum([b.tree.n for b in banks])[:-1])


def sweep(banks, scores, taus=TAUS):
    """{tau: (pooled PQ, (frames, 4) totals)} of the exact selection under `scores` at each tau."""
    res = {}
    for tau in taus:
        rows = np.array([b.totals(select(b.tree, s, tau)) for b, s in zip(banks, scores)])
        tot = rows.sum(0)
        res[float(tau)] = (tot[3] / (tot[0] + 0.5 * tot[1] + 0.5 * tot[2]), rows)
    return res


def main():
    """Parse the arguments, cross-fit the scorer, sweep tau, and print and write
    `<run>/decide_<bank>_<scheme>_<model>.json` (the first partition's scores to
    `scores_<bank>_<scheme>_<model>.npz`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--scheme", default="obs")
    ap.add_argument("--model", default="hurdle", choices=["hurdle", "direct"])
    ap.add_argument("--bank", default="bank", help="bank folder inside the run (bank, bank_split)")
    ap.add_argument("--repeats", type=int, default=1, help="random scorer partitions (scheme random only)")
    a = ap.parse_args()
    run = Path(a.runs) / a.run
    banks = load_banks(run / a.bank)
    curves = []
    for rep in range(a.repeats if a.scheme == "random" else 1):
        if a.scheme == "random":  # observation-level random folds for the scorer
            rng = np.random.default_rng(rep)
            folds = {b.stem: int(k) for b, k in zip(banks, rng.permutation(len(banks)) % 5)}
        else:
            folds = load_folds(a.scheme)
        scores = crossfit(banks, folds, a.model)
        if rep == 0:
            np.savez_compressed(run / f"scores_{a.bank}_{a.scheme}_{a.model}.npz",
                                **{b.stem: s for b, s in zip(banks, scores)})
            res = sweep(banks, scores)
            curves.append({t: v[0] for t, v in res.items()})
        else:
            curves.append({t: v[0] for t, v in sweep(banks, scores).items()})
    mean_curve = {t: float(np.mean([c[t] for c in curves])) for t in curves[0]}
    sd_curve = {t: float(np.std([c[t] for c in curves])) for t in curves[0]}
    best = max(res, key=lambda t: res[t][0])
    lo, hi = metric.bootstrap_pq(res[best][1])
    tot = res[best][1].sum(0)
    # oracle: the true u, as an upper bound for this candidate set
    oracle = sweep(banks, [b.T.mean(1) for b in banks], taus=[0.20, 0.24, 0.28])
    summary = dict(run=a.run, bank=a.bank, scheme=a.scheme, model=a.model, frames=len(banks),
                   nodes=int(sum(b.tree.n for b in banks)), features=len(FEATURES),
                   pq_by_tau={t: round(v[0], 4) for t, v in res.items()}, best_tau=best, pq=round(res[best][0], 4),
                   repeats=len(curves), mean_pq_by_tau={t: round(v, 4) for t, v in mean_curve.items()},
                   sd_pq_by_tau={t: round(v, 4) for t, v in sd_curve.items()},
                   mean_best=round(max(mean_curve.values()), 4),
                   ci90=[round(lo, 4), round(hi, 4)], tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]),
                   sq=round(tot[3] / tot[0], 4), oracle_u={t: round(v[0], 4) for t, v in oracle.items()},
                   gt_coverage_iou_gt_05=round(float(np.mean(np.concatenate([b.G for b in banks]) > 0.5)), 4))
    print(json.dumps(summary, indent=1))
    (run / f"decide_{a.bank}_{a.scheme}_{a.model}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

"""tau by minimax regret over the scorer cross-fitting schemes (random partitions and leave-annotation-group-out).

For each scheme s, regret_s(tau) = max PQ_s - PQ_s(tau); the chosen tau minimises the largest regret, so it is
safe whether the test annotators behave like seen or like unseen annotation groups.

Usage: choose_tau.py --run e3_effv2s_n [--bank bank_split]
"""
import argparse
import json
from pathlib import Path

from solarfil.paths import RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--runs", default=str(RUNS))
ap.add_argument("--bank", default="bank_split")
a = ap.parse_args()
run = Path(a.runs) / a.run
curves = {}
for scheme in ("random", "group"):
    d = json.loads((run / f"decide_{a.bank}_{scheme}_hurdle.json").read_text())
    curves[scheme] = {float(t): v for t, v in (d.get("mean_pq_by_tau") or d["pq_by_tau"]).items()}
taus = sorted(set.intersection(*(set(c) for c in curves.values())))
regret = {t: max(max(c.values()) - c[t] for c in curves.values()) for t in taus}
best = min(regret, key=regret.get)
print(json.dumps(dict(run=a.run, tau=best, max_regret=round(regret[best], 4),
                      pq_at_tau={s: round(c[best], 4) for s, c in curves.items()},
                      best_by_scheme={s: max(c, key=c.get) for s, c in curves.items()})))

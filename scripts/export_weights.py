"""Collect a run's fold models and its fitted scorer (`scorer_<bank>.npz`, written by `decide_test.py`) into a
weights folder for `solarfil.pipeline` (and for the release assets). With `--members`, the models of several runs
(seeds of one recipe) and the scorer of `--run`, the run of their blended maps.

Usage: export_weights.py --run e3_effv2s_n --tau 0.26 --out weights/effv2s [--models full] [--members r_s0 r_s1]
       [--alpha 0.5]   (each selected filament redrawn at alpha times its own peak, `tree.relevel`)
       [--instances inst_maskrcnn/fold0 ... inst_maskrcnn/fold4 --scorer scorer_bank_split+inst_maskrcnn_split.npz]
       (the instance model's fold models, averaged at inference, and the scorer fitted with their out-of-fold
       columns by decide_test.py)
       [--instances inst_maskrcnn/fold0,inst_maskrcnn_grp/fold0 ... inst_maskrcnn/fold4,inst_maskrcnn_grp/fold4
        --scorer scorer_bank_split+inst_maskrcnn+inst_maskrcnn_grp_split.npz]
       (two instance models whose columns the scorer read side by side: one fold model of each per set)
       [--det-scorer det_scorer_det_split_gated.npz --det-config '{"margin": 0.15, "min_score": 0.3, "overlap": 0.2,
        "gated": true, "candidate_min_score": 0.05}']   (detections as candidates too; fit_det_scorer.py)
"""
import argparse
import json
from pathlib import Path

from solarfil.paths import RUNS
from solarfil.pipeline import export_weights

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--runs", default=str(RUNS))
ap.add_argument("--bank", default="bank_split")
ap.add_argument("--tau", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--models", choices=["folds", "full"], default="folds")
ap.add_argument("--members", nargs="*", default=None, help="runs whose models are averaged (default: --run)")
ap.add_argument("--member-weights", nargs="*", type=float, default=None,
                help="each member's share of the average, as blend_maps.py --weights (default equal)")
ap.add_argument("--alpha", type=float, default=None)
ap.add_argument("--instances", nargs="+", default=None,
                help="instance model folders (one, or the fold models; a,b: one fold model of each of two)")
ap.add_argument("--scorer", default=None, help="scorer file in --run (default: scorer_<bank>.npz)")
ap.add_argument("--det-scorer", default=None, help="detection scorer file in --run (fit_det_scorer.py)")
ap.add_argument("--det-config", default=None, help="JSON settings of the detection candidates")
a = ap.parse_args()
run = Path(a.runs) / a.run
members = [Path(a.runs) / r for r in a.members] if a.members else None
sets = [[Path(a.runs) / d for d in s.split(",")] for s in a.instances or []]
sets = [s if len(s) > 1 else s[0] for s in sets]
man = export_weights(run, run / (a.scorer or f"scorer_{a.bank}.npz"), Path(a.out), a.tau, models=a.models,
                     members=members, member_weights=a.member_weights, alpha=a.alpha,
                     instances=(sets if len(sets) > 1 or isinstance(sets[0], list) else sets[0]) if sets else None,
                     detections=dict(json.loads(a.det_config), scorer=run / a.det_scorer) if a.det_scorer else None)
print(json.dumps(man, indent=1))

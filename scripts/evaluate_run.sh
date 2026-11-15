#!/bin/bash
# Out-of-fold evaluation of a finished dense run (all five folds present): the threshold decoder sweep, the
# candidate bank with split rows, and the decision layer over three random scorer partitions plus the
# group-out scheme. Results land in the run folder as JSON. Usage: scripts/evaluate_run.sh <run> [workers]
set -eu
RUN=${1:?run name}
W=${2:-16}
export OMP_NUM_THREADS=$W
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY="uv run python"
$PY scripts/oof_eval.py --run "$RUN" --thresholds 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55 --min-areas 0,200,400 --workers "$W" 2>&1 | grep -v Warn | tail -1
$PY scripts/build_bank.py --run "$RUN" --splits --workers "$W" 2>&1 | grep -v Warn | tail -1
$PY scripts/decide.py --run "$RUN" --bank bank_split --scheme random --repeats 3 2>&1 | grep -v Warn \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in ('run','bank','scheme','pq','best_tau','mean_best','ci90','tp','fp','fn','sq','gt_coverage_iou_gt_05')}))"
$PY scripts/decide.py --run "$RUN" --bank bank_split --scheme group 2>&1 | grep -v Warn \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in ('run','bank','scheme','pq','best_tau','ci90')}))"

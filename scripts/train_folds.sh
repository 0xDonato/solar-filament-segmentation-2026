#!/bin/bash
# Train the listed folds of one recipe in sequence, skipping a fold whose model.pt exists (so a requeued job
# resumes where it stopped; an interrupted fold resumes from its checkpoint). Usage: train_folds.sh <config> <fold>...
set -eu
cfg=$1; shift
name=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['name'])" "$cfg")
for f in "$@"; do
  d=$([ "$f" = all ] && echo full || echo "fold$f")
  if [ -f "${FILAMENT_RUNS:?}/$name/$d/model.pt" ]; then echo "skip $name/$d: done"; continue; fi
  uv run --frozen python -m solarfil.train --config "$cfg" --fold "$f"
done

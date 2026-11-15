"""Where the competition data, the preprocessed cache and the run folders live: the environment variables
`FILAMENT_DATA` (the competition's `MAGFiLO_1.0_Kaggle_2026` folder), `FILAMENT_CACHE` and `FILAMENT_RUNS`, else
`data/`, `cache/` and `runs/` in the repository folder."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FILAMENT_DATA", REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))
CACHE = Path(os.environ.get("FILAMENT_CACHE", REPO / "cache" / "v1"))
RUNS = Path(os.environ.get("FILAMENT_RUNS", REPO / "runs"))

"""Cross-validation folds over observations (`configs/folds.json`).

* ``obs``: 5 folds over observations, stratified by station x number of records. The test frames' median gap to
  the nearest training frame is 36.7 h; an out-of-fold frame's to the frames of the other folds is 45.4 h, so these
  folds borrow no more temporal neighbourhood than the test has.
* ``group``: 5 folds over the 4-digit annotation groups, balanced by record count. Five of MAGFiLO's 24 groups
  have no training records, so test annotators are probably unseen; this scheme measures that shift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
SEED = 42
FOLDS_FILE = Path(__file__).resolve().parents[2] / "configs" / "folds.json"


def make_folds(ds, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, dict[str, int]]:
    """Both fold schemes as {scheme: {stem: fold}}: ``obs`` stratified by station and record count, ``group`` assigning
    the annotation groups greedily, largest record count first, to the least-loaded fold."""
    stems = ds.train_stems
    strata = [f"{ds.observations[s].station}{ds.observations[s].n_records}" for s in stems]
    obs = {}
    for k, (_, va) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(stems, strata)):
        for i in va:
            obs[stems[i]] = k
    recs: dict[str, int] = {}
    for s in stems:
        g = ds.observations[s].group
        recs[g] = recs.get(g, 0) + ds.observations[s].n_records
    load = np.zeros(n_folds, int)
    gfold = {}
    for g in sorted(recs, key=lambda g: (-recs[g], g)):
        k = int(np.argmin(load))
        gfold[g] = k
        load[k] += recs[g]
    group = {s: gfold[ds.observations[s].group] for s in stems}
    return {"obs": dict(sorted(obs.items())), "group": dict(sorted(group.items()))}


def load_folds(scheme: str = "obs", path: Path = FOLDS_FILE) -> dict[str, int]:
    """{stem: fold} of one scheme from `configs/folds.json`."""
    return json.loads(Path(path).read_text())[scheme]


def group_partition(rep: int, ds=None, n_folds: int = N_FOLDS, tries: int = 100) -> dict[str, int]:
    """Partition `rep` of the frames into folds that hold out whole annotation groups, for the scorer's group-out
    cross-fitting. Partition 0 is the fixed ``group`` folds. Partition r > 0 deals the groups, in shuffled orders
    drawn with seed r, each to the fold with the fewest records so far, and keeps the order whose folds' record counts
    spread least of `tries`, so its folds stay about as balanced as the fixed ones."""
    if rep == 0:
        return load_folds("group")
    if ds is None:
        from solarfil.data import Dataset
        ds = Dataset()
    recs: dict[str, int] = {}
    for o in ds.observations.values():
        recs[o.group] = recs.get(o.group, 0) + o.n_records
    groups = sorted(recs)
    rng = np.random.default_rng(rep)
    best, best_spread = {}, None
    for _ in range(tries):
        load = np.zeros(n_folds, int)
        gfold = {}
        for i in rng.permutation(len(groups)):
            k = int(np.argmin(load))
            gfold[groups[i]] = k
            load[k] += recs[groups[i]]
        if best_spread is None or load.max() - load.min() < best_spread:
            best, best_spread = gfold, load.max() - load.min()
    return {s: best[o.group] for s, o in ds.observations.items()}

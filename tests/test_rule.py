"""The release rule's partitions and its verdict: group partitions keep every annotation group whole and stay
balanced, random partitions are the ones the studies have always drawn, and a clear gain passes where none fails."""
from types import SimpleNamespace

import numpy as np

from solarfil.folds import group_partition, load_folds
from solarfil.rule import TAUS, partitions, release_rule


def fake_dataset(n_groups=19, seed=0):
    rng = np.random.default_rng(seed)
    obs = {}
    for g in range(n_groups):
        for i in range(int(rng.integers(5, 60))):
            obs[f"f{g:02d}_{i:03d}"] = SimpleNamespace(group=f"{1000 + g}", n_records=int(rng.integers(1, 4)))
    return SimpleNamespace(observations=obs)


def test_group_partitions_keep_groups_whole_balanced_and_distinct():
    ds = fake_dataset()
    recs = {}
    for o in ds.observations.values():
        recs[o.group] = recs.get(o.group, 0) + o.n_records
    seen = []
    for rep in (1, 2, 3, 4):
        folds = group_partition(rep, ds)
        assert folds == group_partition(rep, ds)  # seeded
        fold_of_group = {}
        for s, k in folds.items():
            assert fold_of_group.setdefault(ds.observations[s].group, k) == k  # a group sits in one fold
        load = np.zeros(5, int)
        for g, k in fold_of_group.items():
            load[k] += recs[g]
        assert load.min() > 0 and load.max() - load.min() <= max(recs.values())
        seen.append(tuple(sorted(fold_of_group.items())))
    assert len(set(seen)) == 4


def test_partition_zero_is_the_fixed_group_folds_and_random_ones_are_unchanged():
    stems = sorted(load_folds("group"))
    parts = list(partitions(stems, n_random=2, n_group=1))
    assert [(s, r) for s, r, _ in parts] == [("random", 0), ("random", 1), ("group", 0)]
    assert parts[2][2] == load_folds("group")
    for rep in (0, 1):  # the draw agree_study.evaluate has used since 10-15
        want = {s: int(k) for s, k in zip(stems, np.random.default_rng(rep).permutation(len(stems)) % 5)}
        assert parts[rep][2] == want


def test_a_clear_gain_passes_and_no_gain_fails():
    rng = np.random.default_rng(0)
    n = 200
    base = {t: np.c_[rng.integers(3, 9, n), rng.integers(0, 4, n), rng.integers(0, 4, n), np.zeros(n)].astype(float)
            for t in TAUS}
    for t in TAUS:
        base[t][:, 3] = 0.7 * base[t][:, 0]
    better = {}
    for t in TAUS:  # every fourth frame turns one miss into a match at IoU 0.7
        b = base[t].copy()
        b[::4] += [1, 0, -1, 0.7]
        better[t] = b
    keys = [("random", r) for r in range(3)] + [("group", r) for r in range(2)]
    gain = release_rule({(s, r, arm): (better if arm == "agree" else base) for s, r in keys
                         for arm in ("base", "agree")}, n)
    group = gain["group_0.26"]["per_partition"]
    assert gain["release_rule"]["passes"] and group == [group[0]] * 2  # identical partitions, identical gains
    none = release_rule({(s, r, arm): base for s, r in keys for arm in ("base", "agree")}, n)
    assert not none["release_rule"]["passes"] and none["release_rule"]["schemes_mean"]["0.26"]["gain"] == 0

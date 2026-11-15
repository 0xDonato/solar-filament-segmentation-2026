"""Component tree, exact selection and bank scoring against direct thresholding and the host-exact metric."""
import itertools

import cv2
import numpy as np
import pytest

from solarfil import metric
from solarfil.bank import FEATURES, build_bank, load_banks
from solarfil.data import DEFAULT_ROOT, Dataset
from solarfil.postprocess import threshold_labels
from solarfil.preprocess import Disk
from solarfil.tree import LEVELS, MIN_AREA, build_tree, relevel, select

DISK = Disk(1023.0, 1024.0, 904.0)


def _smooth_map(seed, n=2048):
    rng = np.random.default_rng(seed)
    z = rng.random((n // 16, n // 16)).astype(np.float32)
    z = cv2.resize(z, (n, n), interpolation=cv2.INTER_CUBIC)
    z = cv2.GaussianBlur(z, (0, 0), 12)
    z = (z - z.min()) / (z.max() - z.min())
    return np.clip((z - 0.45) * 2.5, 0, 1).astype(np.float32)


def _brute_force(tree, score, tau):
    """Best antichain by enumeration (small forests only)."""
    n = tree.n
    anc = []
    for v in range(n):
        a, u = set(), tree.parent[v]
        while u >= 0:
            a.add(u)
            u = tree.parent[u]
        anc.append(a)
    best = 0.0
    for r in range(n + 1):
        for comb in itertools.combinations(range(n), r):
            if any(u in anc[v] for v in comb for u in comb):
                continue
            best = max(best, sum(score[v] - tau for v in comb))
    return best


def test_levels_match_thresholding():
    p = _smooth_map(0)
    tree = build_tree(p)
    for k in (0, 5, 9):
        lab, n = threshold_labels(p, LEVELS[k], MIN_AREA, True)
        alive = np.unique(tree.node_of[k][tree.node_of[k] >= 0])
        assert len(alive) == n
        assert sorted(tree.area[alive].tolist()) == sorted(np.bincount(lab.ravel())[1:].tolist())


def test_select_is_optimal_on_small_forests():
    rng = np.random.default_rng(1)
    tested = 0
    for seed in range(200):
        p = _smooth_map(seed, 64)
        tree = build_tree(p, levels=(0.1, 0.3, 0.5, 0.7))
        if not 3 <= tree.n <= 13:
            continue
        for tau in (0.2, 0.4, 0.6):
            score = rng.random(tree.n)
            sel = select(tree, score, tau)
            got = sum(score[v] - tau for v in sel)
            assert got == pytest.approx(_brute_force(tree, score, tau))
        tested += 1
        if tested == 8:
            break
    assert tested >= 3


def test_bank_totals_equal_host_metric():
    if not (DEFAULT_ROOT / "train").exists():
        pytest.skip("data not present")
    ds = Dataset()
    obs = next(o for o in ds.observations.values() if o.n_records == 3)
    gt = metric.GTIndex(obs)
    # a probability map built from the records themselves, so that many nodes match
    p = np.zeros((2048, 2048), np.float32)
    for rec in obs.records:
        for f in rec.filaments:
            p += f.mask() / 3.0
    p = cv2.GaussianBlur(p, (0, 0), 2.0)
    bank = build_bank(obs.stem, p, np.ones_like(p), DISK, gt)
    assert bank.X.shape == (bank.tree.n, len(__import__("solarfil.bank", fromlist=["FEATURES"]).FEATURES))
    assert np.isfinite(bank.X).all()
    for k in (1, 4, 8):
        alive = np.unique(bank.tree.node_of[k][bank.tree.node_of[k] >= 0])
        lab = bank.tree.labels(alive)
        want = gt.score(lab).as_array()
        assert np.allclose(bank.totals(alive), want, atol=1e-6), (k, bank.totals(alive), want)


def test_split_rows_nest_and_score():
    """With split rows every node lies inside its parent, the DP stays optimal, and level rows still score exactly."""
    p = _smooth_map(3, 1024)
    tree = build_tree(p, splits=True)
    assert 1 in tree.kinds
    for v in range(tree.n):
        pa = tree.parent[v]
        if pa >= 0:
            assert np.isin(tree.node_pixels(v), tree.node_pixels(pa)).all()
    rng = np.random.default_rng(2)
    tested = 0
    for seed in range(300):
        q = _smooth_map(seed, 64)
        tr = build_tree(q, levels=(0.1, 0.3, 0.5, 0.7), splits=True)
        if not (1 in tr.kinds and 3 <= tr.n <= 13):
            continue
        score = rng.random(tr.n)
        sel = select(tr, score, 0.3)
        assert sum(score[v] - 0.3 for v in sel) == pytest.approx(_brute_force(tr, score, 0.3))
        tested += 1
        if tested == 5:
            break
    assert tested >= 2


def test_old_bank_files_load_with_the_current_descriptors(tmp_path):
    """Banks written before 2026-10-12 hold four spine-map descriptors (columns 15-18) and possibly six station
    columns; loading keeps exactly the current descriptors, in order."""
    old_names = FEATURES[:15] + ["s_mean", "s_max", "s_area_frac", "s_len_ratio"] + FEATURES[15:]
    for extra in (0, 6):
        X = np.tile(np.arange(len(old_names) + extra, dtype=np.float32), (3, 1))
        np.savez(tmp_path / f"b{extra}.npz", X=X, level=np.zeros(3, int), last_level=np.zeros(3, int),
                 parent=np.full(3, -1), area=np.ones(3, int))
    for bank in load_banks(tmp_path):
        assert bank.X.shape == (3, len(FEATURES))
        assert [old_names[int(j)] for j in bank.X[0]] == FEATURES


def test_relevel_cuts_each_filament_at_its_own_share_of_its_peak():
    """Two blurred filaments, one at the plateau all annotators give and one at a third of it: alpha 0.5 redraws each
    near half its own peak, keeps them apart, and is idempotent."""
    p = np.zeros((256, 256), np.float32)
    p[60:70, 40:200] = 1.0
    p[160:170, 40:200] = 1.0
    p = cv2.GaussianBlur(p, (0, 0), 3)
    p[:128] *= 0.95 / p[:128].max()
    p[128:] *= 0.35 / p[128:].max()
    tree = build_tree(p)
    roots = [v for v in range(tree.n) if tree.parent[v] < 0]
    assert len(roots) == 2
    out = relevel(tree, p, roots, 0.5)
    levels = sorted(tree.levels[tree.level[w]] for w in out)
    assert levels[0] in (0.15, 0.20) and levels[1] in (0.45, 0.50)
    assert tree.labels(out).max() == 2 and len(np.unique(tree.labels(out))) == 3
    assert relevel(tree, p, out, 0.5) == out
    assert relevel(tree, p, [], 0.5) == []

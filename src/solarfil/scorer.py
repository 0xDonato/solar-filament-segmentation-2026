"""Node scorer: the expected best strict-match IoU ``u = h * m`` of a candidate against a random annotator record.

``h`` is the share of the frame's records that contain a filament matching the node (IoU > 0.5) and ``m`` the
mean best strict-match IoU over those records. A hurdle model fits ``h`` with a gradient-boosted classifier on soft labels
(each node enters as a positive with weight ``h R`` and a negative with weight ``(1 - h) R``) and ``m`` with a
gradient-boosted regressor on the nodes with ``h > 0`` (weight ``h R``). The weight ``R`` (records of the frame)
makes each node count as it does in the pooled metric.

For inference, `FrozenHurdle` holds a fitted model as plain arrays (every tree's nodes), saved and loaded with numpy
alone, so a weights folder holds no pickle; its predictions equal the fitted model's bit for bit.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=False, random_state=0)


def targets(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h, m, u of nodes from their (n, R) matched-IoU rows."""
    h = (T > 0).mean(1)
    u = T.mean(1)
    m = np.where(h > 0, u / np.maximum(h, 1e-9), 0.0)
    return h, m, u


class Hurdle:
    """The hurdle scorer of the module docstring: a classifier of h and a regressor of m, whose product is u."""
    def __init__(self, params: dict | None = None):
        self.params = dict(PARAMS, **(params or {}))

    def fit(self, X: np.ndarray, h: np.ndarray, m: np.ndarray, R: np.ndarray) -> Hurdle:
        """Fit the classifier on `h` as soft labels and the regressor on `m` over the nodes with h > 0, each weighted by
        the frame's record count `R` as described above. Returns self."""
        n = len(X)
        Xd = np.concatenate([X, X])
        yd = np.concatenate([np.ones(n), np.zeros(n)])
        wd = np.concatenate([h * R, (1 - h) * R])
        keep = wd > 0
        self.clf = HistGradientBoostingClassifier(**self.params).fit(Xd[keep], yd[keep], sample_weight=wd[keep])
        pos = h > 0
        self.reg = HistGradientBoostingRegressor(**self.params).fit(X[pos], m[pos], sample_weight=(h * R)[pos])
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """u = h m, h and m of each row; m is clipped to [0.5, 1]."""
        h = self.clf.predict_proba(X)[:, 1]
        m = np.clip(self.reg.predict(X), 0.5, 1.0)
        return h * m, h, m

    def fit_draws(self, Xs: list[np.ndarray], h: np.ndarray, m: np.ndarray, R: np.ndarray) -> Hurdle:
        """Draw augmentation (UM MCTS 1st place, write-up 549801): `Xs` holds the same nodes' rows under each
        combination of detector draws, one draw per slot. Every node enters once per combination with that share of
        its weight, and `min_samples_leaf` grows by the same factor, so a leaf holds as many nodes as before."""
        k, base = len(Xs), self.params
        self.params = dict(base, min_samples_leaf=base["min_samples_leaf"] * k)
        try:
            return self.fit(np.concatenate(Xs), np.tile(h, k), np.tile(m, k), np.tile(R, k) / k)
        finally:
            self.params = base


def predict_draws(scorer, Xs: list[np.ndarray]) -> np.ndarray:
    """The utility averaged over the combinations of draws (`Hurdle` or `FrozenHurdle`): the products h m are
    averaged, not h and m apart."""
    return np.mean([scorer.predict(X)[0] for X in Xs], 0)


class Trees:
    """One fitted HistGradientBoosting model as arrays: the nodes of all its trees (child indices global, `roots`
    the first node of each tree) and the baseline. `raw` gives the model's raw prediction exactly as scikit-learn
    computes it: the baseline plus each tree's leaf value, added in tree order."""

    FIELDS = ("baseline", "roots", "feature", "threshold", "missing_left", "left", "right", "is_leaf", "value")

    def __init__(self, baseline, roots, feature, threshold, missing_left, left, right, is_leaf, value):
        self.baseline = float(baseline)
        self.roots = np.asarray(roots, np.int64)
        self.feature = np.asarray(feature, np.int64)
        self.threshold = np.asarray(threshold, np.float64)
        self.missing_left = np.asarray(missing_left, bool)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.is_leaf = np.asarray(is_leaf, bool)
        self.value = np.asarray(value, np.float64)

    @classmethod
    def from_sklearn(cls, est) -> Trees:
        """From a fitted HistGradientBoosting classifier (binary) or regressor; categorical splits are refused."""
        trees = [pred.nodes for it in est._predictors for pred in it]  # one tree per iteration (binary or regression)
        nodes = np.concatenate(trees)
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported")
        sizes = [len(t) for t in trees]
        roots = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        shift = np.repeat(roots, sizes)
        return cls(est._baseline_prediction.ravel()[0], roots, nodes["feature_idx"], nodes["num_threshold"],
                   nodes["missing_go_to_left"], nodes["left"].astype(np.int64) + shift,
                   nodes["right"].astype(np.int64) + shift, nodes["is_leaf"], nodes["value"])

    def arrays(self) -> dict[str, np.ndarray]:
        """The fields as arrays, keyed by name, for saving."""
        return {f: np.asarray(getattr(self, f)) for f in self.FIELDS}

    def raw(self, X: np.ndarray) -> np.ndarray:
        """The raw prediction (log-odds for the classifier) of each row of `X`; a NaN feature follows the split's
        missing-value direction."""
        X = np.asarray(X, np.float64)
        rows = np.arange(len(X))[:, None]
        node = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        while True:  # all trees at once, one level per step
            inner = ~self.is_leaf[node]
            if not inner.any():
                break
            x = X[rows, self.feature[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)
        terms = np.column_stack([np.full(len(X), self.baseline), self.value[node]])
        return np.cumsum(terms, axis=1)[:, -1]  # a running sum adds in the same order, so the result is identical


class FrozenHurdle:
    """A fitted `Hurdle` for inference: the same `predict`, from arrays that `save` and `load` keep in one `.npz`
    file (numpy only, no pickle)."""

    def __init__(self, h: Trees, m: Trees):
        self.h, self.m = h, m

    @classmethod
    def from_fitted(cls, hurdle: Hurdle) -> FrozenHurdle:
        """Freeze a fitted `Hurdle`."""
        return cls(Trees.from_sklearn(hurdle.clf), Trees.from_sklearn(hurdle.reg))

    def save(self, path) -> None:
        """Write both models' arrays to one `.npz` (keys ``h_<field>`` and ``m_<field>``)."""
        np.savez(path, **{f"{k}_{f}": v for k, t in (("h", self.h), ("m", self.m)) for f, v in t.arrays().items()})

    @classmethod
    def load(cls, path) -> FrozenHurdle:
        """Read a model written by `save` (no pickle allowed)."""
        with np.load(path, allow_pickle=False) as z:
            return cls(*(Trees(**{f: z[f"{k}_{f}"] for f in Trees.FIELDS}) for k in ("h", "m")))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """u = h m, h and m of each row, equal to the fitted `Hurdle.predict`."""
        h = expit(self.h.raw(X))
        m = np.clip(self.m.raw(X), 0.5, 1.0)
        return h * m, h, m

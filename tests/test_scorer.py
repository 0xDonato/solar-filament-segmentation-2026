"""The frozen scorer (plain arrays in an .npz file) predicts exactly what the fitted scikit-learn models predict."""
import numpy as np

from solarfil.scorer import FrozenHurdle, Hurdle


def test_frozen_scorer_matches_the_fitted_one_bit_for_bit(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 3000, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan  # missing values take each split's learned direction
    h = np.clip(0.5 + 0.3 * np.nan_to_num(X[:, 0]) + 0.1 * rng.normal(size=n), 0, 1)
    m = np.clip(0.7 + 0.1 * np.nan_to_num(X[:, 1]), 0.5, 1)
    hurdle = Hurdle(dict(max_iter=60)).fit(X, h, m, rng.integers(1, 4, n).astype(float))
    FrozenHurdle.from_fitted(hurdle).save(tmp_path / "scorer.npz")
    frozen = FrozenHurdle.load(tmp_path / "scorer.npz")
    Xt = rng.normal(size=(1000, d)).astype(np.float32)
    Xt[rng.random(Xt.shape) < 0.1] = np.nan
    Xt[:3] = np.inf
    for want, got in zip(hurdle.predict(Xt), frozen.predict(Xt)):
        assert np.array_equal(want, got)
    assert all(len(v) == 0 for v in frozen.predict(Xt[:0]))


def test_draw_augmentation_with_identical_draws_learns_the_single_draw_model():
    rng = np.random.default_rng(1)
    n, d = 2000, 6
    X = rng.normal(size=(n, d))
    h = np.clip(0.5 + 0.3 * X[:, 0] + 0.1 * rng.normal(size=n), 0, 1)
    m = np.clip(0.7 + 0.1 * X[:, 1], 0.5, 1)
    R = rng.integers(1, 4, n).astype(float)
    one = Hurdle(dict(max_iter=40)).fit(X, h, m, R)
    two = Hurdle(dict(max_iter=40)).fit_draws([X, X], h, m, R)
    assert two.params == one.params  # the scaled min_samples_leaf held only during the fit
    assert np.allclose(two.predict(X)[0], one.predict(X)[0], atol=1e-9)


def test_utilities_over_draws_average_the_products():
    from solarfil.scorer import predict_draws
    rng = np.random.default_rng(2)
    n, d = 1500, 5
    Xa, Xb = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    h = np.clip(0.5 + 0.3 * Xa[:, 0], 0, 1)
    m = np.clip(0.7 + 0.1 * Xa[:, 1], 0.5, 1)
    hurdle = Hurdle(dict(max_iter=30)).fit_draws([Xa, Xb], h, m, np.ones(n))
    u = predict_draws(hurdle, [Xa, Xb])
    assert np.array_equal(u, (hurdle.predict(Xa)[0] + hurdle.predict(Xb)[0]) / 2)
    frozen = FrozenHurdle.from_fitted(hurdle)
    assert np.array_equal(predict_draws(frozen, [Xa, Xb]), u)

"""The per-group annotator model (HierEM): its posterior, its refit and the population map."""
import numpy as np
import torch

from solarfil.annotators import GroupAnnotators
from solarfil.frames import CropSampler, Frame
from solarfil.model import PopulationMap
from solarfil.preprocess import Disk


def test_posterior_is_bayes_rule_over_the_frames_records():
    """From the soft mean alone it equals the posterior computed record by record."""
    rng = np.random.default_rng(0)
    alpha, beta, n = 0.7, 0.97, 3
    ann = GroupAnnotators(["g0"], [10], alpha=alpha, beta=beta)
    z = torch.from_numpy(rng.normal(0, 2, (1, 16, 16)).astype(np.float32))
    drawn = rng.random((n, 16, 16)) < 0.4
    q = ann.posterior(z, torch.from_numpy(drawn.mean(0).astype(np.float32))[None], torch.tensor([n]),
                      torch.tensor([0]))
    pi = 1 / (1 + np.exp(-z.numpy()[0]))
    on = pi * np.prod(np.where(drawn, alpha, 1 - alpha), 0)
    off = (1 - pi) * np.prod(np.where(drawn, 1 - beta, beta), 0)
    np.testing.assert_allclose(q.numpy()[0], on / (on + off), atol=1e-5)


def test_refits_recover_each_groups_sensitivity_and_specificity():
    """EM on three records per frame from a known latent mask finds the groups' rates."""
    rng = np.random.default_rng(1)
    truth = {"g0": (0.9, 0.98), "g1": (0.6, 0.95)}
    ann = GroupAnnotators(list(truth), [10**9, 10**9])  # no pooling: the groups' own estimates
    frames = []
    for gi, (a, b) in enumerate(truth.values()):
        for _ in range(4):
            t = rng.random((64, 64)) < 0.3
            drawn = np.where(t, rng.random((3, 64, 64)) < a, rng.random((3, 64, 64)) < 1 - b)
            frames.append((gi, torch.from_numpy(drawn.mean(0).astype(np.float32))))
    z = torch.full((1, 64, 64), float(np.log(0.3 / 0.7)))  # the true prior
    for _ in range(40):
        for gi, soft in frames:
            ann.posterior(z, soft[None], torch.tensor([3]), torch.tensor([gi]))
        ann.refit()
    summary = ann.summary()["groups"]
    for g, (a, b) in truth.items():
        assert abs(summary[g]["alpha"] - a) < 0.02 and abs(summary[g]["beta"] - b) < 0.01, (g, summary[g])


def test_a_group_that_draws_everything_keeps_finite_logits():
    ann = GroupAnnotators(["a"], [5])
    ann.stats[:] = [[10, 10, 100, 100]]
    ann.refit()
    assert np.isfinite(ann.la).all() and np.isfinite(ann.lb).all() and np.isfinite(ann.mu).all()


def test_a_group_with_few_records_is_pooled_towards_the_population():
    ann = GroupAnnotators(["big", "small"], [1000, 2], pool=10.0)
    ann.stats[:] = [[900, 1000, 990, 1000], [5, 10, 990, 1000]]  # sensitivity 0.9 and 0.5, specificity 0.99
    ann.refit()
    big, small = ann.summary()["groups"]["big"]["alpha"], ann.summary()["groups"]["small"]["alpha"]
    assert abs(big - 0.9) < 0.01 and 0.5 < small < ann.population()[0] < 0.9


def test_state_round_trip():
    ann = GroupAnnotators(["a", "b"], [3, 4])
    ann.stats[:] = [[8, 10, 95, 100], [3, 5, 90, 100]]
    ann.refit()
    other = GroupAnnotators(["a", "b"], [3, 4])
    other.load_state_dict(ann.state_dict())
    assert other.summary() == ann.summary()


def test_population_map_is_the_identity_until_set_and_then_the_population_annotator():
    net = torch.nn.Identity()
    m = PopulationMap(net)
    z = torch.linspace(-6, 6, 13).reshape(1, 1, 13, 1)
    torch.testing.assert_close(m(z), z, atol=1e-4, rtol=0)
    m.annotator.copy_(torch.tensor([0.6, 0.99]))
    pi = torch.sigmoid(z)
    torch.testing.assert_close(torch.sigmoid(m(z)), 0.6 * pi + 0.01 * (1 - pi), atol=1e-6, rtol=0)
    assert set(m.state_dict()) == {"annotator"}  # the buffer travels with the weights


def test_crop_batches_can_name_their_frames():
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 256, (128, 128), dtype=np.uint8)  # noqa: E731
    frames = [Frame(s, Disk(64, 64, 50), 1, img(), p=img(), spine=img()) for s in "ab"]
    xs, ys, idx = CropSampler(frames, 32, seed=2).batch(6, frames=True)
    assert xs.shape[0] == ys.shape[0] == 6 and set(idx.tolist()) <= {0, 1}
    plain = CropSampler(frames, 32, seed=2).batch(6)  # the same draws without the indices
    np.testing.assert_array_equal(plain[0], xs)

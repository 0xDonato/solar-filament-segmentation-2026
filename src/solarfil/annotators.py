"""Per-group annotator model for training a dense network on the latent filament mask (HierEM, arXiv 2603.14418).

Each record is a noisy view of a latent mask T: an annotator of group g draws a pixel of T with probability (the
sensitivity) alpha_g and a pixel outside T with probability 1 - beta_g (beta_g, the specificity). The network
predicts pi = P(T = 1) from the image. All records of a frame come from one group, so the posterior of T at a
pixel drawn by k of the frame's n records is

    logit q = logit pi + k (log alpha - log(1 - beta)) + (n - k) (log(1 - alpha) - log beta),

which needs only the soft mean k / n of the cached targets. The network trains on q (the E-step), and every few
hundred steps alpha and beta are refitted from the summed counts q k, q n, (1 - q)(n - k) and (1 - q) n (the
M-step), each group's logit pooled towards all groups' by its number of records, n_g / (n_g + pool). The map the
tree reads is the population annotator's, sigma(mu_a) pi + (1 - sigma(mu_b))(1 - pi) (`model.PopulationMap`), the
probability that an annotator of an average group draws the pixel, as the soft-mean target is.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class GroupAnnotators:
    """Sensitivity and specificity of each annotation group of the training frames."""

    def __init__(self, groups: list[str], records: list[int], alpha: float = 0.8, beta: float = 0.99,
                 pool: float = 10.0):
        """`groups`: the groups in a fixed order; `records`: each group's number of training records."""
        self.groups = list(groups)
        self.records = np.asarray(records, np.float64)
        self.pool = pool
        self.la = np.full(len(groups), _logit(alpha))  # logit alpha_g
        self.lb = np.full(len(groups), _logit(beta))  # logit beta_g
        self.mu = np.array([_logit(alpha), _logit(beta)])  # the population's logits
        self.stats = np.zeros((len(groups), 4))  # per group: sum q k, q n, (1 - q)(n - k), (1 - q) n

    @torch.no_grad()
    def posterior(self, z: torch.Tensor, soft: torch.Tensor, n: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """q = P(T = 1 | the records) of a batch: `z` (B, H, W) the network's logits of pi, `soft` (B, H, W) the
        records' soft mean k / n, `n` (B,) each crop's record count and `g` (B,) its group index. Adds the batch's
        counts to the M-step's sums."""
        la = torch.tensor(self.la, dtype=torch.float32).to(z.device)[g][:, None, None]  # float32: MPS has no float64
        lb = torch.tensor(self.lb, dtype=torch.float32).to(z.device)[g][:, None, None]
        n = n.to(z.dtype)[:, None, None]
        k = soft * n
        drawn = F.logsigmoid(la) - F.logsigmoid(-lb)  # log alpha - log(1 - beta)
        undrawn = F.logsigmoid(-la) - F.logsigmoid(lb)  # log(1 - alpha) - log beta
        q = torch.sigmoid(z + k * drawn + (n - k) * undrawn)
        sums = torch.stack([(q * k).sum((1, 2)), (q * n).sum((1, 2)), ((1 - q) * (n - k)).sum((1, 2)),
                            ((1 - q) * n).sum((1, 2))], 1)
        np.add.at(self.stats, g.cpu().numpy(), sums.cpu().double().numpy())
        return q

    def refit(self) -> None:
        """The M-step on the counts summed since the last refit; groups without counts keep their values."""
        tp, p, tn, neg = self.stats.T
        seen = (p > 0) & (neg > 0)
        if not seen.any():
            return
        self.mu = np.array([_logit(tp[seen].sum() / p[seen].sum()), _logit(tn[seen].sum() / neg[seen].sum())])
        w = self.records / (self.records + self.pool)
        eps = 1e-6
        for mu, rate, out in ((self.mu[0], tp / np.maximum(p, eps), self.la),
                              (self.mu[1], tn / np.maximum(neg, eps), self.lb)):
            r = np.clip(rate, eps, 1 - eps)
            out[seen] = mu + w[seen] * (np.log(r[seen] / (1 - r[seen])) - mu)
        self.stats[:] = 0

    def population(self) -> tuple[float, float]:
        """alpha and beta of the population annotator."""
        a, b = 1 / (1 + np.exp(-self.mu))
        return float(a), float(b)

    def summary(self) -> dict:
        """The population's and each group's alpha and beta (rounded to six decimals), with each group's record count,
        as a JSON-ready dict."""
        alpha, beta = 1 / (1 + np.exp(-self.la)), 1 / (1 + np.exp(-self.lb))
        a0, b0 = self.population()
        return dict(population=dict(alpha=round(a0, 6), beta=round(b0, 6)),
                    groups={g: dict(alpha=round(float(a), 6), beta=round(float(b), 6), records=int(r))
                            for g, a, b, r in zip(self.groups, alpha, beta, self.records)})

    def state_dict(self) -> dict:
        """The fitted logits and the pending M-step sums as lists, for a checkpoint."""
        return dict(la=self.la.tolist(), lb=self.lb.tolist(), mu=self.mu.tolist(), stats=self.stats.tolist())

    def load_state_dict(self, st: dict) -> None:
        """Restore the logits and the M-step sums from `state_dict`'s output."""
        self.la, self.lb = np.asarray(st["la"]), np.asarray(st["lb"])
        self.mu, self.stats = np.asarray(st["mu"]), np.asarray(st["stats"])

"""Chirality read from a mask's barbs, and a test of what that reading adds beyond the latitude rule.

A filament's chirality shows in the lean of its barbs against its spine (Hao et al. 2015, arXiv:1506.08490: a
right-bearing filament is dextral, a left-bearing one sinistral). MAGFiLO's reviewers kept a Left or Right label only
when they could tell the chirality from the segmentation alone, so a mask that keeps the label's barbs keeps the label
legible. The reading uses the mask only, never the image:

- the mask is skeletonized; the longest geodesic path of the skeleton's largest component is the spine;
- every other end point of the skeleton is a barb tip; the barb's length L is its geodesic distance to the spine and
  it counts when L >= `BRANCH` px;
- a barb's bearing is s = sign(cross(t, b) * dot(t, b)), with t the spine's direction across `K` path nodes on each
  side of the attachment and b the attachment-to-tip vector (x right, y up). s = sign(sin 2 theta) for the barb's
  angle theta from the spine line, so it does not depend on which way the spine runs;
- the score is S = sum of L * s over the barbs. A mask with S != 0 is legible, and the call is Right when C * S > 0.

C = -1 is the convention that reads the annotators' own Left/Right polygons best: Right for barbs that sit
clockwise of the spine line by an acute angle, a backslash lean on a horizontal spine, Left for a slash lean.

The conditional test (`beyond_latitude`) asks whether the bearing predicts the label once the hemispheric rule is
known: a logistic prior on hemisphere and hemisphere x |latitude| against the same model plus the bearing features
(`bearing_features`), each fitted leaving one annotation group out, compared row by row on held-out accuracy and log
loss, with 90% intervals from a bootstrap over groups and over frames.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize
from sklearn.linear_model import LogisticRegression

from .preprocess import Disk

BRANCH = 8  # px, the barb length fixed before the test
THRESHOLDS = (4, 8, 12, 16)  # px, the sensitivity sweep of the barb length
K = 10  # spine path nodes on each side of a barb's attachment for the tangent
C = -1  # sign convention: Right when C * S > 0
PAD = 4  # px of background around the mask's bounding box before skeletonizing
LEFT, RIGHT = 1, 2  # the competition's category ids
PRIOR = ("h", "hx")  # hemisphere, hemisphere x |latitude|
FULL = ("h", "hx", "m", "ls", "cons")  # the prior plus the bearing features


def crop(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """The mask's bounding box padded by `PAD` px of background, and the frame coordinates (y0, x0) of the crop's
    top-left pixel. An empty mask gives an empty crop."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 0), bool), 0, 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return np.pad(mask[y0:y1, x0:x1].astype(bool), PAD), int(y0) - PAD, int(x0) - PAD


def skeleton_graph(sk: np.ndarray) -> tuple[np.ndarray, np.ndarray, csr_matrix | None]:
    """The 8-connected graph of a skeleton: pixel rows and columns in raster order, and the symmetric adjacency with
    edge lengths 1 (side) and sqrt 2 (diagonal); None when no two pixels touch."""
    ys, xs = np.nonzero(sk)
    idx = -np.ones(sk.shape, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < sk.shape[0]) & (x2 >= 0) & (x2 < sk.shape[1])
        a = np.arange(len(ys))[ok]
        b = idx[y2[ok], x2[ok]]
        a, b = a[b >= 0], b[b >= 0]
        d = np.sqrt(2.0) if dy and dx else 1.0
        rows += [a, b]
        cols += [b, a]
        w += [np.full(len(a), d), np.full(len(a), d)]
    if not sum(len(r) for r in rows):
        return ys, xs, None
    r, c, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    return ys, xs, coo_matrix((ww, (r, c)), shape=(len(ys), len(ys))).tocsr()


def barbs(mask: np.ndarray, min_length: float = BRANCH, k: int = K) -> list[tuple[float, float]]:
    """(length L, bearing s) of every barb of a mask at least `min_length` px long, in skeleton raster order. The
    mask may be a whole frame or a crop; it is cropped and padded first (`crop`), so the result does not depend on
    where the mask sits. A mask without a branching skeleton has no barbs."""
    p, _, _ = crop(mask)
    if p.size == 0:
        return []
    ys, xs, g = skeleton_graph(skeletonize(p))
    if g is None or len(ys) < 3:
        return []
    _, lab = connected_components(g, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    g, ys, xs = g[keep][:, keep], ys[keep], xs[keep]
    d0 = dijkstra(g, indices=0)
    far = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1, pred = dijkstra(g, indices=far, return_predecessors=True)
    path = [int(np.argmax(np.where(np.isfinite(d1), d1, -1)))]
    while path[-1] != far and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    path = np.array(path)
    pos = -np.ones(len(ys), np.int64)
    pos[path] = np.arange(len(path))
    ends = np.flatnonzero(np.diff(g.indptr) == 1)
    ends = ends[pos[ends] < 0]
    if len(ends) == 0:
        return []
    dist, _, src = dijkstra(g, indices=path, return_predecessors=True, min_only=True)
    out = []
    for e in ends:
        length = dist[e]
        if not np.isfinite(length) or length < min_length:
            continue
        a = int(src[e])
        j0, j1 = max(0, pos[a] - k), min(len(path) - 1, pos[a] + k)
        if j1 == j0:
            continue
        t = (xs[path[j1]] - xs[path[j0]], -(ys[path[j1]] - ys[path[j0]]))
        b = (xs[e] - xs[a], -(ys[e] - ys[a]))
        s = np.sign(float(t[0] * b[1] - t[1] * b[0]) * float(t[0] * b[0] + t[1] * b[1]))
        out.append((float(length), float(s)))
    return out


def score(barb_list: list[tuple[float, float]], branch: float = BRANCH) -> dict:
    """The barb score of the barbs at least `branch` px long: their count, S = sum of L * s, Sabs = sum of L and the
    sign of S."""
    kept = [(length, s) for length, s in barb_list if length >= branch]
    S = sum(length * s for length, s in kept)
    return dict(barbs=len(kept), S=float(S), Sabs=float(sum(length for length, _ in kept)), Sn=int(np.sign(S)))


def call(sc: dict, c: int = C) -> int:
    """The chirality the barbs show: `RIGHT` when c * S > 0, `LEFT` when c * S < 0, 0 for an illegible mask."""
    if sc["Sn"] == 0:
        return 0
    return RIGHT if c * sc["Sn"] > 0 else LEFT


def latitude(mask: np.ndarray, disk: Disk) -> float:
    """Heliographic latitude of the mask's centroid in degrees, north up, without the B0 tilt: the arcsine of its
    height above the disk centre over the disk radius."""
    p, y0, _ = crop(mask)  # the centroid taken through the crop, which fixes its floating-point value to the last bit
    cy = y0 + np.nonzero(p)[0].mean()
    return float(np.degrees(np.arcsin(np.clip((disk.cy - cy) / disk.r, -1, 1))))


def bearing_features(sc: dict, lat: float, c: int = C) -> dict:
    """The conditional test's features of one reading: hemisphere h (+1 north), hemisphere x |latitude| hx, and the
    bearing's sign m, signed log score ls and consistency cons (S over Sabs), all with the convention c and all 0 for
    an illegible mask."""
    h = 1.0 if lat > 0 else -1.0
    s = c * sc["S"] if sc["Sn"] != 0 else 0.0
    sa = sc["Sabs"] if sc["Sn"] != 0 else 0.0
    return dict(h=h, hx=h * abs(lat), m=float(np.sign(s)), ls=float(np.sign(s) * np.log1p(abs(s))),
                cons=s / sa if sa > 0 else 0.0)


def leave_group_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Probability of y = 1 for every row from a logistic regression (C = 1) fitted on the other groups' rows; a
    group whose training rows hold one class gets that class's rate."""
    p = np.zeros(len(y))
    for q in np.unique(groups):
        tr, te = groups != q, groups == q
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cluster_interval(d: np.ndarray, ids: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """90% interval of the mean of d under a bootstrap that resamples whole clusters (`ids`) with replacement: the
    5th and 95th percentiles of the resampled sums over the resampled row counts."""
    u = np.unique(ids)
    tot = np.array([d[ids == q].sum() for q in u])
    cnt = np.array([(ids == q).sum() for q in u])
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    bt = tot[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(bt, 5)), float(np.percentile(bt, 95))


def beyond_latitude(rows: list[dict], n_groups: int = 4000, n_frames: int = 2000) -> dict:
    """What the bearing adds beyond the latitude prior on labelled rows. Each row holds the label y (1 for Right),
    its annotation group g, its frame f and the features of `bearing_features`. Both models are fitted leaving one
    group out; accuracy counts p > 0.5 as Right and log loss clips p to [1e-6, 1 - 1e-6]. Changes are full minus
    prior, so a gain in accuracy is positive and a gain in log loss negative."""
    y = np.array([r["y"] for r in rows])
    g = np.array([r["g"] for r in rows])
    fr = np.array([r["f"] for r in rows])
    p0, p1 = (leave_group_out(np.array([[r[c] for c in cols] for r in rows], float), y, g) for cols in (PRIOR, FULL))
    q0, q1 = np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)
    dl = -(y * np.log(q1) + (1 - y) * np.log(1 - q1)) + (y * np.log(q0) + (1 - y) * np.log(1 - q0))
    a0, a1 = ((p0 > 0.5) == (y == 1)).astype(float), ((p1 > 0.5) == (y == 1)).astype(float)
    return dict(rows=len(rows), groups=len(np.unique(g)), frames=len(np.unique(fr)),
                legible=float(np.mean([r["m"] != 0 for r in rows])),
                accuracy_prior=float(a0.mean()), accuracy_full=float(a1.mean()), accuracy_gain=float((a1 - a0).mean()),
                accuracy_gain_groups90=cluster_interval(a1 - a0, g, n_groups),
                accuracy_gain_frames90=cluster_interval(a1 - a0, fr, n_frames),
                log_loss_change=float(dl.mean()), log_loss_change_groups90=cluster_interval(dl, g, n_groups))

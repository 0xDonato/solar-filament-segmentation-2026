"""Component tree of a probability map and the exact selection of an antichain.

Thresholding ``p`` at increasing levels ``t_1 < ... < t_K`` (with holes filled, an increasing operator) gives
nested binary masks, so every 8-connected component at level ``k + 1`` lies inside exactly one component at
level ``k``: a forest. Every antichain of the forest is a pixel-disjoint segmentation, and every single-threshold
segmentation is one of them.

Selection maximises ``sum over selected nodes of (u(c) - tau)`` exactly over the candidate forest:
``best(v) = max(0, u(v) - tau, sum of best(children))``. This is the learned additive objective.
With unique matches and known expected matched IoUs, fractional optimisation connects its cutoff with
``tau = PQ* / 2``. Learned utility error, missing candidates and multiple matches caused by overlapping
reference polygons limit that connection. Exact tree selection does not establish optimal host PQ.

Nodes are stored on the sparse support ``S = {p > t_1}`` (cropped to its bounding box) so that per-node
statistics are bincounts over a few hundred thousand pixels instead of full frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .postprocess import fill_holes

LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
MARGIN = 8  # px around the support kept in the crop (rings, dilations)
MIN_AREA = 20  # px at 2048; smaller components are not candidates (GT areas: 5th percentile 317 px)


@dataclass
class Tree:
    """The component tree of a frame's probability map: every connected component above each threshold level, as
    nodes linked to the component that contains them one level down, stored by pixel over the map's support crop."""
    shape: tuple[int, int]  # full frame
    box: tuple[int, int, int, int]  # y0, y1, x0, x1 of the crop in the frame
    support: np.ndarray  # flat indices (in the crop) of S, int64
    node_of: np.ndarray  # (K, |S|) int32: node id of each support pixel at each level, -1 outside
    level: np.ndarray  # (n,) level index of the node's first appearance
    last_level: np.ndarray  # (n,) last level at which the node's pixel set is unchanged
    parent: np.ndarray  # (n,) parent node id, -1 for roots
    area: np.ndarray  # (n,)
    levels: tuple[float, ...]  # threshold of every row
    kinds: tuple[int, ...] = ()  # 0: a threshold level, 1: a watershed partition row

    @property
    def n(self) -> int:
        """Number of nodes."""
        return len(self.level)

    @property
    def crop_shape(self) -> tuple[int, int]:
        """Height and width of the crop."""
        return self.box[1] - self.box[0], self.box[3] - self.box[2]

    def children(self) -> list[list[int]]:
        """Each node's child node ids."""
        ch: list[list[int]] = [[] for _ in range(self.n)]
        for v, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(v)
        return ch

    def node_pixels(self, v: int) -> np.ndarray:
        """Flat crop indices of node v."""
        return self.support[self.node_of[self.level[v]] == v]

    def node_mask(self, v: int) -> np.ndarray:
        """Full-frame boolean mask of node v."""
        m = np.zeros(self.shape, bool)
        y0, y1, x0, x1 = self.box
        c = np.zeros(self.crop_shape, bool).ravel()
        c[self.node_pixels(v)] = True
        m[y0:y1, x0:x1] = c.reshape(self.crop_shape)
        return m

    def labels(self, nodes) -> np.ndarray:
        """Full-frame int32 label image of a set of (disjoint) nodes, labelled 1..len(nodes) in the given order."""
        lab = np.zeros(self.shape, np.int32)
        y0, y1, x0, x1 = self.box
        c = np.zeros(self.crop_shape, np.int32).ravel()
        for i, v in enumerate(nodes, start=1):
            c[self.node_pixels(v)] = i
        lab[y0:y1, x0:x1] = c.reshape(self.crop_shape)
        return lab


def _level_rows(pc: np.ndarray, support: np.ndarray, levels, holes: bool, min_area: int) -> list[np.ndarray]:
    """Component labels on the support at every level (0 = not in the mask or a component below min_area)."""
    rows = []
    for t in levels:
        b = pc > t
        if holes:
            b = fill_holes(b)
        n, lab = cv2.connectedComponents(b.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        lab_s = lab.ravel()[support].astype(np.int64)
        small = np.bincount(lab_s, minlength=n) < min_area
        small[0] = False
        if small.any():  # nesting survives: a child is never larger than its parent
            lab_s = np.where(small[lab_s], 0, lab_s)
        rows.append(lab_s)
    return rows


def _partition_row(lab_a: np.ndarray, lab_b: np.ndarray, pc: np.ndarray, support: np.ndarray, ww: int):
    """Split candidates between two levels: every component of `lab_a` that holds two or more components of `lab_b`
    is partitioned by a watershed of -p seeded with them, so each part gets the full extent its seed would have
    without the neighbour. Returns the labels of the intermediate row, or None when nothing splits."""
    from skimage.segmentation import watershed

    on = lab_b > 0
    pairs = np.unique(np.stack([lab_a[on], lab_b[on]], 1), axis=0)
    n_child = np.bincount(pairs[:, 0], minlength=int(lab_a.max()) + 1)
    split = np.flatnonzero(n_child >= 2)
    split = split[split > 0]
    if len(split) == 0:
        return None
    out = lab_a.copy()
    nxt = int(lab_a.max()) + 1
    order = np.argsort(lab_a, kind="stable")
    starts = np.searchsorted(lab_a[order], split, side="left")
    ends = np.searchsorted(lab_a[order], split, side="right")
    for s0, s1 in zip(starts, ends):
        pos = order[s0:s1]
        yy, xx = np.divmod(support[pos], ww)
        ya, xa = yy.min(), xx.min()
        h, w = yy.max() - ya + 1, xx.max() - xa + 1
        mask = np.zeros((h, w), bool)
        mask[yy - ya, xx - xa] = True
        markers = np.zeros((h, w), np.int64)
        markers[yy - ya, xx - xa] = lab_b[pos]
        elev = -pc[ya: ya + h, xa: xa + w]
        reg = watershed(elev, markers, mask=mask, connectivity=2)[yy - ya, xx - xa]
        kids, inv = np.unique(reg, return_inverse=True)
        out[pos] = nxt + inv
        nxt += len(kids)
    return out


def _nest(rows: list[np.ndarray], n_support: int):
    """Nodes of nested label rows: a component identical to its only parent continues the parent's node."""
    node_of = np.full((len(rows), n_support), -1, np.int32)
    level, last, parent, area = [], [], [], []
    prev_ids = prev_lab = None
    for k, lab_s in enumerate(rows):
        n = int(lab_s.max()) + 1
        cnt = np.bincount(lab_s, minlength=n)
        ids = np.full(n, -1, np.int64)
        if k == 0:
            for c in np.flatnonzero(cnt[1:]) + 1:
                ids[c] = len(level)
                level.append(k), last.append(k), parent.append(-1), area.append(int(cnt[c]))
        else:
            first = np.full(n, -1, np.int64)
            on = np.flatnonzero(lab_s > 0)
            first[lab_s[on][::-1]] = on[::-1]  # a support position of each label
            ok = first >= 0
            ok[0] = False
            par_lab = np.zeros(n, np.int64)
            par_lab[ok] = prev_lab[first[ok]]
            par_node = prev_ids[par_lab]
            n_child = np.bincount(par_lab[ok], minlength=len(prev_ids))
            for c in np.flatnonzero(ok):
                pn = par_node[c]
                if n_child[par_lab[c]] == 1 and area[pn] == cnt[c]:
                    ids[c] = pn  # same pixel set as its parent: one node that persists
                    last[pn] = k
                else:
                    ids[c] = len(level)
                    level.append(k), last.append(k), parent.append(int(pn)), area.append(int(cnt[c]))
        node_of[k] = np.where(lab_s > 0, ids[lab_s], -1)
        prev_ids, prev_lab = ids, lab_s
    arr = lambda v: np.array(v, np.int64)  # noqa: E731
    return node_of, arr(level), arr(last), arr(parent), arr(area)


def build_tree(p: np.ndarray, levels=LEVELS, holes: bool = True, min_area: int = MIN_AREA,
               splits: bool = False) -> Tree:
    """Component tree of a full-frame probability map (float32, 2048 x 2048).

    With `splits`, a watershed partition row is inserted between consecutive levels wherever a component splits
    (`_partition_row`); its rows carry the mid-point threshold and `kinds` = 1.
    """
    b0 = p > levels[0]
    if holes:
        b0 = fill_holes(b0)
    ys, xs = np.nonzero(b0)
    if len(ys) == 0:
        z = np.zeros(0, np.int64)
        return Tree(p.shape, (0, 0, 0, 0), z, np.zeros((len(levels), 0), np.int32), z, z, z, z, tuple(levels),
                    (0,) * len(levels))
    y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + 1 + MARGIN, p.shape[0])
    x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + 1 + MARGIN, p.shape[1])
    pc = p[y0:y1, x0:x1]
    support = np.flatnonzero(b0[y0:y1, x0:x1].ravel())
    rows = _level_rows(pc, support, levels, holes, min_area)
    values, kinds = [levels[0]], [0]
    out_rows = [rows[0]]
    for k in range(1, len(rows)):
        if splits:
            part = _partition_row(rows[k - 1], rows[k], pc, support, x1 - x0)
            if part is not None:
                out_rows.append(part)
                values.append(0.5 * (levels[k - 1] + levels[k]))
                kinds.append(1)
        out_rows.append(rows[k])
        values.append(levels[k])
        kinds.append(0)
    node_of, level, last, parent, area = _nest(out_rows, len(support))
    return Tree(p.shape, (int(y0), int(y1), int(x0), int(x1)), support, node_of, level, last, parent, area,
                tuple(values), tuple(kinds))


def select(tree: Tree, score: np.ndarray, tau: float) -> list[int]:
    """Exact antichain maximising sum(score - tau) (score: one value per node)."""
    n = tree.n
    if n == 0:
        return []
    gain = score - tau
    best = np.zeros(n)
    take = np.zeros(n, bool)  # True: take the node itself; False: defer to its children
    ch = tree.children()
    for v in np.argsort(-tree.level, kind="stable"):  # deepest levels first
        sub = sum(best[w] for w in ch[v])
        own = gain[v]
        if own >= sub and own > 0:
            best[v], take[v] = own, True
        else:
            best[v] = max(sub, 0.0)
    out = []
    stack = [v for v in range(n) if tree.parent[v] < 0]
    while stack:
        v = stack.pop()
        if best[v] <= 0:
            continue
        if take[v]:
            out.append(v)
        else:
            stack.extend(ch[v])
    return sorted(out)


def chain(tree: Tree, ch: list[list[int]], v: int) -> tuple[list[int], int]:
    """The nodes a selected node may be swapped for without touching any other node: its single-child ancestors and
    descendants, ordered from the widest to the narrowest, and v's position among them."""
    up, u = [], v
    while tree.parent[u] >= 0 and len(ch[tree.parent[u]]) == 1:
        u = int(tree.parent[u])
        up.append(u)
    down, u = [], v
    while len(ch[u]) == 1:
        u = ch[u][0]
        down.append(u)
    return up[::-1] + [v] + down, len(up)


def relevel(tree: Tree, p: np.ndarray, nodes, alpha: float) -> list[int]:
    """Redraw each selected instance at its own level: the member of its chain whose threshold is nearest to `alpha`
    times the instance's peak (the 99th percentile of `p` over the chain's widest member); on a tie, the node itself.

    For calibrated probabilities the F1-optimal threshold is half the optimal F1 (Lipton et al., arXiv 1402.1892),
    which for IoU makes the cut J/(1+J) of an object's plateau. On our map the plateau is the share of annotators who
    drew the filament, so a filament drawn by one of three is cut lower than one all three drew. The selection still
    decides which filaments are kept, with alpha validated out of fold."""
    if not len(nodes):
        return []
    ch = tree.children()
    y0, y1, x0, x1 = tree.box
    pc = p[y0:y1, x0:x1].ravel()
    out = []
    for v in nodes:
        members, i = chain(tree, ch, v)
        peak = np.percentile(pc[tree.node_pixels(members[0])], 99)
        d = np.abs(np.asarray([tree.levels[tree.level[w]] for w in members]) - alpha * peak)
        d[i] -= 1e-9
        out.append(members[int(np.argmin(d))])
    return out


def light_tree(level: np.ndarray, last_level: np.ndarray, parent: np.ndarray, area: np.ndarray,
               levels=LEVELS) -> Tree:
    """A tree without pixels (enough for `select`), as stored in a bank file."""
    z = np.zeros(0, np.int64)
    return Tree((0, 0), (0, 0, 0, 0), z, np.zeros((len(levels), 0), np.int32), level, last_level, parent, area,
                tuple(levels))

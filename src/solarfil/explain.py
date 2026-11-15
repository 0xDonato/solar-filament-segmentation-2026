"""The selection made inspectable: why the exact selection (`tree.select`) kept each candidate, and an independent
check that the antichain it returns is the optimum under its constraints.

The constraints are those of `tree.select`: the candidates are the nodes of the frame's component tree, a submission
is an antichain (no node together with any of its ancestors, so the masks are pixel-disjoint), and its value is
`sum over the chosen nodes of (u - tau)`. `dp_table` restates the bottom-up recursion so that every node's numbers
can be printed; `optimum_milp` and `brute_force` compute the optimum without it, as an integer program over the
root-to-leaf chains and by enumeration of every antichain of a small subtree.
"""
from __future__ import annotations

import numpy as np

from .tree import Tree

ATOL = 1e-9  # values closer than this are equal (ties between a node and its children are broken towards the node)


def subtree(tree: Tree, ch: list[list[int]], v: int) -> list[int]:
    """Node v and all its descendants, in depth-first order."""
    out, stack = [], [v]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(reversed(ch[u]))
    return out


def ancestors(tree: Tree, v: int) -> list[int]:
    """v's ancestors from its parent up to its root."""
    out, u = [], int(tree.parent[v])
    while u >= 0:
        out.append(u)
        u = int(tree.parent[u])
    return out


def dp_table(tree: Tree, u: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    """Per node: `gain` = u - tau; `kids` = the sum of the children's best values (what deferring to the children
    is worth); `best` = max(0, gain, kids), the best value of the node's subtree; `take` = the node itself is the
    subtree's best choice (gain >= kids and gain > 0, `tree.select`'s rule)."""
    gain = np.asarray(u, np.float64) - tau
    best, kids = np.zeros(tree.n), np.zeros(tree.n)
    take = np.zeros(tree.n, bool)
    ch = tree.children()
    for v in np.argsort(-tree.level, kind="stable"):
        kids[v] = sum(best[w] for w in ch[v])
        take[v] = gain[v] >= kids[v] and gain[v] > 0
        best[v] = gain[v] if take[v] else max(kids[v], 0.0)
    return dict(gain=gain, kids=kids, best=best, take=take)


def value(u: np.ndarray, tau: float, nodes) -> float:
    """sum(u - tau) over `nodes`."""
    nodes = np.asarray(list(nodes), np.int64)
    return float(np.sum(np.asarray(u, np.float64)[nodes] - tau)) if len(nodes) else 0.0


def is_antichain(tree: Tree, nodes) -> bool:
    """No node of `nodes` is an ancestor of another."""
    s = set(int(v) for v in nodes)
    return all(not (set(ancestors(tree, v)) & s) for v in s)


def optimum_milp(tree: Tree, u: np.ndarray, tau: float) -> tuple[float, list[int]]:
    """The optimum by an integer program, without the recursion: maximise sum((u - tau) x) over binary x with
    sum(x) <= 1 on every root-to-leaf chain (every antichain of a forest satisfies these and only antichains do).
    Returns the optimal value and one optimal node set."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    n = tree.n
    if n == 0:
        return 0.0, []
    ch = tree.children()
    rows, cols = [], []
    for k, leaf in enumerate(v for v in range(n) if not ch[v]):
        chain = [leaf, *ancestors(tree, leaf)]
        rows += [k] * len(chain)
        cols += chain
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(max(rows) + 1, n))
    c = -(np.asarray(u, np.float64) - tau)
    res = milp(c, constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(n), bounds=Bounds(0, 1))
    assert res.success, res.message
    x = np.flatnonzero(res.x > 0.5)
    return value(u, tau, x), sorted(int(v) for v in x)


def brute_force(tree: Tree, u: np.ndarray, tau: float, nodes: list[int]) -> tuple[float, list[int]]:
    """The best antichain among `nodes` (a subtree, at most about 20 nodes) by enumerating every subset."""
    n = len(nodes)
    if n == 0:
        return 0.0, []
    assert n <= 22, n
    pos = {v: i for i, v in enumerate(nodes)}
    anc = np.zeros(n, np.int64)
    for i, v in enumerate(nodes):
        for a in ancestors(tree, v):
            if a in pos:
                anc[i] |= 1 << pos[a]
    S = np.arange(1 << n, dtype=np.int64)
    ok = np.ones(len(S), bool)
    val = np.zeros(len(S))
    g = np.asarray(u, np.float64)[nodes] - tau
    for i in range(n):
        has = (S >> i) & 1 == 1
        ok &= ~(has & (S & anc[i] != 0))
        val += np.where(has, g[i], 0.0)
    val[~ok] = -np.inf
    k = int(np.argmax(val))
    return float(val[k]), [v for i, v in enumerate(nodes) if (k >> i) & 1]


def small_subtrees(tree: Tree, ch: list[list[int]], max_nodes: int) -> list[int]:
    """The roots of the maximal subtrees with 2 to `max_nodes` nodes (their parent's subtree is larger, or they are
    roots)."""
    size = np.ones(tree.n, np.int64)
    for v in np.argsort(-tree.level, kind="stable"):
        if tree.parent[v] >= 0:
            size[tree.parent[v]] += size[v]
    return [v for v in range(tree.n) if 2 <= size[v] <= max_nodes
            and (tree.parent[v] < 0 or size[tree.parent[v]] > max_nodes)]


def check(tree: Tree, u: np.ndarray, tau: float, chosen, max_brute: int = 14) -> dict:
    """Whether `chosen` is the optimum under the selection's constraints: an antichain whose every node has
    u > tau, whose value equals the integer program's optimum, and whose restriction to every maximal small subtree
    with no chosen ancestor equals that subtree's enumerated optimum (a better choice there would raise the total)."""
    chosen = sorted(int(v) for v in chosen)
    ch = tree.children()
    v_sel = value(u, tau, chosen)
    v_opt, x_opt = optimum_milp(tree, u, tau)
    sel = set(chosen)
    n_sub = n_sub_ok = 0
    for r in small_subtrees(tree, ch, max_brute):
        if set(ancestors(tree, r)) & sel:
            continue  # an ancestor holds the region; the subtree must stay empty, which the antichain test covers
        nodes = subtree(tree, ch, r)
        b, _ = brute_force(tree, u, tau, nodes)
        n_sub += 1
        n_sub_ok += abs(value(u, tau, [v for v in nodes if v in sel]) - b) <= ATOL
    antichain = is_antichain(tree, chosen)
    positive = bool(all(u[v] > tau for v in chosen))
    return dict(nodes=tree.n, chosen=len(chosen), value=v_sel, optimum=v_opt, antichain=antichain,
                positive=positive, same_set=chosen == x_opt, subtrees=n_sub, subtrees_agree=n_sub_ok,
                optimal=antichain and positive and abs(v_sel - v_opt) <= ATOL and n_sub_ok == n_sub)


def trace(tree: Tree, u: np.ndarray, tau: float, v: int, chosen) -> list[dict]:
    """The neighbourhood of node v as the selection saw it: v's ancestors up to its root, v, its siblings and its
    descendants, each with its threshold, area, utility, the DP's numbers and whether it was chosen."""
    ch = tree.children()
    t = dp_table(tree, u, tau)
    sel = set(int(x) for x in chosen)
    p = int(tree.parent[v])
    roles = [(a, "ancestor") for a in reversed(ancestors(tree, v))] + [(v, "node")]
    roles += [(s, "sibling") for s in (ch[p] if p >= 0 else []) if s != v]
    roles += [(d, "descendant") for d in subtree(tree, ch, v)[1:]]
    return [dict(node=int(w), role=role, threshold=tree.levels[tree.level[w]], area=int(tree.area[w]),
                 parent=int(tree.parent[w]), children=[int(c) for c in ch[w]], u=float(u[w]),
                 gain=float(t["gain"][w]), kids=float(t["kids"][w]), best=float(t["best"][w]),
                 take=bool(t["take"][w]), chosen=w in sel) for w, role in roles]


def reason(tree: Tree, u: np.ndarray, tau: float, v: int, table: dict | None = None,
           ch: list[list[int]] | None = None) -> str:
    """One line on why chosen node v was kept: its gain u - tau against what its children are worth together, and
    against its parent's gain, which loses to v and v's siblings together."""
    t = table or dp_table(tree, u, tau)
    ch = ch or tree.children()
    g = t["gain"][v]
    kids = ch[v]
    own = f"u {u[v]:.3f} - tau = {g:+.3f}"
    if len(kids) == 1:
        own += f" >= its one smaller part {t['kids'][v]:+.3f}"
    elif kids:
        own += f" >= its {len(kids)} parts together {t['kids'][v]:+.3f}"
    else:
        own += ", no smaller part"
    p = int(tree.parent[v])
    if p < 0:
        return own + "; no wider region"
    n_sib = len(ch[p]) - 1
    up = f"; wider region u {u[p]:.3f} ({t['gain'][p]:+.3f}) < this"
    up += f" with {n_sib} sibling{'s' if n_sib > 1 else ''} {t['kids'][p]:+.3f}" if n_sib else f" {t['kids'][p]:+.3f}"
    return own + up

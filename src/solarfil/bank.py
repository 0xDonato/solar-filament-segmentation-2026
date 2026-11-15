"""Candidate bank: a component tree, node descriptors and per-record overlap targets.

T stores each candidate's largest IoU above 0.5 within each annotation record,
or zero when it has no strict match. The additive totals method treats these
as unique matches. This equals the host counts when both prediction and
reference masks are disjoint within a record. Overlapping reference polygons
can produce multiple strict matches; T then discards pair information. Use
metric.GTIndex.score on the painted masks for native host evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .preprocess import Disk
from .tree import Tree, build_tree, light_tree

FEATURES = [
    "t_first", "t_last", "persistence", "split_part", "log_area", "n_children", "log_area_ratio_parent",
    "log_root_area", "depth",
    "p_mean", "p_max", "p_min", "p_std", "p_ring", "p_contrast",
    "f_mean", "f_min", "f_ring", "f_contrast", "f_std",
    "major", "minor", "elong", "orient_sin2", "orient_cos2", "perim_ratio", "fill_bbox",
    "x_r", "y_r", "rho", "abs_lon", "abs_lat", "lon_to_70",
    "crowd_60", "crowd_150", "crowd_area_150", "nearest_dist",
    "img_n_nodes_05", "img_area_05", "img_f_absdev", "img_p_mass",
]


@dataclass
class Bank:
    """One frame's candidate bank: its component tree, each node's descriptors and, on training frames, each
    node's matched and best IoU against every annotator record."""
    stem: str
    tree: Tree
    X: np.ndarray  # (n, d) float32 descriptors
    T: np.ndarray | None = None  # (n, R) matched IoU (0 when IoU <= 0.5)
    B: np.ndarray | None = None  # (n, R) best IoU (unthresholded)
    n_gt: np.ndarray | None = None  # (R,) GT filaments per record
    G: np.ndarray | None = None  # (sum n_gt,) best IoU of each GT filament over all nodes (candidate coverage)

    def totals(self, sel) -> np.ndarray:
        """Additive surrogate [TP, FP, FN, IoU sum], assuming unique matches per record."""
        sel = np.asarray(sel, int)
        t = self.T[sel] if len(sel) else np.zeros((0, len(self.n_gt)))
        tp = (t > 0).sum(0)
        return np.array([tp.sum(), (len(sel) - tp).sum(), (self.n_gt - tp).sum(), t.sum()], np.float64)


def _ring_and_boundary(lab_crop: np.ndarray, r: int = 3):
    """Ring pixels (outside every node, within r px of one) labelled by a nearby node, and boundary pixels."""
    lf = lab_crop.astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(lf, k)
    ring = (lab_crop == 0) & (dil > 0)
    ero = cv2.erode(lf, np.ones((3, 3), np.uint8))
    boundary = (lab_crop > 0) & (ero != lab_crop)
    return ring, dil.astype(np.int64), boundary


def _add_pixels(acc: dict, ext: tuple, v: np.ndarray, ps: np.ndarray, fs: np.ndarray, sx: np.ndarray,
                sy: np.ndarray, n: int) -> None:
    """Add the pixels of the nodes born at one level (node ids `v`, map values `ps`, flat-field values `fs`, frame
    coordinates `sx`, `sy`) to the running sums `acc` and to the extremes `ext` = (p max, p min, F min)."""
    for key, val in (("p", ps), ("p2", ps ** 2), ("f", fs), ("f2", fs ** 2), ("x", sx), ("y", sy), ("xx", sx ** 2),
                     ("yy", sy ** 2), ("xy", sx * sy)):
        acc[key] += np.bincount(v, weights=val, minlength=n)
    pmax, pmin, fmin = ext
    uv = np.unique(v)
    lab_v = v + 1
    pmax[uv] = ndimage.maximum(ps, lab_v, uv + 1)
    pmin[uv] = ndimage.minimum(ps, lab_v, uv + 1)
    fmin[uv] = ndimage.minimum(fs, lab_v, uv + 1)


def _add_rings(acc: dict, tree: Tree, k: int, pc: np.ndarray, fc: np.ndarray) -> None:
    """Add the rings (pixel count, map and flat-field sums) and boundary lengths of the nodes born at level `k`,
    measured on the tree's crop (`pc` the map, `fc` the flat field minus 1), to the running sums `acc`."""
    n, ids = tree.n, tree.node_of[k]
    hh, ww = tree.crop_shape
    lab_crop = np.zeros(hh * ww, np.int64)
    lab_crop[tree.support[ids >= 0]] = ids[ids >= 0] + 1
    lab_crop = lab_crop.reshape(hh, ww)
    ring, dil, boundary = _ring_and_boundary(lab_crop)
    rv = dil[ring] - 1
    keep = tree.level[rv] == k
    rv = rv[keep]
    acc["ring_n"] += np.bincount(rv, minlength=n)
    acc["ring_p"] += np.bincount(rv, weights=pc[ring][keep], minlength=n)
    acc["ring_f"] += np.bincount(rv, weights=fc[ring][keep], minlength=n)
    bv = lab_crop[boundary] - 1
    bv = bv[tree.level[bv] == k]
    acc["perim"] += np.bincount(bv, minlength=n)


def _pixel_sums(tree: Tree, pc: np.ndarray, fc: np.ndarray):
    """One pass over the tree's levels: every node's pixel sums (`acc`), its map maximum and minimum and flat-field
    minimum, each counted once at the level where the node is born, and the nodes alive at each level."""
    n, K = tree.n, len(tree.levels)
    y0, y1, x0, x1 = tree.box
    ww = tree.crop_shape[1]
    ps, fs = pc.ravel()[tree.support], fc.ravel()[tree.support]
    sy, sx = np.divmod(tree.support, ww)
    sy = sy + y0 + 0.0
    sx = sx + x0 + 0.0
    acc = {k: np.zeros(n) for k in ("p", "p2", "f", "f2", "x", "y", "xx", "yy", "xy", "ring_n", "ring_p", "ring_f",
                                     "perim")}
    pmax, pmin, fmin = np.zeros(n), np.ones(n), np.zeros(n)
    alive_by_level = []
    for k in range(K):
        ids = tree.node_of[k]
        alive_by_level.append(np.unique(ids[ids >= 0]))
        new = (ids >= 0) & (tree.level[np.maximum(ids, 0)] == k)
        if not new.any():
            continue
        _add_pixels(acc, (pmax, pmin, fmin), ids[new], ps[new], fs[new], sx[new], sy[new], n)
        _add_rings(acc, tree, k, pc, fc)
    return acc, pmax, pmin, fmin, alive_by_level


def _roots_and_depths(tree: Tree) -> tuple[np.ndarray, np.ndarray]:
    """Each node's root (its ancestor at the lowest level) and its depth below that root."""
    n, par = tree.n, tree.parent
    root = np.arange(n)
    depth = np.zeros(n)
    for v in np.argsort(tree.level, kind="stable"):
        if par[v] >= 0:
            root[v] = root[par[v]]
            depth[v] = depth[par[v]] + 1
    return root, depth


def _shape(acc: dict, a: np.ndarray):
    """Each node's centroid (x, y), the major and minor axes of its equivalent ellipse (4 sqrt of the eigenvalues of
    its pixel covariance) and the ellipse's orientation, from the pixel sums `acc` and the areas `a`."""
    mx, my = acc["x"] / a, acc["y"] / a
    cxx = acc["xx"] / a - mx**2
    cyy = acc["yy"] / a - my**2
    cxy = acc["xy"] / a - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    major, minor = 4 * np.sqrt(np.maximum(l1, 0)), 4 * np.sqrt(l2)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return mx, my, major, minor, theta


def _position(mx: np.ndarray, my: np.ndarray, disk: Disk) -> dict:
    """Each node's place on the solar disk: centroid over the disk radius, its distance from the centre, and its
    absolute longitude and latitude in degrees (and how far the longitude is from 70)."""
    xr, yr = (mx - disk.cx) / disk.r, (my - disk.cy) / disk.r
    rho = np.hypot(xr, yr)
    lat = np.arcsin(np.clip(yr, -1, 1))
    lon = np.arcsin(np.clip(xr / np.maximum(np.cos(lat), 1e-3), -1, 1))
    return dict(x_r=xr, y_r=yr, rho=rho, abs_lon=np.degrees(np.abs(lon)), abs_lat=np.degrees(np.abs(lat)),
                lon_to_70=70.0 - np.degrees(np.abs(lon)))


def _crowding(tree: Tree, alive_by_level: list, mx: np.ndarray, my: np.ndarray, a: np.ndarray) -> dict:
    """Crowding among the nodes alive at each node's first level: neighbours within 60 and 150 px, the log area
    within 150 px, and the distance to the nearest neighbour (2048 when alone)."""
    n = tree.n
    crowd60, crowd150, crowd_a, nearest = np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 2048.0)
    for k in range(len(tree.levels)):
        al = alive_by_level[k]
        born = al[tree.level[al] == k]
        if len(born) == 0 or len(al) < 2:
            continue
        d = np.hypot(mx[born, None] - mx[None, al], my[born, None] - my[None, al])
        d[born[:, None] == al[None, :]] = np.inf
        crowd60[born] = (d < 60).sum(1)
        crowd150[born] = (d < 150).sum(1)
        crowd_a[born] = np.log1p(((d < 150) * a[None, al]).sum(1))
        nearest[born] = d.min(1)
    return dict(crowd_60=crowd60, crowd_150=crowd150, crowd_area_150=crowd_a, nearest_dist=nearest)


def _image_features(tree: Tree, alive_by_level: list, a: np.ndarray, p: np.ndarray, flat: np.ndarray) -> dict:
    """Frame-level descriptors shared by all its nodes: node count and log area at the level nearest 0.5, the mean
    absolute deviation of the flat field on the disk, and the log mass of the map."""
    k05 = int(np.argmin(np.abs(np.asarray(tree.levels) - 0.5)))
    al05 = alive_by_level[k05]
    on_disk = np.abs(flat[::8, ::8] - 1.0)
    return dict(img_n_nodes_05=len(al05), img_area_05=np.log1p(a[al05].sum()) if len(al05) else 0.0,
                img_f_absdev=float(on_disk[on_disk > 0].mean()) if (on_disk > 0).any() else 0.0,
                img_p_mass=float(np.log1p(p[::4, ::4].sum())))


def _tree_features(tree: Tree, a: np.ndarray) -> dict:
    """Descriptors of each node's place in the tree: first and last level, persistence, split part, log area,
    children, log area over the parent's, log area of its root, and depth."""
    lv, K, par = np.asarray(tree.levels), len(tree.levels), tree.parent
    root, depth = _roots_and_depths(tree)
    return dict(
        t_first=lv[tree.level], t_last=lv[tree.last_level], persistence=tree.last_level - tree.level + 1,
        split_part=np.asarray(tree.kinds or (0,) * K)[tree.level],
        log_area=np.log(a), n_children=np.array([len(c) for c in tree.children()]),
        log_area_ratio_parent=np.where(par >= 0, np.log(a / a[np.maximum(par, 0)]), 0.0),
        log_root_area=np.log(a[root]), depth=depth,
    )


def _intensity_features(acc: dict, a: np.ndarray, pmax: np.ndarray, pmin: np.ndarray, fmin: np.ndarray) -> dict:
    """Descriptors of the map and flat-field values inside each node and in the ring around it, and their
    contrasts."""
    cols = dict(
        p_mean=acc["p"] / a, p_max=pmax, p_min=pmin, p_std=np.sqrt(np.maximum(acc["p2"] / a - (acc["p"] / a) ** 2, 0)),
        p_ring=acc["ring_p"] / np.maximum(acc["ring_n"], 1), f_mean=acc["f"] / a, f_min=fmin,
        f_ring=acc["ring_f"] / np.maximum(acc["ring_n"], 1),
        f_std=np.sqrt(np.maximum(acc["f2"] / a - (acc["f"] / a) ** 2, 0)),
    )
    cols["p_contrast"] = cols["p_mean"] - cols["p_ring"]
    cols["f_contrast"] = cols["f_mean"] - cols["f_ring"]
    return cols


def _shape_features(acc: dict, a: np.ndarray, major: np.ndarray, minor: np.ndarray, theta: np.ndarray) -> dict:
    """Descriptors of each node's shape: ellipse axes, elongation, orientation, perimeter over the root of the area,
    and the share of the ellipse's bounding box the node fills."""
    bbox_area = np.maximum(major * np.maximum(minor, 1.0), 1.0)
    return dict(major=major, minor=minor, elong=major / np.maximum(minor, 1.0), orient_sin2=np.sin(2 * theta),
                orient_cos2=np.cos(2 * theta), perim_ratio=acc["perim"] / np.sqrt(a), fill_bbox=a / bbox_area)


def describe(tree: Tree, p: np.ndarray, flat: np.ndarray, disk: Disk) -> np.ndarray:
    """(n, d) descriptors of every node. `p` is the full-frame filament map, `flat` the flat field F."""
    X = np.zeros((tree.n, len(FEATURES)), np.float32)
    if tree.n == 0:
        return X
    y0, y1, x0, x1 = tree.box
    pc, fc = p[y0:y1, x0:x1], flat[y0:y1, x0:x1] - 1.0
    acc, pmax, pmin, fmin, alive_by_level = _pixel_sums(tree, pc, fc)
    a = tree.area.astype(np.float64)
    mx, my, major, minor, theta = _shape(acc, a)
    cols = (_tree_features(tree, a) | _intensity_features(acc, a, pmax, pmin, fmin)
            | _shape_features(acc, a, major, minor, theta) | _position(mx, my, disk)
            | _crowding(tree, alive_by_level, mx, my, a) | _image_features(tree, alive_by_level, a, p, flat))
    for j, name in enumerate(FEATURES):
        X[:, j] = cols[name]
    return X


def match(tree: Tree, gt) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matched IoU (n, R), best IoU (n, R), GT counts (R,) and each GT filament's best IoU over all nodes, of
    every node against every record of `gt` (a `metric.GTIndex`). IoUs are float32 ratios of the integer counts,
    as the host computes them."""
    n, R = tree.n, len(gt.records)
    T = np.zeros((n, R), np.float32)
    B = np.zeros((n, R), np.float32)
    n_gt = np.array([len(r) for r in gt.records], np.int64)
    G = []
    if n == 0:
        return T, B, n_gt, np.zeros(int(n_gt.sum()), np.float32)
    y0, y1, x0, x1 = tree.box
    hh, ww = tree.crop_shape
    W = tree.shape[1]
    pos = np.full(hh * ww, -1, np.int64)
    pos[tree.support] = np.arange(len(tree.support))
    a_node = tree.area.astype(np.float32)
    for r, rec in enumerate(gt.records):
        for idx in rec:
            gy, gx = np.divmod(idx.astype(np.int64), W)
            inside = (gy >= y0) & (gy < y1) & (gx >= x0) & (gx < x1)
            q = pos[(gy[inside] - y0) * ww + (gx[inside] - x0)]
            q = q[q >= 0]
            if len(q) == 0:
                G.append(0.0)
                continue
            inter = np.zeros(n, np.float32)
            for k in range(len(tree.levels)):
                ids = tree.node_of[k][q]
                ids = ids[ids >= 0]
                ids = ids[tree.level[ids] == k]
                if len(ids):
                    c = np.bincount(ids, minlength=n)
                    nz = np.flatnonzero(c)
                    inter[nz] = c[nz]
            hit = np.flatnonzero(inter)
            iou = inter[hit] / (a_node[hit] + np.float32(len(idx)) - inter[hit])
            B[hit, r] = np.maximum(B[hit, r], iou)
            m = iou > 0.5
            T[hit[m], r] = np.maximum(T[hit[m], r], iou[m])
            G.append(float(iou.max()) if len(iou) else 0.0)
    return T, B, n_gt, np.array(G, np.float32)


def build_bank(stem: str, p: np.ndarray, flat: np.ndarray, disk: Disk, gt=None, splits: bool = False,
               levels=None) -> Bank:
    """The candidate bank of one frame: the component tree of the map `p` (`levels` when given, else the default levels;
    watershed splits with `splits`) and each node's descriptors; with `gt` (a `metric.GTIndex`), also the match arrays
    T, B, n_gt and G."""
    tree = build_tree(p, splits=splits) if levels is None else build_tree(p, levels=levels, splits=splits)
    X = describe(tree, p, flat, disk)
    bank = Bank(stem, tree, X)
    if gt is not None:
        bank.T, bank.B, bank.n_gt, bank.G = match(tree, gt)
    return bank


def save_bank(path: Path | str, bank: Bank) -> None:
    """Write a bank's descriptors, the tree arrays the scorer reads (level, last level, parent, area) and, when matched,
    T, B, n_gt and G to one compressed `.npz`."""
    t = bank.tree
    extra = {} if bank.T is None else dict(T=bank.T, B=bank.B, n_gt=bank.n_gt, G=bank.G)
    np.savez_compressed(path, X=bank.X, level=t.level, last_level=t.last_level, parent=t.parent, area=t.area,
                        **extra)


def _current_columns(X: np.ndarray) -> np.ndarray:
    """The descriptors of `FEATURES` from a bank file. Banks built before 2026-10-12 also hold four spine-map
    descriptors (columns 15-18) and, before the station indicators were dropped, six station columns at the end."""
    if X.shape[1] == len(FEATURES):
        return X
    return np.delete(X[:, :len(FEATURES) + 4], np.s_[15:19], axis=1)


def load_banks(bank_dir: Path | str) -> list[Bank]:
    """Every bank file of a folder, with pixel-free trees (enough for selection and scoring)."""
    banks = []
    for f in sorted(Path(bank_dir).glob("*.npz")):
        with np.load(f) as z:
            t = light_tree(z["level"], z["last_level"], z["parent"], z["area"])
            banks.append(Bank(f.stem, t, _current_columns(z["X"]), z.get("T"), None, z.get("n_gt"), z.get("G")))
    return banks

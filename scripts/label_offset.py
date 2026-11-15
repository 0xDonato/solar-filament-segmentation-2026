"""Is there a sub-pixel offset between the annotators' masks and the image? (Contrails 1st and 4th place, write-ups
430618 and 432998: a constant label offset breaks flip and rotation augmentation for thin structures, and the
winners trained on "symmetric" labels.)

For every ground-truth filament (one record's polygon, rasterised with pycocotools as the metric does) the script
cross-correlates the mask with (a) an out-of-fold probability map of a network trained with all eight flips and
rotations, which cannot learn a constant offset, and (b) the darkness of the flat field, 1 - F clipped at 0, over
integer shifts of up to 3 px, and refines the peak to a sub-pixel shift with a parabola along each axis. A constant
offset between labels and image shows as a mean shift away from zero, of the same sign in every station; (c) the
darkness of the raw JPEG against a 25-px Gaussian background tells whether our flat field moved the image.

Usage: label_offset.py <n frames> [run name] [synthetic dy]   (prints JSON; a synthetic dy shifts the map first,
       to calibrate the estimator)
"""
import json
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_util

from solarfil.cache import DEFAULT_CACHE, load_png
from solarfil.data import Dataset, read_image
from solarfil.folds import load_folds
from solarfil.infer import read_map_png
from solarfil.paths import RUNS

PAD = 4  # pixels around the filament's box
S = 3  # largest integer shift tried


def peak_shift(g: np.ndarray, field: np.ndarray) -> tuple[float, float] | None:
    """Sub-pixel (dx, dy) maximising sum g(y, x) * field(y + dy, x + dx); `field` has S extra pixels on each side."""
    h, w = g.shape
    c = np.empty((2 * S + 1, 2 * S + 1))
    for i, dy in enumerate(range(-S, S + 1)):
        for j, dx in enumerate(range(-S, S + 1)):
            c[i, j] = float((g * field[S + dy: S + dy + h, S + dx: S + dx + w]).sum())
    i, j = np.unravel_index(int(np.argmax(c)), c.shape)
    if i in (0, 2 * S) or j in (0, 2 * S):
        return None  # peak on the edge of the search window: no reliable sub-pixel estimate

    def vertex(a, b, d):  # parabola through (-1, a), (0, b), (1, d)
        """Offset of the parabola's vertex from the middle sample, in samples (0 for a flat triple)."""
        den = a - 2 * b + d
        return 0.0 if den == 0 else 0.5 * (a - d) / den

    return j - S + vertex(c[i, j - 1], c[i, j], c[i, j + 1]), i - S + vertex(c[i - 1, j], c[i, j], c[i + 1, j])


def main():
    """Estimate the offset on the filaments of n random frames against the map, the flat field and the raw JPEG."""
    n = int(sys.argv[1])
    run = sys.argv[2] if len(sys.argv) > 2 else "e3_effv2s_n_long_lean"
    syn = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    folds = load_folds("obs")
    stems = sorted(folds)
    rng = np.random.default_rng(0)
    pick = [stems[i] for i in rng.choice(len(stems), n, replace=False)]
    ds = Dataset()
    rows, row_stems = [], []
    for stem in pick:
        p, _ = read_map_png(RUNS / run / f"fold{folds[stem]}" / "oof" / f"{stem}.png")
        if syn:  # content moves down by syn pixels: map(y) <- map(y - syn)
            m = np.float32([[1, 0, 0], [0, 1, -syn]])
            p = cv2.warpAffine(p, m, (2048, 2048), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        flat = load_png(DEFAULT_CACHE / "flat" / f"{stem}.png").astype(np.float32) / 255.0  # F - 0.5
        dark = np.clip(0.5 - flat, 0, None)  # 1 - F where the pixel is darker than the quiet disk
        raw = cv2.GaussianBlur(read_image(ds.image_path(stem)).astype(np.float32), (0, 0), 25)
        raw_dark = np.clip(1 - read_image(ds.image_path(stem)).astype(np.float32) / np.maximum(raw, 1), 0, None)
        for rec in ds.observations[stem].records:
            for f in rec.filaments:
                g = mask_util.decode(f.rle())
                ys, xs = np.nonzero(g)
                if len(ys) < 150:
                    continue
                y0, y1 = ys.min() - PAD, ys.max() + PAD + 1
                x0, x1 = xs.min() - PAD, xs.max() + PAD + 1
                if y0 < S or x0 < S or y1 > 2048 - S or x1 > 2048 - S:
                    continue
                gc = g[y0:y1, x0:x1].astype(np.float32)
                m = peak_shift(gc, p[y0 - S: y1 + S, x0 - S: x1 + S])
                d = peak_shift(gc, dark[y0 - S: y1 + S, x0 - S: x1 + S])
                r = peak_shift(gc, raw_dark[y0 - S: y1 + S, x0 - S: x1 + S])
                if m is None or d is None or r is None:
                    continue
                rows.append((stem[-2], len(ys), *m, *d, *r))
                row_stems.append(stem)
    a = np.array([r[2:] for r in rows])
    out = dict(frames=n, filaments=len(rows))
    for j, nm in enumerate(["map_dx", "map_dy", "dark_dx", "dark_dy", "raw_dx", "raw_dy"]):
        v = a[:, j]
        lo, hi = np.percentile(v, [10, 90])
        t = v[(v >= lo) & (v <= hi)]
        out[nm] = dict(mean=round(float(v.mean()), 3), trimmed=round(float(t.mean()), 3),
                       median=round(float(np.median(v)), 3), se=round(float(v.std() / np.sqrt(len(v))), 3))
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r[2:])
    out["median_by_station"] = {k: [len(v)] + [round(float(x), 2) for x in np.median(v, axis=0)]
                                for k, v in sorted(by.items())}
    by_year = {}
    for r, stem in zip(rows, row_stems):
        by_year.setdefault(stem[:4], []).append(r[2:])
    out["median_by_year"] = {k: [len(v)] + [round(float(x), 2) for x in np.median(v, axis=0)]
                             for k, v in sorted(by_year.items())}
    print(json.dumps(out))


if __name__ == "__main__":
    main()

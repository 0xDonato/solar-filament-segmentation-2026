"""An image-quality failure audit of the released system on the organizers' anomaly classes.

The classes are those GONG's H-alpha anomaly study labels (Khazaei et al., arXiv 2509.14472, Sec. III): objects
obstructing the disk, shadow (clouds), an uncentred disk, oversaturation and the absence of normal texture. Every
indicator is predefined and simple, and is computed from each competition JPEG alone (train 707, test 180) with the
fitted disk of `cache/v1/disks.json`; nothing is learned and nothing reads an annotation or a prediction.

Indicators (the image is divided by its centre-to-limb profile, the medians of 100 annuli of r/R < 0.98 on the
4x-downscaled frame, to give the *ratio* image; "disk" below is r/R < 0.98 unless stated):

- uncentred: `centre_offset_px`, the fitted centre's distance from the nominal (1023.1, 1024.1); `radius_dev_px`,
  |r - median r over the 887 frames|; `edge_margin_px`, the disk's smallest distance to the frame edge.
- oversaturation: `clipped_frac`, the share of full-resolution disk pixels at 250 or above; `bright_frac`, the share
  of the disk whose ratio, smoothed at 16 px (full resolution), is above 1.4.
- clouds: `nonuniformity`, the 95th minus the 5th percentile, over r/R < 0.9, of the large-scale ratio field (a
  masked Gaussian blur at 90 px of the ratio with features beyond 2.5 robust sigma left out, as the pipeline's own
  flat field builds it).
- missing texture: `hf_energy`, the robust standard deviation of the full-resolution high-pass (image minus its
  3 px Gaussian blur) over r/R < 0.9, divided by the disk median; `low_contrast_frac`, the share of 64 px blocks
  inside r/R < 0.9 whose high-pass robust deviation (same normalisation) is below the 5th percentile of all blocks of
  all 887 frames.
- obstruction: `dark_frac`, the share of the disk whose ratio, smoothed at 16 px, is below 0.6; `limb_outlier_frac`,
  the share of 360 limb rays whose steepest drop lies more than 3 px from the fitted circle.
- grid (the study's own indicator, without its labels): `cell_outliers`, the number of 16 x 16 grid cells lying
  wholly inside the disk whose mean ratio falls outside that cell's 1st-99th percentile range over the 887 frames.

Thresholds, fixed before any error was read. *Flag*: an indicator beyond its 95th percentile over all 887 frames
(its 5th for `hf_energy`, where low is bad), strictly. *Severe*: a physical rule: offset > 0.02 R or the disk
crossing the frame edge; clipped share > 0.1% or bright share > 1%; nonuniformity > 0.3; `hf_energy` below half the
median frame's or more than half the blocks low-contrast; dark share > 1% or more than 10% of limb rays off the
circle; more than 20 grid cells out of range. A class is flagged (severe) when any of its indicators is.

Errors: the release (`ens_v5_inc5`) out of fold at tau 0.26, random partitions 15-19 and group 10-14, from
`report_intervals.py`'s cached frame pass (never random 25-29 or group 20-24): PQ, 1:0 and 0:1 per record on the
flagged and unflagged training frames, with 90% frame-bootstrap intervals (and group-cluster ones) on the flagged
minus unflagged difference (`report_intervals.weights`, 2,000 draws, seed 0), and Spearman's rho of each continuous
indicator against per-frame PQ (the five partitions' mean counts), with a 90% frame-bootstrap interval.

Blinded audit: 30 frames drawn with seed 0 from all 887, 15 flagged in some class and 15 flagged in none, shuffled,
written as 512 px PNGs of the raw JPEG named Q01-Q30 with a blank form (code, class seen, severity 0-2, note); the
key is kept apart, outside the folder the reader is given.

Outputs under --out-dir: `quality_indicators.csv` (one row per frame), `quality_audit.csv` (the blank form, also in
`quality_audit/`), `quality_audit/` (PNGs and form), `quality_audit_key.csv`, `report_quality.md` (thresholds, flag
counts, the error table, the Spearman table and one drafted sentence), and the indicator pass in `quality_cache/`.

Usage: uv run python scripts/report_quality.py
       [--out-dir report] [--workers 3] [--boot 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

import ablation_metrics as am
import report_intervals as ri
from solarfil.data import Dataset, read_image
from solarfil.parallel import pool
from solarfil.paths import REPO
from solarfil.preprocess import NOMINAL_CENTRE, Disk
from solarfil.rule import pq

TAU = 0.26
DOWN = 4  # the downscale of the ratio image
RHO_DISK, RHO_IN = 0.98, 0.9
N_ANNULI = 100
SMOOTH_PX = 16.0  # full-resolution sigma of the bright and dark maps
BG_PX, CLIP = 90.0, 2.5  # the large-scale field, as the pipeline's flat field
HP_PX, BLOCK = 3.0, 64  # high-pass sigma and texture block size (full resolution)
GRID = 16
DISKS_DIR = str(REPO / "cache" / "v1")  # disks.json; FILAMENT_CACHE overrides
CLASSES = {  # class: [(indicator, tail)], tail "high" or "low"
    "uncentred": [("centre_offset_px", "high"), ("radius_dev_px", "high")],
    "oversaturation": [("clipped_frac", "high"), ("bright_frac", "high")],
    "clouds": [("nonuniformity", "high")],
    "texture": [("hf_energy", "low"), ("low_contrast_frac", "high")],
    "obstruction": [("dark_frac", "high"), ("limb_outlier_frac", "high")],
    "grid": [("cell_outliers", "high")],
}
INDICATORS = [i for v in CLASSES.values() for i, _ in v]
_G: dict = {}


# --- one frame -------------------------------------------------------------------------------------------------------

def masked_blur(v: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised Gaussian convolution of v with weights w; 1 where no weighted pixel is near."""
    num = cv2.GaussianBlur(v * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0).astype(np.float32)


def ratio_image(img: np.ndarray, disk: Disk) -> tuple[np.ndarray, np.ndarray]:
    """The downscaled frame divided by its centre-to-limb profile (1 off the disk), and r/R on that grid."""
    small = cv2.resize(img.astype(np.float32), (img.shape[1] // DOWN, img.shape[0] // DOWN),
                       interpolation=cv2.INTER_AREA)
    rho = disk.rho(small.shape, scale=DOWN)
    on = rho < RHO_DISK
    edges = np.linspace(0, RHO_DISK, N_ANNULI + 1)
    idx = np.clip(np.digitize(rho, edges) - 1, 0, N_ANNULI - 1)
    prof = ndimage.median(small, labels=np.where(on, idx + 1, 0), index=np.arange(1, N_ANNULI + 1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(prof)
    clv = np.interp(rho, centres[ok], np.asarray(prof)[ok])
    return np.where(on, small / np.maximum(clv, 1.0), 1.0).astype(np.float32), rho


def large_scale(ratio: np.ndarray, on: np.ndarray) -> np.ndarray:
    """The slowly varying field of the ratio image, features beyond CLIP robust sigma left out (two passes)."""
    bg = np.ones_like(ratio)
    for _ in range(2):
        r = ratio / bg
        med = np.median(r[on])
        sig = 1.4826 * np.median(np.abs(r[on] - med)) + 1e-6
        w = (on & (np.abs(r - med) < CLIP * sig)).astype(np.float32)
        bg = masked_blur(ratio, w, BG_PX / DOWN)
    return bg


def limb_outliers(img: np.ndarray, disk: Disk, n_rays: int = 360) -> float:
    """Share of limb rays whose steepest intensity drop lies more than 3 px from the fitted circle."""
    sm = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(0.85 * disk.r, 1.12 * disk.r, 0.5)
    xs = disk.cx + np.cos(theta)[:, None] * radii[None, :]
    ys = disk.cy + np.sin(theta)[:, None] * radii[None, :]
    prof = ndimage.map_coordinates(sm, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    k = np.argmin(np.gradient(prof, axis=1), axis=1)
    return float(np.mean(np.abs(radii[k] - disk.r) > 3.0))


def robust_sd(x: np.ndarray) -> float:
    """1.4826 times the median absolute deviation."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def texture(img: np.ndarray, disk: Disk) -> tuple[float, np.ndarray]:
    """`hf_energy` and the high-pass robust deviation of every 64 px block wholly inside r/R < 0.9, both divided by
    the disk median."""
    f = img.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), HP_PX)
    rho = disk.rho(img.shape)
    inner = rho < RHO_IN
    med = float(np.median(f[rho < RHO_DISK]))
    hf = robust_sd(hp[inner]) / med
    n = img.shape[0] // BLOCK
    blocks = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK))
            if inner[sl].all():
                blocks.append(robust_sd(hp[sl]) / med)
    return hf, np.array(blocks, np.float32)


def grid_cells(ratio: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Mean ratio of each GRID x GRID cell, NaN for cells not wholly inside the disk."""
    c = ratio.shape[0] // GRID
    out = np.full((GRID, GRID), np.nan, np.float32)
    for i in range(GRID):
        for j in range(GRID):
            sl = (slice(i * c, (i + 1) * c), slice(j * c, (j + 1) * c))
            if (rho[sl] < RHO_DISK).all():
                out[i, j] = ratio[sl].mean()
    return out


def frame_job(job) -> tuple[str, dict, np.ndarray, np.ndarray]:
    """Every per-frame indicator of one JPEG, its texture blocks and its grid cells."""
    stem, path, d = job
    img = read_image(path)
    disk = Disk(d["cx"], d["cy"], d["r"])
    ratio, rho = ratio_image(img, disk)
    on, inner = rho < RHO_DISK, rho < RHO_IN
    sm = masked_blur(ratio, on.astype(np.float32), SMOOTH_PX / DOWN)
    bg = large_scale(ratio, on)
    full_on = disk.rho(img.shape) < RHO_DISK
    hf, blocks = texture(img, disk)
    lo, hi = np.percentile(bg[inner], [5, 95])
    ind = dict(centre_offset_px=float(np.hypot(disk.cx - NOMINAL_CENTRE[0], disk.cy - NOMINAL_CENTRE[1])),
               radius_px=disk.r,
               edge_margin_px=float(min(disk.cx, disk.cy, img.shape[1] - 1 - disk.cx, img.shape[0] - 1 - disk.cy)
                                    - disk.r),
               clipped_frac=float(np.mean(img[full_on] >= 250)), bright_frac=float(np.mean(sm[on] > 1.4)),
               nonuniformity=float(hi - lo), hf_energy=hf, dark_frac=float(np.mean(sm[on] < 0.6)),
               limb_outlier_frac=limb_outliers(img, disk), disk_median=float(np.median(img[full_on])))
    return stem, ind, blocks, grid_cells(ratio, rho)


# --- the indicator pass ----------------------------------------------------------------------------------------------

def indicator_pass(ds: Dataset, disks: dict, workers: int, cache: Path) -> tuple[list[str], dict]:
    """Every frame through `frame_job` (cached): the stems (train then test) and {indicator: (frames,)} with the
    pooled indicators (`radius_dev_px`, `low_contrast_frac`, `cell_outliers`) filled in."""
    stems = ds.train_stems + ds.test_stems
    f = cache / "quality_pass.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            ind, blocks, cells = z["ind"].item(), list(z["blocks"]), z["cells"]
    else:
        jobs = [(s, str(ds.image_path(s)), disks[s]) for s in stems]
        got = {}
        with pool(workers, spawn=True) as w:
            for i, (stem, d, b, c) in enumerate(w.imap_unordered(frame_job, jobs, chunksize=4)):
                got[stem] = (d, b, c)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(jobs)} frames", flush=True)
        ind = {k: np.array([got[s][0][k] for s in stems]) for k in got[stems[0]][0]}
        blocks = [got[s][1] for s in stems]
        cells = np.stack([got[s][2] for s in stems])
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(f, ind=np.array(ind, dtype=object), blocks=np.array(blocks, dtype=object), cells=cells)
    ind["radius_dev_px"] = np.abs(ind["radius_px"] - np.median(ind["radius_px"]))
    cut = np.percentile(np.concatenate(blocks), 5)
    ind["low_contrast_frac"] = np.array([np.mean(b < cut) if len(b) else np.nan for b in blocks])
    lo, hi = np.nanpercentile(cells, 1, axis=0), np.nanpercentile(cells, 99, axis=0)
    with np.errstate(invalid="ignore"):
        ind["cell_outliers"] = ((cells < lo) | (cells > hi)).sum((1, 2)).astype(float)
    return stems, ind


# --- flags -----------------------------------------------------------------------------------------------------------

def tail_flag(x: np.ndarray, tail: str) -> tuple[np.ndarray, float]:
    """Beyond the 95th percentile (5th for a low tail) over all frames, strictly; and the threshold."""
    if tail == "low":
        t = float(np.percentile(x, 5))
        return x < t, t
    t = float(np.percentile(x, 95))
    return x > t, t


def severe_flags(ind: dict) -> dict:
    """The physical rule of each class (see the module docstring)."""
    r = ind["radius_px"]
    return dict(uncentred=(ind["centre_offset_px"] > 0.02 * r) | (ind["edge_margin_px"] < 0),
                oversaturation=(ind["clipped_frac"] > 0.001) | (ind["bright_frac"] > 0.01),
                clouds=ind["nonuniformity"] > 0.3,
                texture=(ind["hf_energy"] < 0.5 * np.median(ind["hf_energy"])) | (ind["low_contrast_frac"] > 0.5),
                obstruction=(ind["dark_frac"] > 0.01) | (ind["limb_outlier_frac"] > 0.1),
                grid=ind["cell_outliers"] > 20)


def flags(ind: dict) -> tuple[dict, dict, dict]:
    """{class: flagged}, {class: severe} (each with "any") and {indicator: threshold}."""
    fl, thr = {}, {}
    for c, items in CLASSES.items():
        m = np.zeros(len(ind[items[0][0]]), bool)
        for name, tail in items:
            f, t = tail_flag(ind[name], tail)
            m |= f
            thr[name] = t
        fl[c] = m
    sv = severe_flags(ind)
    fl["any"] = np.any([fl[c] for c in CLASSES], 0)
    sv["any"] = np.any([sv[c] for c in CLASSES], 0)
    return fl, sv, thr


# --- errors ----------------------------------------------------------------------------------------------------------

def host_rows(sel: dict, scheme: str) -> tuple[list[np.ndarray], np.ndarray]:
    """The scheme's five partitions' per-frame host rows at TAU, and their mean."""
    first = ri.SCHEMES[scheme]
    reps = [sel[scheme, r, TAU]["host"] for r in range(first, first + am.N_PARTITIONS)]
    return reps, np.mean(reps, 0)


def counts(h: np.ndarray) -> dict:
    """PQ and 1:0, 0:1 per record from summed host rows (last axis 12)."""
    comps = h[..., 7]
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(pq=pq(np.moveaxis(h[..., :4], -1, 0)), missed_1_0=h[..., 8] / comps,
                    spurious_0_1=h[..., 9] / comps)


def flag_table(sel: dict, fl: dict, sv: dict, Wt: dict) -> list[dict]:
    """Per tier, class and scheme: flagged and unflagged levels, and flagged minus unflagged with 90% frame and
    group-cluster intervals."""
    out = []
    for scheme in ri.SCHEMES:
        reps, H = host_rows(sel, scheme)
        for tier, F in (("flag", fl), ("severe", sv)):
            for c, m in F.items():
                n = int(m.sum())
                if n == 0:
                    out.append(dict(tier=tier, cls=c, scheme=scheme, frames=0))
                    continue
                a = [counts(r[m].sum(0)) for r in reps]
                b = [counts(r[~m].sum(0)) for r in reps]
                for k in a[0]:
                    va, vb = float(np.mean([x[k] for x in a])), float(np.mean([x[k] for x in b]))
                    fa, fb = counts(Wt["frame"][:, m] @ H[m]), counts(Wt["frame"][:, ~m] @ H[~m])
                    ca, cb = counts(Wt["cluster"][:, m] @ H[m]), counts(Wt["cluster"][:, ~m] @ H[~m])
                    out.append(dict(tier=tier, cls=c, scheme=scheme, frames=n, measure=k, flagged=va,
                                    unflagged=vb, diff=va - vb, ci=ri.ci(fa[k] - fb[k]), cl=ri.ci(ca[k] - cb[k])))
    return out


def spearman_table(sel: dict, ind: dict, n_train: int, n_boot: int) -> list[dict]:
    """Spearman's rho of each continuous indicator against per-frame PQ (frames with a defined PQ), with a 90%
    frame-bootstrap interval (seed 0)."""
    out = []
    rng = np.random.default_rng(0)
    for scheme in ri.SCHEMES:
        _, H = host_rows(sel, scheme)
        y = counts(H)["pq"]
        ok = np.isfinite(y)
        draws = rng.integers(0, ok.sum(), (n_boot, ok.sum()))
        for name in INDICATORS:
            x = ind[name][:n_train][ok]
            yy = y[ok]
            rho = float(spearmanr(x, yy).statistic)
            bs = np.array([spearmanr(x[d], yy[d]).statistic for d in draws])
            out.append(dict(scheme=scheme, indicator=name, frames=int(ok.sum()), rho=rho, ci=ri.ci(bs)))
    return out


# --- audit and outputs -----------------------------------------------------------------------------------------------

def audit(ds: Dataset, stems: list[str], fl: dict, sv: dict, out_dir: Path) -> list[dict]:
    """The blinded sample: 15 frames flagged in some class and 15 in none (seed 0), shuffled, as 512 px PNGs with a
    blank form; returns the key rows."""
    rng = np.random.default_rng(0)
    yes, no = np.flatnonzero(fl["any"]), np.flatnonzero(~fl["any"])
    pick = np.concatenate([rng.choice(yes, 15, replace=False), rng.choice(no, 15, replace=False)])
    rng.shuffle(pick)
    folder = out_dir / "quality_audit"
    folder.mkdir(parents=True, exist_ok=True)
    key = []
    for k, i in enumerate(pick):
        code = f"Q{k + 1:02d}"
        img = read_image(ds.image_path(stems[i]))
        cv2.imwrite(str(folder / f"{code}.png"), cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA))
        key.append(dict(code=code, stem=stems[i], split="train" if stems[i] in ds.observations else "test",
                        flagged=" ".join(c for c in CLASSES if fl[c][i]) or "none",
                        severe=" ".join(c for c in CLASSES if sv[c][i]) or "none"))
    form = [dict(code=r["code"], class_seen="", severity_0_2="", note="") for r in key]
    for p in (out_dir / "quality_audit.csv", folder / "quality_audit.csv"):
        write_rows(p, form)
    return key


def write_rows(path: Path, rows: list[dict]) -> None:
    """A list of dicts as CSV (columns of the first row)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def indicator_rows(ds: Dataset, stems: list[str], ind: dict, fl: dict, sv: dict) -> list[dict]:
    """One row per frame: split, every indicator and every class's flag and severe flag."""
    rows = []
    for i, s in enumerate(stems):
        r = dict(stem=s, split="train" if s in ds.observations else "test")
        r.update({k: float(v[i]) for k, v in ind.items()})
        r.update({f"flag_{c}": int(m[i]) for c, m in fl.items()})
        r.update({f"severe_{c}": int(m[i]) for c, m in sv.items()})
        rows.append(r)
    return rows


def fmt_ci(x: tuple) -> str:
    """'[lo, hi]' with signs."""
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


def sentence(table: list[dict], n_flag: dict, n_test: dict, sv_train: dict, max_offset: float) -> str:
    """The drafted report sentence from the random scheme's any-class row."""
    r = {x["measure"]: x for x in table if x.get("measure") and x["scheme"] == "random" and x["tier"] == "flag"
         and x["cls"] == "any"}
    p = r["pq"]
    q = next(x for x in table if x.get("measure") == "pq" and x["scheme"] == "random" and x["tier"] == "severe"
             and x["cls"] == "any")
    absent = [c for c in CLASSES if sv_train[c] == 0]
    return (f"On predefined image-quality indicators for the five anomaly classes of GONG's H-alpha quality study "
            f"(2509.14472), each flagged beyond its 95th percentile over all 887 frames, the {n_flag['any']} "
            f"flagged training frames read out-of-fold PQ {p['flagged']:.3f} against {p['unflagged']:.3f} on the "
            f"rest (difference {p['diff']:+.3f}, 90% frame interval {fmt_ci(p['ci'])}); {n_test['any']} of the "
            f"180 test frames are flagged. By physical rules no training frame is severely "
            f"{', '.join(absent) if absent else '(none absent)'} (the largest centre offset is {max_offset:.1f} px "
            f"of a 904 px radius), and the {q['frames']} frames the other classes' physical rules call severe read "
            f"PQ {q['flagged']:.3f} against {q['unflagged']:.3f} (difference {q['diff']:+.3f}, "
            f"{fmt_ci(q['ci'])}).")


def write_md(path: Path, thr: dict, fl: dict, sv: dict, n_train: int, table: list[dict], sp: list[dict],
             text: str) -> None:
    """The summary: thresholds, flag counts, errors, Spearman and the sentence."""
    tr, te = slice(0, n_train), slice(n_train, None)
    L = ["# Image-quality failure audit of v6", "",
         "Classes from 2509.14472 (Sec. III). Indicators and thresholds are defined in `scripts/report_quality.py`'s "
         "docstring and were fixed before any error was read. v6 out of fold at tau 0.26; random 15-19, group "
         "10-14; random 25-29 and group 20-24 unread.", "", "## Thresholds (flag tier: 95th/5th percentile of 887)",
         "", "| indicator | tail | threshold |", "|---|---|---|"]
    L += [f"| {n} | {t} | {thr[n]:.4g} |" for c in CLASSES.values() for n, t in c]
    L += ["", "## Flag counts", "", "| class | flag train | flag test | severe train | severe test |",
          "|---|---|---|---|---|"]
    L += [f"| {c} | {int(fl[c][tr].sum())} | {int(fl[c][te].sum())} | {int(sv[c][tr].sum())} | "
          f"{int(sv[c][te].sum())} |" for c in fl]
    L += ["", "## Flagged minus unflagged (training frames, out of fold)", "",
          "PQ pooled over a set's frames, 1:0 and 0:1 per record; intervals on fewer than 10 flagged frames say "
          "little (one frame's bootstrap only reweights it).", "",
          "| tier | class | scheme | frames | measure | flagged | unflagged | diff | 90% frame | 90% group |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if not r["frames"]:
            L.append(f"| {r['tier']} | {r['cls']} | {r['scheme']} | 0 | - | - | - | - | - | - |")
            continue
        L.append(f"| {r['tier']} | {r['cls']} | {r['scheme']} | {r['frames']} | {r['measure']} | "
                 f"{r['flagged']:.3f} | {r['unflagged']:.3f} | {r['diff']:+.3f} | {fmt_ci(r['ci'])} | "
                 f"{fmt_ci(r['cl'])} |")
    L += ["", "## Spearman of each indicator against per-frame PQ", "",
          "| scheme | indicator | frames | rho | 90% frame |", "|---|---|---|---|---|"]
    L += [f"| {r['scheme']} | {r['indicator']} | {r['frames']} | {r['rho']:+.3f} | {fmt_ci(r['ci'])} |" for r in sp]
    L += ["", "## Drafted sentence", "", text, ""]
    path.write_text("\n".join(L))


def main():
    """Compute the indicators, flag, read the release's out-of-fold errors, and write the tables, the audit and the
    summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ri.add_release_args(ap)
    a = ap.parse_args()
    cache_in = ri.apply_release_args(a)
    ds = Dataset()
    disks = json.loads((Path(os.environ.get("FILAMENT_CACHE", DISKS_DIR)) / "disks.json").read_text())
    stems, ind = indicator_pass(ds, disks, a.workers, a.out_dir / "quality_cache")
    fl, sv, thr = flags(ind)  # fixed before any error is read
    n_train = len(ds.train_stems)
    write_rows(a.out_dir / "quality_indicators.csv", indicator_rows(ds, stems, ind, fl, sv))
    write_rows(a.out_dir / "quality_audit_key.csv", audit(ds, stems, fl, sv, a.out_dir))
    scores = ri.load_scores(cache_in)
    train = sorted(scores["random", am.FIRST_RANDOM])
    assert train == stems[:n_train]
    sel, _, _ = ri.run_pass(train, scores, a.workers, a.out_dir / "intervals_cache", "")
    obs = ds.observations
    Wt = ri.weights(train, [obs[s].group for s in train], np.ones(n_train, bool), a.boot)
    fl_tr = {c: m[:n_train] for c, m in fl.items()}
    sv_tr = {c: m[:n_train] for c, m in sv.items()}
    table = flag_table(sel, fl_tr, sv_tr, Wt)
    sp = spearman_table(sel, ind, n_train, a.boot)
    _, H = host_rows(sel, "random")
    print("sanity PQ random tau 0.26:", round(float(counts(H.sum(0))["pq"]), 5))
    n_flag = {c: int(m.sum()) for c, m in fl_tr.items()}
    n_test = {c: int(m[n_train:].sum()) for c, m in fl.items()}
    text = sentence(table, n_flag, n_test, {c: int(m.sum()) for c, m in sv_tr.items()},
                    float(ind["centre_offset_px"].max()))
    write_md(a.out_dir / "report_quality.md", thr, fl, sv, n_train, table, sp, text)
    print(text)


if __name__ == "__main__":
    main()

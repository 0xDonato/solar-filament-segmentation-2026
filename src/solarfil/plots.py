"""The judges' figures (host self-evaluation notebook): IoU and Dice distributions of overlapping pairs, and the
one-to-many / many-to-one counts. Drawn from a `metric.Report`.

The report's figures (`reference_figure`, `pq_intervals_figure`, `morphology_figure`) follow the ACM page (acmart
sigconf, the organizers' template): 8 pt type, single-column or full width, and one colour-blind-safe colour per
series (Okabe-Ito) with a line style, marker or hatch as a second cue; the second annotator is always vermillion and
dotted."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .metric import Report
from .reference import HOST_BINS, legend_counts


def _hist(ax, values: np.ndarray, title: str, xlabel: str) -> None:
    bins = np.linspace(0, 1, 51)
    counts, edges = np.histogram(values, bins=bins)
    prob = counts / max(counts.sum(), 1)
    ax.bar((edges[:-1] + edges[1:]) / 2, prob, width=0.02, color="#2a78d6", edgecolor="#1b4f8a", alpha=0.85)
    if len(values):
        ax.axvline(values.mean(), color="#e34948", ls="--", lw=1.4, label=f"mean {values.mean():.3f}")
        ax.legend(fontsize=8, frameon=False)
    ax.set(title=title, xlabel=xlabel, ylabel="probability")
    ax.grid(axis="y", ls=":", alpha=0.5)


def _degrees(ax, rep: Report) -> None:
    g, p = rep.gt_degree, rep.pred_degree
    for d, c in g.items():
        ax.bar(d + 0.5, c, width=0.8, color="#e34948" if d == 0 else "#008300", alpha=0.8)
    for d, c in p.items():
        ax.bar(-(d + 0.5), c, width=0.8, color="#eda100" if d == 0 else "#4a3aa7", alpha=0.8)
    ticks = sorted([d + 0.5 for d in g] + [-(d + 0.5) for d in p])
    names = [f"1:{int(t - 0.5)}" if t > 0 else f"{int(-t - 0.5)}:1" for t in ticks]
    ax.set_xticks(ticks, names, fontsize=8, rotation=45)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="GT 1:n (right) and predictions n:1 (left), IoU > 0", ylabel="count")
    ax.grid(axis="y", ls=":", alpha=0.5)


def judges_figure(rep: Report, path, title: str = "") -> None:
    """Save the judges' three panels of a report: the IoU and Dice histograms of overlapping pairs and the 1:n / n:1
    degree bars."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained")
    _hist(ax[0], rep.iou_values, "IoU of overlapping pairs", "IoU")
    _hist(ax[1], rep.dice_values, "Dice of overlapping pairs", "Dice")
    _degrees(ax[2], rep)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


PALETTE = [(255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 215, 0), (186, 85, 211), (0, 206, 209),
           (255, 140, 0), (154, 205, 50), (255, 105, 180), (100, 149, 237)]


def overlay(img: np.ndarray, masks: list[np.ndarray], path, gt: list[list[np.ndarray]] | None = None,
            box: tuple[int, int, int, int] | None = None, scale: float = 0.5) -> None:
    """Draw the outline of every predicted mask (one colour each) on the grayscale frame, and optionally the
    outlines of each annotator record's filaments (white, thin). `box` = (x0, y0, x1, y1) crops the view."""
    import cv2

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for rec in gt or []:
        for m in rec:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, (255, 255, 255), 1)
    for i, m in enumerate(masks):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, PALETTE[i % len(PALETTE)], 2)
    if box is not None:
        x0, y0, x1, y1 = box
        rgb = rgb[y0:y1, x0:x1]
    if scale != 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def msiou_figure(model: np.ndarray, human: np.ndarray, sizes, path, title: str = "") -> None:
    """Multiscale IoU of matched filaments, the model's against a second annotator's (rows are MIoU, pixel IoU and r at each cell size). Left: the mean share r of the annotator's edge cells
    that the other outline touches, per cell size. Right: mean MIoU at equal pixel IoU."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    bins = np.linspace(0.5, 1, 11)
    for rows, name, c in ((model, "model", "#2a78d6"), (human, "second annotator", "#e34948")):
        ax[0].plot(sizes, rows[:, 2:].mean(axis=0), marker="o", color=c, label=f"{name} (n {len(rows):,})")
        k = np.clip(np.digitize(rows[:, 1], bins) - 1, 0, len(bins) - 2)
        x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1) if (k == i).sum() >= 20]
        y = [rows[k == i, 0].mean() for i in range(len(bins) - 1) if (k == i).sum() >= 20]
        ax[1].plot(x, y, marker="o", color=c, label=name)
    ax[0].set_xscale("log", base=2)
    ax[0].set(title="Edge cells of the annotator's outline touched, per cell size", xlabel="cell size (px)",
              ylabel="mean r")
    ax[1].set(title="Multiscale IoU at equal pixel IoU (bins with 20+ pairs)", xlabel="pixel IoU",
              ylabel="mean multiscale IoU")
    for a in ax:
        a.grid(ls=":", alpha=0.5)
        a.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


FULL_WIDTH, COLUMN_WIDTH = 7.0, 3.33  # ACM acmart sigconf page, inches (506 pt and 241 pt)
PAPER = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#555555", "grid.linewidth": 0.4,
         "grid.color": "#dddddd", "savefig.dpi": 300}
MODEL_STYLES = ({"color": "#0072B2", "ls": "-", "hatch": None, "marker": "o"},
                {"color": "#56B4E9", "ls": "--", "hatch": "////", "marker": "s"},
                {"color": "#009E73", "ls": "-.", "hatch": "\\\\\\\\", "marker": "^"},
                {"color": "#CC79A7", "ls": (0, (4, 1, 1, 1)), "hatch": "xxxx", "marker": "D"})
ANNOTATOR_STYLE = {"color": "#D55E00", "ls": ":", "hatch": "....", "marker": "v"}
DEGREE_TICKS = ("≥3:1", "2:1", "1:1", "0:1", "1:0", "1:1", "1:2", "1:≥3")


@dataclass
class ReferenceSeries:
    """What `reference_figure` draws of one series: the host's 50-bin counts and the mean of the IoU and of the Dice
    of the overlapping pairs, the filament (1:n) and mask (n:1) degrees, and the number of comparisons. It comes from
    a `metric.Report` or from the summary that `reference.host_summary` stores in `judges_reference.py`'s JSON, so
    the figure can be drawn again from that file alone."""

    iou_counts: np.ndarray
    iou_mean: float
    dice_counts: np.ndarray
    dice_mean: float
    gt_degree: Counter
    pred_degree: Counter
    comparisons: int

    @classmethod
    def of_report(cls, rep: Report) -> ReferenceSeries:
        """From a `metric.Report`."""
        iou, dice = rep.iou_values, rep.dice_values
        return cls(np.histogram(iou, bins=HOST_BINS)[0], float(iou.astype(np.float64).mean()),
                   np.histogram(dice, bins=HOST_BINS)[0], float(dice.astype(np.float64).mean()), rep.gt_degree,
                   rep.pred_degree, len(rep.ious))

    @classmethod
    def of_summary(cls, host: dict) -> ReferenceSeries:
        """From a `host_summary` dict (its means are rounded to four decimals)."""
        def degrees(d: dict) -> Counter:
            """A degree count dict from JSON (string keys) as a Counter."""
            return Counter({int(k): int(v) for k, v in d.items()})

        return cls(np.asarray(host["iou"]["hist50"]), float(host["iou"]["mean"]), np.asarray(host["dice"]["hist50"]),
                   float(host["dice"]["mean"]), degrees(host["filament_degree"]), degrees(host["mask_degree"]),
                   int(host["comparisons"]))


def _outline(ax, counts: np.ndarray, mean: float, style: dict) -> float:
    """The host's normalised 50-bin histogram as an outline, and the mean as a vertical line; returns the mean."""
    ax.stairs(counts / max(counts.sum(), 1), HOST_BINS, color=style["color"], ls=style["ls"], lw=1.0)
    ax.axvline(mean, color=style["color"], ls=style["ls"], lw=0.7)
    return mean


def _degree_counts(c: Counter) -> np.ndarray:
    """Counts of degree 0, 1, 2 and 3 or more."""
    return np.array([c[0], c[1], c[2], sum(v for d, v in c.items() if d >= 3)], np.float64)


def _relation_bars(ax, series: list[tuple[ReferenceSeries, dict]]) -> None:
    """The host's split bar chart with one bar per series in each group, per comparison, on a log scale."""
    width = 0.8 / len(series)
    x = np.arange(4) + 0.5
    lowest = np.inf
    for i, (rep, style) in enumerate(series):
        dx = (i - (len(series) - 1) / 2) * width
        for pos, degrees in ((x, rep.gt_degree), (-x, rep.pred_degree)):
            share = _degree_counts(degrees) / rep.comparisons
            ax.bar(pos + dx, share, width, color=style["color"], hatch=style["hatch"], hatchcolor="white",
                   edgecolor="white", lw=0.4)
            lowest = min(lowest, share[share > 0].min())
    ax.set_yscale("log")
    ax.set_ylim(bottom=lowest / 2)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_xticks(np.r_[-x[::-1], x], DEGREE_TICKS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set(title="(c) Relations (IoU > 0)", xlabel="masks n:1   |   filaments 1:n", ylabel="per comparison")
    ax.grid(axis="y")


def reference_figure(models: list[tuple[str, Report | ReferenceSeries]], annotator: Report | ReferenceSeries,
                     path, font_scale: float = 1.0) -> None:
    """The report's Fig. 2 (a)-(c): the host's three panels, each selection beside the second annotator.

    `models` pairs a label with the report (or `ReferenceSeries`) of a selection scored against every record of its
    frames; `annotator` scores every annotator against every other annotator of the same frames
    (`solarfil.reference`). (a), (b): the
    host's normalised 50-bin histograms of the IoU and Dice of every overlapping pair, as outlines, with each
    series' mean as a vertical line. (c): the host's split bar chart of relations per comparison (record or ordered
    pair of records): the mask degree 0:1, 1:1, 2:1, >=3:1 on the left, the filament degree 1:0, 1:1, 1:2, 1:>=3 on
    the right, on a log scale so that the splits and merges, a few hundredths of the 1:1 bars, stay readable. The
    legend gives each series' comparisons and the four counts: 1:0, 0:1, splits and merges.
    `font_scale` enlarges type when the full-width source is placed smaller in a report.
    """
    assert len(models) <= len(MODEL_STYLES), "at most four selections per figure"
    if not np.isfinite(font_scale) or font_scale <= 0:
        raise ValueError("font_scale must be positive and finite")

    def as_series(x: Report | ReferenceSeries) -> ReferenceSeries:
        """A series as given, or made from a report."""
        return x if isinstance(x, ReferenceSeries) else ReferenceSeries.of_report(x)

    series = [(label, as_series(rep), "records", style) for (label, rep), style in zip(models, MODEL_STYLES)]
    series.append(("second annotator", as_series(annotator), "ordered pairs of records", ANNOTATOR_STYLE))
    if font_scale > 1:
        series[-1] = ("second annotator", series[-1][1], "ordered record pairs", ANNOTATOR_STYLE)
    style = {key: value * font_scale if key.endswith(("size", "fontsize")) else value
             for key, value in PAPER.items()}
    with plt.rc_context(style):
        fig, ax = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5 + max(0, font_scale - 1) * .8),
                               layout="constrained")
        for a, panel, kind, name in ((ax[0], "a", "iou", "IoU"), (ax[1], "b", "dice", "Dice")):
            means = [_outline(a, getattr(rep, f"{kind}_counts"), getattr(rep, f"{kind}_mean"), style)
                     for _, rep, _, style in series]
            a.legend([Line2D([], [], color=s["color"], ls=s["ls"], lw=1.0) for *_, s in series],
                     [f"{m:.3f}" for m in means], title="mean", loc="upper left", frameon=False)
            title = f"({panel}) {name} of overlapping pairs"
            if font_scale > 1:
                title = f"({panel}) {name} of\noverlapping pairs"
            a.set(title=title, xlabel=name, ylabel="share of pairs", xlim=(0, 1))
            a.grid(axis="y")
        _relation_bars(ax[2], [(rep, style) for _, rep, _, style in series])
        if font_scale > 1:
            ax[2].set_xlabel("masks | filaments")
        handles = [Patch(facecolor=s["color"], hatch=s["hatch"], hatchcolor="white", edgecolor="white")
                   for *_, s in series]
        labels = [f"{label}, {rep.comparisons:,} {unit}: "
                  + ", ".join(f"{k} {v:,}" for k, v in legend_counts(rep).items())
                  for label, rep, unit, _ in series]
        fig.legend(handles, labels, loc="outside lower center", frameon=False)
        fig.savefig(path)
    plt.close(fig)


def pq_intervals_figure(categories: list[str], series: list[tuple[str, np.ndarray, np.ndarray, float]], path,
                        xlabel: str, width: float = FULL_WIDTH) -> None:
    """Host-exact PQ with its 90% frame-bootstrap interval in each category of frames (the report's supplement S4
    by annotation group, S5 by time gap): one marker and interval per series side by side, and each series' PQ on
    all frames as a horizontal line in its style. `series` holds (label, PQ per category, (n, 2) intervals, PQ on
    all frames)."""
    assert len(series) <= len(MODEL_STYLES), "at most four selections per figure"
    x = np.arange(len(categories))
    with plt.rc_context(PAPER):
        fig, ax = plt.subplots(figsize=(width, 2.2), layout="constrained")
        for i, ((label, pq, ci, overall), style) in enumerate(zip(series, MODEL_STYLES)):
            dx = (i - (len(series) - 1) / 2) * 0.22
            _, _, (bars,) = ax.errorbar(x + dx, pq, yerr=np.abs(ci.T - pq), fmt=style["marker"], ms=3.5,
                                        color=style["color"], elinewidth=0.9, capsize=1.5, label=label)
            bars.set_linestyle(style["ls"])
            ax.axhline(overall, color=style["color"], ls=style["ls"], lw=0.7)
        ax.set_xticks(x, categories)
        ax.set(xlabel=xlabel, ylabel="PQ")
        ax.grid(axis="y")
        ax.legend(frameon=False)
        fig.savefig(path)
    plt.close(fig)


MORPH_OURS = {"color": "#E69F00", "ls": "solid", "lw": 0.9}  # our masks at tau 0.26
MORPH_OURS_030 = {"color": "#56B4E9", "ls": "dashed", "lw": 0.9}  # at tau 0.30, where they differ
MORPH_RECORDS = ("solid", "dashed", "dotted")  # the annotators' outlines, white, one dash pattern per record
MORPH_SPINE = {"color": "#F0E442", "lw": 0.5}  # the annotators' spines


def morphology_figure(tiles: list[dict], path, cols: int | None = None, width: float = FULL_WIDTH,
                      scale_px: int = 50) -> None:
    """The report's Fig. 3 (or a gallery): square crops of H-alpha frames in rows of `cols` tiles at `width`.

    Each tile holds `image` (the native-resolution crop), `vmin` and `vmax` (the linear stretch), `outlines` (masks of
    the crop with `color`, `ls` and `lw`, drawn at the pixel edges), optional `lines` ((n, 2) x, y polylines in crop
    pixels with `color` and `lw`) and a `title`, printed in the tile's top-left corner. A `scale_px` scale bar sits in
    the bottom-left corner."""
    cols = cols or -(-len(tiles) // 2)
    rows = -(-len(tiles) // cols)
    gap = 0.003
    with plt.rc_context(PAPER):
        fig = plt.figure(figsize=(width, width / cols * rows))
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            ax = fig.add_axes([c / cols + gap, 1 - (r + 1) / rows + gap * cols / rows, 1 / cols - 2 * gap,
                               1 / rows - 2 * gap * cols / rows])
            img = t["image"]
            h, w = img.shape
            ax.imshow(img, cmap="gray", vmin=t["vmin"], vmax=t["vmax"], interpolation="nearest")
            for o in t["outlines"]:
                if o["mask"].any():
                    ax.contour(o["mask"].astype(np.float32), levels=[0.5], colors=[o["color"]], linestyles=[o["ls"]],
                               linewidths=[o["lw"]])
            for line in t.get("lines", []):
                ax.plot(line["xy"][:, 0], line["xy"][:, 1], color=line["color"], lw=line["lw"])
            x0, y0 = 0.06 * w, 0.93 * h
            ax.plot([x0, x0 + scale_px], [y0, y0], color="black", lw=2.8, alpha=0.55, solid_capstyle="butt")
            ax.plot([x0, x0 + scale_px], [y0, y0], color="white", lw=1.4, solid_capstyle="butt")
            ax.text(x0 + scale_px / 2, y0 - 0.025 * h, f"{scale_px} px", color="white", ha="center", va="bottom",
                    fontsize=6, bbox=dict(facecolor="black", alpha=0.55, pad=0.8, lw=0))  # legible on bright tiles
            ax.text(0.03 * w, 0.03 * h, t["title"], color="white", ha="left", va="top", fontsize=7,
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.0, lw=0))
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_axis_off()
        fig.savefig(path)
    plt.close(fig)

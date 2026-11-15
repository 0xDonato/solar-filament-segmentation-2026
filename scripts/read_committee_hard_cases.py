"""Describe fixed native OOF errors on four annotation-derived hard-case sets.

Use supplied fixed predictions and official training records. Categories
are image contrast, annotated spine length, reference skeleton branching and
disk position. These are descriptive proxies, not new physical barb/faintness
labels. Plot median-quality and failure examples using fixed rules; no training,
test images, candidate selection changes or committee-score estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pycocotools import mask as coco
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paired_bootstrap import read_selection
from solarfil import metric
from solarfil.cache import load_disks, load_png
from solarfil.data import Dataset, read_image
from solarfil.morphology import crop_box, disk_stretch, spine_length
from solarfil.parallel import pool
from solarfil.preprocess import from_uint8
from solarfil.reference import mask_shape

_STATE = {}
TITLES = {"low_contrast": "Low contrast", "long_spine": "Long spine",
          "branched_reference": "Branched reference", "limb": "Near limb"}


def branch_count(mask):
    skeleton = skeletonize(mask).astype(np.uint8)
    neighbours = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    junctions = skeleton.astype(bool) & (neighbours > 2)
    return int(cv2.connectedComponents(junctions.astype(np.uint8), connectivity=8)[0] - 1)


def ring_contrast(flat, mask):
    ring = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool) & ~mask
    return float(flat[ring].mean() - flat[mask].mean()) if ring.any() else 0.0


def one(job):
    stem, selection, cache, object_features = job
    if not _STATE:
        _STATE.update(ds=Dataset(), predictions=read_selection(selection), disks=load_disks(Path(cache)))
        if object_features:
            features = {}
            with Path(object_features).open(newline="") as source:
                for row in csv.DictReader(source):
                    for key in ("record_index", "filament_index", "area", "branches", "connected_pieces", "holes"):
                        row[key] = int(row[key])
                    for key in ("spine_length", "contrast", "rho", "best_iou"):
                        row[key] = float(row[key])
                        if not np.isfinite(row[key]):
                            raise ValueError("Nonfinite cached annotation feature")
                    row["detected"] = row["detected"] == "True"
                    key = row["stem"], row["record_index"], row["filament_index"]
                    if key in features:
                        raise ValueError("Duplicate cached annotation feature")
                    features[key] = row
            if len(features) != 8199 or {key[0] for key in features} != set(_STATE["ds"].train_stems):
                raise ValueError("Require all official annotation-derived features")
            _STATE["features"] = features
        cv2.setNumThreads(1)
    ds, predictions, disk = _STATE["ds"], _STATE["predictions"].get(stem, []), _STATE["disks"][stem]
    flat = from_uint8(load_png(Path(cache) / "flat" / f"{stem}.png"))
    total, objects = metric.PQ(), []
    for ri, record in enumerate(ds.observations[stem].records):
        gt = record.rles()
        inter, ag, ap = metric.overlap_counts(gt, predictions)
        iou, _ = metric.iou_dice(inter, ag, ap)
        total = total + metric.PQ().add_matrix(iou)
        best = iou.max(1) if len(predictions) else np.zeros(len(gt))
        for fi, (filament, encoded) in enumerate(zip(record.filaments, gt, strict=True)):
            if object_features:
                feature = _STATE["features"][stem, ri, fi]
                if feature["group"] != record.group or feature["area"] != int(ag[fi]):
                    raise ValueError("Cached feature belongs to another annotation")
                np.testing.assert_allclose(feature["spine_length"], spine_length(filament.spine_xy()),
                                           rtol=0, atol=1e-10)
                objects.append(dict(feature, best_iou=float(best[fi]), detected=bool(best[fi] > .5)))
                continue
            mask = coco.decode(encoded).astype(bool)
            bx, by, bw, bh = coco.toBbox(encoded).astype(int)
            x0, y0 = max(0, bx - 4), max(0, by - 4)
            x1, y1 = min(2048, bx + bw + 4), min(2048, by + bh + 4)
            small = mask[y0:y1, x0:x1]
            assert small.any()
            moments = cv2.moments(small.astype(np.uint8))
            cx, cy = x0 + moments["m10"] / moments["m00"], y0 + moments["m01"] / moments["m00"]
            pieces, holes = mask_shape(small)
            objects.append(dict(stem=stem, group=record.group, record_index=ri, filament_index=fi,
                                area=int(mask.sum()), spine_length=spine_length(filament.spine_xy()),
                                contrast=ring_contrast(flat[y0:y1, x0:x1], small), branches=branch_count(small),
                                rho=float(np.hypot(cx - disk.cx, cy - disk.cy) / disk.r),
                                connected_pieces=pieces, holes=holes, best_iou=float(best[fi]),
                                detected=bool(best[fi] > 0.5)))
    return stem, objects, total.as_array()


def plot_examples(ds, cache, selection, chosen, out):
    predictions, disks = read_selection(selection), load_disks(cache)
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2), constrained_layout=True)
    for column, category in enumerate(TITLES):
        for row, role in enumerate(("median", "failure")):
            ax, case = axes[row, column], chosen[category][role]
            stem = case["stem"]
            observation = ds.observations[stem]
            encoded = observation.records[case["record_index"]].filaments[case["filament_index"]].rle()
            active = coco.decode(encoded).astype(bool)
            x0, y0, x1, y1 = crop_box(active, (2048, 2048), min_side=200, margin=1.5)
            image, disk = read_image(ds.image_path(stem)), disks[stem]
            lo, hi = disk_stretch(image, disk.cx, disk.cy, disk.r)
            ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi)
            for record in observation.records:
                for gt in record.rles():
                    crop = coco.decode(gt)[y0:y1, x0:x1]
                    if crop.any() and not crop.all():
                        ax.contour(crop, levels=[0.5], colors=["white"], linewidths=0.6, alpha=0.75)
            for predicted in predictions.get(stem, []):
                crop = coco.decode(predicted)[y0:y1, x0:x1]
                if crop.any() and not crop.all():
                    ax.contour(crop, levels=[0.5], colors=["#ff9c20"], linewidths=1.2)
            crop = active[y0:y1, x0:x1]
            ax.contour(crop, levels=[0.5], colors=["#00e5dc"], linewidths=1.1)
            ax.set_title(f"{TITLES[category]} / {role}\nbest IoU {case['best_iou']:.3f}", fontsize=10)
            ax.text(0.02, 0.02, f"{stem}\nrecord {case['record_index']}, object {case['filament_index']}",
                    transform=ax.transAxes, color="white", fontsize=6.5, va="bottom",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fixed native OOF examples: cyan = selected reference, white = other references, orange = prediction",
                 fontsize=11)
    fig.savefig(out / "hard_cases.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--audit", type=Path, help="Existing complete native audit with the v7 prediction source")
    source.add_argument("--selection", type=Path, help="Fixed native training-prediction CSV")
    ap.add_argument("--expected-pq", type=float, help="Native control required with --selection")
    ap.add_argument("--object-features", type=Path,
                    help="Complete official annotation/image features; every overlap is recomputed")
    ap.add_argument("--protocol", default="Observation-held R15/tau0.26; nonnested neural/utility folds")
    ap.add_argument("--source-note", default="Neural and utility folds are nonnested; deployment masks differ.")
    for key in ("cache", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--technical", action="store_true")
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Preserve previous reads and use a fresh output folder")
    cv2.setNumThreads(1)
    straight = np.zeros((21, 21), bool)
    straight[3:18, 10] = True
    fork = straight.copy()
    fork[10, 3:18] = True
    assert branch_count(straight) == 0 and branch_count(fork) == 1
    assert ring_contrast(np.ones((21, 21)), straight) == 0
    darker = np.ones((21, 21))
    darker[straight] = 0.8
    np.testing.assert_allclose(ring_contrast(darker, straight), .2, atol=1e-12, rtol=0)
    ds, begun = Dataset(), time.monotonic()
    if args.audit:
        audit = json.loads(args.audit.read_text())
        selection = audit["inputs"]["v7"]["path"]
        expected_pq = audit["methods"]["all_records_pooled"]["v7"]["pq"]
        if args.expected_pq is not None:
            raise ValueError("The audit already defines the native control")
    else:
        if args.expected_pq is None:
            raise ValueError("Supply the saved native PQ control with --selection")
        selection, expected_pq = str(args.selection), args.expected_pq
    stems = ds.train_stems[::90] if args.technical else ds.train_stems
    assert set(read_selection(selection)).issubset(ds.observations)
    args.out.mkdir(parents=True)
    objects, native = [], {}
    with pool(args.workers, spawn=True) as workers:
        jobs = [(stem, selection, str(args.cache), str(args.object_features) if args.object_features else None)
                for stem in stems]
        for i, (stem, rows, counts) in enumerate(workers.imap_unordered(one, jobs), 1):
            objects.extend(rows)
            native[stem] = counts
            if args.technical or i % 100 == 0 or i == len(stems):
                print(json.dumps(dict(complete=i, total=len(stems), objects=len(objects))), flush=True)
    objects.sort(key=lambda r: (r["stem"], r["record_index"], r["filament_index"]))
    total = np.stack([native[stem] for stem in stems]).sum(0)
    native_pq = float(total[3] / (total[0] + .5 * (total[1] + total[2])))
    if not args.technical:
        assert len(objects) == 8199
        np.testing.assert_allclose(native_pq, expected_pq, atol=1e-12, rtol=0)
    area, length, contrast, branches, rho, pieces, holes = (
        np.asarray([r[key] for r in objects])
        for key in ("area", "spine_length", "contrast", "branches", "rho", "connected_pieces", "holes")
    )
    threshold = dict(low_contrast=float(np.quantile(contrast, .1)), long_spine=float(np.quantile(length, .9)),
                     branched_reference=int(max(1, np.ceil(np.quantile(branches, .9)))),
                     branch_minimum_area=float(np.median(area)), limb=0.85)
    membership = dict(low_contrast=contrast <= threshold["low_contrast"], long_spine=length >= threshold["long_spine"],
                      branched_reference=(branches >= threshold["branched_reference"]) &
                      (area >= threshold["branch_minimum_area"]) & (pieces == 1) & (holes == 0), limb=rho >= .85)
    groups = np.asarray([r["group"] for r in objects])
    unique = np.unique(groups)
    draws = np.random.default_rng(110234).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), 4000)
    stats, chosen, used_frames = {}, {}, set()
    for category, take in membership.items():
        assert take.any(), category
        cohort = [r for r, selected in zip(objects, take, strict=True) if selected]
        counts = np.asarray([[sum(r["group"] == g for r in cohort),
                              sum(r["detected"] for r in cohort if r["group"] == g),
                              sum(r["best_iou"] for r in cohort if r["group"] == g)] for g in unique])
        n, hits, overlap = counts.sum(0)
        sampled = draws @ counts
        valid = sampled[:, 0] > 0
        intervals = {name: np.quantile(sampled[valid, j] / sampled[valid, 0], [.05, .95]).tolist()
                     for j, name in ((1, "detected_share"), (2, "mean_best_iou"))}
        stats[category] = dict(annotation_objects=int(n), frames=len({r["stem"] for r in cohort}),
                               annotation_groups=len({r["group"] for r in cohort}), detected_share=float(hits / n),
                               mean_best_iou=float(overlap / n), group_ci90=intervals,
                               empty_group_bootstrap_draws=int((~valid).sum()))
        median = float(np.median([r["best_iou"] for r in cohort]))
        chosen[category] = {}
        for role in ("median", "failure"):
            ordered = sorted(cohort, key=lambda r: (
                abs(r["best_iou"] - median) if role == "median" else r["best_iou"],
                -r["area"], r["stem"], r["record_index"], r["filament_index"]))
            selected = next((r for r in ordered if r["stem"] not in used_frames), ordered[0])
            chosen[category][role] = selected
            used_frames.add(selected["stem"])
    with (args.out / "objects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    result = dict(scope=__doc__.strip(), frames=len(stems), annotation_objects=len(objects), technical=args.technical,
                  native_pq=native_pq, original_native_control_exact=not args.technical,
                  prediction_source=selection, partition=args.protocol, source_note=args.source_note,
                  reused_annotation_features=str(args.object_features) if args.object_features else None,
                  thresholds=threshold, categories=stats, examples=chosen,
                  example_rule=("closest to category median IoU and smallest IoU;"
                                "area descending then stable identifiers;"
                                "prefer frames unused by earlier examples;examples do not estimate prevalence"),
                  uncertainty=("Fixed thresholds/predictions;whole-annotation-group resampling;"
                               "not training or physical uncertainty"),
                  physical_barb_labels_created=False, test_images_read=0, new_fits=0, new_forecasts=0,
                  seconds=time.monotonic() - begun)
    plot_examples(ds, args.cache, selection, chosen, args.out)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    table = ["# Annotation-derived hard cases", "",
             args.protocol + "; official training annotations only.",
             "Cyan is the chosen reference, white the other records, orange the fixed predictions.",
             "Categories overlap. Objects are annotation instances, including repeated structures across records.",
             "", "| Category | Annotation objects | Frames | Detected (IoU>.5) | Mean best IoU |",
             "|---|---:|---:|---:|---:|"]
    for category, row in stats.items():
        table.append(f"| {TITLES[category]} | {row['annotation_objects']} | {row['frames']} | "
                     f"{row['detected_share']:.3f} | {row['mean_best_iou']:.3f} |")
    table.extend(["", "![Fixed representative and failure examples](hard_cases.png)", "",
                  "Faintness and branch counts are annotation/image proxies; physical barbs are unlabelled.",
                  "See result.json for definitions, group intervals and complete selection rules.",
                  args.source_note,
                  "Image and annotation source: official MAGFiLO_1.0_Kaggle_2026/GONG training data;"
                  " preserve attribution and the competition data terms."])
    (args.out / "README.md").write_text("\n".join(table) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}), flush=True)


if __name__ == "__main__":
    main()

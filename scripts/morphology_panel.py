"""The report's morphology panel (Fig. 3): real structures on H-alpha frames, picked by fixed rules from out-of-fold
selections, beside the annotators' outlines.

Rules (a)-(h) (`solarfil.morphology.RULES`) read the selections at tau 0.26 and 0.30 (positional, in that order),
the training records and the run's bank and maps (rule (d)). Rules (i) and (j) draw two seeded test frames from a
test submission (`--test-csv`, a CSV or a folder holding one, our outlines only); they are left out while there is
no file. The panel takes, rule by
rule, the best candidate on a frame that no earlier tile uses; ties go by stem order.

Each tile is a square native-resolution crop of the raw JPEG, stretched linearly between the 1st and 99th
percentiles of the solar disk, with a 50 px scale bar. The annotators' outlines are white, one dash pattern per record
(solid, dashed, dotted in record order). Ours are orange at tau 0.26 and cyan dashed at tau 0.30 where a mask
differs. Tile (a) adds the annotators' spines in yellow, and tile (d) the rejected tree candidate in orange dotted.
The JSON gives every tile's rule, stem, crop box (x0, y0, x1, y1 at 2048 px), the records, the IoUs of its masks
against each record and the values its rule ranks by. With `--n 5`, each rule also gets a gallery of its best five
tiles, one per frame, written beside the figure as `<out>_gallery_<rule>.png`.

Usage: morphology_panel.py A_tau0.26.csv B_tau0.30.csv --run ens_s0_win --out fig3.png --json fig3.json
       [--bank bank_split] [--test-csv test_tau0.26.csv [test_tau0.30.csv]] [--n 5] [--workers 3]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from solarfil import metric, morphology
from solarfil.bank import FEATURES
from solarfil.cache import DEFAULT_CACHE, load_disks
from solarfil.data import Dataset, H, W, read_image
from solarfil.infer import read_map_png
from solarfil.parallel import pool
from solarfil.paths import RUNS
from solarfil.plots import MORPH_OURS, MORPH_OURS_030, MORPH_RECORDS, MORPH_SPINE, morphology_figure
from solarfil.tree import build_tree

PANEL = "abcdefgh"
CANDIDATE = {"color": MORPH_OURS["color"], "ls": "dotted", "lw": 0.9}  # rule (d)'s rejected tree candidate
RECORD_LW = 0.6
_DS: dict = {}


def read_masks(path: str) -> dict[str, list[tuple[str, dict]]]:
    """(filament_id, COCO RLE) of each mask of a selection or submission file, by stem, in file order."""
    out: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filament_id"].rsplit("_", 1)[0]].append(
                (row["filament_id"], {"size": [H, W], "counts": row["segmentation_rle"].encode()}))
    return out


def candidates(args) -> list[dict]:
    """Every rule's candidates on one training frame."""
    stem, sel, sel30, bank_file, map_file, median_area = args
    obs = _DS.setdefault("ds", Dataset()).observations[stem]
    gt = [metric.gt_rles_cached(r) for r in obs.records]
    spines = [[f.spine_xy() for f in r.filaments] for r in obs.records]
    out = morphology.frame_candidates(stem, gt, spines, sel, sel30, median_area)
    with np.load(bank_file) as z:
        X, B = z["X"], z["B"]
    assert X.shape[1] == len(FEATURES), (stem, X.shape)
    nodes = morphology.compact_candidates(stem, X, B, FEATURES)
    if len(nodes):
        tree = build_tree(read_map_png(map_file)[0], splits=True)
        assert tree.n == len(X), (stem, tree.n, len(X))
        union = mask_util.decode(mask_util.merge(sel, intersect=False)).astype(bool) if sel else np.zeros((H, W), bool)
        out += morphology.compact_pick(stem, tree, nodes, union, X, B, FEATURES)
    return out


def decode(r: dict) -> np.ndarray:
    """Boolean mask of one COCO RLE."""
    return mask_util.decode(r).astype(bool)


def tile(c: dict, ds: Dataset, disks: dict, sel: dict, sel30: dict, maps: dict, title: str) -> tuple[dict, dict]:
    """The drawing of one candidate and its JSON entry."""
    stem = c["stem"]
    img = read_image(ds.image_path(stem))
    d = disks[stem]
    lo, hi = morphology.disk_stretch(img, d.cx, d.cy, d.r)
    obs = ds.observations.get(stem)
    records = [metric.gt_rles_cached(r) for r in obs.records] if obs is not None else []
    gt = [[decode(g) for g in rec] for rec in records]
    ours = [decode(m) for _, m in sel.get(stem, [])]
    ids26 = {m["counts"] for _, m in sel.get(stem, [])}
    ours30 = [decode(m) for _, m in sel30.get(stem, []) if m["counts"] not in ids26]
    focus = np.zeros((H, W), bool)
    for k in c.get("masks", []):
        focus |= ours[k]
    if "filament" in c:
        focus |= gt[c["filament"][0]][c["filament"][1]]
    if "pieces" in c:
        for f in c["pieces"][1]:
            focus |= gt[c["pieces"][0]][f]
    for r, f in c.get("versions", []):
        focus |= gt[r][f]
    node = None
    if "node" in c:
        node = build_tree(read_map_png(maps[stem])[0], splits=True).node_mask(c["node"])
        focus |= node
    x0, y0, x1, y1 = morphology.crop_box(focus, (H, W))
    crop = (slice(y0, y1), slice(x0, x1))
    outlines = [dict(mask=g[crop], color="white", ls=MORPH_RECORDS[r % len(MORPH_RECORDS)], lw=RECORD_LW)
                for r, rec in enumerate(gt) for g in rec if g[crop].any()]
    outlines += [dict(mask=m[crop], **MORPH_OURS) for m in ours if m[crop].any()]
    outlines += [dict(mask=m[crop], **MORPH_OURS_030) for m in ours30 if m[crop].any()]
    if node is not None:
        outlines.append(dict(mask=node[crop], **CANDIDATE))
    lines = []
    if c["rule"] == "a" and obs is not None:
        for r in obs.records:
            for f in r.filaments:
                xy = f.spine_xy() - [x0, y0]
                if len(xy) > 1 and ((xy >= 0) & (xy < [x1 - x0, y1 - y0])).all(axis=1).any():
                    lines.append(dict(xy=xy, **MORPH_SPINE))
    drawing = dict(image=img[crop], vmin=lo, vmax=hi, outlines=outlines, lines=lines, title=title)
    entry = dict(rule=c["rule"], title=morphology.RULES[c["rule"]][0], stem=stem, box=[x0, y0, x1, y1],
                 stretch=[round(lo, 1), round(hi, 1)],
                 records=[r.record_id for r in obs.records] if obs is not None else [],
                 masks=[sel[stem][k][0] for k in c.get("masks", [])], ious=c.get("ious", []),
                 values=c.get("values", {}))
    for key in ("filament", "pieces", "versions", "node"):
        if key in c:
            entry[key] = c[key]
    return drawing, entry


def resolve(path: str | None) -> str | None:
    """A submission file, or the one CSV in a submission folder; None when there is none yet."""
    if path is None or not Path(path).exists():
        return None
    if Path(path).is_dir():
        files = sorted(Path(path).glob("*.csv"))
        assert len(files) <= 1, f"{path} holds {len(files)} CSV files: name one"
        return str(files[0]) if files else None
    return path


def test_candidates(path: str, ds: Dataset, n: int) -> list[dict]:
    """Rules (i), (j): seeded test frames, each around its largest selected mask."""
    masks = read_masks(path)
    out = []
    for i, stem in enumerate(morphology.test_frames(ds.test_stems, set(masks), n)):
        area = morphology.rle_areas([m for _, m in masks[stem]])
        k = int(np.argmax(area))
        out.append(dict(stem=stem, rule="ij"[min(i, 1)], key=(i,), masks=[k], ious=[],
                        values=dict(area=int(area[k]), draw_order=i)))
    return out


def main():
    """Pick the tiles by rule, draw the panel (and the galleries with `--n`) and write the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("selections", nargs=2, help="out-of-fold selection CSVs at tau 0.26 and tau 0.30")
    ap.add_argument("--run", required=True, help="the run whose maps and bank made the selections")
    ap.add_argument("--bank", default="bank_split")
    ap.add_argument("--out", required=True, help="the figure (png or pdf)")
    ap.add_argument("--json", required=True, help="the tiles' rules, stems, boxes and IoUs")
    ap.add_argument("--test-csv", nargs="+", default=None,
                    help="test submission at tau 0.26 [and at tau 0.30]: a CSV, or a folder holding one")
    ap.add_argument("--n", type=int, default=0, help="also write a gallery of the best n tiles of every rule")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out_path = Path(a.out)
    gallery_path = {rule: out_path.with_name(f"{out_path.stem}_gallery_{rule}{out_path.suffix}")
                    for rule in [*PANEL, "test"]}
    targets = [out_path, Path(a.json)] + (list(gallery_path.values()) if a.n else [])
    for p in targets:
        assert not p.exists(), f"{p} exists: give a new name"
    ds = Dataset()
    sel26, sel30 = read_masks(a.selections[0]), read_masks(a.selections[1])
    run = RUNS / a.run
    maps = {q.stem: q for q in run.glob("fold*/oof/*.png")}
    stems = sorted(f.stem for f in (run / a.bank).glob("*.npz"))
    median_area = float(np.median([f.area for o in ds.observations.values() for r in o.records for f in r.filaments]))
    jobs = [(s, [m for _, m in sel26.get(s, [])], [m for _, m in sel30.get(s, [])], str(run / a.bank / f"{s}.npz"),
             str(maps[s]), median_area) for s in stems]
    with pool(a.workers) as w:
        found = [c for cs in w.map(candidates, jobs, chunksize=4) for c in cs]
    by_rule = defaultdict(list)
    for c in found:
        by_rule[c["rule"]].append(c)
    test_file = resolve(a.test_csv[0]) if a.test_csv else None
    test_file30 = resolve(a.test_csv[1]) if test_file and len(a.test_csv) > 1 else None
    test_sel30 = read_masks(test_file30) if test_file30 else {}
    test = test_candidates(test_file, ds, max(2, a.n)) if test_file else []
    test_sel = read_masks(test_file) if test_file else {}
    used: set[str] = set()
    panel = [c for rule in PANEL for c in morphology.pick(by_rule[rule], 1, used)] + test[:2]
    disks = load_disks(DEFAULT_CACHE)

    def draw(c: dict, label: str) -> tuple[dict, dict]:
        """The drawing and JSON entry of one tile, against the test selections for rules (i) and (j)."""
        is_test = c["rule"] in "ij"
        return tile(c, ds, disks, test_sel if is_test else sel26, test_sel30 if is_test else sel30, maps, label)

    drawn = [draw(c, f"({c['rule']}) {morphology.LABELS[c['rule']]}") for c in panel]
    morphology_figure([d for d, _ in drawn], out_path, cols=-(-len(drawn) // 2))
    result = dict(selections=a.selections, run=a.run, bank=a.bank, frames=len(stems), median_area=median_area,
                  stretch="linear between the 1st and 99th percentiles of the solar disk", scale_bar_px=50,
                  rules={k: v[1] for k, v in morphology.RULES.items()},
                  candidates={rule: len(by_rule[rule]) for rule in PANEL},
                  test_csv_given=a.test_csv, test_csv=[test_file, test_file30],
                  test_tiles="drawn" if test_file else "left out: no test file",
                  figure=str(out_path), tiles=[e for _, e in drawn])
    if a.n:
        result["galleries"] = {}
        for rule in [*PANEL, "test"]:
            picked = test[: a.n] if rule == "test" else morphology.pick(by_rule[rule], a.n)
            if not picked:
                continue
            tiles = [draw(c, f"({c['rule']}) {morphology.LABELS[c['rule']]}, {i + 1}") for i, c in enumerate(picked)]
            morphology_figure([d for d, _ in tiles], gallery_path[rule], cols=len(tiles))
            result["galleries"][rule] = dict(figure=str(gallery_path[rule]), tiles=[e for _, e in tiles])
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(result, indent=1))
    for e in result["tiles"]:
        print(e["rule"], e["stem"], e["box"], e["values"])
    print(result["test_tiles"])


if __name__ == "__main__":
    main()

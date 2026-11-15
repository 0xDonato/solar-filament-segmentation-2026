"""Draw six report tiles from an existing fixed-rule morphology specification.

The source specification binds the training selections and test submissions.
This CPU-only renderer uses the original rule choices, crop boxes and masks.
It reads official JPEGs and training records; it performs no fitting or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import morphology_panel as mp
from solarfil.cache import DEFAULT_CACHE, load_disks
from solarfil.data import Dataset
from solarfil.paths import RUNS
from solarfil.plots import morphology_figure

KEEP = {
    "a": "(a) irregular",
    "b": "(b) long",
    "e": "(c) disagreement",
    "f": "(d) missed",
    "h": "(e) unannotated",
    "j": "(f) test frame",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    for target in (args.out, args.json):
        if target.exists():
            raise FileExistsError(f"Choose a new output: {target}")
    spec = json.loads(args.source.read_text())
    inputs = [args.source, Path(__file__), Path(mp.__file__)]
    inputs.extend(Path(p) for p in spec["selections"] + spec["test_csv"])
    bound = {str(p.resolve()): sha256(p) for p in inputs}
    ds, disks = Dataset(), load_disks(DEFAULT_CACHE)
    sel26, sel30 = (mp.read_masks(p) for p in spec["selections"])
    test26, test30 = (mp.read_masks(p) for p in spec["test_csv"])
    maps = {p.stem: p for p in (RUNS / spec["run"]).glob("fold*/oof/*.png")}
    mp.MORPH_SPINE["color"] = "#00FFFF"
    tiles = {t["rule"]: t for t in spec["tiles"]}
    drawings, entries = [], []
    for rule, label in KEEP.items():
        tile = tiles[rule]
        s26, s30 = (test26, test30) if rule in "ij" else (sel26, sel30)
        ids = [fid for fid, _ in s26[tile["stem"]]]
        candidate = dict(
            stem=tile["stem"], rule=rule,
            masks=[ids.index(fid) for fid in tile["masks"]],
            ious=tile["ious"], values=tile.get("values", {}),
        )
        for key in ("filament", "pieces", "versions", "node"):
            if key in tile:
                candidate[key] = tile[key]
        image_path = Path(ds.image_path(tile["stem"]))
        bound[str(image_path.resolve())] = sha256(image_path)
        drawing, entry = mp.tile(candidate, ds, disks, s26, s30, maps, label)
        assert entry["box"] == tile["box"], (rule, entry["box"], tile["box"])
        assert entry["records"] == tile["records"], rule
        assert entry["masks"] == tile["masks"], rule
        drawings.append(drawing)
        entries.append(dict(entry, label=label))
    for filename, digest in bound.items():
        assert sha256(Path(filename)) == digest, filename
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    morphology_figure(drawings, args.out, cols=len(drawings))
    result = dict(
        source=str(args.source), kept=list(KEEP), tiles=entries,
        inputs_sha256=bound, figure=str(args.out), figure_sha256=sha256(args.out),
        scope="CPU rendering of six fixed-rule tiles; no fitting or inference",
    )
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(tiles=len(drawings), figure_sha256=result["figure_sha256"])))


if __name__ == "__main__":
    main()

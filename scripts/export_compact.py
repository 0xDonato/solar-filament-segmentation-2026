"""Write the compact release: a weights folder with one detector triple instead of five (optionally without one of its
detectors), every network stored as float32 safetensors, a file over GitHub's 100 MB limit split into shards
(`<name>-00001-of-00002.safetensors`, ...), so the repository holds the validated weights bit for bit as plain files.

Usage: export_compact.py --weights <release weights folder> --out <new folder> [--triple K]
                         [--drop-set S --scorer <scorer.npz>]
`--triple` picks which fold triple of the manifest's `instances` the compact release keeps (0 by default).
`--drop-set` leaves out the S-th detector of the triple, with `--scorer` the scorer fitted without its columns
(`decide_test.py --drop-set`). Otherwise the scorer and the inference settings are copied unchanged.
`Pipeline.load` reads the shards back as one state dict (`pipeline.load_state`). Prints each file's size.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

SHARD_BYTES = 95 * 10**6  # under GitHub's 100 MB file limit, with room for the header


def convert(src: Path, dst: Path) -> list[Path]:
    """Save one `.pt` state dict as safetensors, in shards when it exceeds `SHARD_BYTES`; returns the files."""
    state = {k: v.contiguous() for k, v in torch.load(src, map_location="cpu", weights_only=True).items()}
    shards, size = [{}], 0
    for k, v in state.items():
        n = v.numel() * v.element_size()
        if size + n > SHARD_BYTES and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][k] = v
        size += n
    if len(shards) == 1:
        files = [dst]
    else:
        files = [dst.with_name(f"{dst.stem}-{i + 1:05d}-of-{len(shards):05d}.safetensors") for i in range(len(shards))]
    for f, part in zip(files, shards):
        save_file(part, f)
        assert f.stat().st_size < 100 * 10**6, f"{f.name} is over GitHub's file limit"
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--triple", type=int, default=0)
    ap.add_argument("--drop-set", type=int, default=None)
    ap.add_argument("--scorer", type=Path, default=None)
    args = ap.parse_args()
    src, out = Path(args.weights), Path(args.out)
    assert (args.drop_set is None) == (args.scorer is None), "--drop-set goes with the scorer fitted without that set"
    out.mkdir(parents=True, exist_ok=False)
    man = json.loads((src / "manifest.json").read_text())
    files = {}
    for d in man["dense"]:
        name = Path(d["file"]).with_suffix(".safetensors").name
        files[name] = convert(src / d["file"], out / name)
        d["file"] = name
    triple = [f for i, f in enumerate(man["instances"][args.triple]) if i != args.drop_set]
    man["instances"] = [[Path(f).with_suffix(".safetensors").name for f in triple]]
    for f, name in zip(triple, man["instances"][0]):
        files[name] = convert(src / f, out / name)
    scorer = args.scorer or src / man["scorer"]
    shutil.copy2(scorer, out / man["scorer"])
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    sizes = {f.name: f.stat().st_size for fs in files.values() for f in fs}
    total = sum(sizes.values()) + (out / man["scorer"]).stat().st_size
    for name, n in sizes.items():
        print(f"{name}: {n / 1e6:.1f} MB")
    print(f"total {total / 1e6:.1f} MB in {len(sizes) + 2} files")
    (out / "export_report.json").write_text(json.dumps(dict(source=str(src), triple=args.triple,
                                                            drop_set=args.drop_set, scorer=str(scorer),
                                                            files=sizes, total_bytes=total), indent=1) + "\n")


if __name__ == "__main__":
    main()

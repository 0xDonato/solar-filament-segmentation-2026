"""Out-of-fold detections of a trained instance model with flipped views pooled (`solarfil.instances.detect`), as a
new detector set for the scorer's columns.

Each fold's detector (<runs>/<run>/fold<k>/model.pt, with its anchor and fold scheme from the fold's cfg.json) detects
the fold's held-out frames under the first `views` flips (identity, horizontal, vertical, both). The detections of
every view, flipped back onto the frame, are stored together in <runs>/<run>_f<views>/fold<k>/oof/<stem>.npz, so
`instance_study.py --inst <run>_f<views> --base <run>` judges the views as a replacement for the single view.
Four views gained +0.0020 at tau 0.26 in place of one set's single view, and nothing beside the first two
one-view sets, so the pipeline detects with one view.

Usage: predict_instances.py --run inst_maskrcnn [--views 4] [--folds 0,1,2,3,4] [--max-val N --cpu (smoke)]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from solarfil import paths
from solarfil.cache import DEFAULT_CACHE, load_png
from solarfil.folds import load_folds
from solarfil.instances import FLIPS, build, detect, save_instances
from solarfil.model import device, inference_dtype


def main():
    """Detect each fold's held-out frames under the first `--views` flips and write the pooled detections per frame."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained instance model's run, e.g. inst_maskrcnn")
    ap.add_argument("--views", type=int, default=4, choices=range(1, len(FLIPS) + 1))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--runs", default=str(paths.RUNS))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--max-val", type=int, default=None, help="first N held-out frames only (smoke tests)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu") if a.cpu else device()
    amp = inference_dtype(dev)  # bfloat16 on CUDA, as the out-of-fold detections were made
    for k in map(int, a.folds.split(",")):
        src = Path(a.runs) / a.run / f"fold{k}"
        cfg = json.loads((src / "cfg.json").read_text())
        model = build(coco=False, anchor=cfg.get("anchor", 32))
        model.load_state_dict(torch.load(src / "model.pt", map_location="cpu", weights_only=True))
        model.to(dev).eval()
        out = Path(a.runs) / f"{a.run}_f{a.views}" / f"fold{k}"
        (out / "oof").mkdir(parents=True, exist_ok=True)
        va = [s for s, f in load_folds(cfg.get("scheme", "obs")).items() if f == k][:a.max_val]
        (out / "cfg.json").write_text(json.dumps(dict(cfg, source=a.run, views=a.views, n_val=len(va)), indent=1))
        t0, kept = time.time(), []
        for s in va:
            scores, packed, maskness = detect(model, load_png(Path(a.cache) / "flat" / f"{s}.png"), dev, amp,
                                              views=a.views)
            save_instances(out / "oof" / f"{s}.npz", scores, packed, maskness)
            kept.append(len(scores))
        print(json.dumps(dict(event="fold done", fold=k, frames=len(va),
                              detections_per_frame=round(float(np.mean(kept)), 1) if kept else 0.0,
                              sec_per_frame=round((time.time() - t0) / max(1, len(va)), 2))), flush=True)


if __name__ == "__main__":
    main()

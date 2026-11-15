"""Re-predict the out-of-fold maps of a run with other test-time settings, into a new run folder that the
evaluation scripts accept (each frame still mapped by the model of its own fold).

Usage: predict_oof.py --run e3_r34_n --views 8 --out e3_r34_n_tta8 [--zoom 1.25] [--folds 0,1]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from solarfil.cache import DEFAULT_CACHE, load_disks
from solarfil.frames import load_frame
from solarfil.infer import predict_frame, write_map_png
from solarfil.model import device, inference_dtype, model_from_cfg, uses_coords
from solarfil.paths import RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--runs", default=str(RUNS))
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--zoom", type=float, default=1.0, help="enlarge the input by this factor (see predict_frame)")
ap.add_argument("--folds", default="", help="comma-separated folds (default: all of the run)")
a = ap.parse_args()
src, dst = Path(a.runs) / a.run, Path(a.runs) / a.out
dev = device()
amp = inference_dtype(dev)
disks = load_disks(DEFAULT_CACHE)
t0, n = time.time(), 0
for fd in sorted(src.glob("fold*")):
    if a.folds and fd.name[4:] not in a.folds.split(","):
        continue
    cfg = json.loads((fd / "cfg.json").read_text())
    m = model_from_cfg(cfg)
    m.load_state_dict(torch.load(fd / "model.pt", map_location="cpu", weights_only=True))
    m = m.to(dev).eval()
    (dst / fd.name / "oof").mkdir(parents=True, exist_ok=True)
    shutil.copy(fd / "cfg.json", dst / fd.name / "cfg.json")
    for png in sorted((fd / "oof").glob("*.png")):
        fr = load_frame(png.stem, cfg["scale"], DEFAULT_CACHE, disks, targets=False)
        fr.coords = uses_coords(cfg)
        pm = predict_frame([m], fr, dev, n_flips=a.views, amp_dtype=amp,
                           label_dys=[cfg.get("label_dy", 0.0) / cfg["scale"]], zoom=a.zoom,
                           tiles=[cfg.get("infer_tile", 0)])
        write_map_png(dst / fd.name / "oof" / png.name, pm)
        n += 1
print(json.dumps(dict(frames=n, views=a.views, zoom=a.zoom, sec_per_frame=round((time.time() - t0) / max(n, 1), 3))))

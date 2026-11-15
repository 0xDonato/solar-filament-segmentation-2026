"""Predict the test frames end to end with an exported weights folder and write the submission file.

Usage: predict.py --images <folder of .jpeg> --weights <weights folder> --out submission.csv [--tau T] [--limit N]
                  [--alpha A] [--outline [--t-low 0.1]] [--workers 4]
`--tau` replaces the weights folder's selection threshold, `--alpha` its level rule (`tree.relevel`; 0 turns it off),
and `--outline` grows the selected filaments by thin protrusions of their map at level `--t-low` or above
(`morph.grow_thin`; 0.1 by default, the level DESIGN.md reports on). Our final selection is the pipeline at tau
0.30 with the outline step at `--t-low 0.3`; the 0.26 and 0.30 references leave it off.
Prints the wall time per frame and each stage's own seconds per frame (flat field, dense maps, tree, scorer,
selection); with workers the CPU stages overlap the dense maps, so the stages add up to more than the wall time.
`--trace PATH` also writes one CSV row per frame with its seconds and the memory in use (`FrameTrace`); the
submission file is the same with or without it.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import torch

from solarfil import submission
from solarfil.pipeline import Pipeline

TRACE_COLUMNS = ("frame", "stem", "seconds", "rss_mb", "rss_main_mb", "cuda_alloc_peak_mb", "cuda_reserved_peak_mb")


def rss_mb(pids: list[int]) -> dict[int, float]:
    """Resident memory in MiB of each live process in `pids`: from /proc on Linux, from `ps` elsewhere."""
    if Path("/proc/self/statm").exists():
        page, out = os.sysconf("SC_PAGE_SIZE"), {}
        for pid in pids:
            try:
                out[pid] = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * page / 2**20
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass  # a worker that has just exited, or a restricted process reading
        return out
    try:
        ps = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True)
    except OSError:
        return {}  # optional telemetry must not stop prediction when the OS denies process readings
    return {int(a): int(b) / 1024 for a, b in (line.split() for line in ps.stdout.splitlines())}


class FrameTrace:
    """Per-frame seconds and memory of a pipeline run, for the efficiency filter's two failures (slow, leaking).

    A row is taken each time the main process finishes a frame's GPU stages (`Pipeline.detect`, called once per frame
    in order), or each whole frame (`Pipeline.instances`) when the run has no workers. `seconds` is the time since the
    previous row (the first row includes the pool's start). `rss_mb` is the resident memory of this process and its
    live worker processes, `rss_main_mb` that of this process alone; on CUDA the peaks of PyTorch's allocated and
    reserved memory since the previous row (reset after each row). With workers the last frames' decision stages
    finish after their rows, inside the total that predict.py prints. RSS fields are empty if the OS denies the
    process reading; an unavailable measurement is never reported as zero memory."""

    def __init__(self, pipe: Pipeline, stems: list[str], workers: int):
        self.stems, self.rows, self.cuda = stems, [], pipe.dev.type == "cuda"
        hook = "detect" if workers > 0 else "instances"
        inner = getattr(pipe, hook)

        def traced(*args, **kwargs):
            out = inner(*args, **kwargs)
            self.record()
            return out

        setattr(pipe, hook, traced)  # an instance attribute: the class and the default path are untouched
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(pipe.dev)
        self.t = time.perf_counter()

    def record(self) -> None:
        """Append the row of the frame just finished and reset the clock and the CUDA peaks."""
        now, me = time.perf_counter(), os.getpid()
        rss = rss_mb([me] + [c.pid for c in mp.active_children()])
        mib = 2**20
        alloc = torch.cuda.max_memory_allocated() / mib if self.cuda else None
        reserved = torch.cuda.max_memory_reserved() / mib if self.cuda else None
        i = len(self.rows)
        self.rows.append(dict(frame=i, stem=self.stems[i] if i < len(self.stems) else "", seconds=now - self.t,
                              rss_mb=sum(rss.values()) if me in rss else None, rss_main_mb=rss.get(me),
                              cuda_alloc_peak_mb=alloc, cuda_reserved_peak_mb=reserved))
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.t = time.perf_counter()

    def write(self, path: Path | str) -> None:
        """Write the rows as CSV, numbers to three decimals and empty cells where a reading does not apply."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, TRACE_COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: ("" if v is None else round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def main():  # a function: the worker processes import this module
    """Run the pipeline on the test frames, write and validate the submission, and print its checks and timings as
    JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None, help="selection threshold (default: the weights folder's)")
    ap.add_argument("--alpha", type=float, default=None, help="level rule (default: the weights folder's)")
    ap.add_argument("--outline", action="store_true", help="the thin-protrusion outline step (morph.grow_thin)")
    ap.add_argument("--t-low", type=float, default=0.1, help="the outline step's map level (morph.grow_thin's t_low)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="processes for the CPU stages; 0 runs frame by frame")
    ap.add_argument("--trace", default=None, help="CSV of each frame's seconds and memory (FrameTrace)")
    a = ap.parse_args()
    pipe = Pipeline.load(a.weights)
    if a.tau is not None:
        pipe.tau = a.tau
    if a.alpha is not None:
        pipe.alpha = a.alpha or None
    if a.outline:
        pipe.outline = {"t_low": a.t_low}
    paths = sorted(Path(a.images).glob("*.jpeg"))[: a.limit]
    if not paths:
        ap.error(f"no .jpeg files in {a.images}")
    trace = FrameTrace(pipe, [p.stem for p in paths], a.workers) if a.trace else None
    t0 = time.perf_counter()
    inst = pipe.run(paths, workers=a.workers)
    total = time.perf_counter() - t0
    if trace:
        trace.write(a.trace)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    submission.write(inst, a.out)
    check = submission.validate(a.out, [p.stem for p in paths])
    per = {k: round(v / len(paths), 3) for k, v in pipe.timings.items()}
    print(json.dumps(dict(**check, tau=pipe.tau, alpha=pipe.alpha, outline=pipe.outline,
                          sec_per_frame=round(total / len(paths), 3), stages=per, workers=a.workers,
                          device=str(pipe.dev))))


if __name__ == "__main__":
    main()

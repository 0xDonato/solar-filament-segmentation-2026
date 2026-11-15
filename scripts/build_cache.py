"""Build the flat-field and target cache (see `solarfil.cache`). Usage: build_cache.py [--workers N]"""
import argparse
import json
import time
from pathlib import Path

from solarfil.cache import DEFAULT_CACHE, build_one
from solarfil.data import DEFAULT_ROOT, Dataset
from solarfil.parallel import pool


def main():
    """Parse the arguments, build the cache of every training and test frame, and write the disks to `disks.json`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    ds = Dataset(a.root)
    cache = Path(a.cache)
    for sub in ("flat", "p", "spine"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(a.root, a.cache, s, True) for s in ds.train_stems] + [(a.root, a.cache, s, False) for s in ds.test_stems]
    t0, disks = time.time(), {}
    with pool(a.workers) as workers:
        for i, (stem, disk) in enumerate(workers.imap_unordered(build_one, jobs, chunksize=4)):
            disks[stem] = disk
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    (cache / "disks.json").write_text(json.dumps(dict(sorted(disks.items())), indent=0))
    print(f"done {len(disks)} frames in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Process pools with one thread per worker.

OpenCV, OpenMP (scikit-learn) and BLAS each start one thread per core inside every process, so W pool workers
would otherwise run about W x cores threads; on a shared machine that starves the other jobs.
"""
from __future__ import annotations

import multiprocessing as mp

_LIMITS = None  # keeps the thread limits of a worker alive


def _single_thread() -> None:
    global _LIMITS
    import sys

    import cv2
    import numpy  # noqa: F401  (threadpoolctl limits only the libraries already loaded: BLAS here,
    import sklearn.ensemble  # noqa: F401  OpenMP here)
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    _LIMITS = threadpool_limits(1)


def _start(init, initargs: tuple) -> None:
    _single_thread()
    if init is not None:
        init(*initargs)


def pool(workers: int, spawn: bool = False, init=None, initargs: tuple = ()):
    """A pool of `workers` single-threaded processes, each running `init(*initargs)` once. Use `spawn=True` when
    the parent has already run OpenMP code (a scorer fit) or CUDA: forked children can deadlock."""
    ctx = mp.get_context("spawn") if spawn else mp.get_context()
    return ctx.Pool(workers, initializer=_start, initargs=(init, initargs))

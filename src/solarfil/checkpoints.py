"""Write a checkpoint without replacing a usable file with a partial save."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def save(state, path: Path | str) -> None:
    """Serialize beside `path`, flush it, then replace `path` atomically.

    An interrupted or failed write leaves the previous checkpoint available for
    resumption. The temporary file stays on the destination filesystem so the
    rename cannot become a copy across filesystems.
    """
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

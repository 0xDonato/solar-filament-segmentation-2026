"""A failed checkpoint write must preserve the previous resumable state."""
from unittest.mock import patch

import pytest
import torch

from solarfil.checkpoints import save


def test_save_replaces_a_complete_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    save({"step": 1, "w": torch.ones(2)}, path)
    save({"step": 2, "w": torch.zeros(2)}, path)
    state = torch.load(path, weights_only=True)
    assert state["step"] == 2
    torch.testing.assert_close(state["w"], torch.zeros(2))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_keeps_previous_checkpoint(tmp_path, existing):
    path = tmp_path / "ckpt.pt"
    if existing:
        save({"step": 1}, path)
    before = path.read_bytes() if existing else None

    def partial_write(state, stream):
        stream.write(b"partial checkpoint")
        raise OSError("No space left on device")

    with patch("solarfil.checkpoints.torch.save", side_effect=partial_write), pytest.raises(OSError):
        save({"step": 2}, path)
    assert path.read_bytes() == before if existing else not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])

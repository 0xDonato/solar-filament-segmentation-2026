"""The compact release's weight files: a state dict exported as safetensors shards reads back bit for bit."""
import importlib.util
import json
from pathlib import Path

import torch

from solarfil.pipeline import load_state

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_compact", REPO / "scripts" / "export_compact.py")
export_compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_compact)


def test_shards_read_back_identical(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(0)
    state = {"a.weight": torch.randn(64, 32, generator=g), "a.bias": torch.randn(64, generator=g),
             "b.weight": torch.randn(128, 64, generator=g), "bn.num_batches_tracked": torch.tensor(7)}
    torch.save(state, tmp_path / "net.pt")
    monkeypatch.setattr(export_compact, "SHARD_BYTES", 20_000)  # forces three shards
    files = export_compact.convert(tmp_path / "net.pt", tmp_path / "net.safetensors")
    assert len(files) > 1 and not (tmp_path / "net.safetensors").exists()
    back = load_state(tmp_path / "net.safetensors")
    assert back.keys() == state.keys()
    assert all(torch.equal(back[k], v) and back[k].dtype == v.dtype for k, v in state.items())


def test_one_file_when_small(tmp_path):
    state = {"w": torch.ones(3, 3)}
    torch.save(state, tmp_path / "net.pt")
    assert export_compact.convert(tmp_path / "net.pt", tmp_path / "net.safetensors") == [tmp_path / "net.safetensors"]
    assert torch.equal(load_state(tmp_path / "net.safetensors")["w"], state["w"])


def test_release_files_under_github_limit():
    weights = REPO / "weights"
    man = json.loads((weights / "manifest.json").read_text())
    names = [d["file"] for d in man["dense"]] + [f for group in man["instances"] for f in group]
    for name in names:
        stem = Path(name).stem
        files = [weights / name] if (weights / name).exists() else sorted(weights.glob(f"{stem}-*-of-*.safetensors"))
        assert files, name
        assert all(f.stat().st_size < 100 * 10**6 for f in files), name

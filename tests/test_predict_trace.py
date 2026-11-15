"""`predict.py --trace`: one row per frame with its seconds and memory, in both the frame-by-frame and the overlapped
path, and the same submission file as without the trace."""
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest
import torch

from test_pipeline import _pipeline
from test_preprocess import synthetic_frame

PREDICT = Path(__file__).parents[1] / "scripts" / "predict.py"


@pytest.mark.parametrize("workers", [0, 2])
def test_trace_writes_a_row_per_frame_and_leaves_the_submission_unchanged(tmp_path, workers):
    weights, images = tmp_path / "weights", tmp_path / "images"
    weights.mkdir()
    images.mkdir()
    _pipeline(weights, lean=True)
    stems = ["20140609195854Bh", "20140610195854Bh"]
    for seed, stem in enumerate(stems):
        cv2.imwrite(str(images / f"{stem}.jpeg"), synthetic_frame(seed=seed)[0])

    def predict(out: str, *extra: str) -> bytes:
        cmd = [sys.executable, str(PREDICT), "--images", str(images), "--weights", str(weights),
               "--out", str(tmp_path / out), "--workers", str(workers), *extra]
        run = subprocess.run(cmd, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert run.returncode == 0, run.stderr.decode(errors="replace")
        return (tmp_path / out).read_bytes()

    plain = predict("plain.csv")
    traced = predict("traced.csv", "--trace", str(tmp_path / "trace.csv"))
    assert traced == plain
    rows = list(csv.DictReader(open(tmp_path / "trace.csv")))
    assert [r["stem"] for r in rows] == stems and [int(r["frame"]) for r in rows] == [0, 1]
    assert all(float(r["seconds"]) > 0 for r in rows)
    for r in rows:
        if r["rss_main_mb"]:
            assert float(r["rss_mb"]) >= float(r["rss_main_mb"]) > 0
        else:
            assert r["rss_mb"] == ""  # the OS may deny process readings in a restricted session
    assert all(r["cuda_alloc_peak_mb"] == "" for r in rows)  # the CPU run has no CUDA readings


def test_denied_process_reading_keeps_trace_and_predictions(monkeypatch):
    spec = importlib.util.spec_from_file_location("predict_trace_test", PREDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def denied(*args, **kwargs):
        raise PermissionError("process access denied")

    monkeypatch.setattr(mod.subprocess, "run", denied)
    monkeypatch.setattr(mod.Path, "exists", lambda path: False)
    assert mod.rss_mb([os.getpid()]) == {}
    pipe = SimpleNamespace(dev=torch.device("cpu"), instances=lambda: "prediction")
    trace = mod.FrameTrace(pipe, ["frame"], workers=0)
    assert pipe.instances() == "prediction"
    assert len(trace.rows) == 1
    assert trace.rows[0]["rss_mb"] is None and trace.rows[0]["rss_main_mb"] is None


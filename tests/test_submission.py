"""Submission CSV: write, validate, and refuse overlapping masks."""
import numpy as np
import pytest

from solarfil import rle, submission


def _mask(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_write_and_validate(tmp_path):
    inst = {"20140609195854Bh": [rle.encode(_mask(10, 20, 10, 40)), rle.encode(_mask(30, 40, 10, 40))],
            "20110120105534Ch": []}
    path = tmp_path / "sub.csv"
    assert submission.write(inst, path) == 2
    check = submission.validate(path, list(inst))
    assert check["rows"] == 2 and check["images_with_instances"] == 1


def test_overlap_is_refused(tmp_path):
    inst = {"20140609195854Bh": [rle.encode(_mask(10, 20, 10, 40)), rle.encode(_mask(15, 25, 10, 40))]}
    path = tmp_path / "sub.csv"
    submission.write(inst, path)
    with pytest.raises(AssertionError, match="overlapping"):
        submission.validate(path, list(inst))

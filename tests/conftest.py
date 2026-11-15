"""Puts the tests folder on the import path, so that tests share helpers (`from test_preprocess import ...`)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

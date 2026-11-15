"""The code never reads the public MAGFiLO release, only the competition's own data.

The competition's data is the folder `MAGFiLO_1.0_Kaggle_2026` (its annotation file
`MAGFiLO_1.0_Annotations_kaggle2026_train.json`); it carries the dataset's name and is allowed. The public release
(Harvard Dataverse doi:10.7910/DVN/J6JNVK, mlecofi.net/magfilo, the Bitbucket repository `mleco-magfilodatacode`,
and notebooks and datasets built on it) contains the test labels, and the host disregards submissions that use it
(forum 729396, 746318).

Two checks:

* static: no release address appears anywhere in `src/`, `scripts/` or `reproduce.ipynb`; every string in the code
  that names MAGFiLO names the competition's Kaggle 2026 data; the code holds no web address but the weights
  release's; and every absolute path in it is a process file, a temporary file, or a folder this repository's own
  steps write (runs, cache, weights, submissions, report), never a data folder;
* dynamic: with `FILAMENT_DATA` at a temporary tree that holds only the competition's layout (and a decoy release
  folder beside it), parsing the annotations, listing the test frames and building the cache of a training and a
  test frame open nothing outside that tree and the cache folder (an audit hook on every open and directory
  listing, and a wrapper on OpenCV's image reads, which bypass Python's file layer).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

from test_preprocess import synthetic_frame

REPO = Path(__file__).resolve().parents[1]
COMPETITION = "MAGFiLO_1.0_Kaggle_2026"

# What identifies the public release and what is built on it (DESIGN "Rules kept").
RELEASE = re.compile(
    r"10\.7910/DVN/J6JNVK|J6JNVK|dataverse\.harvard\.edu|mlecofi|mleco-magfilodatacode|lamhuy8904"
    r"|solar-filament-unet-segmentation-0-55|swefil",
    re.IGNORECASE,
)
NAMES_DATASET = re.compile(r"magfilo", re.IGNORECASE)
COMPETITION_DATA = re.compile(r"kaggle_?2026", re.IGNORECASE)
URL = re.compile(r"[a-z][a-z0-9+.-]*://", re.IGNORECASE)
WEIGHTS_RELEASE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/releases/download/")
OWN_OUTPUT = re.compile(r"/(runs|report|submissions|cache/v\d+|weights/[\w.+-]+)/?$")
SYSTEM = ("/proc/", "/tmp/", "/dev/null")
FILESYSTEM = re.compile(
    r"^(~/|/(Volumes|Users|home|mnt|media|data|srv|opt|root|kaggle|workspace|content|tmp|proc|dev|var|private)(/|$))"
)


# Job-queue scripts for the original machines, which the public repository leaves out (machine paths, no data reads).
LAB_ONLY = {"box_queue.sh", "screen_eval.sh", "screen_watch.sh", "finish_run.sh", "drip_folds.py", "launch_folds.sh"}


def code_files(public: bool = False) -> list[Path]:
    """The Python and shell files of the package and the scripts; `public`: without the job-queue scripts."""
    files = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    files += sorted((REPO / "scripts").glob("*.sh"))
    return [f for f in files if not (public and f.name in LAB_ONLY)]


def notebook_code() -> str:
    """The code cells of `reproduce.ipynb`, joined."""
    nb = json.loads((REPO / "reproduce.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def string_constants(source: str) -> list[tuple[int, str]]:
    """Every string constant of a Python source except docstrings, with its line."""
    tree = ast.parse(source)
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        first = body[0] if isinstance(body, list) and body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            docs.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def strings_of(path: Path) -> list[tuple[int, str]]:
    """The strings a file's code uses: AST constants for Python, quoted words and bare paths for shell."""
    text = path.read_text()
    if path.suffix == ".py":
        return string_constants(text)
    return [(i, tok) for i, line in enumerate(text.splitlines(), 1) if not line.lstrip().startswith("#")
            for tok in re.findall(r"[^\s\"'=]+", line)]


def bad_string(s: str) -> str | None:
    """Why a string in the code is not allowed, or None."""
    if NAMES_DATASET.search(s) and not COMPETITION_DATA.search(s):
        return "names MAGFiLO but not the competition's Kaggle 2026 data"
    if URL.search(s) and not WEIGHTS_RELEASE.match(s):
        return "a web address other than the weights release"
    if FILESYSTEM.match(s):
        if not (s.startswith(SYSTEM) or OWN_OUTPUT.search(s)) or "/data" in s or NAMES_DATASET.search(s):
            return "an absolute path outside the repository's own outputs"
    return None


def test_no_release_address_in_the_code():
    hits = [f"{f.relative_to(REPO)}:{i}: {m.group(0)}" for f in code_files()
            for i, line in enumerate(f.read_text().splitlines(), 1) for m in RELEASE.finditer(line)]
    hits += [f"reproduce.ipynb: {m.group(0)}" for m in RELEASE.finditer(notebook_code())]
    assert not hits, hits


def test_strings_name_only_the_competition_data():
    hits = [f"{f.relative_to(REPO)}:{i}: {s!r}: {why}" for f in code_files(public=True) for i, s in strings_of(f)
            if (why := bad_string(s))]
    hits += [f"reproduce.ipynb:{i}: {s!r}: {why}" for i, s in string_constants(notebook_code())
             if (why := bad_string(s))]
    assert not hits, "\n".join(hits)


def test_the_rule_tells_the_competition_folder_from_the_release():
    assert bad_string(COMPETITION) is None
    assert bad_string("train/MAGFiLO_1.0_Annotations_kaggle2026_train.json") is None
    assert bad_string("/mnt/x/filament/runs") is None
    assert bad_string("MAGFiLO_1.0") is not None
    assert bad_string("/data/MAGFiLO/masks") is not None
    assert bad_string("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J6JNVK") is not None
    assert bad_string("https://github.com/someone/solarfil/releases/download/v6") is None
    assert RELEASE.search("doi:10.7910/DVN/J6JNVK") and not RELEASE.search(COMPETITION)


LOADER = r"""
import json, os, sys
from pathlib import Path
import cv2
from solarfil import cache, data, paths

root, cache_dir, train_stem, test_stem = sys.argv[1:5]
opened = []
def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir", "glob.glob") and args \
            and isinstance(args[0], (str, bytes, os.PathLike)):
        opened.append(os.fsdecode(args[0]))
read = cv2.imread
def imread(path, *a):
    opened.append(str(path))
    return read(path, *a)
cv2.imread = imread
sys.addaudithook(hook)

assert str(paths.DATA) == root, (paths.DATA, root)
ds = data.Dataset()
assert list(ds.observations) == [train_stem], list(ds.observations)
assert ds.test_stems == [test_stem], ds.test_stems
for sub in ("flat", "p", "spine"):
    (Path(cache_dir) / sub).mkdir(parents=True, exist_ok=True)
cache.build_one((ds.root, cache_dir, train_stem, True))
cache.build_one((ds.root, cache_dir, test_stem, False))
data.read_image(ds.image_path(test_stem))
sys.stdout.write("OPENED " + json.dumps(opened))
"""


def test_loader_opens_only_the_competition_tree(tmp_path):
    root = tmp_path / COMPETITION
    train_stem, test_stem = "20140609195854Bh", "20150101120000Lh"
    img, _ = synthetic_frame()
    for split, stem in (("train/train_images", train_stem), ("test/test_images", test_stem)):
        (root / split).mkdir(parents=True)
        cv2.imwrite(str(root / split / f"{stem}.jpeg"), img)
    coco = {
        "images": [{"id": f"100101-{train_stem}", "file_name": f"{train_stem}.jpeg"}],
        "annotations": [{"image_id": f"100101-{train_stem}",
                         "segmentation": "[[900, 800, 1150, 860, 1150, 870, 900, 810]]",
                         "spine": "[905, 805, 1145, 865]", "category_id": 3, "area": 2000.0}],
    }
    (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    decoy = tmp_path / "MAGFiLO_1.0"  # stands for the public release beside the competition folder
    (decoy / "test").mkdir(parents=True)
    (decoy / "test" / "masks.json").write_text("{}")
    cache_dir = tmp_path / "cache"

    env = {"FILAMENT_DATA": str(root), "FILAMENT_CACHE": str(cache_dir), "FILAMENT_RUNS": str(tmp_path / "runs"),
           "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-c", LOADER, str(root), str(cache_dir), train_stem, test_stem],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    opened = [Path(p).resolve() for p in json.loads(out.stdout.split("OPENED ", 1)[1])]

    allowed = (root.resolve(), cache_dir.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
               (REPO / "src").resolve())
    outside = [p for p in opened if not any(p == a or a in p.parents for a in allowed)]
    assert not outside, outside
    assert not [p for p in opened if decoy.resolve() in p.parents]
    data_files = {p for p in opened if root.resolve() in p.parents}
    assert root.resolve() / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json" in data_files
    assert root.resolve() / "train/train_images" / f"{train_stem}.jpeg" in data_files
    assert root.resolve() / "test/test_images" / f"{test_stem}.jpeg" in data_files

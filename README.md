# solarfil: compact solar filament instance segmentation

This repository converts one GONG H-alpha JPEG into separate, pixel-disjoint
filament masks for the Solar Filament Segmentation Challenge 2026. It contains
the prediction code, configurations, trained weights and reference predictions.
The compact pipeline uses four networks and 546 MB of weights.

Read [the method overview](docs/method-overview.md) for the reasoning,
[DESIGN.md](DESIGN.md) for parameters and validation, and
[VERIFICATION.md](VERIFICATION.md) for execution evidence.

## Reproduce the CPU submission

Use Python 3.12 and the pinned environment in `uv.lock`. From the repository,
prepare the notebook environment with `uv sync --group notebook`, then start
Jupyter with `uv run --group notebook jupyter lab` and open `reproduce.ipynb`.
Set `DATA` to the official competition data folder. The notebook reads its
`test/test_images/` directory and loads the contained `weights/` folder.

The settings explicitly select `DEVICE = "cpu"`, four PyTorch threads and the
manifest's cutoff 0.30. Keep `LIMIT = None` to process all 180 images. Running
the four code cells writes and validates `submission.csv`, then plots sample
outlines. `TAU = 0.26` selects the other retained operating point.

| CPU reference | Cutoff | Masks | Public PQ |
|---|---:|---:|---:|
| `release/submission_cpu_tau0.30.csv` | 0.30 | 1,152 | 0.41 |
| `release/submission_cpu_tau0.26.csv` | 0.26 | 1,210 | 0.40 |
| `release/submission_cpu_tau0.30_outline0.3.csv` | 0.30, outline step at `t_low` 0.3 | 1,152 | 0.41 |

The scored entries are 57854255, 57887823 and 58281141, respectively. The third, our final selection, adds the outline step (`OUTLINE = True`, `T_LOW = 0.3` in the notebook; `--outline --t-low 0.3` on the command line), which regrows thin side branches (barbs) that the selection trims; its masks keep the same objects and grow by 0.32% of their total area. Replaying the decision stage with this step on the cached CPU maps of the 0.30 execution reproduces its reference file on all 180 images. Complete CPU
notebook executions at 0.30 take 3.94 and 3.70 hours on an Apple M-series Mac
Pro; both reproduce the scored reference byte for byte. The second execution
checks the current explicit device/thread settings and all four cells on all
180 images. The separate 0.26 execution takes 4.59 hours. These are observations
on a shared machine, with no controlled speed comparison.

## GPU execution and reference scope

Set `DEVICE = "cuda"` for a CUDA GPU or `"mps"` for an Apple GPU. CUDA uses
bfloat16 dense maps; CPU and MPS use float32. Device arithmetic can change masks.
A complete explicit-MPS notebook run takes 284.30 seconds (4.74 minutes),
including loading and its figure. It produces 1,152 masks, with identical masks
to the scored CPU 0.30 reference on 179 of 180 images. One mask adds 51 pixels;
all 1,152 masks match above IoU 0.5 at mean IoU 0.999978. That MPS output is
unscored.

A complete 180-image notebook run on an RTX PRO 6000 Blackwell Max-Q takes
119.15 seconds, including loading and its figure, and produces 1,119 masks.
That output is unscored. Cross-device PQ equivalence is unmeasured.

The separately named `release/submission_tau0.26.csv` and
`release/submission_tau0.30.csv` contain NVIDIA GB10 references with 1,193 and
1,120 masks. Complete command-line runs on two GB10 machines reproduce those
files exactly. Against the GB10 0.30 file, 1,115 RTX masks match above IoU 0.5
at mean IoU 0.99062; five reference and four RTX masks are unmatched.

The command-line entrypoint is `uv run python scripts/predict.py --images
<test_images> --weights weights --out <new.csv>`. The notebook exposes the
backend directly for reference-specific reproduction. Outline growth is on only in
`release/submission_cpu_tau0.30_outline0.3.csv`.

## How the model makes a decision

Two native-resolution U-Nets with EfficientNetV2-S encoders produce an equally
weighted map, each averaged over four flips. The second recipe uses rotation,
zoom, longer training and a small target displacement with its opposite
inference correction. Two Mask R-CNN models provide instance evidence: one
learns from individual annotation records; the other samples one outline per
cross-record filament cluster.

Thresholding the dense map creates nested candidates and watershed split
parts. Each candidate has 41 image/tree descriptors and 12 detector columns.
A boosted classifier estimates matching frequency across annotation records;
a regressor estimates overlap conditional on a match. Their product scores
candidate utility. A bottom-up tree pass selects the exact best disjoint set
for the learned additive objective at the chosen cutoff.

The [one-image explanation](docs/selection-example.md) follows a scored CPU
prediction through actual candidate utilities and whole-versus-parts decisions.
Its reader reproduces all 16 masks from one JPEG and checks the additive
optimum independently.

The eight weight/scorer/manifest files total 546,224,480 bytes; the largest is
94,544,744 bytes. Networks load from float32 safetensors shards. Utility trees
load from primitive arrays. Inference needs one official JPEG. Training uses
the official competition images and labels with generic ImageNet/COCO
initialization. The decision layer follows the public filseg approach (`THIRD_PARTY_NOTICES.md`) credited
in the competition report.

## What the local evaluation measures

All 707 labelled observations enter the host's record-pooled native evaluation.
Four annotation-group blocks hold 214, 146, 180 and 167 observations. Each block
has its own two dense networks and two detectors: sixteen neural fits in total.
Every queried group is excluded from all four neural ancestors. Its utility
pair is fitted on the other groups inside the same neural held-out block.
Nineteen utility pairs cover every observation once. CPU float32 maps use four
flips per dense model and one rounding after their equal floating mean.

| Cutoff | Native group-confined PQ |
|---|---:|
| 0.26 | 0.4577 |
| 0.30 | 0.4576 |

At 0.30, the conditional 90% group-bootstrap interval is 0.4462–0.4737.
Removing both detector feature sets lowers PQ by 0.0096; the best of the same
14 fixed map thresholds scores 0.3990. Its threshold is selected in sample.
The report also compares identical 1,196 target-record weights over 296
multiply annotated observations: mean overlapping-pair IoU is 0.605 for the
model and 0.517 between annotators. Uncovered-reference differences remain
unresolved. The intervals condition on fitted networks, selected recipes and
observed groups; retraining, search and unknown test annotation styles are
outside their scope. Temporal proximity between observations remains possible.

The contained inference models use two full-data dense fits, fixed fold-0
models from each detector family, and one utility pair fitted on the complete
out-of-fold bank. They differ from the group-confined validation models. The
local PQ and its interval do not directly predict the private leaderboard.
See [report measures](docs/report-measures.md) for exact training overrides,
calibration roles, sufficient-statistic schema and table reproduction.

The [systematic hard-case read](docs/hard-cases.md) covers low contrast, long
spines, branched reference masks and the limb, including fixed failure examples.
Its saved group-confined native predictions and CPU reader reproduce the
complete annotation-derived breakdown, using the same exclusions as the report.

Annotation agreement measures consistency. Physical truth, hidden-reference
aggregation and a physical-chirality ceiling remain unmeasured. Public scores
round to two decimals and do not summarize the committee's full rubric.

## Training, analysis and licences

`FILAMENT_DATA`, `FILAMENT_CACHE` and `FILAMENT_RUNS` select the official data,
preprocessed cache and training products. `scripts/build_cache.py` prepares
flat fields and mean-record targets. The two deployment dense recipes are
`configs/e3_effv2s_n_long_lean*.json`; their default scheme is observation folds.
For the report's group exclusions, use `configs/report_group_n.json`,
`configs/report_group_w.json`, `configs/report_group_folds.json` and
`scripts/train_report_group.py`. Its validation-only mode checks the pools
without constructing a model. [Report measures](docs/report-measures.md)
explains all four roles and the nineteen utility splits.

`scripts/report_tables.py --out <new-results.json>` recomputes both report
tables and the full-system group interval from stored native per-observation
counts in `report/`. It performs no training or inference. Inference with the
supplied `weights/` requires no retraining. The shared exploratory analysis
helpers, including `ablation_metrics.py`, use their separately named banks;
those defaults do not construct the report's group-confined experiment.

`src/solarfil/` separates preprocessing, models, candidates, scoring, selection,
metrics and morphology. Tests cover the host metric, tree selection and safe
weight loading; data-dependent tests use the official data. Our code is MIT.
The timm encoders/initializers use Apache-2.0, and TorchVision uses BSD-3-Clause.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

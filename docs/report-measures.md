# Report validation and measures

The report's native validation uses all 707 official training observations,
1,154 annotation records, 8,199 polygons and 19 annotation groups. Every query
group is excluded from all upstream neural fits and its utility calibration.
The test set supplies no labels to this analysis.

## Neural training pools

`configs/report_group_folds.json` supplies the exact observation-to-block map.
Block IDs are 0, 2, 3 and 4. Their held-out sizes are 214, 146, 180 and 167;
their training sizes are 493, 561, 527 and 540. The same block exclusions apply
to N, W, the record detector and the across-record detector. Four blocks each
use four networks, giving sixteen neural fits.

The deployment recipe files `e3_effv2s_n_long_lean*.json` default to observation
folds. The explicit report recipes are `report_group_n.json` and
`report_group_w.json`, both with `scheme: group`. The report training entrypoint
overrides the fold loader with the four-block manifest for every neural role.
It never changes the deployment fold file.

After preparing the official-data cache, use `scripts/train_report_group.py`
with `--role n`, `w`, `record` or `union`, the required `--fold`, an output
root and cache path. For example, `uv run python scripts/train_report_group.py
--role n --fold 0 --out <report-runs> --cache <cache>` trains the first dense
role on 493 observations and predicts its 214 held observations. Apply each
role to each of the four block IDs. Adding `--validate-only` checks complete
frame coverage and group disjointness without constructing a model.

N uses the 12,000-update EfficientNetV2-S recipe with seed zero. W uses 24,000
updates, seed one, rotation/zoom and the 0.24-pixel target shift. The trainer
seeds each fold as `100 * seed + fold`. Both start from generic ImageNet
weights. Both detectors use the generic COCO initializer and the 24-epoch
defaults of `train_instances.py`, with group folds and seed zero. The `union`
role samples one outline per cross-record IoU cluster; it does not train on a
pixelwise OR of touching filaments. Their model settings are in `DESIGN.md`
and the source. Retraining constructs new fitted models and can change results.

## Native maps, candidates and utility roles

Each held-out observation reads its block's four models. CPU float32 inference
uses four flips per dense model, inverse displacement for W, equal averaging
of floating probability maps and one final eight-bit rounding. Detectors use
one view. The candidate bank has 41 image/tree columns, six record-detector
columns and six across-record-detector columns, in that order. Native target
and candidate intersections use the supplied polygons and host evaluation.

For one query group, calibration includes only the other groups in that same
neural held-out block. `report/calibration_roles.json` provides all nineteen
calibration/query frame-identifier sets and the complete ordered frame list. The utility
classifier and conditional-IoU regressor use record-count weights, 400
iterations, seven leaves, minimum leaf 40, learning rate 0.05 and L2 penalty
one. A query receives one prediction. Whole-group bootstrap draws preserve
all repeated record comparisons within an observation and group.

For component removals, retain the exact maps, candidates, native targets and
nineteen roles, remove the appropriate six-column detector block and refit
both utility estimators. Removing both detectors retains columns 0–40. The
fixed-threshold comparator uses the same dense map, hole filling and minimum
area of 20 pixels. Its threshold 0.55 is the best in-sample value among the
fourteen candidate levels; it is not an independently tuned test.

## Recompute the two tables

`uv run python scripts/report_tables.py --out <new-results.json>` recomputes
both tables directly from the included native sufficient statistics. This
requires neither a GPU nor new fitting or inference. The arrays contain
per-observation metric sums, frame IDs and group labels; no raw image or
polygon pixels are redistributed.

`report/component_rows.npz` contains all full/removal/candidate-threshold
arms. Each row is `[TP, FP, FN, matched-IoU sum]`. `report/native_rows.npz`
adds the number of positive-overlap pairs, IoU sum, Dice sum, record-comparison
count, uncovered references, uncovered predictions, splits and merges, in
that order. A split touches one reference with at least two predictions; a
merge touches one prediction with at least two references. These relations
use any positive overlap. PQ matches use strict IoU above 0.5 and complete
pair matrices, including overlapping reference polygons.

The annotation comparison covers 296 multi-record observations and twelve
groups. Every ordered pair of distinct records gives 1,196 comparisons. A
model-to-record row receives weight R−1 for R records, giving identical target
weights in both arms. Repetition adds no independent observation. Mean pair
IoU and Dice exclude zero-overlap pairs; uncovered-object counts and PQ retain
the associated misses.

Intervals are 90% percentile ranges from 2,000 whole-group resamples. Component
differences use seed zero; full-system native summaries use seed 20261102.
Paired differences reuse identical group draws. They condition on fitted
networks, reused-data recipe selection and the observed groups. They exclude
retraining, search and unknown test annotation styles. Annotation groups do
not provide independent temporal blocks.

## Fine-structure assessment

The report's exploratory chirality check uses a separate observation-held-out,
nonnested neural/utility construction. It has 1,935 model and 2,720 human
detected-reference readings from 269 observations, with unequal arm weights.
It does not isolate barb quality.

`solarfil.chirality` skeletonizes the largest mask component and uses a longest
geodesic path as the spine. Branches have length at least eight pixels. The
tangent spans ten spine nodes on each side of attachment; the signed
length-weighted branch-bearing sum defines an outline feature. A logistic
reader leaves one annotation group out, comparing hemisphere/latitude with
the same inputs plus branch-bearing features. At cutoff 0.26, accuracy gains
beyond latitude are 4.4 percentage points for model outlines and 11.4 for human
outlines. These are outline-consistency proxies; magnetic chirality remains
unvalidated. `scripts/chirality_legibility.py` defines the reading cohorts.

The stored inference models are separate from these validation fits: two
full-data dense models, fixed fold-0 detectors and one utility pair on the
full out-of-fold bank. Their complete scored CPU replay is documented in
`VERIFICATION.md`. Local validation intervals do not directly predict its
private leaderboard result.

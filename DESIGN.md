# Solar filament instance segmentation: design

The pipeline converts one official 2048 by 2048 grayscale H-alpha JPEG into
pixel-disjoint filament masks. The official training set contains 707 frames,
1154 annotation records and 8199 polygons. Different annotators can mark different
filaments or different outlines of one structure. The model learns agreement
with those records. Independent annotation agreement supplies a consistency
reference; physical truth and the ceiling for an image predictor remain unmeasured.

## Image evidence

A fitted solar limb defines the disk. Dividing the image by its median
centre-to-limb profile and a smooth background produces a flat field F. The
dense networks read 5(F-1), stored through the same 8-bit quantization used to
make their training caches. Each target is the per-pixel mean of the frame's
record-union masks. The target retains annotator disagreement. A mixture of
binary cross-entropy and soft Dice trains the dense map; the map is an image
feature and its probability calibration is not assumed.

The lean EfficientNetV2-S recipe uses 12000 crop updates. Network W uses 24000,
rotation/zoom and a 0.24 pixel downward target displacement, followed by the
opposite map displacement at inference. Cross-correlation on 1992 filaments
supports the small label/image offset. These changes form one recipe; an
isolated causal effect for the displacement is unmeasured. Four flip views
are averaged for each network. Full-data networks generate test maps;
corresponding neural-fold networks generate each training frame's OOF map.

This system loads two EfficientNetV2-S dense networks, lean seed 0 and W,
with equal map weights. Two Mask R-CNN families provide instance evidence at
1024 pixels: a random annotator record per sample on annotation-group folds,
and one sampled outline per cross-record filament cluster on frame folds.
The deployment training bank uses five folds per family; inference uses
fold 0 of each. The report uses the separate four-block group design below.
Thus inference loads
2 dense and 2 detector networks. The utility model reads 41 image/tree descriptors
and 12 detector columns, 53 in all.

Each detector contributes six columns per candidate: best mask IoU, the
associated detection score, maximum score times IoU, maximum score-weighted
coverage, the number of detections above IoU 0.3, and maximum score times
maskness times IoU. Detection scores enter as learned features; their empirical
calibration as an annotator-frequency probability is unproved. Detectors
supply evidence for existing tree candidates in this pipeline.

## Utility and exact selection

Fourteen map levels from 0.05 to 0.80, hole filling, 8-connected components and
watershed split parts define a component tree. A candidate is one connected,
hole-free mask. Its 41 descriptors summarize tree position, map values,
flat-field contrast, shape, disk position, crowding and frame statistics.
Filename, observation time and station metadata do not enter these features.

The utility target is the largest IoU above 0.5 within each annotation record,
averaged over a frame's records, with zero for an unmatched record. A classifier estimates the fraction h of
records with a match and a regressor estimates mean IoU m conditional on a
match. Their product estimates utility. The fitted gradient-boosted trees
use 400 iterations, 7 leaves, minimum leaf 40, learning rate 0.05 and L2 penalty 1.
Record-count weighting matches the pooled evaluation's comparison units.
The inference scorer stores primitive arrays.

The bottom-up recurrence compares zero, a node's utility minus tau, and the
sum of its children's optima. It solves the learned additive objective over
tree antichains exactly. Achieved PQ also depends on candidate coverage, utility-estimation error and
the evaluation distribution. The ratio-to-additive reduction
assumes unique matches. Overlapping reference polygons can let one prediction
match multiple references, so reported evaluation uses the host's actual
pair-counting matrices. The default tau is 0.30; 0.26 is the other retained operating
point. The optional outline-growth step is off in the reference CSVs.

## What validation measures

The report uses all 707 official training observations in four annotation-group
blocks, with 214, 146, 180 and 167 held observations. All four networks exclude
the same block before generating its dense maps and detector evidence. Every
query group has a separate utility pair trained on the remaining groups within
that neural held-out block. Nineteen utility pairs cover all observations.
Four blocks each contain four fitted networks, giving sixteen neural fits.

The numerical path matches CPU inference: float32 maps, four flip views per
dense model, inverse displacement for W, equal floating mean and one rounding.
Native masks are evaluated against all records using the host pair matrices.
At cutoff 0.30, PQ is 0.457556 and its conditional 90% group-bootstrap interval
is 0.446198–0.473700. Cutoff 0.26 gives 0.457718; its paired difference has an
interval crossing zero. Both detector feature sets together contribute
0.009631 PQ. The best fixed threshold among the same fourteen levels gives
0.398995; this in-sample choice favors that comparator.

The matched annotation comparison uses 296 observations from twelve groups.
Every ordered pair of distinct records yields 1,196 human readings. Repeating
each model-to-record comparison R−1 times, for R records, gives identical
target weights in both arms. Model/human mean overlapping-pair IoU is
0.605370/0.517448; split relations occur 0.170569/0.211538 times per comparison.
Uncovered-reference and uncovered-prediction differences have group intervals
crossing zero. These measures describe agreement with supplied annotations.
Physical completeness and magnetic chirality remain unmeasured.

Intervals use 2,000 whole-group percentile draws, paired for differences.
They condition on the shared fitted networks, selected configurations and
observed groups. Retraining variability, recipe search and unknown test
annotation styles are excluded. Group exclusion does not establish independent
future-period generalization because nearby observations remain possible.
`docs/report-measures.md` specifies the exact folds and training overrides;
`scripts/report_tables.py` recomputes both results tables from native counts.

The inference payload totals 546,224,480 bytes: six safetensors shards, one
primitive utility-array file and one manifest; the largest is 94,544,744 bytes.
The complete explicit-default CPU notebook executes all four cells and all
180 JPEGs in 13,305.27 seconds (3.70 hours), reproducing its 1,152-mask reference
CSV scored 0.41 byte for byte. Complete MPS and RTX notebook executions take
284.30 and 119.15 seconds, respectively, and produce unscored outputs.
`VERIFICATION.md` gives their environments and mask comparisons. These shared
machine observations establish completion; cross-device score equivalence and
a controlled speed ratio remain unmeasured.

## Data, licensing and execution

The deployed models use official competition labels and cleared generic
ImageNet/COCO initialization. Inference reads one official JPEG. Public MAGFiLO
test annotations, temporal neighbors, external filament labels and test
self-training do not enter the pipeline. The repository contains the trained
weights, configurations and prediction sources. See `THIRD_PARTY_NOTICES.md` for
software and initialization sources and the README for English environment
and execution instructions.

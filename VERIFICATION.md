# Verification

Execution scope for the contained models and reference predictions.

## Clean clone (2026-10-25, Apple M-series Mac Pro, macOS, MPS)

1. `git clone` of this repository into an empty folder (no other files on the machine were used).
2. `uv sync`, then `uv run pytest -q`: 107 tests passed.
3. `uv sync --group notebook`, then `reproduce.ipynb` executed with Jupyter (`jupyter nbconvert --execute`) with
   `LIMIT = 2`, `DATA` pointing at the competition's `MAGFiLO_1.0_Kaggle_2026` folder. It wrote a valid submission
   (3 filaments in 2 frames, 13.4 s per frame on MPS).
4. Those 3 masks against the same frames of `release/submission_tau0.26.csv` (`scripts/compare_submissions.py`,
   one-to-one matches at IoU > 0.5): 3 of 3 matched, mean IoU 0.963, none unmatched on either side.

## Rerun on a second NVIDIA GB10 (2026-10-25)

This repository's files on another GB10 (DGX Spark, CUDA 13), `uv sync`, then `scripts/predict.py --images
<test_images> --weights weights` at tau 0.26 and 0.30: both CSVs equal `release/` byte for byte (the SHA-256 sums in
`release/SHA256SUMS`).

## Complete notebook on CPU (2026-10-31)

All four code cells of the contained notebook completed on an Apple M-series
Mac Pro, using the existing pinned Python environment and contained weights.
The execution copy changed only the CSV output path and explicitly selected
the supported CPU device. All 180 official test JPEGs were processed; no
package was installed. The complete execution took 16,534.6 seconds (4.59 h),
including startup, weight loading, inference and the final three-image figure.
The notebook reports 91.72 seconds per frame for its inference section.

The resulting CSV validates: 1,210 masks on 177 nonempty frames. Against the
GB10 tau 0.26 reference, 1,175 masks match one to one at IoU > 0.5, with mean
IoU 0.96554; 18 reference masks and 35 CPU masks are unmatched. The executed
notebook has no cell errors, and its final contour figure was inspected.
This establishes complete CPU execution, not cross-device PQ equality.

## Complete scored-cutoff notebook on CPU (2026-11-01)

All four code cells complete on the Apple M-series Mac Pro at tau0.30,
processing all180 official test JPEGs. The existing pinned environment and
contained weights are used, with explicit CPU selection and tau0.30. Execution
takes14,201.9seconds (3.94hours), including startup, weight loading, inference
and final figure; the inference section reports78.78seconds per frame.

The valid CSV has1,152masks on177nonempty frames and is byte-identical to
`release/submission_cpu_tau0.30.csv`, public.41 (submission57854255). All four
execution counts are sequential, no cell error occurs, and the final three-image
contour figure is inspected. The explicit cutoff matches the current manifest's
default. Neural weights and utility arrays are unchanged. Shared-Mac wall times
do not establish a speed improvement over the separate tau0.26 execution.

## Complete explicit-default CPU notebook (2026-11-02)

All four cells of the current notebook complete sequentially with explicit
`DEVICE = "cpu"`, four Torch threads and the default cutoff 0.30. The existing
pinned environment and all contained weights are used. Only input/output paths
and per-frame completion logging change in the execution copy. All 180 official
test JPEGs complete in 13,305.266 seconds (3.70 hours), including startup,
loading, inference and the final figure. The inference section reports 73.80
seconds per frame.

The valid CSV contains 1,152 masks on 177 nonempty images. It equals the scored
`release/submission_cpu_tau0.30.csv` byte for byte; exact RLE sets also agree on
all 180 images, including empty images. No cell error occurs, and the embedded
three-image contour figure is inspected. This closes the full replay of the
newly exposed settings. It adds no score or controlled speed comparison.

## Complete notebook on RTX CUDA (2026-11-01)

All four code cells of the current contained notebook execute on an NVIDIA
RTX PRO 6000 Blackwell Max-Q at the default tau 0.30, processing all 180 official
test JPEGs. Only the data and output paths change. The existing scientific
environment uses PyTorch 2.14.1+cu130 and Python 3.12.3; the notebook runner
installs no packages. A separate Jupyter overlay supplies notebook execution.
The complete execution takes 119.148 seconds, including kernel startup, weight
loading, inference and the final three-image figure.

The CSV validates with 1,119 masks on 177 nonempty frames. All four execution
counts are sequential, no cell error occurs, and the final figure is inspected.
The CSV is unscored. Against the 1,120-mask GB10 tau 0.30 reference, 1,115 masks
match one to one above IoU 0.5, with mean IoU 0.990621; five reference and four
RTX masks are unmatched. Any RLE difference occurs in 163 of 180 frames.
This execution establishes completion on the RTX host. Reference byte parity,
cross-device PQ equivalence and a controlled speed ratio remain unmeasured.

## Complete explicit-MPS notebook (2026-11-02)

All four code cells complete with the notebook's explicit `DEVICE = "mps"`
setting and four Torch threads on the Apple M-series Mac Pro. The execution
uses the existing Python environment and contained weights, tau 0.30 and all
180 official JPEGs. Data/output paths and completion logging change in the
execution copy. No package is installed. Total execution takes 284.301 seconds;
the inference section reports 1.47 seconds per frame.

The valid output has 1,152 masks on 177 nonempty frames. All execution counts
are sequential, there are no cell errors, and the final figure is inspected.
Exact RLE sets equal the scored CPU 0.30 reference on 179 of all 180 images,
including empty images. The remaining image, `20130315063154Uh`, has one mask
with 51 added pixels, area 1,953 to 2,004 and IoU 0.974551; all other masks are
unchanged. Across both outputs all 1,152 masks match uniquely above IoU 0.5,
with mean IoU 0.999978 and none unmatched. The generic comparison script counts
177 nonempty images and reports 176 identical; the all-image check counts 180.

The MPS CSV is unscored. This measures full execution and prediction agreement
with the CPU reference, without assigning it the CPU score. Device and software
conditions are not controlled independently in these observations.

## Fresh environment on a second Mac (2026-11-03)

The committed code/model snapshot `9751434` is unpacked into an empty directory
on a second Apple M-series Mac Pro. A fresh virtual environment uses Python
3.12.14 and the release's pinned package versions. The machine's stricter
seven-day package cooldown remains in force. Resolving that policy changes only
the copied lockfile's cooldown metadata; all 162 package versions remain equal
to the source lockfile. The original release lockfile is unchanged.

All four notebook cells execute sequentially with the contained weights,
explicit CPU device, four Torch threads and cutoff 0.30. The execution copy
changes the input/output paths and sets `LIMIT = 2`. It uses the fresh virtual
environment's own kernel and excludes the user Python site. The complete
two-image execution takes 213.943 seconds, including startup, loading, inference
and the final figure. Its three masks on two nonempty images exactly equal
the corresponding RLE sets in the scored CPU reference. The figure is inspected.

The full fresh test run reports 104 passes, two data-dependent skips and one
attribution-string failure. The attribution is corrected to name the official
Kaggle 2026 dataset explicitly. A focused rerun of all four related guards and
the two real-data checks passes all six. Together these cover all 107 distinct
release tests; this is not a claim that the initial run was entirely green.

This is a bounded fresh-environment and second-machine check. The earlier
complete 180-image CPU executions establish the full replay scope. No full
second-machine replay, new score or controlled speed ratio follows from two images.

## Numerical scope across devices

The two unqualified `submission_tau*.csv` references use an NVIDIA GB10 with CUDA and bfloat16 dense maps.
MPS and CPU compute dense maps in float32. The complete RTX CUDA run also
differs from the GB10 reference, so hardware, software versions and numerical
paths are jointly relevant to these observations. Candidate thresholds and
selection can turn small numerical changes into different masks. The individual
contribution of each factor is unmeasured. The GB10 reference files retain their
own `release/SHA256SUMS`.

## Files

- `release/submission_tau0.26.csv` and `release/submission_tau0.30.csv`: public leaderboard 0.40 and 0.41;
  every mask is one 8-connected piece without holes (1,193 and 1,120 masks).

## Current notebook scope

The complete two-operating-point parity result uses the command-line pipeline.
The notebook execution checks cover all 180 frames on MPS at 0.30, all 180
frames on CPU at both cutoffs, and all 180 frames on RTX CUDA at 0.30.
The CPU .30 execution reproduces its scored reference exactly. The RTX run is
unscored and differs from the GB10 reference.

The saved October 27 notebook execution is also complete: four code cells,
all 180 JPEGs, 1,193 masks at tau 0.26, and a CSV byte-identical to the reference.
A Mac-side data read on October 31 checked its sequential execution counts,
absence of cell errors, completion summary, CSV and embedded three-image figure.
The saved October 25 and October 27 command-line CSVs match both operating-point
references byte for byte. These are historical executions; the current CPU
notebook result above remains the complete demonstration of today's notebook.

Saved frame traces record GPU-stage completions. Their 2.39/2.21 s means exclude
model loading and final CPU-stage drainage; no complete notebook wall-time claim
is inferred from those trace means.

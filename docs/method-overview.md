# How the compact pipeline works

From one H-alpha JPEG, draw a separate mask for each filament.
People disagree about which faint structures to include and where their
outlines end. We train against the official annotations and report that
disagreement. We do not equate agreement with physical truth.

**The image.** Fit the solar disk and remove the broad centre-to-limb brightness
profile and smooth background. This makes local contrast more comparable
across the disk. The model still reads the image's pixels. Filenames, dates,
station identities, neighbouring observations and test labels are not inputs.

**The four networks.** Two U-Nets estimate a filament map at the original
resolution, using different training recipes and four flip views each. Their
mean supplies candidate outlines. Two Mask R-CNN detectors provide evidence
about individual structures: one learns from individual annotation records,
the other from cross-record filament clusters. Their detections help score
candidates. Final masks come from the candidate tree. The second dense recipe includes a small
training-label displacement and the opposite inference correction. Its
combined recipe was evaluated; the displacement's isolated effect is unknown.

**The candidates.** Threshold the dense map at several levels. A low threshold
can join structures that separate at a higher threshold. Keep this nesting in
a component tree and add watershed split candidates. The candidate set offers
both whole structures and their parts, with connected masks and no holes.

**The score.** For each candidate, use its image contrast, shape, tree position
and detector agreement to estimate its best strict-match IoU in an annotation
record, assigning zero when it is absent. One boosted-tree model
estimates the chance of a match; another estimates overlap conditional on a
match. Their product is a learned utility. This does not prove that either
probability is calibrated on the hidden test.

**The decision.** At each tree node, choose the best of keeping nothing,
keeping that node, or keeping its selected children. Score each retained node
as utility minus a fixed cutoff. This gives the exact best solution for that
learned additive tree objective and prevents overlapping masks. It does not
give the exact best physical segmentation or guarantee optimal host PQ:
utility errors, missing candidates and overlapping reference polygons matter.
The demonstration notebook defaults to cutoff 0.30; 0.26 is the other documented point.
Outline growth is off.

**The evidence.** Evaluate all 707 labelled observations with the host's native
PQ and IoU/Dice/split/merge measures. Every query group is excluded from all
four neural models and its utility fit. Four held-out blocks require sixteen
neural fits; nineteen utility pairs cover all observations. PQ is 0.4576 at
the default cutoff, with a conditional 90% group interval of 0.4462–0.4737.
The matched annotation comparison uses identical target weights. Its overlap
and continuity measures describe consistency with expert drawings. Physical
completeness and magnetic chirality require separate evidence. Recipes share
reused observations and nearby observing times remain possible. Public 0.41
and the committee's quantitative/qualitative rubric supply further evidence.

**The practical choice.** Four networks and their scorer need 546 MB. A complete
contained CPU notebook reproduces the scored.41file over 180 images in 3.70 h.
The explicit MPS notebook finishes in 284 s and changes 51 pixels of one mask;
its score is unmeasured. The RTX notebook finishes in 119 s and produces a
different, unscored CSV. Device agreement and annotation agreement are measured
separately. Larger 22- and 7-network
alternatives offer quantitative/complexity tradeoffs. The committee weighs
quantitative results 70% and qualitative criteria 30%; speed is a feasibility
filter. The compact recommendation balances those criteria.

The decision layer follows the public filseg approach (`THIRD_PARTY_NOTICES.md`) credited in the competition report. See `DESIGN.md` for the complete parameters and `docs/report-measures.md` for evaluation definitions.

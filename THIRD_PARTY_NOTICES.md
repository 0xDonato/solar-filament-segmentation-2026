# Third-party notices

The code in this repository is MIT-licensed (`LICENSE`). It builds on the following
third-party work.

## Code and methods

- **filseg** (Kaggle notebook by tushhaaarrrr, "Solar Filament Segmentation 2026 (0.40)",
  https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-40;
  public Kaggle notebooks are released under Apache-2.0). The component-tree candidates,
  hurdle utility and dynamic-programming selection follow its approach; our
  implementation is written independently (`src/solarfil/instances.py`, `tree.py`).
- **Host self-evaluation notebook** (Azim Ahmadzadeh,
  https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook, Apache-2.0).
  `tests/host_reference.py` copies its scoring code, used only as the reference in the
  parity tests. A copy of the Apache License 2.0 is at
  https://www.apache.org/licenses/LICENSE-2.0.

## Libraries and initial weights

- **timm** (Apache-2.0): EfficientNetV2-S encoders and their ImageNet initialization.
- **segmentation-models-pytorch** (MIT): U-Net decoders.
- **TorchVision** (BSD-3-Clause): Mask R-CNN ResNet-50-FPN v2 and its COCO initialization.
- Other Python dependencies and their versions are pinned in `uv.lock`; each keeps its
  own licence.

## Data

Training uses the Solar Filament Segmentation Challenge 2026 competition data
(MAGFiLO annotations of GONG H-alpha images; GONG data courtesy of the NSO Integrated
Synoptic Program). The competition data are not included. The figures in `analysis/`
and the statistics in `analysis/` and `report/` are derived from the training data and
are provided for non-commercial academic research and education, as the competition
rules permit. The trained weights in `weights/` were fitted to that data.

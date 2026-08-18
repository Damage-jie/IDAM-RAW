# IDAM-RAW

Official implementation of **“IDAM-RAW: a sensor-aware RAW imaging framework for low-light perception via illumination decoupling and adaptive feature modulation,”** accepted by the *Journal of Electronic Imaging* (2026).

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Built%20on-Ultralytics-111F68.svg)](https://github.com/ultralytics/ultralytics)

[中文说明](README_zh-CN.md)

## Overview

IDAM-RAW performs object detection directly from Bayer RAW measurements under extremely low illumination. It jointly optimizes a physical sensor-domain front-end and detection-oriented representations instead of treating low-light enhancement as an independent preprocessing step.

The public code uses the terminology of the final paper:

| Paper name | Role |
|---|---|
| **DetISP** | Task-driven differentiable physical front-end |
| **RIDM** | Residual illumination decoupling module |
| **AFMM** | Adaptive feature modulation module |
| **Self-Boost** | Training-only consistency regularization |

Earlier internal implementation names have been replaced by these formal names.

## Installation

Python 3.9 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/Damage-jie/IDAM-RAW.git
cd IDAM-RAW
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Dataset preparation

Only source code is hosted in this GitHub repository. Dataset files are distributed separately.

> **CR7-RAW download (Baidu Netdisk): `BAIDU_NETDISK_LINK_TO_BE_ADDED`**

Replace the placeholder above with the final Baidu Netdisk URL and extraction code before announcing the dataset as public. Dataset acquisition, annotation, split, and integrity information is provided in the [CR7-RAW dataset card](docs/datasets/CR7_RAW.md).

Public benchmark sources:

- [LOD Dataset — official repository](https://github.com/ying-fu/LODDataset)
- [BDD100K — official repository](https://github.com/bdd100k/bdd100k)

Expected CR7-RAW layout:

```text
datasets/CR7_RAW/
├── images/
│   ├── train/       # 1,019 four-channel Bayer RAW samples
│   └── val/         # 140 four-channel Bayer RAW samples
└── labels/
    ├── train/       # 1,019 YOLO-format labels
    └── val/         # 140 YOLO-format labels
```

Validate the downloaded archive before training:

```bash
python scripts/check_dataset.py --root datasets/CR7_RAW \
  --expected-train 1019 --expected-val 140
```

If the dataset is stored elsewhere, update the `path` field in `configs/datasets/cr7_raw.yaml`.

## Training

The controlled experiments in the paper are trained from scratch with AdamW, an initial learning rate of `1e-3`, cosine annealing, random horizontal flipping, and no mosaic, mixup, copy-paste, or HSV color distortion. The three independent seeds are 2024, 2025, and 2026.

```bash
python scripts/train.py \
  --model configs/models/idam_raw_yolov8s.yaml \
  --data configs/datasets/cr7_raw.yaml \
  --device 0 --seed 2024
```

For distributed training, pass a comma-separated device list:

```bash
python scripts/train.py --device 0,1,2,3,4,5 --seed 2024
```

Repeat with `--seed 2025` and `--seed 2026` for the reported statistical protocol.

## Evaluation

```bash
python scripts/val.py \
  --weights path/to/best.pt \
  --data configs/datasets/cr7_raw.yaml \
  --device 0
```

Available YOLOv8s configurations:

| Configuration | File |
|---|---|
| Full IDAM-RAW | `configs/models/idam_raw_yolov8s.yaml` |
| DetISP only | `configs/models/detisp_yolov8s.yaml` |
| DetISP + RIDM | `configs/models/detisp_ridm_yolov8s.yaml` |
| DetISP + AFMM | `configs/models/detisp_afmm_yolov8s.yaml` |

## Main results

Mean ± sample standard deviation over three independent seeds for experiments under our control:

| Dataset | mAP@50 | mAP@50–95 |
|---|---:|---:|
| LOD | 0.727 ± 0.006 | 0.487 ± 0.002 |
| CR7-RAW | 0.713 ± 0.002 | 0.484 ± 0.001 |
| BDD-Night | 0.479 ± 0.005 | 0.261 ± 0.000 |

## Citation

```bibtex
@article{dai2026idamraw,
  title   = {IDAM-RAW: A Sensor-Aware RAW Imaging Framework for Low-Light Perception via Illumination Decoupling and Adaptive Feature Modulation},
  author  = {Dai, Minjie and Lai, Xingyu},
  journal = {Journal of Electronic Imaging},
  year    = {2026},
  note    = {Accepted}
}
```

The final DOI and bibliographic details will be added after publication.

## License and acknowledgments

The code is released under the [GNU Affero General Public License v3.0](LICENSE), consistent with the included Ultralytics-derived codebase. Dataset files are not covered by this code license and remain subject to their own terms. This project builds on [Ultralytics](https://github.com/ultralytics/ultralytics).

## Contact

Please open a GitHub issue or contact Minjie Dai at `2023021001000611@ecjtu.edu.cn`.

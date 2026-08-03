# Research Internship — Daily Log & Implementation

This repository tracks my day-to-day work during my research internship — tasks completed, experiments conducted, results, challenges, and next steps — and contains the working implementation for the Grounding DINO research task.

## Overview

The internship covers two parallel workstreams:

1. **PyTorch Fundamentals** — working through the [PyTorch for Deep Learning](https://www.udemy.com/course/pytorch-for-deep-learning/) course on Udemy.
2. **Grounding DINO Research** — exploring [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), an open-vocabulary object detection model, as a potential first stage in a deepfake-detection pipeline (i.e., locating faces/facial regions before applying a dedicated forensic model).

## Repository Structure

Each working day gets its own folder containing that day's materials (papers, notes, code, screenshots, results) plus a short log of what was done. The Grounding DINO implementation itself lives alongside the daily logs.

```
.
├── README.md
├── LOG_TEMPLATE.md
├── Day01/
│   ├── Grounding_DINO_Summary.md
│   └── log.md              # daily log entry (see format below)
├── Day02/
│   └── ...
├── Day03/
│   └── ...
├── ...
├── GroundingDINO/           # cloned official repo (its own README.md included)
├── weights/                 # pretrained checkpoints (gitignored — not committed)
├── src/                     # scripts (e.g. batch_inference.py)
├── data/                    # test images, organized by category
├── results/                 # annotated images + prediction files (JSON/CSV)
└── reports/                 # weekly progress reports, short report, presentation
```

## Environment

| | Version used |
|---|---|
| Python | 3.10 |
| PyTorch | 2.13.0 |
| transformers | 4.37 (pinned down from whatever newer default pip installed) |
| CUDA | N/A (macOS, no NVIDIA GPU) |
| GPU/chip | Apple M2 |
| MPS available | True (ran inference in `--cpu-only` mode for reliability/accuracy) |
| OS | macOS 26.6 |

## Installation

```bash
# Create environment
conda create -n groundingdino python=3.10 -y
conda activate groundingdino

# Install PyTorch (match to your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install GroundingDINO (vendored in this repo under GroundingDINO/)
cd GroundingDINO
pip install -e .
cd ..

# Download the pretrained Swin-T checkpoint
mkdir -p weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Installation errors and their fixes are logged day-by-day in the relevant `DayXX/log.md`.

## Usage

### Basic inference (single image)

```bash
python GroundingDINO/demo/inference_on_a_image.py \
  -c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p weights/groundingdino_swint_ogc.pth \
  -i <path-to-image> \
  -o <output-dir> \
  -t "human face" \
  --box_threshold 0.35 \
  --text_threshold 0.25
```

### Batch inference script

```bash
python src/batch_inference.py \
  --input_folder <path-to-image-folder> \
  --prompt "human face . eyes . mouth" \
  --box_threshold 0.35 \
  --text_threshold 0.25 \
  --output_folder <path-to-output>
```

Outputs: annotated images → `results/images/`, predictions (boxes, labels, confidence scores) → `results/predictions.json` (or `.csv`), with per-image inference time logged.

## Daily Log Format

Each `DayXX/log.md` follows this template:

```markdown
# Day XX — [Weekday], [Date]

## To-Do
- [ ] [Task 1]
  - Notes:
- [ ] [Task 2]
  - Notes:
- [ ] [Task 3]
  - Notes:

## Experiments conducted
-

## Results obtained
-

## Challenges encountered
| # | Challenge | Fix |
|---|-----------|-----|
|   |           |     |

## Planned next steps
-
```

## Weekly Progress Reports

A brief weekly progress report is prepared before each supervisor meeting, covering tasks completed, experimental results/code updates, problems or unsuccessful attempts, plans for the following week, and any support/resources required. These live in `reports/`.
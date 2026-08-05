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
│   └── subsets_for_thresholds/  # hand-picked image subset used for the Task D threshold sweep (not a primary category)
├── results/                 # annotated images + prediction files (JSON/CSV)
└── Reports/                 # weekly progress reports, short report, presentation
```

## Test Image Set

A curated set of 21 images spans four categories under `data/`:

| Category | Folder | Count |
|---|---|---|
| Multiple faces | `data/multiple/` | 4 |
| Occluded / blurred face | `data/occluded/` | 2 |
| Profile / angled face | `data/profile/` | 5 |
| Single frontal face | `data/single/` | 10 |

Sourcing: `single` images are pulled from the `logasja/lfw` dataset on Hugging Face via the `datasets` library:

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('logasja/lfw', split='train')
for i in range(10):
    ds[i]['image'].save(f'data/single/lfw_{i:02d}.jpg')
"
```

`multiple`, `occluded`, and `profile` images are stock photos from Unsplash/Pexels. Deepfake dataset frames (`data/deepfake/`) are planned but not yet implemented, as access to FaceForensics++ has been requested but not yet granted.

## Environment

| | Version used |
|---|---|
| Python | 3.10 |
| PyTorch | 2.13.0 |
| transformers | 4.37 |
| CUDA | N/A (macOS, no NVIDIA GPU) |
| GPU/chip | Apple M2 |
| MPS available | True (ran inference in `--cpu-only` mode for reliability/accuracy) |
| OS | macOS 26.6 |

## Installation

```bash
# Create environment
conda create -n groundingdino python=3.10 -y
conda activate groundingdino

# Install PyTorch (this project ran on macOS/Apple Silicon via MPS, no CUDA —
# if you have an NVIDIA GPU, install the matching CUDA build from pytorch.org instead)
pip install torch==2.13.0 torchvision

# Install GroundingDINO (vendored in this repo under GroundingDINO/)
# --no-build-isolation is required: plain `pip install -e .` fails with
# `ModuleNotFoundError: No module named 'torch'` inside pip's isolated build env
cd GroundingDINO
pip install -e . --no-build-isolation
cd ..

# GroundingDINO's setup can pull in a newer transformers than tested; pin if you hit issues
pip install transformers==4.37

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
  --text_threshold 0.25 \
  --cpu-only
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

Outputs: annotated images → `<path-to-output>/images/`, predictions (boxes, labels, confidence scores) → `<path-to-output>/predictions.json`, with per-image inference time logged.

This repo's `results/` folder contains the actual output from the experiments documented in Day05/log.md — e.g. `results/prompt_human_face/`, `results/thresh_box025_text020/`.

### Summarizing results

Use `src/summarize_results.py` to consolidate `num_boxes` (and related stats) from one or more `predictions.json` files, grouped by category and prompt — useful after running multiple prompts across the full image set.

### Running the threshold sweep

To compare how `--box_threshold` and `--text_threshold` affect detections on the same set of images, run `src/batch_inference.py` once per threshold combination against a fixed input folder, pointing each run at a distinct `--output_folder`:

```bash
python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box025_text020 --prompt "human face" \
  --box_threshold 0.25 --text_threshold 0.20 --cpu_only

python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box035_text025 --prompt "human face" \
  --box_threshold 0.35 --text_threshold 0.25 --cpu_only

python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box045_text030 --prompt "human face" \
  --box_threshold 0.45 --text_threshold 0.30 --cpu_only
```

Each run needs its own `--output_folder` — `predictions.json` is overwritten (not appended) if two runs share an output folder.

Use `src/compare_thresholds.py` to print `num_boxes`/scores across all `thresh_*` result folders side by side for comparison.

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

A brief weekly progress report is prepared before each supervisor meeting, covering tasks completed, experimental results/code updates, problems or unsuccessful attempts, plans for the following week, and any support/resources required. These live in `Reports/`.
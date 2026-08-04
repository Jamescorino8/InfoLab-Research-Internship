# Day 04 — Monday, 8/3

**Focus:** Batch inference script — build, debug, and test (Task E & Task C); begin image curation

## To-Do
- [ ✅ ] Curate 20+ test images: single frontal face, multiple faces, profile/occluded faces, small/low-res faces, real images
    - Note: Deepfake frames pending (FaceForensics++/DFDC access request submitted)
- [ ✅ ] Organize images into labeled folders by category
- [ ✅ ] Build initial batch-inference script (`src/batch_inference.py`) — accepts input folder, prompt, and thresholds as CLI args; saves annotated images + predictions.json; records per-image inference time
- [ ✅ ] Debug and resolve script errors (see Challenges below)
- [ ✅ ] Test script on a small known image before running full experiments

## Experiments conducted
- Wrote `src/batch_inference.py`, wrapping `load_model`, `load_image`, `predict`, and `annotate` from `groundingdino.util.inference` in a loop over an image folder, with `argparse` for `--input_folder`, `--output_folder`, `--prompt`, `--box_threshold`, `--text_threshold`, and `--cpu_only`.
- Ran a smoke test on a small `data/test_mini/` folder (reused Day03's sample image) to validate the script end-to-end before scaling up:
  - Confirmed clean exit with no traceback.
  - Confirmed `results/test_mini/images/` contained a correctly annotated output image.
  - Validated `predictions.json` with `python -m json.tool` and a manual script checking `num_boxes`, `scores`, and `inference_time_sec` per entry for sane values.
- Tested edge cases:
  - Empty input folder → exits cleanly, produces `predictions.json` as `[]`.
  - Non-image files mixed into the folder (e.g. `.DS_Store`) → correctly skipped by the extension filter.
  - Deliberately mismatched prompt (e.g. "bicycle" on a face photo) → correctly returns `num_boxes: 0` with no crash on drawing zero boxes.
  - Re-running into the same `--output_folder` with a different prompt → confirmed `predictions.json` is overwritten, not appended (noted as a design consideration for Task D, where threshold sweeps will need distinct output folders per run).
- Visually spot-checked annotated output images for correct box placement and normal color channels (no BGR/RGB mismatch).
- Curated and organized images for four of five categories (`single_frontal`, `multiple_faces`, `profile_occluded`, `small_lowres`) into `data/` subfolders:
  - Attempted to source `single_frontal` faces from LFW (official `vis-www.cs.umass.edu` host), but the server was unresponsive.
  - Used stock portrait/group photos from Unsplash and Pexels instead for `single_frontal`, `multiple_faces`, and `profile_occluded` (sunglasses, hats, side angles, partial hand occlusion).
  - Generated `small_lowres` images by downscaling existing images to a small resolution then upscaling back, to simulate real low-resolution capture rather than just shrinking the display size.

## Results obtained
- All smoke tests and edge-case tests passed. Script reliably produces annotated images and a valid `predictions.json` with `image`, `category` (inferred from subfolder name), `prompt`, thresholds, `num_boxes`, `phrases`, `scores`, and `inference_time_sec` per entry.
- Script is ready to run against the curated set.
- Four of five image categories are curated and organized (`single_frontal`, `multiple_faces`, `profile_occluded`, `small_lowres`). The fifth category, deepfake dataset frames, is not yet available — see note below.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | `TypeError: load_model() got an unexpected keyword argument 'cpu_only'` | Assumed a `cpu_only` param by analogy to the CLI demo flag, but the actual function signature differs. Checked with `inspect.signature(load_model)` and switched to the correct `device` parameter. |
| 2 | `AssertionError: Torch not compiled with CUDA enabled` inside `predict()`, even after fixing `load_model` | `predict()` has its own independent `device` parameter defaulting to `"cuda"`, unrelated to the device passed into `load_model()`. Fixed by explicitly passing `device="cpu"` into the `predict()` call as well. Grepped `inference.py` for all `device` occurrences to confirm no other function had the same silent default. |
| 3 | LFW official host (`vis-www.cs.umass.edu`) unresponsive when attempting download | Used Unsplash/Pexels stock photos as an alternate source for the affected categories instead of pursuing an LFW mirror, since only a small number of images were needed. |

## Planned next steps
- Run the tested script across the four available categories with the five specified prompts and log per-category observations for Task C.
- Begin threshold sweep (Task D) on a representative subset of the available images.
- Follow up on FaceForensics++/DFDC access requests; incorporate deepfake frames once available.
# Day 05 — Tuesday, 8/4

**Focus:** Task C prompt experiments + begin Task D threshold sweep, on available categories (deepfake frames still pending)
## To-Do
- [ ✅ ] Confirm final image counts per category before running experiments
- [ ✅ ] Run detections with prompts "human face" and "person" across the full available set (single_frontal, multiple_faces, profile_occluded, small_lowres)
- [ ✅ ] Save annotated outputs and note observations per category
- [ ✅ ] Run detections with prompts "face . eyes . mouth", "real face", "manipulated face"
- [ ✅ ] Document results per category; note that "real face" / "manipulated face" results are exploratory, not reliable forensic signals
- [ ✅ ] Consolidate per-category/per-prompt results (num_boxes, scores) into a summary
- [  ] Run initial threshold sweep (Task D) on a representative subset (e.g. 1 image each from single_frontal, profile_occluded, small_lowres) across 3 box/text threshold combinations
- [  ] Compare threshold sweep results for missed faces, duplicate boxes, false detections, confidence scores
- [  ] Update README to reflect final image counts/categories used, note on pending deepfake category, and instructions for running the threshold sweep
- [  ] Write short report: model summary, experiment methodology, Task C results (prompt behavior patterns), Task D threshold findings, limitations, encountered problems
- [  ] Organize experiment notes and screenshots into a running report draft

## Experiments conducted

## Results obtained

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
|  |  |  |

## Planned next steps
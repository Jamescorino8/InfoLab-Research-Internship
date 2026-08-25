# Weekly Progress Report — Tuesday, 8/18 (before next meeting)

## Tasks completed this period

- Reviewed Tamer's Aug 2026 feedback on the initial findings report and translated it into a concrete work plan (Fri 8/14 – Tue 8/18).
- Researched RWDF-23 and FakeAVCeleb (DASH Lab) as alternative deepfake-video sources: both are Google Form-gated (FakeAVCeleb also requires a license agreement) — submitted Fri 8/14.
- Sent a follow-up on the still-pending FaceForensics++ request.
- Added a 12-image Kaggle deepfake-image sample (`manjilkarki/deepfake-and-real-images`, Test/Fake split) to `data/deepfake/` as a clearly-labeled, non-forensic placeholder pending approval on one of the three real datasets.
- Expanded `occluded` from 2 → 7 images (masks, hands, hair, sunglasses+scarf) and `multiple` from 4 → 5 images (one new dense crowd scene).
- Added a new `small_lowres` category (6 images) — thumbnail-resolution downscales of already-curated photos, isolating resolution as a variable.
- Re-ran `src/batch_inference.py` across all new/expanded categories with the standard 5-prompt set, and ran the threshold sweep (0.25/0.20, 0.35/0.25, 0.45/0.30) on occluded, multiple, and small_lowres.
- Built `Reports/GroundingDINO_Report_v2.pdf` — a standalone update to the original findings report (not an addendum to paste into the old one) covering all of the above, including the crowd-density finding below.
- [PyTorch course: log sections completed]

## Results / code updates

- **New failure mode found:** the new, denser `multiple` addition (hundreds of people, individual faces ~30–40px) returned **zero detections** across 4 of 5 prompts at default threshold, vs. 18 boxes on the original Day05 crowd outlier. The threshold sweep confirmed this isn't a tuning problem — even the most permissive setting (0.25/0.20) recovers only 2 very-low-confidence boxes out of hundreds of visible faces. This is a more significant limitation than the duplicate-box behavior documented previously, and is written up as its own section in Report v2.
- **Deepfake-sample prompt results (exploratory, 12-image Kaggle placeholder):** "real face" (1.25 avg boxes/image) and "manipulated face" (1.17) both track closely with plain "human face" (1.33) — consistent with the Day05 finding on ordinary photos, now replicated on a (still non-forensic) deepfake-labeled sample. No evidence the model is distinguishing genuine from manipulated content via prompt wording alone.
- **Threshold sweep on new categories:** occluded and small_lowres (single/profile-derived) behave like the original Day05 categories — threshold mainly trades duplicate boxes vs. missed genuine ones, 0.35/0.25 remains a reasonable default. Dense-scene images don't respond the same way — lowering threshold recovers duplicates on a scene the model already detected, but does little for one it fundamentally missed.

## Problems or unsuccessful attempts

- First threshold-sweep script had a shell-portability bug (word-splitting behaves differently in zsh vs. bash on macOS) that silently failed all 9 runs with an argparse error — caught it from the error output, fixed, re-ran successfully. Documented in Day 12 log and Report v2's Challenges section.

## Plan for the following week

- Continue following up on FaceForensics++/RWDF-23/FakeAVCeleb access.
- Once one of the three datasets is granted, re-run the "real face"/"manipulated face" prompt comparison on genuine manipulated content and replace the Kaggle placeholder.
- Investigate the crowd-density detection failure further (e.g. tiling the image before inference, or a higher input resolution) as a follow-up technical direction.
- Continue PyTorch course.

## Support / resources needed

- **Flagging for possible help expediting:** all three dataset requests (FaceForensics++, RWDF-23, FakeAVCeleb) are still pending as of this report. If there's a faster path — e.g. Tamer/the lab vouching directly for RWDF-23/FakeAVCeleb since they're DASH Lab's own datasets — that would unblock replacing the current Kaggle placeholder with forensic-quality frames.
- Possibly worth a quick sanity check with Firuz/Abdenour on the crowd-density zero-detection finding — is this expected model behavior at this face-density/scale, or worth investigating further (e.g. tiling the image before inference)?

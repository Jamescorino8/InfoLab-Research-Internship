# src/make_small_lowres.py
#
# Generates data/small_lowres/ from existing curated photos by saving them at
# genuinely reduced pixel dimensions (not just visually blurred) -- simulates
# thumbnail/distant-camera resolution faces, the condition Tamer flagged as
# missing from the test image set (Aug 2026 feedback).
#
# Re-run with: python src/make_small_lowres.py

from pathlib import Path
from PIL import Image

# (source image, output filename, target size of the longest edge in pixels)
JOBS = [
    ("data/single/lfw_00.jpg", "thumb_single_lfw00.jpg", 40),
    ("data/single/lfw_03.jpg", "thumb_single_lfw03.jpg", 56),
    ("data/profile/pexels-angela-roma-7480302.jpg", "thumb_profile.jpg", 48),
    ("data/occluded/pexels-celine-3776818-11430254.jpg", "thumb_occluded.jpg", 48),
    ("data/multiple/tim-mossholder-hOF1bWoet_Q-unsplash.jpg", "thumb_multiple_01.jpg", 90),
    ("data/multiple/windows-p74ndnYWRY4-unsplash.jpg", "thumb_multiple_02.jpg", 90),
]

def main():
    out_dir = Path("data/small_lowres")
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, name, longest in JOBS:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        scale = longest / max(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        im.resize(new_size, Image.LANCZOS).save(out_dir / name, quality=90)
        print(f"{src} {w}x{h} -> {out_dir/name} {new_size[0]}x{new_size[1]}")

if __name__ == "__main__":
    main()

# src/batch_inference.py
import argparse
import json
import time
from pathlib import Path

import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

def main():
    # Define what flags the script accepts (required/type/default) from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    # Convert string args into Path objects and create output directory stricture
    input_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # Load the model once outside the loop
    model = load_model(args.config, args.checkpoint, device="cpu" if args.cpu_only else "cuda")

    # Iterate over every file in input_dir and keep only image files
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)

    # 
    results = []
    for img_path in image_paths:
        # image_source = raw image ; image = tensor version
        image_source, image = load_image(str(img_path)) 

        # runs model and return three lists per detected box: coordinates (boxes), confidence score (logits), and matched text phrase (phrases)
        t0 = time.time()
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device="cpu" if args.cpu_only else "cuda",
        )
        # Inference time covers only the predict() call
        elapsed = time.time() - t0 

        # Draws bounding boxes and labels onto original image and returns image arra
        annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        out_img_path = out_dir / "images" / img_path.name
        cv2.imwrite(str(out_img_path), annotated)

        # Builds one dictionary per image and appends to results list
        results.append({
            "image": str(img_path.relative_to(input_dir)), # stores relative path not absolute
            "category": img_path.parent.name,  # e.g. "single_frontal", "occluded"
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "num_boxes": len(boxes),
            "boxes": boxes.tolist(),
            "phrases": phrases,
            "scores": [round(float(l), 3) for l in logits],
            "inference_time_sec": round(elapsed, 3),
        })
        print(f"{img_path.name}: {len(boxes)} boxes, {elapsed:.2f}s")

    # write list of dictionaries to JSON file
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
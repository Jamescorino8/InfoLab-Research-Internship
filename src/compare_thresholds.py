import json
from pathlib import Path

for run_dir in sorted(Path("results").glob("thresh_*")):
    data = json.load(open(run_dir / "predictions.json"))
    print(run_dir.name)
    for entry in data:
        print(f"  {entry['image']}: {entry['num_boxes']} boxes, scores={entry['scores']}")
    print()
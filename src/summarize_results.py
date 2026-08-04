import json
from pathlib import Path

summary = {}
for run_dir in Path("results").glob("prompt_*"):
    data = json.load(open(run_dir / "predictions.json"))
    prompt = data[0]["prompt"] if data else run_dir.name
    for entry in data:
        cat = entry["category"]
        summary.setdefault(cat, {}).setdefault(prompt, []).append(entry["num_boxes"])

for cat, prompts in summary.items():
    print(cat)
    for prompt, counts in prompts.items():
        print(f"  {prompt}: {counts}")
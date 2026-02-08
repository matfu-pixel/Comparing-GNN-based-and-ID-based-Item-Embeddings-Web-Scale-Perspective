import json
import os
from statistics import mean


FOLDER = "./logs"

for filename in os.listdir(FOLDER):
    if not (filename.startswith("finetune") and filename.endswith(".json")):
        continue

    filepath = os.path.join(FOLDER, filename)

    with open(filepath, "r") as f:
        data = json.load(f)
    if not data:
        continue

    metrics = data[0].keys()
    averages = {}
    for metric in metrics:
        values = [entry[metric] for entry in data if metric in entry]
        if values:
            averages[metric] = mean(values)

    print(f"Average statistics from {filename} across {len(data)} runs:")
    for metric, avg_value in averages.items():
        print(f"- {metric}: {avg_value}")
    print()

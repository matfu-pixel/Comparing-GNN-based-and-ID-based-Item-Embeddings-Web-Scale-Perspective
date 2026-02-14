import json
import logging
import os
from collections import defaultdict
from statistics import mean


FOLDER = "./logs"
OUTPUT_PATH = os.path.join(FOLDER, "averaged.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

results = defaultdict(dict)

for filename in sorted(os.listdir(FOLDER)):
    if not (filename.startswith("finetune") and filename.endswith(".json")):
        continue

    filepath = os.path.join(FOLDER, filename)
    run_name = filename.removesuffix(".json")

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

    for metric, avg_value in averages.items():
        results[metric][run_name] = avg_value

with open(OUTPUT_PATH, "w") as f:
    json.dump(dict(results), f, indent=4)

logger.info(f"Averaged results saved to {OUTPUT_PATH}")

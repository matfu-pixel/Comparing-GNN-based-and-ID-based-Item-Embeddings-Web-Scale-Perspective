#!/bin/bash
set -e

N_RUNS=${1:-10}

echo "Running $N_RUNS iterations"

for run in $(seq 1 "$N_RUNS"); do
    ./run_pipeline.sh --train-twhin
    ./run_pipeline.sh --train-pretrain
    ./run_pipeline.sh --train-finetune
done

uv run scripts/auxiliary/average_results.py
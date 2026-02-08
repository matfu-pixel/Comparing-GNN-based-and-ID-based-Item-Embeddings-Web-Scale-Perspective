#!/bin/bash
set -e

for run in {1..10}; do
    ./run_pipeline.sh --train-twhin
    ./run_pipeline.sh --train-pretrain
    ./run_pipeline.sh --train-finetune
done

uv run scripts/auxiliary/average_results.py
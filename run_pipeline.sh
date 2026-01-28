#!/bin/bash
set -e

DATA_DIR="./data"
MODEL_DIR="./models"
LOGS_DIR="./logs"

mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR
mkdir -p $LOGS_DIR

RUN_PREPARE=false
RUN_TWHIN=false
RUN_PRETRAIN=false
RUN_FINETUNE=false

# --------------------
# Parse arguments
# --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare)
      RUN_PREPARE=true
      ;;
    --train-twhin)
      RUN_TWHIN=true
      ;;
    --train-pretrain)
      RUN_PRETRAIN=true
      ;;
    --train-finetune)
      RUN_FINETUNE=true
      ;;
    --train)
      RUN_TWHIN=true
      RUN_PRETRAIN=true
      RUN_FINETUNE=true
      ;;
    --all)
      RUN_PREPARE=true
      RUN_TWHIN=true
      RUN_PRETRAIN=true
      RUN_FINETUNE=true
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

# If no args given → default to all
if ! $RUN_PREPARE && ! $RUN_TWHIN && ! $RUN_PRETRAIN && ! $RUN_FINETUNE; then
  RUN_PREPARE=true
  RUN_TWHIN=true
  RUN_PRETRAIN=true
  RUN_FINETUNE=true
fi

# --------------------
# Preprocessing
# --------------------
if $RUN_PREPARE; then
    echo "Start preprocessing..."

    python scripts/data/preprocess.py \
        --input ./data/dataset.parquet \
        --output-train-twhin $DATA_DIR/train_twhin.parquet \
        --output-test-twhin $DATA_DIR/test_twhin.parquet \
        --output-train-pretrain $DATA_DIR/train_pretrain.parquet \
        --output-test-pretrain $DATA_DIR/test_pretrain.parquet \
        --output-train-finetune $DATA_DIR/train_finetune.parquet \
        --output-test-finetune $DATA_DIR/test_finetune.parquet \
        --time-split 1703451224
fi

# --------------------
# Train TWHIN
# --------------------
if $RUN_TWHIN; then
    echo "Training TWHIN..."

    uv run scripts/train/train_twhin.py \
        --train-data $DATA_DIR/train_twhin.parquet \
        --test-data $DATA_DIR/test_twhin.parquet \
        --user-cardinality 3315 \
        --item-cardinality 25834 \
        --relation-cardinality 3 \
        --embedding-dim 64 \
        --reg-weight 0.1 \
        --batch-size 8192 \
        --learning-rate 0.001 \
        --num-epochs 15 \
        --device cuda \
        --output-log-dir $LOGS_DIR \
        --output-model-dir $MODEL_DIR
fi

# --------------------
# Pretrain
# --------------------
if $RUN_PRETRAIN; then
    echo "Training pretrain..."

    python train_pretrain.py \
        --train-data $DATA_DIR/train_pretrain.parquet \
        --test-data $DATA_DIR/test_pretrain.parquet \
        --item-cardinality 25834 \
        --embedding-dim 64 \
        --batch-size 1024 \
        --learning-rate 0.001 \
        --num-epochs 15 \
        --device cuda \
        --output-log-dir $LOGS_DIR \
        --output-model-dir $MODEL_DIR
fi

# --------------------
# Finetune
# --------------------
if $RUN_FINETUNE; then
    echo "Training finetune..."

    python train_finetune.py \
        --train-data $DATA_DIR/train_finetune.parquet \
        --test-data $DATA_DIR/test_finetune.parquet \
        --batch-size 1024 \
        --learning-rate 0.001 \
        --num-epochs 15 \
        --device cuda \
        --output-log-dir $LOGS_DIR \
        --output-model-dir $MODEL_DIR
fi
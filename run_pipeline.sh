#!/bin/bash
set -e

DATA_DIR="./tmp/data"
MODEL_DIR="./tmp/models"
LOGS_DIR="./tmp/logs"

mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR
mkdir -p $LOGS_DIR

echo "Start preprocessing..."

python preprocess.py \
    --input ./data/dataset.parquet \
    --output-train-twhin $DATA_DIR/train_twhin.parquet \
    --output-test-twhin $DATA_DIR/test_twhin.parquet \
    --output-train-pretrain $DATA_DIR/train_pretrain.parquet \
    --output-test-pretrain $DATA_DIR/test_pretrain.parquet \
    --output-train-finetune $DATA_DIR/train_finetune.parquet \
    --output-test-finetune $DATA_DIR/test_finetune.parquet \
    --time-split 1703451224

echo "End preprocessing..."

echo "Start training..."

python train_twhin.py \
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

python train_finetune.py \
    --train-data $DATA_DIR/train_finetune.parquet \
    --test-data $DATA_DIR/test_finetune.parquet \
    --batch-size 1024 \
    --learning-rate 0.001 \
    --num-epochs 15 \
    --device cuda \
    --output-log-dir $LOGS_DIR \
    --output-model-dir $MODEL_DIR

echo "End training..."
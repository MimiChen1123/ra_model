#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEVEL="${LEVEL:-I}"
TRAIN_DATA="${TRAIN_DATA:-data/${LEVEL}_train.json}"
QUESTION_DATA="${QUESTION_DATA:-data/question.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${LEVEL}_translation}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-5}"
TEST_SIZE="${TEST_SIZE:-0.1}"
SEED="${SEED:-0}"

"$PYTHON_BIN" train.py \
  --model "$MODEL" \
  --essay_data "$TRAIN_DATA" \
  --question_data "$QUESTION_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --test_size "$TEST_SIZE" \
  --seed "$SEED" \
  "$@"

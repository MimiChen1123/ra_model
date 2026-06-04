#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEVEL="${LEVEL:-HI}"
TEST_DATA="${TEST_DATA:-/home/mimi911123/ra_model/eval_data/HI_translation_answers.json}"
QUESTION_DATA="${QUESTION_DATA:-/home/mimi911123/ra_model/eval_data/HI_translation_questions.json}"
CHECKPOINT="${CHECKPOINT:-/tmp/models/mistral_with_gemma/translation/${LEVEL}_translation.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${LEVEL}_eval}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-0}"

"$PYTHON_BIN" eval.py \
  --model "$MODEL" \
  --test_data "$TEST_DATA" \
  --question_data "$QUESTION_DATA" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  "$@"

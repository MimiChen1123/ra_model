#!/usr/bin/env bash
set -euo pipefail

# Real-world inference pipeline:
# raw test JSON -> rubric scores -> Gemma CEFR -> word_count -> PyCaret inference

conda activate gept_writing_eval

LEVEL="${LEVEL:-HI}"
TASK_TYPE="${TASK_TYPE:-classification}"

INPUT_DATA="${INPUT_DATA:-./data/${LEVEL}_test.json}"
QUESTION_PATH="${QUESTION_PATH:-./data/${LEVEL}_train_question.json}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_preprocess}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-12b-it}"
MODEL_PATH="${MODEL_PATH:-./model/classification/level_${LEVEL}/ada.pkl}"
PREDICTION_OUTPUT="${PREDICTION_OUTPUT:-./output/${LEVEL}_test_prediction.json}"

mkdir -p "${WORK_DIR}"

RUBRIC_OUTPUT="${WORK_DIR}/${LEVEL}_test_rubric.json"
CEFR_OUTPUT="${WORK_DIR}/${LEVEL}_test_rubric_cefr.json"
READY_INPUT="${WORK_DIR}/${LEVEL}_test_ready.json"

python rubric_inference.py \
  --input "${INPUT_DATA}" \
  --output "${RUBRIC_OUTPUT}" \
  --question_path "${QUESTION_PATH}"

python cefr_label_infer_extract.py \
  --input "${RUBRIC_OUTPUT}" \
  --output "${CEFR_OUTPUT}" \
  --model "${GEMMA_MODEL}"

python length_feature_extraction.py \
  --input_file "${CEFR_OUTPUT}" \
  --output_file "${READY_INPUT}" \
  --level "${LEVEL}"

python pycaret_inference.py \
  --input_data_path "${READY_INPUT}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${PREDICTION_OUTPUT}" \
  --type "${TASK_TYPE}"

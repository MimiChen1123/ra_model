#!/usr/bin/env bash
set -euo pipefail

# Real-world inference pipeline:
# raw test JSON -> rubric scores -> Gemma CEFR -> word_count -> PyCaret inference

source ~/anaconda3/bin/activate pycaret

LEVEL="${LEVEL:-HI}"
TASK_TYPE="${TASK_TYPE:-classification}"

INPUT_DATA="${INPUT_DATA:-/home/mimi911123/ra_model/eval_data/HI_essay_answers.json}"
QUESTION_PATH="${QUESTION_PATH:-/home/mimi911123/ra_model/eval_data/HI_essay_questions.json}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_preprocess}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-12b-it}"
MODEL_PATH="${MODEL_PATH:-/tmp/models/pycaret/essay/classification/level_HI/lda.pkl}"
PREDICTION_OUTPUT="${PREDICTION_OUTPUT:-./output/${LEVEL}_prediction.json}"

mkdir -p "${WORK_DIR}"

RUBRIC_OUTPUT="${WORK_DIR}/${LEVEL}_rubric.json"
CEFR_OUTPUT="${WORK_DIR}/${LEVEL}_rubric_cefr.json"
READY_INPUT="${WORK_DIR}/${LEVEL}_ready.json"

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

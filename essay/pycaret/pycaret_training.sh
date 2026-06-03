#!/usr/bin/env bash
set -euo pipefail

# Labeled-data pipeline:
# raw train/test JSON -> rubric scores -> CEFR extraction -> word_count -> PyCaret training

conda activate gept_writing_eval

LEVEL="${LEVEL:-HI}"
TASK_TYPE="${TASK_TYPE:-regression}"

TRAIN_INPUT="${TRAIN_INPUT:-./data/${LEVEL}_train.json}"
TEST_INPUT="${TEST_INPUT:-./data/${LEVEL}_test.json}"
QUESTION_PATH="${QUESTION_PATH:-./data/${LEVEL}_train_question.json}"

WORK_DIR="${WORK_DIR:-./output/${LEVEL}_training_preprocess}"
MODEL_OUTPUT_PATH="${MODEL_OUTPUT_PATH:-./model_full_data}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-12b-it}"
MISTRAL_MODEL="${MISTRAL_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

mkdir -p "${WORK_DIR}"

TRAIN_RUBRIC="${WORK_DIR}/${LEVEL}_train_rubric.json"
TEST_RUBRIC="${WORK_DIR}/${LEVEL}_test_rubric.json"

TRAIN_GEMMA="${WORK_DIR}/${LEVEL}_train_gemma_cefr.jsonl"
TEST_GEMMA="${WORK_DIR}/${LEVEL}_test_gemma_cefr.jsonl"
TRAIN_MISTRAL="${WORK_DIR}/${LEVEL}_train_mistral_cefr.jsonl"
TEST_MISTRAL="${WORK_DIR}/${LEVEL}_test_mistral_cefr.jsonl"

TRAIN_CEFR="${WORK_DIR}/${LEVEL}_train_cefr.json"
TEST_CEFR="${WORK_DIR}/${LEVEL}_test_cefr.json"

TRAIN_READY="${WORK_DIR}/${LEVEL}_train_ready.json"
TEST_READY="${WORK_DIR}/${LEVEL}_test_ready.json"

: > "${TRAIN_GEMMA}"
: > "${TEST_GEMMA}"
: > "${TRAIN_MISTRAL}"
: > "${TEST_MISTRAL}"

python rubric_inference.py \
  --input "${TRAIN_INPUT}" \
  --output "${TRAIN_RUBRIC}" \
  --question_path "${QUESTION_PATH}"

python rubric_inference.py \
  --input "${TEST_INPUT}" \
  --output "${TEST_RUBRIC}" \
  --question_path "${QUESTION_PATH}"

python cefr_label_inference.py \
  --input "${TRAIN_RUBRIC}" \
  --output "${TRAIN_GEMMA}" \
  --model "${GEMMA_MODEL}"

python cefr_label_inference.py \
  --input "${TEST_RUBRIC}" \
  --output "${TEST_GEMMA}" \
  --model "${GEMMA_MODEL}"

python cefr_label_inference.py \
  --input "${TRAIN_RUBRIC}" \
  --output "${TRAIN_MISTRAL}" \
  --model "${MISTRAL_MODEL}"

python cefr_label_inference.py \
  --input "${TEST_RUBRIC}" \
  --output "${TEST_MISTRAL}" \
  --model "${MISTRAL_MODEL}"

python cefr_label_extraction.py \
  --gemma_input_file "${TRAIN_GEMMA}" \
  --mistral_input_file "${TRAIN_MISTRAL}" \
  --input_file "${TRAIN_RUBRIC}" \
  --output_file "${TRAIN_CEFR}"

python cefr_label_extraction.py \
  --gemma_input_file "${TEST_GEMMA}" \
  --mistral_input_file "${TEST_MISTRAL}" \
  --input_file "${TEST_RUBRIC}" \
  --output_file "${TEST_CEFR}"

python length_feature_extraction.py \
  --input_file "${TRAIN_CEFR}" \
  --output_file "${TRAIN_READY}" \
  --level "${LEVEL}"

python length_feature_extraction.py \
  --input_file "${TEST_CEFR}" \
  --output_file "${TEST_READY}" \
  --level "${LEVEL}"

python pycaret_training.py \
  --train_data_path "${TRAIN_READY}" \
  --test_data_path "${TEST_READY}" \
  --model_output_path "${MODEL_OUTPUT_PATH}" \
  --level "${LEVEL}" \
  --type "${TASK_TYPE}"

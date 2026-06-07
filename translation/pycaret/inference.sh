#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/bin/activate trans

LEVEL="${LEVEL:-HI}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
INPUT_DATA="${INPUT_DATA:-/home/mimi911123/ra_model/eval_data/HI_translation_answers.json}"
QUESTION_DATA="${QUESTION_DATA:-/home/mimi911123/ra_model/eval_data/HI_translation_questions.json}"
CHECKPOINT="${CHECKPOINT:-/tmp/models/pycaret/translation/${LEVEL}_translation.pt}"
OUTPUT="${OUTPUT:-./output/${LEVEL}_translation_predictions.json}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_translation_preprocess}"
LLM_MODEL="${LLM_MODEL:-google/gemma-3-12b-it}"

ALIGNMENT_OUTPUT="${ALIGNMENT_OUTPUT:-${WORK_DIR}/${LEVEL}_alignment.jsonl}"
PSEUDO_LABEL_OUTPUT="${PSEUDO_LABEL_OUTPUT:-${WORK_DIR}/${LEVEL}_pseudo_labels.jsonl}"
MERGED_INPUT="${MERGED_INPUT:-${WORK_DIR}/${LEVEL}_merged.json}"

mkdir -p "${WORK_DIR}"

: > "${ALIGNMENT_OUTPUT}"
: > "${PSEUDO_LABEL_OUTPUT}"

python alignment_inference.py \
  --subject "${QUESTION_DATA}" \
  --input "${INPUT_DATA}" \
  --output "${ALIGNMENT_OUTPUT}" \
  --model "${LLM_MODEL}"

python pseudo_label_inference.py \
  --input "${ALIGNMENT_OUTPUT}" \
  --output "${PSEUDO_LABEL_OUTPUT}" \
  --model "${LLM_MODEL}"

python data/merge_pseudo_labels.py \
  --input "${INPUT_DATA}" \
  --pseudo_labels "${PSEUDO_LABEL_OUTPUT}" \
  --output "${MERGED_INPUT}"

python inference.py \
  --input "${MERGED_INPUT}" \
  --output "${OUTPUT}" \
  --question_data "${QUESTION_DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --model "${MODEL}"

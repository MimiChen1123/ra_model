#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/bin/activate gemma

LEVEL="${LEVEL:-HI}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
LLM_MODEL="${LLM_MODEL:-google/gemma-3-12b-it}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-5}"

TRAIN_INPUT="${TRAIN_INPUT:-./data/${LEVEL}_train.json}"
QUESTION_DATA="${QUESTION_DATA:-./data/reference-answers/question_prompts.json}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_translation_training_preprocess}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${LEVEL}_translation}"

ALIGNMENT_OUTPUT="${ALIGNMENT_OUTPUT:-${WORK_DIR}/${LEVEL}_train_alignment.jsonl}"
PSEUDO_LABEL_OUTPUT="${PSEUDO_LABEL_OUTPUT:-${WORK_DIR}/${LEVEL}_train_pseudo_labels.jsonl}"
MERGED_TRAIN="${MERGED_TRAIN:-${WORK_DIR}/${LEVEL}_train_merged.json}"

mkdir -p "${WORK_DIR}"

: > "${ALIGNMENT_OUTPUT}"
: > "${PSEUDO_LABEL_OUTPUT}"

python alignment_inference.py \
  --subject "${QUESTION_DATA}" \
  --input "${TRAIN_INPUT}" \
  --output "${ALIGNMENT_OUTPUT}" \
  --model "${LLM_MODEL}"

python pseudo_label_inference.py \
  --input "${ALIGNMENT_OUTPUT}" \
  --output "${PSEUDO_LABEL_OUTPUT}" \
  --model "${LLM_MODEL}"

python data/merge_pseudo_labels.py \
  --input "${TRAIN_INPUT}" \
  --pseudo_labels "${PSEUDO_LABEL_OUTPUT}" \
  --output "${MERGED_TRAIN}"

python train.py \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --essay_data "${MERGED_TRAIN}" \
  --question_data "${QUESTION_DATA}" \
  --output_dir "${OUTPUT_DIR}"

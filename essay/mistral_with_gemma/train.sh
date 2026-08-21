#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Training pipeline:
# raw train JSON -> Gemma pseudo labels -> merge features -> Mistral regression training

source ~/anaconda3/bin/activate gemma

LEVEL="${LEVEL:-I}"

if [[ "${LEVEL}" == "HI" ]]; then
  DEFAULT_INPUT_DATA="data/HI_train.json"
  DEFAULT_QUESTION_PATH="data/HI_train_question.json"
else
  DEFAULT_INPUT_DATA="data/I_train.json"
  DEFAULT_QUESTION_PATH="data/I_train_question.json"
fi

INPUT_DATA="${INPUT_DATA:-${DEFAULT_INPUT_DATA}}"
QUESTION_PATH="${QUESTION_PATH:-${DEFAULT_QUESTION_PATH}}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_train_preprocess}"
PSEUDO_OUTPUT="${PSEUDO_OUTPUT:-${WORK_DIR}/${LEVEL}_train_pseudo.jsonl}"
MERGED_DATA="${MERGED_DATA:-${WORK_DIR}/${LEVEL}_train_ready.json}"

PSEUDO_MODEL="${PSEUDO_MODEL:-google/gemma-3-12b-it}"
TRAIN_MODEL="${TRAIN_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${LEVEL}_train}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
TEST_SIZE="${TEST_SIZE:-0.03}"
SEED="${SEED:-0}"
START="${START:-0}"

mkdir -p "${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"

PSEUDO_ARGS=(
  --input "${INPUT_DATA}"
  --output "${PSEUDO_OUTPUT}"
  --question_path "${QUESTION_PATH}"
  --level "${LEVEL}"
  --model "${PSEUDO_MODEL}"
  --start "${START}"
)

if [[ -n "${LIMIT:-}" ]]; then
  PSEUDO_ARGS+=(--limit "${LIMIT}")
fi

python run_pseudo.py "${PSEUDO_ARGS[@]}"

python run_pseudo.py \
  --merge \
  --input "${INPUT_DATA}" \
  --output "${PSEUDO_OUTPUT}" \
  --save_path "${MERGED_DATA}"

TRAIN_ARGS=(
  --essay_data "${MERGED_DATA}"
  --question_data "${QUESTION_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --model "${TRAIN_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --test_size "${TEST_SIZE}"
  --seed "${SEED}"
)

if [[ "${NO_QUANTIZE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no_quantize)
fi

python train.py "${TRAIN_ARGS[@]}"

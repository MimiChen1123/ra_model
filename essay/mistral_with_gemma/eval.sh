#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Real-world inference pipeline:
# raw test JSON -> Gemma pseudo labels -> merge features -> Mistral essay inference

source ~/anaconda3/bin/activate gemma

LEVEL="${LEVEL:-HI}"

INPUT_DATA="${INPUT_DATA:-/home/mimi911123/ra_model/eval_data/HI_essay_answers.json}"
QUESTION_PATH="${QUESTION_PATH:-/home/mimi911123/ra_model/eval_data/HI_essay_questions.json}"
WORK_DIR="${WORK_DIR:-./output/${LEVEL}_preprocess}"
PSEUDO_OUTPUT="${PSEUDO_OUTPUT:-${WORK_DIR}/${LEVEL}_pseudo.jsonl}"
MERGED_DATA="${MERGED_DATA:-${WORK_DIR}/${LEVEL}_ready.json}"

PSEUDO_MODEL="${PSEUDO_MODEL:-google/gemma-3-12b-it}"
EVAL_MODEL="${EVAL_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
CHECKPOINT="${CHECKPOINT:-/tmp/models/mistral_with_gemma/essay/HI_essay.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_SCORE="${MIN_SCORE:-0.0}"
SCORE_STEP="${SCORE_STEP:-0.5}"
NUM_CLASSES="${NUM_CLASSES:-11}"
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

python eval.py \
  --test_data "${MERGED_DATA}" \
  --question_data "${QUESTION_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model "${EVAL_MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --min_score "${MIN_SCORE}" \
  --score_step "${SCORE_STEP}" \
  --num_classes "${NUM_CLASSES}"

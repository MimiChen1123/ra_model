#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEVEL="${LEVEL:-HI}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${DATA_DIR:-/home/mimi911123/ra_model/eval_data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/models/deberta/}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

QUESTIONS="${QUESTIONS:-${DATA_DIR}/${LEVEL}_essay_questions.json}"
ANSWERS="${ANSWERS:-${DATA_DIR}/${LEVEL}_essay_answers.json}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/${LEVEL}_essay_prediction.json}"

CONFIG_PATH="${CONFIG_PATH:-${ARTIFACT_DIR}/${LEVEL}_model_hybrid_rel_0.5_config.json}"
MODEL_PATH="${MODEL_PATH:-${ARTIFACT_DIR}/${LEVEL}_model_hybrid_rel_0.5_xgb.ubj}"
TFIDF_VECTORIZER_PATH="${TFIDF_VECTORIZER_PATH:-${ARTIFACT_DIR}/${LEVEL}_model_hybrid_rel_0.5_tfidf_vectorizer.pkl}"

for required_path in \
    "${QUESTIONS}" \
    "${ANSWERS}" \
    "${CONFIG_PATH}" \
    "${MODEL_PATH}" \
    "${TFIDF_VECTORIZER_PATH}"
do
    if [[ ! -f "${required_path}" ]]; then
        echo "Missing required file: ${required_path}" >&2
        echo "Override paths with QUESTIONS, ANSWERS, CONFIG_PATH, MODEL_PATH, or TFIDF_VECTORIZER_PATH." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON_BIN}" eval.py \
    --seed "${SEED}" \
    --questions "${QUESTIONS}" \
    --answers "${ANSWERS}" \
    --output "${OUTPUT}" \
    --config_path "${CONFIG_PATH}" \
    --model_path "${MODEL_PATH}" \
    --tfidf_vectorizer_path "${TFIDF_VECTORIZER_PATH}" \
    "$@"

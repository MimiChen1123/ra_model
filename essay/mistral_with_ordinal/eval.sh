#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEVEL="${LEVEL:-HI}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${DATA_DIR:-/home/mimi911123/ra_model/eval_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

QUESTIONS="${QUESTIONS:-${DATA_DIR}/${LEVEL}_essay_questions.json}"
ANSWERS="${ANSWERS:-${DATA_DIR}/${LEVEL}_essay_answers.json}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/${LEVEL}_prediction.json}"

if [[ "${LEVEL}" == "I" ]]; then
    DEFAULT_CHECKPOINT="/tmp/models/essay/mistral_with_ordinal/model_I.pt"
elif [[ "${LEVEL}" == "HI" ]]; then
    DEFAULT_CHECKPOINT="/tmp/models/essay/mistral_with_ordinal/model_HI.pt"
else
    echo "LEVEL must be I or HI, got: ${LEVEL}" >&2
    exit 1
fi

CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

for required_path in "${QUESTIONS}" "${ANSWERS}"
do
    if [[ ! -f "${required_path}" ]]; then
        echo "Missing required file: ${required_path}" >&2
        echo "Override paths with QUESTIONS, ANSWERS, or DATA_DIR." >&2
        exit 1
    fi
done

if [[ -n "${CHECKPOINT:-}" && ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint file: ${CHECKPOINT}" >&2
    echo "Set CHECKPOINT to a valid .pt checkpoint path, or unset it to use eval.py defaults for LEVEL." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON_BIN}" eval.py \
    --level "${LEVEL}" \
    --seed "${SEED}" \
    --question_json "${QUESTIONS}" \
    --answer_json "${ANSWERS}" \
    --output-json "${OUTPUT}" \
    --batch_size "${BATCH_SIZE}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"

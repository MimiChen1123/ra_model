#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEVEL="${LEVEL:-HI}"
OUTPUT="${OUTPUT:-./output/merged_predictions.json}"

echo "======== Deberta Predictions ========"
python merge_predictions.py \
    --input /home/mimi911123/ra_model/essay/deberta/output/HI_essay_prediction.json \
    --output ${OUTPUT} \
    --version deberta \
    --level ${LEVEL}

echo "======== Mistral with Ordinal Predictions ========"
python merge_predictions.py \
    --input /home/mimi911123/ra_model/essay/mistral_with_ordinal/output/HI_prediction.json \
    --output ${OUTPUT} \
    --version mistral_with_ordinal \
    --level ${LEVEL}

echo "======== Mistral with Gemma Predictions ========"
python merge_predictions.py \
    --input /home/mimi911123/ra_model/essay/mistral_with_gemma/output/predictions.json \
    --output ${OUTPUT} \
    --version mistral_with_gemma \
    --level ${LEVEL}

echo "======== PyCaret Predictions ========"
python merge_predictions.py \
    --input /home/mimi911123/ra_model/essay/pycaret/output/HI_prediction_xgb.json \
    --output ${OUTPUT} \
    --version pycaret \
    --level ${LEVEL}
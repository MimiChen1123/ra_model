#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEVEL="${LEVEL:-HI}"
ESSAY_OUTPUT="${OUTPUT:-./output/essay_predictions.json}"
TRANSLATION_OUTPUT="${OUTPUT:-./output/translation_predictions.json}"
OUTPUT="${OUTPUT:-./output/students_predictions.xlsx}"


echo "======== Essay Predictions ========"

echo "======== Deberta Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/essay/deberta/output/HI_essay_prediction.json \
    --output ${ESSAY_OUTPUT} \
    --version deberta \
    --level ${LEVEL}

echo "======== Mistral with Ordinal Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/essay/mistral_with_ordinal/output/HI_prediction.json \
    --output ${ESSAY_OUTPUT} \
    --version mistral_with_ordinal \
    --level ${LEVEL}

echo "======== Mistral with Gemma Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/essay/mistral_with_gemma/output/predictions.json \
    --output ${ESSAY_OUTPUT} \
    --version mistral_with_gemma \
    --level ${LEVEL}

echo "======== PyCaret Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/essay/pycaret/output/HI_prediction.json \
    --output ${ESSAY_OUTPUT} \
    --version pycaret \
    --level ${LEVEL}

echo "======== Translation Predictions ========"
echo "======== Mistral with Gemma Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/translation/mistral_with_gemma/outputs/HI_eval/predictions.json \
    --output ${TRANSLATION_OUTPUT} \
    --version mistral_with_gemma \
    --level ${LEVEL}

echo "======== PyCaret Predictions ========"
python postprocess.py \
    --input /home/mimi911123/ra_model/translation/pycaret/output/HI_translation_predictions.json \
    --output ${TRANSLATION_OUTPUT} \
    --version pycaret \
    --level ${LEVEL}


python json_to_xlsx.py \
    --essay_input ${ESSAY_OUTPUT} \
    --translation_input ${TRANSLATION_OUTPUT} \
    --output ${OUTPUT} 

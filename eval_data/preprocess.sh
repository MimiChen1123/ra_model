#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEVEL="${LEVEL:-HI}"
DATA_DIR="${DATA_DIR:-.}"
INPUT_XLSX="${INPUT_XLSX:-1150328中高一日CBT寫作作答_.xlsx}"

TRANSLATION_OUTPUT="${TRANSLATION_OUTPUT:-${LEVEL}_translation_answers.json}"
ESSAY_OUTPUT="${ESSAY_OUTPUT:-${LEVEL}_essay_answers.json}"

SEAT_COLUMN="${SEAT_COLUMN:-座位號碼}"
SUBJECT_COLUMN="${SUBJECT_COLUMN:-寫作卷別}"
TRANSLATION_COLUMN="${TRANSLATION_COLUMN:-翻譯}"
ESSAY_COLUMN="${ESSAY_COLUMN:-作文}"

python preprocess.py \
  --data_dir "${DATA_DIR}" \
  --input_xlsx "${INPUT_XLSX}" \
  --level "${LEVEL}" \
  --translation_output "${TRANSLATION_OUTPUT}" \
  --essay_output "${ESSAY_OUTPUT}" \
  --seat_column "${SEAT_COLUMN}" \
  --subject_column "${SUBJECT_COLUMN}" \
  --translation_column "${TRANSLATION_COLUMN}" \
  --essay_column "${ESSAY_COLUMN}"

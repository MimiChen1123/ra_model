#!/usr/bin/env python3
"""Convert essay and translation prediction JSON files into an XLSX workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMON_COLUMNS = ["seat_number", "subject", "level"]
EXCLUDED_COLUMNS = {"document_id"}
ESSAY_MODEL_COLUMNS = [
    "deberta",
    "mistral_with_ordinal",
    "mistral_with_gemma",
    "pycaret",
]
TRANSLATION_MODEL_COLUMNS = ["mistral_with_gemma", "pycaret"]
COLUMN_RENAMES = {
    "seat_number": "座位號碼",
    "subject": "寫作卷別",
    "level": "等級"
}


def load_predictions(input_path: Path, model_columns: list[str]) -> pd.DataFrame:
    df = pd.read_json(input_path)
    df = pd.json_normalize(df.to_dict(orient="records"))
    df.columns = [column.removeprefix("scores.") for column in df.columns]

    preferred_columns = COMMON_COLUMNS + model_columns
    existing_preferred = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_preferred and column not in EXCLUDED_COLUMNS
    ]
    df = df[existing_preferred + remaining_columns]
    return df.rename(columns=COLUMN_RENAMES)


def convert(essay_input: Path, translation_input: Path, output_path: Path) -> None:
    essay_df = load_predictions(essay_input, ESSAY_MODEL_COLUMNS)
    translation_df = load_predictions(translation_input, TRANSLATION_MODEL_COLUMNS)

    with pd.ExcelWriter(output_path) as writer:
        essay_df.to_excel(writer, index=False, sheet_name="essay")
        translation_df.to_excel(writer, index=False, sheet_name="translation")

    print(
        f"Wrote {len(essay_df)} essay rows and "
        f"{len(translation_df)} translation rows to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert essay and translation prediction JSON files to one XLSX."
    )
    
    parser.add_argument("--essay_input", type=str, default="./output/essay_predictions.json", help="Input JSON file for essay predictions.")
    parser.add_argument(
        "--translation_input",
        type=str,
        default="./output/translation_predictions.json",
        help="Input JSON file for translation predictions."
    )
    parser.add_argument("--output", type=str, default="./output/students_predictions.xlsx", help="Output XLSX file.")
    args = parser.parse_args()

    convert(args.essay_input, args.translation_input, args.output)


if __name__ == "__main__":
    main()

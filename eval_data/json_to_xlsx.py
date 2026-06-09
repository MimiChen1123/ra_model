#!/usr/bin/env python3
"""Convert essay and translation prediction JSON files into an XLSX workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ESSAY_INPUT = PROJECT_ROOT / "essay/result/merged_essay_predictions.json"
DEFAULT_TRANSLATION_INPUT = PROJECT_ROOT / "translation/result/merged_translation_predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "result_predictions.xlsx"

COMMON_COLUMNS = ["document_id", "seat_number", "subject", "level"]
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
    remaining_columns = [column for column in df.columns if column not in existing_preferred]
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
    parser.add_argument("essay_input", nargs="?", type=Path, default=DEFAULT_ESSAY_INPUT)
    parser.add_argument(
        "translation_input",
        nargs="?",
        type=Path,
        default=DEFAULT_TRANSLATION_INPUT,
    )
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    convert(args.essay_input, args.translation_input, args.output)


if __name__ == "__main__":
    main()

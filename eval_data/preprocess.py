#!/usr/bin/env python3
"""Convert raw writing XLSX answers into essay and translation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def save_json(data: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def build_answer_item(
    document_id: int,
    seat_number,
    subject,
    content,
    level: str,
) -> dict:
    return {
        "document_id": document_id,
        "seat_number": seat_number,
        "subject": subject,
        "content": str(content),
        "level": level,
    }


def preprocess(
    input_xlsx: Path,
    translation_output: Path,
    essay_output: Path,
    level: str,
    seat_column: str,
    subject_column: str,
    translation_column: str,
    essay_column: str,
) -> None:
    import pandas as pd

    df = pd.read_excel(input_xlsx)

    required_columns = [seat_column, subject_column, translation_column, essay_column]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"缺少欄位：{missing_columns}")

    translation_data = []
    essay_data = []

    for idx, row in df.iterrows():
        document_id = idx + 1
        seat_number = row[seat_column]
        subject = row[subject_column]

        if not pd.isna(row[translation_column]):
            translation_data.append(
                build_answer_item(
                    document_id=document_id,
                    seat_number=seat_number,
                    subject=subject,
                    content=row[translation_column],
                    level=level,
                )
            )

        if not pd.isna(row[essay_column]):
            essay_data.append(
                build_answer_item(
                    document_id=document_id,
                    seat_number=seat_number,
                    subject=subject,
                    content=row[essay_column],
                    level=level,
                )
            )

    save_json(translation_data, translation_output)
    save_json(essay_data, essay_output)

    print(f"讀取檔案：{input_xlsx}")
    print(f"level：{level}")
    print(f"翻譯輸出：{translation_output} ({len(translation_data)} 筆)")
    print(f"作文輸出：{essay_output} ({len(essay_data)} 筆)")


def data_path(path: str, data_dir: str | None) -> Path:
    target_path = Path(path)

    if data_dir and not target_path.is_absolute():
        return Path(data_dir) / target_path

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw writing XLSX answers into essay and translation JSON files."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing input/output files. Relative file paths are resolved from this directory."
    )
    parser.add_argument(
        "--input_xlsx",
        default="1150328中高一日CBT寫作作答_.xlsx",
        help="Input XLSX file."
    )
    parser.add_argument(
        "--level",
        default="HI",
        help="Level name used in output records and default output filenames."
    )
    parser.add_argument(
        "--translation_output",
        default=None,
        help="Output JSON file for translation answers. Default: <level>_translation_answers.json"
    )
    parser.add_argument(
        "--essay_output",
        default=None,
        help="Output JSON file for essay answers. Default: <level>_essay_answers.json"
    )
    parser.add_argument(
        "--seat_column",
        default="座位號碼",
        help="Column name for seat number."
    )
    parser.add_argument(
        "--subject_column",
        default="寫作卷別",
        help="Column name for writing subject/type."
    )
    parser.add_argument(
        "--translation_column",
        default="翻譯",
        help="Column name for translation answers."
    )
    parser.add_argument(
        "--essay_column",
        default="作文",
        help="Column name for essay answers."
    )
    args = parser.parse_args()

    level = args.level
    translation_output = args.translation_output or f"{level}_translation_answers.json"
    essay_output = args.essay_output or f"{level}_essay_answers.json"

    preprocess(
        input_xlsx=data_path(args.input_xlsx, args.data_dir),
        translation_output=data_path(translation_output, args.data_dir),
        essay_output=data_path(essay_output, args.data_dir),
        level=level,
        seat_column=args.seat_column,
        subject_column=args.subject_column,
        translation_column=args.translation_column,
        essay_column=args.essay_column,
    )


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


def load_json(path: str):
    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def validate_input_item(item: dict, idx: int):
    required_keys = ["document_id", "subject", "predicted_score", "seat_number"]

    for key in required_keys:
        if key not in item:
            raise ValueError(f"Input item at index {idx} missing required key: {key}")


def merge_predictions(input_path: str, output_path: str, version: str, level: str):
    input_data = load_json(input_path)
    output_data = load_json(output_path)

    if not isinstance(input_data, list):
        raise ValueError("Input JSON must be a List[Dict].")

    if not isinstance(output_data, list):
        raise ValueError("Output JSON must be a List[Dict].")

    merged_map = {}

    # 先讀取既有 output
    for idx, item in enumerate(output_data):
        if "document_id" not in item or "subject" not in item or "seat_number" not in item:
            raise ValueError(f"Output item at index {idx} missing required key.")

        document_id = item["document_id"]
        subject = item["subject"]
        seat_number = item["seat_number"]

        key = (document_id, subject, seat_number)
        merged_map[key] = item

        if "scores" not in merged_map[key]:
            merged_map[key]["scores"] = {}

        if "level" not in merged_map[key]:
            merged_map[key]["level"] = level

    # 合併新的 input prediction
    for idx, item in enumerate(input_data):
        validate_input_item(item, idx)

        document_id = item["document_id"]
        subject = item["subject"]
        predicted_score = item["predicted_score"]
        seat_number = item["seat_number"]

        key = (document_id, subject, seat_number)

        if key not in merged_map:
            merged_map[key] = {
                "document_id": document_id,
                "subject": subject,
                "seat_number": seat_number,
                "level": level,
                "scores": {}
            }

        merged_map[key]["level"] = level
        merged_map[key]["scores"][version] = predicted_score

    merged_data = list(merged_map.values())

    merged_data.sort(key=lambda x: (x["document_id"], x["subject"], x["seat_number"]))

    save_json(merged_data, output_path)

    print(f"已合併 {len(input_data)} 筆資料")
    print(f"目前總共有 {len(merged_data)} 筆資料")
    print(f"level：{level}")
    print(f"版本名稱：{version}")
    print(f"輸出檔案：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge prediction results from different model versions into one JSON file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input prediction JSON file. Must be List[Dict] with document_id, subject, predicted_score."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output merged JSON file."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Model version name, e.g. deberta_v3_large, xgb_baseline, gemma_12b."
    )

    parser.add_argument(
        "--level",
        default="HI",
        help="Level name, e.g. HI or I."
    )

    args = parser.parse_args()

    merge_predictions(
        input_path=args.input,
        output_path=args.output,
        version=args.version,
        level=args.level
    )


if __name__ == "__main__":
    main()
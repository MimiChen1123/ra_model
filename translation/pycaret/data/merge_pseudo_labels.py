import argparse
import json
import math
from pathlib import Path


IDENTIFIER_KEYS = ("id", "document_id")


def normalize_identifier(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def get_identifier(item):
    for key in IDENTIFIER_KEYS:
        if key in item:
            item_id = normalize_identifier(item[key])
            if item_id is not None:
                return item_id
    return None


def load_pseudo_labels(pseudo_labels_path: Path):
    pseudo_labels_by_id = {}
    pseudo_labels_by_position = []
    with open(pseudo_labels_path, "r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            item_id = get_identifier(item)
            pseudo_label_item = {
                "alignment": item.get("llm_alignment", {}).get("alignment"),
                "pseudo_labels": item.get("llm_pseudo_labels", {}).get("pseudo_labels", []),
            }
            if item_id is not None:
                pseudo_labels_by_id[item_id] = pseudo_label_item
            pseudo_labels_by_position.append((line_number, item_id, pseudo_label_item))
    return pseudo_labels_by_id, pseudo_labels_by_position


def merge_pseudo_labels(input_path: Path, pseudo_labels_path: Path, output_path: Path) -> None:
    pseudo_labels_by_id, pseudo_labels_by_position = load_pseudo_labels(pseudo_labels_path)
    with open(input_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON list.")

    merged_data = []
    used_position_fallback = False
    for row_index, item in enumerate(data):
        merged_item = dict(item)
        item_id = get_identifier(merged_item)
        pseudo_labels_item = pseudo_labels_by_id.get(item_id) if item_id is not None else None

        if pseudo_labels_item is None:
            if row_index >= len(pseudo_labels_by_position):
                identifier = item_id if item_id is not None else f"row {row_index + 1}"
                print(f"Warning: {identifier} not found in pseudo labels.")
                continue
            _, _, pseudo_labels_item = pseudo_labels_by_position[row_index]
            used_position_fallback = True

        merged_item["alignment"] = pseudo_labels_item["alignment"]
        merged_item["pseudo_labels"] = pseudo_labels_item["pseudo_labels"]
        merged_data.append(merged_item)

    if used_position_fallback:
        print("Warning: Some records were merged by row order because no matching id/document_id was available.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(merged_data, fp, ensure_ascii=False, indent=4)

    print(f"✅ Wrote {output_path} ({len(merged_data)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LLM alignment and pseudo-label JSONL into translation JSON data.")
    parser.add_argument("--input", type=Path, required=True, help="Original translation JSON file.")
    parser.add_argument("--pseudo_labels", type=Path, required=True, help="Pseudo-label JSONL from pseudo_label_inference.py.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output JSON file.")
    args = parser.parse_args()

    merge_pseudo_labels(
        input_path=args.input,
        pseudo_labels_path=args.pseudo_labels,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

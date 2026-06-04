import argparse
import json
from pathlib import Path

import pandas as pd


def load_pseudo_labels(pseudo_labels_path: Path):
    pseudo_labels = {}
    with open(pseudo_labels_path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            pseudo_labels[item["id"]] = {
                "alignment": item.get("llm_alignment", {}).get("alignment"),
                "pseudo_labels": item.get("llm_pseudo_labels", {}).get("pseudo_labels", []),
            }
    return pseudo_labels


def merge_pseudo_labels(input_path: Path, pseudo_labels_path: Path, output_path: Path) -> None:
    pseudo_labels = load_pseudo_labels(pseudo_labels_path)
    data = pd.read_json(input_path)

    merged_data = []
    for _, row in data.iterrows():
        item_id = row["id"]
        if item_id not in pseudo_labels:
            print(f"Warning: ID {item_id} not found in pseudo labels.")
            continue

        pseudo_labels_item = pseudo_labels[item_id]
        merged_item = row.to_dict()
        merged_item["alignment"] = pseudo_labels_item["alignment"]
        merged_item["pseudo_labels"] = pseudo_labels_item["pseudo_labels"]
        merged_data.append(merged_item)

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

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from cefr_label_inference import (  # noqa: E402
    SYSTEM_INSTRUCTIONS,
    build_user_prompt,
    chat_once,
    make_client,
    parse_cefr_from_response,
)


def load_input(input_path: Path, start: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON list of essay records.")

    if limit is not None:
        return data[start : start + limit]

    if start:
        return data[start:]

    return data


def run_inference(
    input_path: Path,
    output_path: Path,
    api_key: Optional[str],
    model: str,
    limit: Optional[int],
    start: int,
    keep_raw_output: bool,
) -> None:
    data = load_input(input_path, start=start, limit=limit)
    client = make_client(api_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for item in tqdm(data):
        content = str(item.get("content", "")).strip()

        if content:
            user_prompt = build_user_prompt(content)
            raw_output = chat_once(
                client=client,
                model=model,
                system_prompt=SYSTEM_INSTRUCTIONS,
                user_prompt=user_prompt,
                max_tokens=1024,
            )
            cefr_prediction = parse_cefr_from_response(raw_output)
        else:
            raw_output = ""
            cefr_prediction = None

        new_item = dict(item)
        new_item["cefr_prediction"] = cefr_prediction

        if keep_raw_output:
            new_item["cefr_model"] = model
            new_item["cefr_raw_output"] = raw_output

        results.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Real-world CEFR predictions saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma-only CEFR label inference for real-world essay data without gold scores."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON list of essay records.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSON file.")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it", help="DeepInfra model name.")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--limit", type=int, default=None, help="Only process N records.")
    parser.add_argument("--start", type=int, default=0, help="Start index for partial runs.")
    parser.add_argument(
        "--keep_raw_output",
        action="store_true",
        help="Keep cefr_model and cefr_raw_output fields for debugging.",
    )
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        start=args.start,
        keep_raw_output=args.keep_raw_output,
    )


if __name__ == "__main__":
    main()

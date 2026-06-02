"""
Pseudo-label essays with CEFR prediction and rubric scores (RELEVANCE, COHERENCE, ORGANIZATION).

Usage:
    python pseudo_label.py \
        --input data/HI_2641A.json \
        --output output/HI_2641A.jsonl \
        --question_path data/test_question.json \
        --level HI \
        --model google/gemma-3-12b-it

After inference, convert JSONL → JSON:
    python pseudo_label.py \
        --merge \
        --input data/HI_2641A.json \
        --output output/HI_2641A.jsonl \
        --save_path data/HI_2641A.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

# ---------------------------------------------------------------------------
# Path setup – make sure prompt package is importable
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from prompt.rubric_prompt import (
    SYSTEM_INSTRUCTIONS as RUBRIC_SYSTEM,
    build_rubric_prompt,
)
from prompt.cefr_prompt import (
    SYSTEM_INSTRUCTIONS as CEFR_SYSTEM,
    build_cefr_prompt,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
_SCORE_KEYS = ("RELEVANCE", "COHERENCE", "ORGANIZATION")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _coerce_score(v: Any) -> Optional[int]:
    """Convert a value to an int score in [0, 5], or None."""
    try:
        iv = int(v)
        return iv if 0 <= iv <= 5 else None
    except Exception:
        return None


def parse_scores_from_response(text: str) -> Optional[Dict[str, int]]:
    """Extract RELEVANCE / COHERENCE / ORGANIZATION from LLM response."""
    # Try JSON block first
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            out: Dict[str, int] = {}
            for k in _SCORE_KEYS:
                if k in obj:
                    val = _coerce_score(obj[k])
                    if val is not None:
                        out[k] = val
            if len(out) == 3:
                return out
        except Exception:
            pass

    # Fallback: regex per key
    out = {}
    for k in _SCORE_KEYS:
        mm = re.search(rf"{k}\s*[:]\s*([0-5])", text, flags=re.IGNORECASE)
        if mm:
            out[k.upper()] = int(mm.group(1))
    if len(out) == 3:
        return out
    return None


def parse_cefr_from_response(text: str) -> Optional[str]:
    """Extract a single CEFR level string from LLM response."""
    # Try JSON first
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            lvl = obj.get("final_level") or obj.get("final", {}).get("level")
            if isinstance(lvl, str):
                lvl = lvl.strip().upper()
                if lvl in CEFR_LEVELS:
                    return lvl
        except Exception:
            pass

    # "CEFR level: B2"
    matches = re.findall(r"CEFR\s*level\s*:\s*([ABC][12])", text, flags=re.IGNORECASE)
    if matches:
        lvl = matches[-1].upper()
        if lvl in CEFR_LEVELS:
            return lvl

    # Bare level mention
    matches = re.findall(r"\b(A1|A2|B1|B2|C1|C2)\b", text)
    if matches:
        return matches[-1]

    return None


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------
def make_client(api_key: Optional[str]) -> OpenAI:
    key = api_key or os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError(
            "Please set --api_key or environment variable DEEPINFRA_API_KEY"
        )
    return OpenAI(api_key=key, base_url="https://api.deepinfra.com/v1/openai")


def chat_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Single chat completion with retry."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, RateLimitError, APIStatusError):
            time.sleep(1.5 ** attempt)
            if attempt == 3:
                raise
    return ""


# ---------------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------------
def count_words(text: str) -> int:
    """Simple whitespace-based word count."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference(
    input_path: Path,
    output_path: Path,
    question_path: Path,
    api_key: Optional[str],
    model: str,
    level: str,
    limit: Optional[int],
    start: int,
) -> None:
    data: List[Dict[str, Any]] = json.load(open(input_path, "r", encoding="utf-8"))
    if limit is not None:
        data = data[start : start + limit]
    elif start > 0:
        data = data[start:]

    # Load question / topic text
    try:
        qobj = json.load(open(question_path, "r", encoding="utf-8"))
        subject_map: Dict[str, str] = qobj.get("subject", {})
    except FileNotFoundError:
        subject_map = {}
        print(f"⚠️  Question file not found: {question_path}; proceeding without topic text")

    client = make_client(api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-processed document_ids
    processed_ids: set = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        processed_ids.add(item.get("document_id"))
                    except Exception:
                        pass
        print(f"✅ Found {len(processed_ids)} already processed items, resuming…")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for item in tqdm(data, desc="Pseudo-labeling"):
            doc_id = item.get("document_id")
            if doc_id in processed_ids:
                continue

            content: str = (item.get("content") or "").strip()
            new_item = dict(item)
            new_item["word_count"] = count_words(content)

            if not content:
                new_item["cefr_prediction"] = None
                new_item["RELEVANCE"] = None
                new_item["COHERENCE"] = None
                new_item["ORGANIZATION"] = None
            else:
                subject_id = item.get("subject")  # e.g. "HW2641A"
                subject_text = subject_map.get(subject_id)

                # ---- Rubric scoring ----
                rubric_user_prompt = build_rubric_prompt(
                    subject_text=subject_text,
                    content=content,
                    level=level,
                )
                rubric_resp = chat_once(
                    client=client,
                    model=model,
                    system_prompt=RUBRIC_SYSTEM,
                    user_prompt=rubric_user_prompt,
                    max_tokens=512,
                )
                scores = parse_scores_from_response(rubric_resp)
                if scores:
                    new_item["RELEVANCE"] = scores["RELEVANCE"]
                    new_item["COHERENCE"] = scores["COHERENCE"]
                    new_item["ORGANIZATION"] = scores["ORGANIZATION"]
                else:
                    new_item["RELEVANCE"] = None
                    new_item["COHERENCE"] = None
                    new_item["ORGANIZATION"] = None
                    print(f"⚠️  Failed to parse rubric for doc {doc_id}: {rubric_resp[:120]}")

                # ---- CEFR prediction ----
                cefr_user_prompt = build_cefr_prompt(content)
                cefr_resp = chat_once(
                    client=client,
                    model=model,
                    system_prompt=CEFR_SYSTEM,
                    user_prompt=cefr_user_prompt,
                    max_tokens=1024,
                )
                cefr_pred = parse_cefr_from_response(cefr_resp)
                new_item["cefr_prediction"] = cefr_pred
                if cefr_pred is None:
                    print(f"⚠️  Failed to parse CEFR for doc {doc_id}: {cefr_resp[:120]}")

            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"✅ Inference done! JSONL written to {output_path}")


# ---------------------------------------------------------------------------
# Merge JSONL results back into the original JSON
# ---------------------------------------------------------------------------
def merge_results(
    input_path: Path,
    output_jsonl_path: Path,
    save_path: Path,
) -> None:
    """Read JSONL output and merge pseudo-labels back into original JSON format."""
    # Load original data as base (to keep ordering)
    original: List[Dict[str, Any]] = json.load(open(input_path, "r", encoding="utf-8"))
    orig_map = {item["document_id"]: item for item in original}

    # Read JSONL predictions
    predictions: Dict[int, Dict[str, Any]] = {}
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                predictions[item["document_id"]] = item

    # Merge
    merged = []
    for item in original:
        doc_id = item["document_id"]
        if doc_id in predictions:
            merged.append(predictions[doc_id])
        else:
            # Keep original (no prediction available)
            merged.append(item)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged {len(predictions)} predictions into {save_path}")
    print(f"   Total items: {len(merged)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudo-label essays with CEFR + rubric scores via DeepInfra"
    )
    parser.add_argument("--input", type=Path, default=Path("data/HI_2641A.json"))
    parser.add_argument("--output", type=Path, default=Path("output/HI_2641A.jsonl"))
    parser.add_argument("--question_path", type=Path, default=Path("data/test_question.json"))
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--level", type=str, default="HI", help="Level: I or HI")
    parser.add_argument("--limit", type=int, default=None, help="Only process N records")
    parser.add_argument("--start", type=int, default=0, help="Start index")

    # Merge mode
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge JSONL output back into JSON format instead of running inference",
    )
    parser.add_argument(
        "--save_path", type=Path, default=None,
        help="Where to save merged JSON (used with --merge)",
    )

    args = parser.parse_args()

    if args.merge:
        save = args.save_path or args.input  # overwrite original by default
        merge_results(
            input_path=args.input,
            output_jsonl_path=args.output,
            save_path=save,
        )
    else:
        run_inference(
            input_path=args.input,
            output_path=args.output,
            question_path=args.question_path,
            api_key=args.api_key,
            model=args.model,
            level=args.level,
            limit=args.limit,
            start=args.start,
        )


if __name__ == "__main__":
    main()

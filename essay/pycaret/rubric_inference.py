import argparse
import json
import os
import re
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
_CURRENT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _CURRENT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
_PROJECT_ROOT = _PARENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from prompt.rubric_prompt import SYSTEM_INSTRUCTIONS, build_content_scoring_prompt

load_dotenv()

_SCORE_KEYS = ("RELEVANCE", "COHERENCE", "ORGANIZATION")

def _coerce_score(v: Any) -> Optional[int]:
    try:
        iv = int(v)
        return iv if 0 <= iv <= 5 else None
    except Exception:
        return None

def parse_scores_from_response(text: str) -> Optional[Dict[str, int]]:
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

    out: Dict[str, int] = {}
    for k in _SCORE_KEYS:
        mm = re.search(rf"{k}\s*[:]\s*([0-5])", text, flags=re.IGNORECASE)
        if mm:
            out[k.upper()] = int(mm.group(1))
    if len(out) == 3:
        return out
    return None


def make_client(api_key: Optional[str]) -> OpenAI:
    key = api_key or os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("Please set --api_key or environment variable DEEPINFRA_API_KEY")
    return OpenAI(api_key=key, base_url="https://api.deepinfra.com/v1/openai")

def chat_once(client: OpenAI, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
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

def run_inference(
    input_path: Path,
    output_path: Path,
    question_path: Path,
    api_key: Optional[str],
    model: str,
    limit: Optional[int],
    start: int,
) -> None:
    data: List[Dict[str, Any]] = json.load(open(input_path, "r", encoding="utf-8"))
    data = data[start : start + limit] if limit is not None else data[start:]

    # 讀題目字典：questions["subject"]["IW-0901"] -> 中文題目
    try:
        qobj = json.load(open(question_path, "r", encoding="utf-8"))
        
        subject_map: Dict[str, str] = qobj.get("subject", {})
    except FileNotFoundError:
        subject_map = {}
        print(f"⚠️ question file not found: {question_path}; proceeding without topic text")

    client = make_client(api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("LEVEL: ", input_path.stem.split('_')[0])
    results = []
    if start > 0 and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f_in:
            existing_results = json.load(f_in)
        if not isinstance(existing_results, list):
            raise ValueError(f"{output_path} must contain a JSON list to resume.")
        results.extend(existing_results)
        print(f"Loaded {len(existing_results)} existing records from {output_path}")

    for item in tqdm(data, desc="Scoring"):
        content: str = (item.get("content") or "").strip()
        if not content:
            pred = None
            resp_text = ""
        else:
            level = input_path.stem.split('_')[0] if '_' in input_path.stem else 'I'
            subject_id = item.get("subject")  # e.g., "IW-0901"
            subject_text = subject_map.get(subject_id)

            user_prompt = build_content_scoring_prompt(
                subject_text=subject_text,
                content=content,
                level=level,
            )

            resp_text = chat_once(
                client=client,
                model=model,
                system_prompt=SYSTEM_INSTRUCTIONS,
                user_prompt=user_prompt,
                max_tokens=512,
            )
            pred = parse_scores_from_response(resp_text)

        new_item = dict(item)
        if pred:
            new_item["RELEVANCE"] = pred["RELEVANCE"]
            new_item["COHERENCE"]  = pred["COHERENCE"]
            new_item["ORGANIZATION"] = pred["ORGANIZATION"]
        else:
            new_item["RELEVANCE"] = None
            new_item["COHERENCE"]  = None
            new_item["ORGANIZATION"] = None

        results.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)


# ===== CLI =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Rubric scoring via DeepInfra (OpenAI-compatible)")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--question_path", type=Path)
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        question_path=args.question_path,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        start=args.start,
    )

if __name__ == "__main__":
    main()

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
from prompt.cefr_prompt import SYSTEM_INSTRUCTIONS, build_user_prompt

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

load_dotenv()

def parse_cefr_from_response(text: str) -> Optional[str]:
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

    matches = re.findall(r"CEFR\s*level\s*:\s*([ABC][12])", text, flags=re.IGNORECASE)
    if matches:
        lvl = matches[-1].upper()   # take the last one
        if lvl in CEFR_LEVELS:
            return lvl

    matches = re.findall(r"\b(A1|A2|B1|B2|C1|C2)\b", text)
    if matches:
        return matches[-1]

    return None


def make_client(api_key: Optional[str]) -> OpenAI:
    """
    Create an OpenAI client that talks to DeepInfra's OpenAI-compatible endpoint.
    """
    key = api_key or os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("Please set --api_key or environment variable DEEPINFRA_API_KEY")
    return OpenAI(
        api_key=key,
        base_url="https://api.deepinfra.com/v1/openai"
    )

def chat_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """
    One-shot chat completion via DeepInfra (OpenAI-compatible).
    Includes light retry on transient errors.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # simple retry/backoff
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            # brief exponential backoff
            sleep_s = 1.5 ** attempt
            time.sleep(sleep_s)
            if attempt == 3:
                raise
    return ""  # unreachable in normal cases

def run_inference(
    input_path: Path,
    output_path: Path,
    api_key: Optional[str],
    model: str,
    limit: Optional[int],
    start: int,
) -> None:
    data: List[Dict[str, Any]] = json.load(open(input_path))
    if limit is not None:
        data = data[start : start + limit]

    client = make_client(api_key)

    # ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:  # append 模式
        for idx, item in enumerate(tqdm(data), start=1):
            content: str = item.get("content", "").strip()
            if not content:
                pred = None
                resp_text = ""
            else:
                user_prompt = build_user_prompt(content)
                resp_text = chat_once(
                    client=client,
                    model=model,
                    system_prompt=SYSTEM_INSTRUCTIONS,
                    user_prompt=user_prompt,
                    max_tokens=1024,
                )
                pred = parse_cefr_from_response(resp_text)

            new_item = dict(item)
            new_item["cefr_prediction"] = {
                "model": model,
                "prediction": pred,
                "raw_output": resp_text,
            }
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            f_out.flush()

# ===== CLI =====

def main() -> None:
    parser = argparse.ArgumentParser(description="CEFR inference over essays via DeepInfra (OpenAI-compatible)")
    parser.add_argument("--input", type=Path, default=Path("../data/sampled_IHI_train.json"))
    parser.add_argument("--output", type=Path, default=Path("output/gemma-3-12b.jsonl"))
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        start=args.start,
    )

if __name__ == "__main__":
    main()
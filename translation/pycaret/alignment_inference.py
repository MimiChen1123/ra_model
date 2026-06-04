import argparse
import json
import os
import re
import time
import sys
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from prompt.alignment_prompt import ALIGNMENT_SYSTEM_PROMPT, ALIGNEMTN_USER_PROMPT

load_dotenv()

def parse_alignment_from_response(text: str) -> Optional[List[List[str]]]:
    """
    Extract the JSON alignment array from the model's response.

    The expected format is a JSON array of arrays, e.g.:
    [
      ["Chinese part", "English part"],
      ["Another", ""]
    ]

    The function attempts multiple extraction strategies robustly.
    """

    # 1. Try fenced ```json ... ``` blocks
    fenced = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # 2. Try fenced ``` ... ``` blocks without json tag
    fenced_generic = re.search(r"```[\s\S]*?(\[[\s\S]*?\])[\s\S]*?```", text)
    if fenced_generic:
        try:
            return json.loads(fenced_generic.group(1))
        except Exception:
            pass

    # 3. Try direct JSON array inside the text
    array_match = re.search(r"(\[[\s\S]*?\])", text)
    if array_match:
        try:
            return json.loads(array_match.group(1))
        except Exception:
            pass

    # 4. Final fallback: extract top-level [...] carefully
    #    (Find the first '[' and last matching ']')
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return "Parsing Error: Incorrect JSON Format."

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
    max_tokens: int = 2048,
) -> str:

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
                timeout=60,
            )
            # print("Here is the raw output:\n",resp)
            return resp.choices[0].message.content.strip()
        
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            time.sleep(2 ** attempt)
            if attempt == 3:
                return f"Reached maximum retries. Last error: {e}"

def run_inference(
    subject_path: Path,
    input_path: Path,
    output_path: Path,
    api_key: Optional[str],
    model: str,
    limit: Optional[int],
    start: int,
) -> None:
    
    # load subjects mapping file
    with open(subject_path, "r", encoding="utf-8") as f:
        subjects = json.load(f)["subject"]
    
    if input_path.suffix == ".csv":
        # use pandas for csv data file
        raw_data = pd.read_csv(input_path)
        raw_data["question"] = raw_data["subject"].map(subjects)
        data = raw_data.to_dict(orient="records")
    elif input_path.suffix == ".json":
        # load jsonl file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["question"] = subjects.get(item["subject"], "")
    else:
        raise ValueError("Unsupported input file format. Use .csv or .json")

    if start is not None:
        data = data[start:]
    if limit is not None:
        data = data[start : start + limit]

    client = make_client(api_key)

    # ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:  # append 模式
        for idx, item in enumerate(tqdm(data), start=1):
            content: str = item.get("content", "").strip()
            question: str = item.get("question", "").strip()
            
            if not content:
                alignment = None
                resp_text = ""
            else:
                user_prompt = ALIGNEMTN_USER_PROMPT.format(
                    chinese_question=question,
                    english_translation=content,
                )

                resp_text = chat_once(
                    client=client,
                    model=model,
                    system_prompt=ALIGNMENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=1024,
                )
                alignment = parse_alignment_from_response(resp_text)

            new_item = dict(item)
            new_item["llm_alignment"] = {
                "model": model,
                "alignment": alignment,
                "raw_output": resp_text,
            }
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            f_out.flush()

# ===== CLI =====

def main() -> None:
    parser = argparse.ArgumentParser(description="Alignment via DeepInfra (OpenAI-compatible)")
    parser.add_argument("--subject", type=Path, default=Path("./data/reference-answers/question_prompts.json"))
    parser.add_argument("--input", type=Path, default=Path("./data/HI_lttc_test.json"))
    parser.add_argument("--output", type=Path, default=Path("./data/alignment/gemma-3-12b-it-HI-lttc-test.jsonl"))
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    args = parser.parse_args()

    run_inference(
        subject_path=args.subject,
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        start=args.start,
    )

if __name__ == "__main__":
    main()
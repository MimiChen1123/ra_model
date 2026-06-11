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
from prompt.pseudo_label_prompt import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT

load_dotenv()

def parse_labels_from_response(text: str) -> Optional[Dict[str, int]]:
    """
    Extract and validate the JSON evaluation output from the model's response.

    Args:
        text (str): The raw output from the LLM.

    Returns:
        Optional[Dict[str, int]]: A dictionary with the counts for each category if valid, otherwise None.
    """
    try:
        # 1. Try fenced ```json ... ``` blocks
        fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fenced:
            result = json.loads(fenced.group(1))
            if validate_evaluation_output(result):
                return result

        # 2. Try fenced ``` ... ``` blocks without json tag
        fenced_generic = re.search(r"```[\s\S]*?(\{[\s\S]*?\})[\s\S]*?```", text)
        if fenced_generic:
            result = json.loads(fenced_generic.group(1))
            if validate_evaluation_output(result):
                return result

        # 3. Try direct JSON object inside the text
        json_match = re.search(r"(\{[\s\S]*?\})", text)
        if json_match:
            result = json.loads(json_match.group(1))
            if validate_evaluation_output(result):
                return result

    except (json.JSONDecodeError, TypeError):
        pass

    # If all parsing attempts fail, return None
    return {}

def validate_evaluation_output(result: Dict[str, int]) -> bool:
    """
    Validate the JSON evaluation output.

    Args:
        result (Dict[str, int]): The parsed JSON object.

    Returns:
        bool: True if the output is valid, False otherwise.
    """
    required_keys = ["missing translation", "over-translation", "mistranslation", "grammar errors", "spelling errors", "explanation"]
    return (
        isinstance(result, dict)
        and all(key in result for key in required_keys)
        and all(isinstance(result[key], int) for key in required_keys[:-1])
        and isinstance(result["explanation"], str)
    )

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

def is_chinese(text: str) -> bool:
    """
    Check if the given text contains Chinese characters.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text contains Chinese characters, False otherwise.
    """
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Unicode range for Chinese characters
            return True
    return False

def run_inference(
    input_path: Path,
    output_path: Path,
    api_key: Optional[str],
    model: str,
    limit: Optional[int],
    start: int,
) -> None:
    
    # load jsonl file
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    if start is not None:
        data = data[start:]
    if limit is not None:
        data = data[start : start + limit]

    client = make_client(api_key)

    # ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:  # append 模式
        for idx, item in enumerate(tqdm(data), start=1):
            alignments = item.get("llm_alignment", {}).get("alignment", [])
            
            new_item = dict(item)
            new_item["llm_pseudo_labels"] = {
                "model": model,
                "pseudo_labels": [],
                "raw_output": [],
            }
            
            if not isinstance(alignments, list):
                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                f_out.flush()
                continue
            
            for alignment in alignments:
                try:
                    chinese_sent = alignment[0].strip()
                    english_sent = alignment[1].strip()
                except IndexError:
                    if len(alignment) == 1:
                        if is_chinese(alignment[0].strip()):
                            chinese_sent = alignment[0].strip()
                            english_sent = ""
                        else:
                            chinese_sent = ""
                            english_sent = alignment[0].strip()
                    else:
                        continue
                
                if not chinese_sent and english_sent:
                    pseudo_labels = {
                        "missing translation": 0,
                        "over-translation": len(english_sent.split(" ")),
                        "mistranslation": 0,
                        "grammar errors": 0,
                        "spelling errors": 0,
                        "explanation": "Over-translation detected."
                    }
                    resp_text = "Over-translation detected."
                elif not english_sent and chinese_sent:
                    pseudo_labels = {
                        "missing translation": len(chinese_sent.split(" ")),
                        "over-translation": 0,
                        "mistranslation": 0,
                        "grammar errors": 0,
                        "spelling errors": 0,
                        "explanation": "Missing translation detected."
                    }
                    resp_text = "Missing translation detected."
                else:
                    user_prompt = EVALUATION_USER_PROMPT.format(
                        chinese_sentence=chinese_sent,
                        english_sentence=english_sent,
                    )

                    resp_text = chat_once(
                        client=client,
                        model=model,
                        system_prompt=EVALUATION_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        max_tokens=2048,
                    )
                    pseudo_labels = parse_labels_from_response(resp_text)
                
                new_item["llm_pseudo_labels"]["pseudo_labels"].append(pseudo_labels)
                new_item["llm_pseudo_labels"]["raw_output"].append(resp_text)
                
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            f_out.flush()

# ===== CLI =====
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo labels via DeepInfra (OpenAI-compatible)")
    parser.add_argument("--input", type=Path, default=Path("./data/alignment/gemma-3-12b-it-I-lttc-test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("./data/pseudo_labels/gemma-3-12b-it-I-lttc-test.jsonl"))
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
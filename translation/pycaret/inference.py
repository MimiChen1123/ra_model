import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tqdm import tqdm
from transformers import AutoTokenizer

from alignment_inference import parse_alignment_from_response
from prompt.alignment_prompt import ALIGNEMTN_USER_PROMPT, ALIGNMENT_SYSTEM_PROMPT
from prompt.pseudo_label_prompt import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT
from prompt.translation_prompt import build_translation_error_prompt, build_translation_scoring_prompt
from pseudo_label_inference import parse_labels_from_response
from train import MistralWithRegression

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PSEUDO_LABEL_KEYS = (
    ("missing_translation", "missing translation"),
    ("over-translation", "over-translation"),
    ("mistranslation", "mistranslation"),
    ("grammar_errors", "grammar errors"),
    ("spelling_errors", "spelling errors"),
)


def load_question_map(question_path: Path) -> Dict[str, str]:
    with open(question_path, "r", encoding="utf-8") as f:
        question_data = json.load(f)
    return question_data.get("subject", {})


def make_client(api_key: Optional[str]) -> OpenAI:
    key = api_key or os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("Please set --api_key or environment variable DEEPINFRA_API_KEY")
    return OpenAI(api_key=key, base_url="https://api.deepinfra.com/v1/openai")


def chat_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> str:
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
                timeout=60,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, RateLimitError, APIStatusError):
            time.sleep(2 ** attempt)
            if attempt == 3:
                raise
    return ""


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def generate_alignment(client: OpenAI, model: str, source_text: str, translation_text: str) -> Optional[List[Any]]:
    user_prompt = ALIGNEMTN_USER_PROMPT.format(
        chinese_question=source_text,
        english_translation=translation_text,
    )
    raw_output = chat_once(
        client=client,
        model=model,
        system_prompt=ALIGNMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1024,
    )
    return parse_alignment_from_response(raw_output)


def generate_pseudo_labels(client: OpenAI, model: str, alignments: List[Any]) -> List[Dict[str, Any]]:
    pseudo_labels = []
    for alignment in alignments:
        try:
            chinese_sent = alignment[0].strip()
            english_sent = alignment[1].strip()
        except IndexError:
            if len(alignment) == 1 and is_chinese(alignment[0].strip()):
                chinese_sent = alignment[0].strip()
                english_sent = ""
            elif len(alignment) == 1:
                chinese_sent = ""
                english_sent = alignment[0].strip()
            else:
                continue

        if not chinese_sent and english_sent:
            pseudo_labels.append({
                "missing translation": 0,
                "over-translation": len(english_sent.split()),
                "mistranslation": 0,
                "grammar errors": 0,
                "spelling errors": 0,
                "explanation": "Over-translation detected.",
            })
            continue

        if chinese_sent and not english_sent:
            pseudo_labels.append({
                "missing translation": len(chinese_sent.split()),
                "over-translation": 0,
                "mistranslation": 0,
                "grammar errors": 0,
                "spelling errors": 0,
                "explanation": "Missing translation detected.",
            })
            continue

        user_prompt = EVALUATION_USER_PROMPT.format(
            chinese_sentence=chinese_sent,
            english_sentence=english_sent,
        )
        raw_output = chat_once(
            client=client,
            model=model,
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
        )
        parsed = parse_labels_from_response(raw_output)
        if parsed:
            pseudo_labels.append(parsed)

    return pseudo_labels


def aggregate_pseudo_labels(pseudo_labels: List[Dict[str, Any]]) -> Dict[str, float]:
    valid_labels = [item for item in pseudo_labels if "missing translation" in item]
    if not valid_labels:
        return {output_key: 0.0 for output_key, _ in PSEUDO_LABEL_KEYS}

    aggregated = {}
    for output_key, source_key in PSEUDO_LABEL_KEYS:
        aggregated[output_key] = sum(float(item.get(source_key, 0.0) or 0.0) for item in valid_labels) / len(valid_labels)
    return aggregated


def clamp_and_round_score(score: float) -> float:
    score = max(0.0, min(5.0, score))
    return round(score * 2) / 2.0


def load_model(model_name: str, checkpoint_path: Path, no_quantize: bool):
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MistralWithRegression(
        model_name=model_name,
        no_quantize=no_quantize,
        pseudo_labels_dim=5,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    logging.info("Loaded checkpoint from %s; missing=%s unexpected=%s", checkpoint_path, len(missing), len(unexpected))
    model.eval()
    return tokenizer, model


def predict_one(
    item: Dict[str, Any],
    tokenizer,
    model,
    question_map: Dict[str, str],
    max_length: int,
    api_client: Optional[OpenAI],
    llm_model: str,
    generate_labels: bool,
) -> Dict[str, Any]:
    level = str(item.get("level", "I")).upper()
    source_text = item.get("question") or item.get("requirement") or question_map.get(item.get("subject"), "")
    translation_text = item.get("input") or item.get("content", "")

    new_item = dict(item)
    pseudo_labels = item.get("pseudo_labels") or []

    if generate_labels and not pseudo_labels:
        if api_client is None:
            raise RuntimeError("API client is required when --generate_pseudo_labels is enabled.")
        alignments = item.get("alignment") or generate_alignment(api_client, llm_model, source_text, translation_text)
        new_item["alignment"] = alignments
        if isinstance(alignments, list):
            pseudo_labels = generate_pseudo_labels(api_client, llm_model, alignments)
            new_item["pseudo_labels"] = pseudo_labels

    pseudo_features = aggregate_pseudo_labels(pseudo_labels)

    prompt_score = build_translation_scoring_prompt(
        source_text=source_text,
        translation_text=translation_text,
        level=level,
    )
    prompt_err = build_translation_error_prompt(
        source_text=source_text,
        translation_text=translation_text,
    )

    tokens_score = tokenizer(
        prompt_score,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    tokens_err = tokenizer(
        prompt_err,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    pseudo_labels_score = torch.tensor(
        [
            pseudo_features["missing_translation"],
            pseudo_features["over-translation"],
            pseudo_features["mistranslation"],
            pseudo_features["grammar_errors"],
            pseudo_features["spelling_errors"],
        ],
        dtype=torch.float,
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(
            input_ids_score=tokens_score["input_ids"],
            input_ids_err=tokens_err["input_ids"],
            attention_mask_score=tokens_score["attention_mask"],
            attention_mask_err=tokens_err["attention_mask"],
            pseudo_labels_score=pseudo_labels_score,
        )

    raw_score = float(prediction.squeeze(0).cpu().numpy().item())
    new_item["predicted_score_raw"] = raw_score
    new_item["predicted_score"] = clamp_and_round_score(raw_score)
    new_item["pseudo_label_features"] = pseudo_features
    return new_item


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference with a trained translation scoring model.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON list.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument("--question_data", type=Path, default=Path("./data/reference-answers/question_prompts.json"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint .pt file.")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--batch_size", type=int, default=1, help="Reserved for CLI compatibility; inference runs item-by-item.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--generate_pseudo_labels", action="store_true", help="Generate alignment and pseudo labels if missing.")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPINFRA_API_KEY"))
    parser.add_argument("--llm_model", type=str, default="google/gemma-3-12b-it")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{args.input} must contain a JSON list.")

    data = data[args.start : args.start + args.limit] if args.limit is not None else data[args.start:]
    question_map = load_question_map(args.question_data)
    tokenizer, model = load_model(args.model, args.checkpoint, args.no_quantize)
    api_client = make_client(args.api_key) if args.generate_pseudo_labels else None

    results = []
    for item in tqdm(data, desc="Predicting"):
        results.append(
            predict_one(
                item=item,
                tokenizer=tokenizer,
                model=model,
                question_map=question_map,
                max_length=args.max_length,
                api_client=api_client,
                llm_model=args.llm_model,
                generate_labels=args.generate_pseudo_labels,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info("Saved %s predictions to %s", len(results), args.output)


if __name__ == "__main__":
    main()

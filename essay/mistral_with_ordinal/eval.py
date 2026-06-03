import sys
import json
import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from model import MistralWithOrdinalRegression
from utils import set_seed, decode_scores_from_logits, PROMPT_TEMPLATE


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["I", "HI"],
        help="Level of the model (I or HI). Required only when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If provided, overrides the default checkpoint for the selected level.",
    )
    parser.add_argument(
        "--question_json",
        type=str,
        default=None,
        help="Path to question.json. Enables batch inference when used with --answer_json.",
    )
    parser.add_argument(
        "--answer_json",
        type=str,
        default=None,
        help="Path to answer.json. Enables batch inference when used with --question_json.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="prediction.json",
        help="Path to write batch inference results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for JSON batch inference.",
    )

    
    return parser.parse_args()


def get_checkpoint_path(args):
    if args.checkpoint is not None:
        return args.checkpoint
    if args.level is None:
        raise ValueError("--level is required when --checkpoint is not provided.")
    if args.level == "I":
        return "/home/b10902133/gept-analysis/llm/outputs/Mistral-7B-Instruct-v0.2/ordinal_regression_train/model_epoch_6.pt"
    return "/home/b10902133/gept-analysis/llm/outputs/Mistral-7B-Instruct-v0.2/ordinal_regression_train_HI/model_epoch_1.pt"


def load_questions(question_json):
    with Path(question_json).open("r", encoding="utf-8") as file:
        data = json.load(file)
    subjects = data.get("subject")
    if not isinstance(subjects, dict):
        raise ValueError("question.json must contain a 'subject' object.")
    return subjects


def load_answers(answer_json):
    with Path(answer_json).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("answer.json must be a list of objects.")
    return data


def build_prompt_from_answer(answer, subjects):
    subject_id = answer.get("subject")
    content = answer.get("content")
    if subject_id not in subjects:
        raise ValueError(f"Unknown subject: {subject_id}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Missing content for document_id={answer.get('document_id')}")
    return PROMPT_TEMPLATE.format(requirement=subjects[subject_id], input=content)



def load_model_and_tokenizer(args, device, model_name, num_classes):
    checkpoint_path = get_checkpoint_path(args)
    level_display = args.level if args.level is not None else "custom checkpoint"

    print(f"Loading '{model_name}' with ordinal regression head (Level {level_display})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = MistralWithOrdinalRegression(model_name, num_classes=num_classes)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Loaded model state dict from checkpoint [model_state_dict]")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded model state dict from checkpoint dict structure")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model state dict directly")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def run_batch_inference(args, model, tokenizer, device, score_step):
    subjects = load_questions(args.question_json)
    answers = load_answers(args.answer_json)
    results = []

    for start in tqdm(range(0, len(answers), args.batch_size), desc="Processing batches"):
        batch_answers = answers[start:start + args.batch_size]
        prompts = [build_prompt_from_answer(answer, subjects) for answer in batch_answers]

        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=2048,
            padding="max_length",
            truncation=True,
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_classes, pred_scores = decode_scores_from_logits(logits, score_step=score_step)
            cumulative_probs = torch.sigmoid(logits).cpu().tolist()

        for answer, pred_class, pred_score, probs in zip(
            batch_answers,
            pred_classes.cpu().tolist(),
            pred_scores.cpu().tolist(),
            cumulative_probs,
        ):
            result = dict(answer)
            result["predicted_score"] = round(float(pred_score), 1)
            result["predicted_class"] = int(pred_class)
            result["cumulative_probs"] = [round(float(prob), 4) for prob in probs]
            results.append(result)

        print(f"Processed {min(start + args.batch_size, len(answers))}/{len(answers)} answers")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Saved batch predictions to: {output_path}")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    num_classes = 11
    score_step = 0.5

    model, tokenizer = load_model_and_tokenizer(args, device, model_name, num_classes)

    if args.question_json is None or args.answer_json is None:
        raise ValueError("--question_json and --answer_json must be provided together.")
    
    run_batch_inference(args, model, tokenizer, device, score_step)
    

if __name__ == "__main__":
    main()

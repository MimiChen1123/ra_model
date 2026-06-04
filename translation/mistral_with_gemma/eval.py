import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import logging

from utils import set_seed, load_translation
from model import MistralWithRegression
from dataset import TranslationDataset

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it to your Hugging Face API token.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model name or path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training or testing",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable quantization (use full precision)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (4 or 8, used in training mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        default="./data/I_test.json",
        help="Path to the test data JSON file (used in test mode)",
    )
    parser.add_argument(
        "--question_data",
        type=Path,
        default="./data/question.json",
        help="Path to the question data JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to save model checkpoints (training) or predictions (testing)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the trained model checkpoint (required in test mode)",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    if not args.output_dir:
        curr_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            Path("outputs") / f"{args.model.replace('/', '_')}" / curr_datetime
        )
        logging.info(f"No output directory specified, using default: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args

def main():
    """Run training or testing based on arguments."""
    args = parse_args()
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:  # FIX: ensure pad token
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Running in test mode")
    essays = load_translation(args.test_data, args.question_data)

    data_dicts = [
        {
            "requirement": c["question"],  # source text
            "input": c["content"],         # translation text
            "level": c["level"],
            "score": c.get("score", 0),
            "id": c.get("id", i),
        }
        for i, c in enumerate(essays)
    ]

    test_data = TranslationDataset(data_dicts, tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = MistralWithRegression(
        model_name=args.model,
        no_quantize=args.no_quantize
    )

    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        model.load_state_dict(checkpoint, strict=False)
        logging.info(f"Loaded model from {args.checkpoint}")

    model.eval()

    predictions = []
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
        for batch in pbar:
            input_ids_score = batch["input_ids_score"].cuda()
            input_ids_err = batch["input_ids_err"].cuda()
            attention_mask_score = batch["attention_mask_score"].cuda()
            attention_mask_err = batch["attention_mask_err"].cuda()
            target = batch["score"].cuda()
            doc_ids = batch["document_id"]

            pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err)

            if torch.any(target != 0.0):
                loss = criterion(pred, target)
                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({"test_loss": f"{avg_loss:.4f}"})

            for doc_id, pred_score, true_score in zip(doc_ids, pred.cpu().numpy(), target.cpu().numpy()):
                predictions.append({
                    "document_id": doc_id.item() if hasattr(doc_id, "item") else doc_id,
                    "predicted_score": float(pred_score),
                    "true_score": float(true_score)
                })

    if total_loss > 0:
        avg_mse = total_loss / len(test_loader)
        logging.info(f"Test MSE: {avg_mse:.4f}")
    else:
        logging.info("No ground truth scores found in test data; MSE not computed")

    predictions_file = args.output_dir / "predictions.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"Saved predictions to {predictions_file}")

    metrics_file = args.output_dir / "test_metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        if total_loss > 0:
            f.write(f"Test MSE: {avg_mse:.4f}\n")
        f.write(f"Number of predictions: {len(predictions)}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
    logging.info(f"Saved metrics to {metrics_file}")



if __name__ == "__main__":
    main()

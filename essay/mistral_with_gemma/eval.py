import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import f1_score
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import logging

from utils import set_seed, load_essays
from model import MistralWithRegression
from dataset import EssayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/mimi911123/ra_model/.env")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it to your Hugging Face API token.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def score_to_class(score, min_score, score_step, num_classes):
    idx = int(round((float(score) - min_score) / score_step))
    return max(0, min(num_classes - 1, idx))


def class_to_score(class_idx, min_score, score_step):
    return float(class_idx) * score_step + min_score


def batch_values_to_list(values):
    if torch.is_tensor(values):
        return values.cpu().tolist()
    return list(values)


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
        default="./data/I_train_question.json",
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
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum possible score for converting regression output to classes",
    )
    parser.add_argument(
        "--score_step",
        type=float,
        default=0.5,
        help="Score interval for converting regression output to classes",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=11,
        help="Number of score classes after discretization",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required when --test is specified")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)
    if tokenizer.pad_token is None:  # FIX: ensure pad token
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info("Running in test mode")
    # Load test data
    essays = load_essays(args.test_data, args.question_data)
    
    data_dicts = [
        {
            "requirement": c["question"],
            "input": c["content"],
            "level": c["level"],
            "cefr_prediction": c["cefr_prediction"],
            "word_count": c["word_count"],
            "RELEVANCE": c["RELEVANCE"],
            "COHERENCE": c["COHERENCE"],
            "ORGANIZATION": c["ORGANIZATION"],
            "score": c.get("score"),
            "document_id": c.get("document_id", i),
            "seat_number": c.get("seat_number", ""),
            "subject": c.get("subject", "")
        }
        for i, c in enumerate(essays)
    ]

    test_data = EssayDataset(data_dicts, tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Use the SAME custom model class as training
    model = MistralWithRegression(
        model_name=args.model, 
        no_quantize=args.no_quantize  # Use same quantization setting as training
    )
    
    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        # Load the checkpoint into the custom model
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # strict=False to handle any minor mismatches
        logging.info(f"Loaded model from {args.checkpoint}")
    
    # Do not move the model explicitly. It is already sharded per device via device_map="auto".
    model.eval()

    # Run inference using the custom model's forward method
    predictions = []
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            other_feature = batch["other_feature"].to(device)
            target = batch["score"].to(device)
            doc_ids = batch["document_id"]
            seat_numbers = batch["seat_number"]
            subjects = batch["subject"]

            pred = model(input_ids=input_ids, attention_mask=attention_mask, other_feature=other_feature)
            
            # Calculate loss if targets are available
            if torch.all(target != 0.0):  # Only if we have real targets
                loss = criterion(pred, target)
                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({"test_loss": f"{avg_loss:.4f}"})

            # Store predictions
            for seat_number, doc_id, pred_score, true_score, subject in zip(
                seat_numbers,
                doc_ids,
                pred.cpu().numpy(),
                target.cpu().numpy(),
                subjects,
            ):
                pred_class = score_to_class(
                    pred_score,
                    args.min_score,
                    args.score_step,
                    args.num_classes,
                )
                result = {
                    "document_id": int(doc_id),
                    "seat_number": str(seat_number),
                    "subject": subject,
                    "predicted_score": class_to_score(pred_class, args.min_score, args.score_step),
                    "true_score": float(true_score)
                }
                predictions.append(result)

    # Compute average MSE only if we have real targets
    if total_loss > 0:
        avg_mse = total_loss / len(test_loader)
        logging.info(f"Test MSE: {avg_mse:.4f}")
    else:
        logging.info("No ground truth scores found in test data; MSE not computed")

    valid_predictions = [
        prediction for prediction in predictions
        if prediction["true_score"] != 0.0
    ]
    if valid_predictions:
        y_true = [
            score_to_class(
                prediction["true_score"],
                args.min_score,
                args.score_step,
                args.num_classes,
            )
            for prediction in valid_predictions
        ]
        y_pred = [
            score_to_class(
                prediction["predicted_score"],
                args.min_score,
                args.score_step,
                args.num_classes,
            )
            for prediction in valid_predictions
        ]
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        logging.info(f"Test Macro F1: {macro_f1:.4f}")
        logging.info(f"Test Weighted F1: {weighted_f1:.4f}")
    else:
        logging.info("No ground truth scores found in test data; F1 not computed")

    # Save predictions
    predictions_file = args.output_dir / "predictions.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"Saved predictions to {predictions_file}")

    # Save metrics
    metrics_file = args.output_dir / "test_metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        if total_loss > 0:
            f.write(f"Test MSE: {avg_mse:.4f}\n")
        if valid_predictions:
            f.write(f"Test Macro F1: {macro_f1:.4f}\n")
            f.write(f"Test Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"F1 Min Score: {args.min_score}\n")
            f.write(f"F1 Score Step: {args.score_step}\n")
            f.write(f"F1 Num Classes: {args.num_classes}\n")
        f.write(f"Number of predictions: {len(predictions)}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
    logging.info(f"Saved metrics to {metrics_file}")

    
if __name__ == "__main__":
    main()

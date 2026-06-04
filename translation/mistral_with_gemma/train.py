import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import wandb
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
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (used in training mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training or testing",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer (used in training mode)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test split (used in training mode)",
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
        "--essay_data",
        type=Path,
        default="./data/I_train.json",
        help="Path to the training data JSON file (used in training mode)",
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
    args = parser.parse_args()


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

    logging.info("Running in training mode")
    wandb.init(
        project="gept-analysis",
        name=args.output_dir.name,
        config=args,
        dir=args.output_dir,
    )

    essays = load_translation(args.essay_data, args.question_data)

    data_dicts = [
        {
            "requirement": c["question"],  # source text
            "input": c["content"],         # translation text
            "level": c["level"],
            "score": c["score"],
            "id": c.get("id", i),
        }
        for i, c in enumerate(essays)
    ]


    train, eval = train_test_split(
        data_dicts, test_size=args.test_size, random_state=args.seed
    )
    data = {"train": train, "eval": eval}

    train_data = TranslationDataset(data["train"], tokenizer)
    eval_data = TranslationDataset(data["eval"], tokenizer)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    model = MistralWithRegression(args.model, no_quantize=args.no_quantize)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1} / {args.epochs}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            input_ids_score = batch["input_ids_score"].cuda()
            input_ids_err = batch["input_ids_err"].cuda()
            attention_mask_score = batch["attention_mask_score"].cuda()
            attention_mask_err = batch["attention_mask_err"].cuda()
            target = batch["score"].cuda()
            
            # print(tokenizer.decode(input_ids_score[0]))
            # print('=================================')
            # print(tokenizer.decode(input_ids_err[0]))

            # exit()
            pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            wandb.log({"train_loss": loss.item()})

        tqdm.write(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")
        wandb.log({"epoch": epoch + 1, "train_avg_loss_epoch": total_loss / len(train_loader)})

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids_score = batch["input_ids_score"].cuda()
                input_ids_err = batch["input_ids_err"].cuda()
                attention_mask_score = batch["attention_mask_score"].cuda()
                attention_mask_err = batch["attention_mask_err"].cuda()
                target = batch["score"].cuda()

                pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err)
                loss = criterion(pred, target)
                eval_loss += loss.item()

        tqdm.write(f"Epoch {epoch+1} - Eval Loss: {eval_loss / len(eval_loader):.4f}")
        wandb.log({"epoch": epoch + 1, "eval_avg_loss_epoch": eval_loss / len(eval_loader)})

        # Save the model after each epoch
        model_save_path = args.output_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")

        # Write eval loss to a file
        with open(args.output_dir / "eval_loss.txt", "a") as f:
            f.write(f"Epoch {epoch+1} - Eval Loss: {eval_loss / len(eval_loader):.4f}\n")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import wandb
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
from prompt.translation_prompt import build_translation_scoring_prompt, build_translation_error_prompt

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it to your Hugging Face API token.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        """Initialize dataset for training or testing."""
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]  # FIX: example must be defined before use
        level = example.get("level", "I")

        # For translation scoring: source = requirement/question, translation = content/input
        source_text = example.get("requirement", "")
        translation_text = example.get("input", "")

        prompt_score = build_translation_scoring_prompt(
            source_text=source_text,
            translation_text=translation_text,
            level=level
        )

        prompt_err = build_translation_error_prompt(
            source_text=source_text,
            translation_text=translation_text,
        )

        tokens_score = self.tokenizer(
            prompt_score,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )   

        tokens_err = self.tokenizer(
            prompt_err,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        missing_translation = example.get("missing_translation", 0.0)
        over_translation = example.get("over-translation", 0.0)
        mistranslation = example.get("mistranslation", 0.0)
        grammar_errors = example.get("grammar_errors", 0.0)
        spelling_errors = example.get("spelling_errors", 0.0)

        pseudo_labels_score = torch.tensor([float(missing_translation), float(over_translation), float(mistranslation), float(grammar_errors), float(spelling_errors)], dtype=torch.float)

        return {
            "input_ids_score": tokens_score["input_ids"].squeeze(0),
            "input_ids_err": tokens_err["input_ids"].squeeze(0),
            "attention_mask_score": tokens_score["attention_mask"].squeeze(0),
            "attention_mask_err": tokens_err["attention_mask"].squeeze(0),
            "pseudo_labels_score": pseudo_labels_score,
            "score": torch.tensor(example["score"], dtype=torch.float),
            "document_id": example.get("id", idx),  # For tracking in test mode if needed
        }

class MistralWithRegression(nn.Module):
    def __init__(self, model_name, no_quantize=False, pseudo_labels_dim=5):
        """Initialize model with LoRA and regression head."""
        super().__init__()
        if not no_quantize:
            try:
                import bitsandbytes  # noqa: F401
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logging.warning("bitsandbytes not installed. Falling back to full precision.")
                bnb_config = None
        else:
            bnb_config = None

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_TOKEN
        )
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base) if bnb_config else base
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora_config)

        hidden_size = self.model.config.hidden_size  # e.g., 4096

        F = pseudo_labels_dim  # Assuming 5-dimensional pseudo labels
        self.feature_extractor_score= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.feature_extractor_err= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.pseudo_labels_processor = nn.Sequential(
            nn.Linear(F, 16),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids_score, input_ids_err, attention_mask_score, attention_mask_err, pseudo_labels_score):
        """Forward pass to predict score."""
        # Ensure input tensors are located on the same device as the LLM's input embeddings
        try:
            llm_device = self.model.model.get_input_embeddings().weight.device
        except Exception:
            llm_device = next(self.model.model.parameters()).device

        # === 推 score prompt ===
        input_ids_score = input_ids_score.to(llm_device)
        attention_mask_score = attention_mask_score.to(llm_device)
        outputs_score = self.model.model(
            input_ids=input_ids_score,
            attention_mask=attention_mask_score,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_score = outputs_score.hidden_states[-1]  # (B, T, H)
        cls_score = last_hidden_score[:, -1, :]  # 取最後 token 的表示 (B, H)

        # === 推 error prompt ===
        input_ids_err = input_ids_err.to(llm_device)
        attention_mask_err = attention_mask_err.to(llm_device)
        outputs_err = self.model.model(
            input_ids=input_ids_err,
            attention_mask=attention_mask_err,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_err = outputs_err.hidden_states[-1]
        cls_err = last_hidden_err[:, -1, :]

        head_device = cls_err.device

        # Move small custom heads to the LLM output device if they're not already there
        for module in (
            self.feature_extractor_score,
            self.feature_extractor_err,
            self.pseudo_labels_processor,
            self.regressor,
        ):
            # Only move if the module has parameters (avoid unnecessary ops)
            try:
                params = next(module.parameters())
            except StopIteration:
                params = None
            if params is not None and params.device != head_device:
                module.to(head_device)


        feat_score = self.feature_extractor_score(cls_score)  # (B, 512)
        feat_err = self.feature_extractor_err(cls_err)

        pseudo_labels_score = pseudo_labels_score.to(head_device)
        pseudo_labels_score = self.pseudo_labels_processor(pseudo_labels_score)  # (B, F)

        combined_feat = torch.cat([feat_score, feat_err, pseudo_labels_score], dim=-1)  # (B, 1024+F)
        out = self.regressor(combined_feat).squeeze(-1)  # (B,)

        return out

def load_translation(essay_path: Path, question_path: Path):
    """Load essay data and map subjects to questions."""
    essay = json.load(open(essay_path))
    question = json.load(open(question_path))
    subject2question = question["subject"]
    for item in essay:
        item["question"] = subject2question[item["subject"]]
    return essay

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
        "--test_data",
        type=Path,
        default="./data/I_test.json",
        help="Path to the test data JSON file (used in test mode)",
    )
    parser.add_argument(
        "--question_data",
        type=Path,
        default="./data/reference-answers/question_prompts.json",
        help="Path to the question data JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to save model checkpoints (training) or predictions (testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode instead of training mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the trained model checkpoint (required in test mode)",
    )
    args = parser.parse_args()

    if args.test and not args.checkpoint:
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:  # FIX: ensure pad token
        tokenizer.pad_token = tokenizer.eos_token

    if args.test:
        # (Kept for completeness; minimal edits to avoid undefined symbols)
        logging.info("Running in test mode")
        essays = load_translation(args.test_data, args.question_data)

        data_dicts = []
        for i, c in enumerate(essays):
            datum = {
                "requirement": c["question"],  # source text
                "input": c["content"],         # translation text
                "level": c["level"],
                "score": c.get("score", 0.0),
                "id": c.get("id", i),
            }

            num_labeled_sentences = sum([1 if "missing translation" in item else 0 for item in c["pseudo_labels"]])
            if num_labeled_sentences == 0:
                continue  # Skip essays with no pseudo labels
            datum["missing_translation"] = sum([item.get("missing translation", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["over-translation"] = sum([item.get("over-translation", 0.0)for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["mistranslation"] = sum([item.get("mistranslation", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["grammar_errors"] = sum([item.get("grammar errors", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["spelling_errors"] = sum([item.get("spelling errors", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences

            data_dicts.append(datum)

        test_data = TranslationDataset(data_dicts, tokenizer)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        model = MistralWithRegression(
            model_name=args.model,
            no_quantize=args.no_quantize,
            pseudo_labels_dim=5
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
                pseudo_labels_score = batch["pseudo_labels_score"].cuda()
                target = batch["score"].cuda()
                doc_ids = batch["document_id"]

                pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err, pseudo_labels_score=pseudo_labels_score)

                if torch.any(target != 0.0):
                    loss = criterion(pred, target)
                    total_loss += loss.item()
                    avg_loss = total_loss / (pbar.n + 1)
                    pbar.set_postfix({"test_loss": f"{avg_loss:.4f}"})

                for doc_id, pred_score, true_score in zip(doc_ids, pred.cpu().numpy(), target.cpu().numpy()):
                    predictions.append({
                        "document_id": str(doc_id),
                        "predicted_score": float(pred_score),
                        "true_score": float(true_score)
                    })

        if total_loss > 0:
            avg_mse = total_loss / len(test_loader)
            logging.info(f"Test MSE: {avg_mse:.4f}")
        else:
            logging.info("No ground truth scores found in test data; MSE not computed")

        os.makedirs(args.output_dir, exist_ok=True)
        predictions_file = os.path.join(args.output_dir, "predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            try:
                json.dump(predictions, f, indent=2)
            except:
                print(len(predictions))
                print(predictions[0])
                print(type(predictions[0]), type(predictions[0]['predicted_score']))
        logging.info(f"Saved predictions to {predictions_file}")

        metrics_file = os.path.join(args.output_dir, "test_metrics.txt")
        with open(metrics_file, "w", encoding="utf-8") as f:
            if total_loss > 0:
                f.write(f"Test MSE: {avg_mse:.4f}\n")
            f.write(f"Number of predictions: {len(predictions)}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
        logging.info(f"Saved metrics to {metrics_file}")

    else:
        # Training mode
        logging.info("Running in training mode")
        wandb.init(
            project="gept-analysis",
            name=args.output_dir.name,
            config=args,
            dir=args.output_dir,
        )

        essays = load_translation(args.essay_data, args.question_data)

        data_dicts = []
        for i, c in enumerate(essays):
            datum = {
                "requirement": c["question"],  # source text
                "input": c["content"],         # translation text
                "level": c["level"],
                "score": c["score"],
                "id": c.get("id", i),
            }

            num_labeled_sentences = sum([1 if "missing translation" in item else 0 for item in c["pseudo_labels"]])
            if num_labeled_sentences == 0:
                continue  # Skip essays with no pseudo labels
            datum["missing_translation"] = sum([item.get("missing translation", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["over-translation"] = sum([item.get("over-translation", 0.0)for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["mistranslation"] = sum([item.get("mistranslation", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["grammar_errors"] = sum([item.get("grammar errors", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences
            datum["spelling_errors"] = sum([item.get("spelling errors", 0.0) for item in c["pseudo_labels"]])/num_labeled_sentences

            data_dicts.append(datum)

        train, eval = train_test_split(
            data_dicts, test_size=args.test_size, random_state=args.seed
        )
        data = {"train": train, "eval": eval}

        train_data = TranslationDataset(data["train"], tokenizer)
        eval_data = TranslationDataset(data["eval"], tokenizer)


        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

        model = MistralWithRegression(args.model, no_quantize=args.no_quantize, pseudo_labels_dim=5)
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
                pseudo_labels_score = batch["pseudo_labels_score"].cuda()
                target = batch["score"].cuda()
                
                # print(tokenizer.decode(input_ids_score[0]))
                # print('=================================')
                # print(tokenizer.decode(input_ids_err[0]))

                # exit()
                pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err, pseudo_labels_score=pseudo_labels_score)
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
                    pseudo_labels_score = batch["pseudo_labels_score"].cuda()
                    target = batch["score"].cuda()

                    pred = model(input_ids_score=input_ids_score, input_ids_err=input_ids_err, attention_mask_score=attention_mask_score, attention_mask_err=attention_mask_err, pseudo_labels_score=pseudo_labels_score)
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
import torch
from torch.utils.data import Dataset
from prompt.translation_prompt import build_translation_scoring_prompt, build_translation_error_prompt

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

        return {
            "input_ids_score": tokens_score["input_ids"].squeeze(0),
            "input_ids_err": tokens_err["input_ids"].squeeze(0),
            "attention_mask_score": tokens_score["attention_mask"].squeeze(0),
            "attention_mask_err": tokens_err["attention_mask"].squeeze(0),
            "score": torch.tensor(example["score"], dtype=torch.float),
            "document_id": example.get("document_id", idx),  # For tracking in test mode if needed
            "subject": example.get("subject", "unknown")  # For potential analysis by subject
        }

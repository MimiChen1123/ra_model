import torch
from torch.utils.data import Dataset

from utils import PROMPT_TEMPLATE

class EssayDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        """Initialize dataset for training or testing."""
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = PROMPT_TEMPLATE.format(
            requirement=example["requirement"], input=example["input"]
        )
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Build other_feature tensor: [cefr_mapped, word_count, relevance, coherence, organization]
        cefr_label = example.get("cefr_prediction")
        level = example.get("level")
        # Map CEFR label per level
        cefr_map_I = {"A1": 1, "A2": 2, "B1": 3.5, "B2": 4, "C1": 5, "C2": 5}
        cefr_map_HI = {"A1": 1, "A2": 2, "B1": 3,  "B2": 3.5, "C1": 4, "C2": 5}
        cefr_mapped = 0
        if isinstance(cefr_label, str):
            cefr_label = cefr_label.strip().upper()
            if level == "I":
                cefr_mapped = cefr_map_I.get(cefr_label, 0)
            elif level == "HI":
                cefr_mapped = cefr_map_HI.get(cefr_label, 0)
            else:
                cefr_mapped = cefr_map_HI.get(cefr_label, 0)
        
        # word_count (clamp to >=0)
        if level == "I":
            wc = example.get("word_count", -100)
        elif level == "HI":
            wc = example.get("word_count", -130)
        else:
            wc = example.get("word_count", -130)

        # rubric scores
        rel = int(example.get("RELEVANCE", 0) or 0)
        coh = int(example.get("COHERENCE", 0) or 0)
        org = int(example.get("ORGANIZATION", 0) or 0)
        other_feature = torch.tensor([float(cefr_mapped), float(wc), float(rel), float(coh), float(org)], dtype=torch.float)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "other_feature": other_feature,
            "score": torch.tensor(example.get("score") or 0.0, dtype=torch.float),
            "document_id": example.get("document_id", idx),
            "subject": example.get("subject", "")
        }


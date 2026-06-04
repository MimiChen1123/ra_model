import random
import numpy as np
import torch
import json
from pathlib import Path

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
        
def load_translation(essay_path: Path, question_path: Path):
    """Load essay data and map subjects to questions."""
    essay = json.load(open(essay_path))
    question = json.load(open(question_path))
    subject2question = question["subject"]
    for item in essay:
        item["question"] = subject2question[item["subject"]]
    return essay
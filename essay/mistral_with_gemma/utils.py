import torch
import numpy as np
import random
from pathlib import Path
import json

PROMPT_TEMPLATE = '''<|im_start|>user
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a professional English writing evaluator. Your task is to assess the quality of the given composition and the corresponding requirement, and then provide a score from 0.0 to 5.0, in increments of 0.5...

Below is the requirement wrapped in triple backticks:
"""
{requirement}
"""

Provide with a floating point score of the composition only in your response.

### Input:
{input}

### Response:<|im_end|>
<|im_start|>assistant
'''

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
        
def load_essays(essay_path: Path, question_path: Path):
    """Load essay data and map subjects to questions."""
    essay = json.load(open(essay_path))
    question = json.load(open(question_path))
    subject2question = question["subject"]

    for item in essay:
        item["question"] = subject2question[item["subject"]]
    
    return essay
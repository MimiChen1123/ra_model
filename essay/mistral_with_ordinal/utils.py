import random
import numpy as np
import torch

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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def decode_scores_from_logits(logits, score_step=0.5):
    probs = torch.sigmoid(logits)
    pred_class = (probs > 0.5).sum(dim=1)
    pred_score = pred_class.float() * score_step
    return pred_class, pred_score
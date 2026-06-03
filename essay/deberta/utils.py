import random
import numpy as np
import torch
import nltk

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def score_to_class(score, min_score, score_step, num_classes):
    idx = int(round((float(score) - min_score) / score_step))
    return max(0, min(num_classes - 1, idx))


def class_to_score(class_idx, min_score, score_step):
    return float(class_idx) * score_step + min_score


def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)
            

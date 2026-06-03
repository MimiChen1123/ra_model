import json
import spacy
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
parser.add_argument("--output_file", type=str, help="Path to the output JSON file")
parser.add_argument("--level", type=str, choices=["I", "HI"], help="Level of the task")
args = parser.parse_args()

LENGTH_THRESHOLD = {"I": 100, "HI": 130}    # {level: length_threshold}

nlp = spacy.load("en_core_web_sm")


with open(args.input_file, "r") as f:
    data = json.load(f)

for item in tqdm(data):
    doc = nlp(item["content"])
    word_count = sum(1 for token in doc if token.is_alpha)
    item["word_count"] = word_count - LENGTH_THRESHOLD[item.get("level", args.level)]  # Add normalized word count as a new column
    

with open(args.output_file, "w") as f:
    json.dump(data, f, indent=4)
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gemma_input_file", type=str, default="/home/jammy910316/gept_writing_eval/llm_output/cefr/sample/gemma-3-12b.jsonl")
parser.add_argument("--mistral_input_file", type=str, default="/home/jammy910316/gept_writing_eval/llm_output/cefr/sample/mistral-7b.jsonl")
parser.add_argument("--input_file", type=str, default="/home/jammy910316/gept_writing_eval/data/sampled_HI_test.json")
parser.add_argument("--output_file", type=str, default="/home/jammy910316/gept_writing_eval/data/cefr_extraction/cefr_sample_HI_test.json")
args = parser.parse_args()

CEFR_LEVELS = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
GEPT_LEVELS = {"I":3, "HI":4}
PASS_THRESHOLD = 4.0

with open(args.input_file, "r") as f:
    data = json.load(f)
    
with open(args.gemma_input_file, "r") as f:
    gemma_output = {}
    for line in f.readlines():
        obj = json.loads(line)
        gemma_output[f"{obj["level"]}_{obj["document_id"]}"] = obj    

with open(args.mistral_input_file, "r") as f:
    mistral_output = {}
    for line in f.readlines():
        obj = json.loads(line)
        mistral_output[f"{obj["level"]}_{obj["document_id"]}"] = obj


no_result_ctr = 0
gemma_correct = 0
mistral_correct = 0
gemma_incorrect = 0
mistral_incorrect = 0

for item in tqdm(data):
    doc_id = item["document_id"]
    level = item["level"]
    score = item["score"]
    ground_truth = GEPT_LEVELS[level]
    pass_or_not = True if score >= PASS_THRESHOLD else False

    try:
        gemma_cefr = gemma_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
        gemma_cefr = CEFR_LEVELS[gemma_cefr]
    except:
        gemma_cefr = None
    
    try:
        mistral_cefr = mistral_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
        mistral_cefr = CEFR_LEVELS[mistral_cefr]
    except:
        mistral_cefr = None
    
    if gemma_cefr and ((gemma_cefr >= ground_truth and pass_or_not) or (gemma_cefr < ground_truth and not pass_or_not)):
        item["cefr_prediction"] = gemma_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
        gemma_correct += 1
    elif mistral_cefr and ((mistral_cefr >= ground_truth and pass_or_not) or (mistral_cefr < ground_truth and not pass_or_not)):
        item["cefr_prediction"] = mistral_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
        mistral_correct += 1
    else:
        if gemma_cefr:
            item["cefr_prediction"] = gemma_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
            gemma_incorrect += 1
        elif mistral_cefr:
            item["cefr_prediction"] = mistral_output[f"{level}_{doc_id}"]["cefr_prediction"]["prediction"]
            mistral_incorrect += 1
        else:            
            item["cefr_prediction"] = None
            no_result_ctr += 1
            
print(f"Gemma correct: {gemma_correct}")
print(f"Mistral correct: {mistral_correct}")
print(f"Gemma incorrect: {gemma_incorrect}")
print(f"Mistral incorrect: {mistral_incorrect}")
print(f"No result samples: {no_result_ctr}")

with open(args.output_file, "w") as f:
    json.dump(data, f, indent=4)
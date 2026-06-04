# Translation Scoring Pipeline

This folder contains the translation scoring pipeline for training and real-world
inference.

There are two main shell pipelines:

1. **Training**: `train.sh` generates alignment and pseudo-label features, merges
   them back into the labeled training JSON, then trains `train.py`.
2. **Real-world inference**: `inference.sh` generates alignment and pseudo-label
   features, merges them back into the input JSON, then runs the trained model
   with `inference.py`.

## Environment Setup

Install dependencies in the project environment, then activate it:

```bash
cd handover_jammy/translation
conda activate gept_writing_eval
```

The LLM preprocessing steps require DeepInfra:

```bash
export DEEPINFRA_API_KEY="your_deepinfra_key"
```

Model loading/training requires a Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Required Input Format

Raw translation data should be a JSON list. Each item should contain:

```text
subject
id
content
level
```

Training data additionally requires:

```text
score
```

The preprocessing pipeline adds:

```text
alignment
pseudo_labels
```

The final training/inference format is like:

```json
{
  "subject": "HW-1701",
  "id": "example-id",
  "content": "English translation text...",
  "level": "HI",
  "score": 3.5,
  "alignment": [["中文句子", "English sentence"]],
  "pseudo_labels": [
    {
      "missing translation": 0,
      "over-translation": 0,
      "mistranslation": 1,
      "grammar errors": 1,
      "spelling errors": 0,
      "explanation": "..."
    }
  ]
}
```

For real-world inference, omit `score`.

## Training Pipeline

Run the full training pipeline:

```bash
bash train.sh
```

Equivalent explicit settings:

```bash
LEVEL=HI \
TRAIN_INPUT=./data/HI_train.json \
QUESTION_DATA=./data/reference-answers/question_prompts.json \
OUTPUT_DIR=./outputs/HI_translation \
bash train.sh
```

`train.sh` runs:

```text
alignment_inference.py
-> pseudo_label_inference.py
-> data/merge_pseudo_labels.py
-> train.py
```

Useful variables:

```text
LEVEL          I or HI
MODEL          base Hugging Face model
LLM_MODEL      DeepInfra model for alignment and pseudo-labeling
EPOCHS         training epochs
BATCH_SIZE     training batch size
LR             learning rate
TRAIN_INPUT    labeled input JSON
QUESTION_DATA  subject-to-question mapping JSON
WORK_DIR       preprocessing output folder
OUTPUT_DIR     trained checkpoint output folder
```

Intermediate files are written under:

```text
./output/<LEVEL>_translation_training_preprocess/
```

`train.py` saves checkpoints such as:

```text
<OUTPUT_DIR>/model_epoch_1.pt
```

## Real-World Inference Pipeline

Run the full inference pipeline:

```bash
bash inference.sh
```

Equivalent explicit settings:

```bash
LEVEL=HI \
INPUT_DATA=./data/HI_lttc_test.json \
CHECKPOINT=./models/HI_translation.pt \
OUTPUT=./output/HI_translation_predictions.json \
bash inference.sh
```

`inference.sh` runs:

```text
alignment_inference.py
-> pseudo_label_inference.py
-> data/merge_pseudo_labels.py
-> inference.py
```

Useful variables:

```text
LEVEL          I or HI
MODEL          base Hugging Face model
LLM_MODEL      DeepInfra model for alignment and pseudo-labeling
INPUT_DATA     raw real-world input JSON
QUESTION_DATA  subject-to-question mapping JSON
CHECKPOINT     trained .pt checkpoint
WORK_DIR       preprocessing output folder
OUTPUT         final prediction JSON
```

The final output preserves the input fields and adds:

```text
predicted_score_raw
score_prediction
pseudo_label_features
```

## Manual Preprocessing Steps

If you want to run preprocessing manually:

```bash
python alignment_inference.py \
  --subject ./data/reference-answers/question_prompts.json \
  --input ./data/HI_lttc_test.json \
  --output ./output/HI_translation_preprocess/HI_alignment.jsonl \
  --model google/gemma-3-12b-it

python pseudo_label_inference.py \
  --input ./output/HI_translation_preprocess/HI_alignment.jsonl \
  --output ./output/HI_translation_preprocess/HI_pseudo_labels.jsonl \
  --model google/gemma-3-12b-it

python data/merge_pseudo_labels.py \
  --input ./data/HI_lttc_test.json \
  --pseudo_labels ./output/HI_translation_preprocess/HI_pseudo_labels.jsonl \
  --output ./output/HI_translation_preprocess/HI_merged.json
```

Then run the trained model:

```bash
python inference.py \
  --input ./output/HI_translation_preprocess/HI_merged.json \
  --output ./output/HI_translation_predictions.json \
  --question_data ./data/reference-answers/question_prompts.json \
  --checkpoint ./models/HI_translation.pt \
  --model mistralai/Mistral-7B-Instruct-v0.2
```

## Existing Prepared Data

Examples of already prepared files:

```text
data/HI_train.json
data/HI_test.json
data/HI_lttc_test.json
data/HI_lttc_test_with_scores.json
```

`*_with_scores.json` files include gold `score` and can be used for evaluation.
Files without `score` are for real-world inference.

## Troubleshooting

- Missing `DEEPINFRA_API_KEY`: alignment and pseudo-label inference will fail.
- Missing `HF_TOKEN`: model/tokenizer loading may fail.
- Repeated JSONL records: rerun through `train.sh` or `inference.sh`; both scripts truncate generated JSONL files before writing.
- Missing `alignment` or `pseudo_labels`: run `data/merge_pseudo_labels.py` after `pseudo_label_inference.py`.

# Essay Scoring Pipeline

This folder contains the essay scoring pipeline for training and real-world
inference.

There are two main shell pipelines:

1. **Training/evaluation**: `pycaret_training.sh` preprocesses labeled train/test
   JSON files, then trains PyCaret models.
2. **Real-world test/inference**: `pycaret_inference.sh` preprocesses data
   without gold `score`, then runs a trained PyCaret model.

## Environment Setup

Create and activate the conda environment:

```bash
cd handover_jammy/essay
conda env create -f environment.yml
conda activate gept_writing_eval
```

Set the DeepInfra API key for CEFR inference:

```bash
export DEEPINFRA_API_KEY="your_api_key"
```

`pycaret_training.py` and `pycaret_inference.py` require PyCaret. If PyCaret is
not installed in the active environment, install the project-compatible PyCaret
version before running those scripts.

## Required Input Columns

For real-world inference, the input JSON must be a list of records with:

```text
level
content
```

For PyCaret training or final PyCaret inference, the prepared data should contain:

```text
level
cefr_prediction
word_count
RELEVANCE
COHERENCE
ORGANIZATION
```

Optional metadata columns are preserved when present:

```text
document_id
source_id
subject
content
```

Training data additionally requires:

```text
score
```

## Training Pipeline

Use this workflow for labeled train/test data. The script runs:

```text
rubric_inference.py
-> cefr_label_inference.py with Gemma
-> cefr_label_inference.py with Mistral
-> cefr_label_extraction.py
-> length_feature_extraction.py
-> pycaret_training.py
```

Default command:

```bash
bash pycaret_training.sh
```

Equivalent explicit settings:

```bash
LEVEL=HI \
TASK_TYPE=regression \
TRAIN_INPUT=./data/HI_train.json \
TEST_INPUT=./data/HI_test.json \
QUESTION_PATH=./data/HI_train_question.json \
MODEL_OUTPUT_PATH=./model_full_data \
bash pycaret_training.sh
```

Useful variables:

```text
LEVEL              I or HI
TASK_TYPE          regression or classification
TRAIN_INPUT        labeled train JSON
TEST_INPUT         labeled test JSON
QUESTION_PATH      question/topic JSON
WORK_DIR           preprocessing output folder
MODEL_OUTPUT_PATH  trained model output folder
GEMMA_MODEL        Gemma model for CEFR inference
MISTRAL_MODEL      Mistral model for CEFR inference
```

The script writes intermediate files under:

```text
./output/<LEVEL>_training_preprocess/
```

and trained models under:

```text
<MODEL_OUTPUT_PATH>/<TASK_TYPE>/level_<LEVEL>/<model_name>.pkl
```

## Real-World Test / Inference

Use this workflow when the test data does not have a gold `score`. The script
runs:

```text
rubric_inference.py
-> cefr_label_infer_extract.py
-> length_feature_extraction.py
-> pycaret_inference.py
```

Default command:

```bash
bash pycaret_inference.sh
```

Equivalent explicit settings:

```bash
LEVEL=HI \
TASK_TYPE=classification \
INPUT_DATA=./data/HI_test.json \
QUESTION_PATH=./data/HI_train_question.json \
MODEL_PATH=./model/classification/level_HI/ada.pkl \
PREDICTION_OUTPUT=./output/HI_test_prediction.json \
bash pycaret_inference.sh
```

Useful variables:

```text
LEVEL              I or HI
TASK_TYPE          regression or classification
INPUT_DATA         real-world input JSON
QUESTION_PATH      question/topic JSON
WORK_DIR           preprocessing output folder
GEMMA_MODEL        Gemma model for CEFR inference
MODEL_PATH         trained PyCaret model path
PREDICTION_OUTPUT  final prediction output path
```

The script writes intermediate files under:

```text
./output/<LEVEL>_preprocess/
```

and the final prediction file to `PREDICTION_OUTPUT`.

## Manual Real-World Steps

If you want to run each step manually, use the same sequence as
`pycaret_inference.sh`.

### 1. Predict Rubric Scores

```bash
python rubric_inference.py \
  --input ./data/HI_test.json \
  --output ./output/HI_preprocess/HI_test_rubric.json \
  --question_path ./data/HI_train_question.json
```

This adds:

```text
RELEVANCE
COHERENCE
ORGANIZATION
```

### 2. Predict CEFR With Gemma

```bash
python cefr_label_infer_extract.py \
  --input ./output/HI_preprocess/HI_test_rubric.json \
  --output ./output/HI_preprocess/HI_test_rubric_cefr.json \
  --model google/gemma-3-12b-it
```

This adds a simple CEFR label:

```json
"cefr_prediction": "A2"
```

### 3. Add Normalized Word Count

Skip this step only if `word_count` already exists in the input file.

```bash
python length_feature_extraction.py \
  --input_file ./output/HI_preprocess/HI_test_rubric_cefr.json \
  --output_file ./output/HI_preprocess/HI_test_ready.json \
  --level HI
```

### 4. Run PyCaret Inference

```bash
python pycaret_inference.py \
  --input_data_path ./output/HI_preprocess/HI_test_ready.json \
  --model_path ./model/classification/level_HI/ada.pkl \
  --output_path ./output/HI_test_prediction.json \
  --type classification
```

## Manual Training Command

If your data is already fully prepared with `cefr_prediction`, `word_count`,
`RELEVANCE`, `COHERENCE`, `ORGANIZATION`, and `score`, you can call
`pycaret_training.py` directly:

```bash
python pycaret_training.py \
  --train_data_path ./data/HI_train.json \
  --test_data_path ./data/HI_test.json \
  --model_output_path ./model_full_data \
  --level HI \
  --type regression
```

Trained models are saved under:

```text
<model_output_path>/<type>/level_<I|HI>/<model_name>.pkl
```

Example:

```text
model_full_data/regression/level_HI/br.pkl
```

## Test Inference With Existing Models

If your test data is already fully prepared, use `pycaret_inference.py` directly:

```bash
python pycaret_inference.py \
  --input_data_path ./data/HI_test.json \
  --model_path ./model/classification/level_HI/ada.pkl \
  --output_path ./output/HI_test_prediction.json \
  --type classification
```

The output file contains the original fields, PyCaret prediction columns, and a
rounded `score_prediction` field.

## Legacy CEFR Extraction

`cefr_label_extraction.py` is for the older labeled-data workflow that combines
Gemma and Mistral outputs. It uses gold `score` to choose the CEFR label that
matches pass/fail behavior, so it should not be used for real-world test data.

For real-world inference, use:

```text
cefr_label_infer_extract.py
```

## Troubleshooting

- Missing API key: set `DEEPINFRA_API_KEY`.
- Missing spaCy model: install or load `en_core_web_sm`.
- Missing PyCaret: install PyCaret in the active conda environment.
- Missing `word_count`: run `length_feature_extraction.py`.
- Missing `cefr_prediction`: run `cefr_label_infer_extract.py`.

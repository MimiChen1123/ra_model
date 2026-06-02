#!/usr/bin/env bash
set -euo pipefail

python eval.py \
    --test_data data/HI_2641A.json \
    --question_data data/test_question.json \
    --output_dir output/HI_2641A \
    --checkpoint model/HI_essay.pt \
    --batch_size 1 \
    --min_score 0.0 \
    --score_step 0.5 \
    --num_classes 11

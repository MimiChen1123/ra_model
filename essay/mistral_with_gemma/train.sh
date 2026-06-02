#!/usr/bin/env bash
set -euo pipefail

python train.py \
    --essay_data data/I_train.json \
    --question_data data/I_train_question.json \
    --output_dir output/I_train \
    --batch_size 2 \
    --epochs 5 \
    --lr 2e-5

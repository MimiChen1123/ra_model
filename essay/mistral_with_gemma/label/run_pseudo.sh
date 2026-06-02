python run_pseudo.py \
    --input data/HI_2641A.json \
    --output data/HI_2641A_pseudo.jsonl \
    --question_path data/test_question.json \
    --level HI \
    --model google/gemma-3-12b-it

python run_pseudo.py \
    --merge \
    --input data/HI_2641A.json \
    --output data/HI_2641A_pseudo.jsonl \
    --save_path data/HI_2641A.json

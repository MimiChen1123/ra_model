"""
Interactive inference script for the hybrid RoBERTa + XGBoost model.

Please use english for the requirement and omit unnecessary instruction.
For example, the original requirement may be:

> 說明：請依下面所提供的文字提示寫一篇英文作文，長度約120字（8至12個句子）。作文可以是一個完整的段落，也可以分段。（評分重點包括內容、組織、文法、用字遣詞、標點符號、大小寫。）\n提示：台灣小學生近視 (nearsightedness) 的問題越來越嚴重。請寫一篇文章\n(1) 說明造成這個現象的可能原因； \n(2) 並提出可以有效預防近視的方法。

You can simply input:
> The problem of nearsightedness among elementary school students in Taiwan is getting worse. Please write an essay to:\n(1) explain the possible reasons for this phenomenon;\n(2) and propose effective ways to prevent nearsightedness.

"""

import json
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

import language_tool_python

from metric import build_handcrafted_features_single
from utils import (
    class_to_score,
    ensure_nltk_resources,
    score_to_class,
    set_seed,
)

def parse_args():
    parser = ArgumentParser(
        description="AES inference for hybrid DeBERTa + XGBoost."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to question.json. Enables batch mode when used with --answers.",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=None,
        help="Path to answer.json. Enables batch mode when used with --questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write batch prediction results as JSON.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        required=True,
        help="Path to the model config JSON.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        required=True,
        help="Path to the XGBoost model file.",
    )
    parser.add_argument(
        "--tfidf_vectorizer_path",
        type=Path,
        default=None,
        required=True,
        help="Path to the TF-IDF vectorizer pickle.",
    )
    args = parser.parse_args()
    return args


def load_text_encoder(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model


def clamp_max_length(model, max_length):
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is None:
        return max_length

    effective_max = model_max - 2
    if max_length > effective_max:
        print(
            f"[Warning] max_length={max_length} exceeds model limit ({effective_max}). "
            f"Clamping to {effective_max}."
        )
        return effective_max
    return max_length


def extract_roberta_embedding_single(text, tokenizer, model, max_length, device):
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        cls_emb = hidden[:, 0, :]

    return cls_emb.cpu().numpy().astype(np.float32)


def load_artifacts(config_path, model_path, tfidf_vectorizer_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    xgb_model = XGBRegressor()
    xgb_model.load_model(model_path)

    with open(tfidf_vectorizer_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    return config, xgb_model, tfidf_vectorizer


def predict_single(
    requirement,
    essay,
    xgb_model,
    tfidf_vectorizer,
    tokenizer,
    encoder,
    max_length,
    device,
    sbert_model,
    grammar_tool,
    min_score,
    score_step,
    num_classes,
    thresholds,
    no_relevance,
    use_cos_sim_rel,
    use_hybrid_rel,
    alpha,
):
    embedding = extract_roberta_embedding_single(
        essay, tokenizer, encoder, max_length=max_length, device=device
    )
    handcrafted = build_handcrafted_features_single(
        requirement=requirement,
        essay=essay,
        no_relevance=no_relevance,
        use_cos_sim_rel=use_cos_sim_rel,
        use_hybrid_rel=use_hybrid_rel,
        alpha=alpha,
        sbert_model=sbert_model,
        grammar_tool=grammar_tool,
        tfidf_vectorizer=tfidf_vectorizer,
    )

    x = np.concatenate([embedding, handcrafted], axis=1)
    expected_features = getattr(xgb_model, "n_features_in_", None)
    if expected_features is not None and x.shape[1] != expected_features:
        raise ValueError(
            "Feature dimension mismatch: "
            f"extracted {x.shape[1]} features, "
            f"but model expects {expected_features}. "
            "Check relevance flags and model config alignment."
        )

    raw_pred = float(xgb_model.predict(x)[0])
    scaled = (raw_pred - min_score) / score_step

    if thresholds is not None:
        pred_class = int(
            np.digitize(
                np.asarray([scaled], dtype=np.float32),
                bins=np.asarray(thresholds),
            )[0]
        )
    else:
        pred_class = score_to_class(raw_pred, min_score, score_step, num_classes)

    pred_score = class_to_score(pred_class, min_score, score_step)
    return {
        "predicted_score": pred_score,
        "predicted_class": pred_class,
        "raw_regressed_score": raw_pred,
    }


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subjects = data.get("subject")
    if not isinstance(subjects, dict):
        raise ValueError("question.json must contain a 'subject' object.")
    return subjects


def load_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("answer.json must be a list of objects or a single object.")


def run_batch(
    args,
    xgb_model,
    tfidf_vectorizer,
    tokenizer,
    encoder,
    max_length,
    device,
    sbert_model,
    grammar_tool,
    min_score,
    score_step,
    num_classes,
    thresholds,
    no_relevance,
    use_cos_sim_rel,
    use_hybrid_rel,
    alpha,
):
    questions = load_questions(args.questions)
    answers = load_answers(args.answers)
    results = []

    for index, answer in enumerate(answers, start=1):
        subject_id = answer.get("subject")
        essay = answer.get("content", "")

        if subject_id not in questions:
            raise KeyError(
                f"Answer item {index} references unknown subject: {subject_id}"
            )
        if not essay:
            raise ValueError(f"Answer item {index} has empty content.")

        prediction = predict_single(
            requirement=questions[subject_id],
            essay=essay,
            xgb_model=xgb_model,
            tfidf_vectorizer=tfidf_vectorizer,
            tokenizer=tokenizer,
            encoder=encoder,
            max_length=max_length,
            device=device,
            sbert_model=sbert_model,
            grammar_tool=grammar_tool,
            min_score=min_score,
            score_step=score_step,
            num_classes=num_classes,
            thresholds=thresholds,
            no_relevance=no_relevance,
            use_cos_sim_rel=use_cos_sim_rel,
            use_hybrid_rel=use_hybrid_rel,
            alpha=alpha,
        )
        results.append({**answer, **prediction})
        # print(
        #     f"[{index}/{len(answers)}] document_id={answer.get('document_id')} "
        #     f"subject={subject_id} "
        #     f"score={prediction['predicted_score']:.1f}"
        # )

    output_text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"Wrote batch results to {args.output}")
    else:
        print(output_text)


def main():
    ensure_nltk_resources()

    args = parse_args()
    set_seed(args.seed)

    roberta_model = "microsoft/deberta-v3-large"
    max_length = 1024
    min_score = 0.0
    score_step = 0.5
    num_classes = 11
    thresholds = None

    no_relevance = False
    use_cos_sim_rel = False
    use_hybrid_rel = True
    alpha = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, encoder = load_text_encoder(roberta_model, device)
    max_length = clamp_max_length(encoder, max_length)

    sbert_model = None
    if use_cos_sim_rel or use_hybrid_rel:
        sbert_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if device.type == "cuda" else "cpu",
        )

    grammar_tool = language_tool_python.LanguageTool("en-US")
    _config, xgb_model, tfidf_vectorizer = load_artifacts(
        args.config_path,
        args.model_path,
        args.tfidf_vectorizer_path,
    )

    try:
        if args.questions is None or args.answers is None:
            raise ValueError("--questions and --answers must be provided together.")
        run_batch(
            args=args,
            xgb_model=xgb_model,
            tfidf_vectorizer=tfidf_vectorizer,
            tokenizer=tokenizer,
            encoder=encoder,
            max_length=max_length,
            device=device,
            sbert_model=sbert_model,
            grammar_tool=grammar_tool,
            min_score=min_score,
            score_step=score_step,
            num_classes=num_classes,
            thresholds=thresholds,
            no_relevance=no_relevance,
            use_cos_sim_rel=use_cos_sim_rel,
            use_hybrid_rel=use_hybrid_rel,
            alpha=alpha,
        )
    
    finally:
        if grammar_tool is not None:
            try:
                grammar_tool.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

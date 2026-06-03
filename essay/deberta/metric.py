import re
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from language_tool_python import LanguageTool
from nltk import pos_tag, word_tokenize

def count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def compute_readability_features(words, sentences):
    n_words = max(1, len(words))
    n_sents = max(1, len(sentences))
    syllable_count = sum(count_syllables(w) for w in words)
    asl = n_words / n_sents
    asw = syllable_count / n_words
    flesch_reading_ease = 206.835 - 1.015 * asl - 84.6 * asw
    fk_grade = 0.39 * asl + 11.8 * asw - 15.59
    return flesch_reading_ease, fk_grade


def compute_prompt_relevance_single(requirement, essay, tfidf_vectorizer=None):
    try:
        vecs = tfidf_vectorizer.transform([requirement, essay])
        return float(cosine_similarity(vecs[0], vecs[1])[0, 0])
    except ValueError:
        print(
            "[Warning] TF-IDF vectorization failed. Returning relevance score of 0.0."
        )
        return 0.0


def compute_cos_sim_relevance_single(requirement, essay, sbert_model):
    with torch.no_grad():
        req_emb = sbert_model.encode([requirement], convert_to_tensor=True)
        essay_emb = sbert_model.encode([essay], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(req_emb, essay_emb)[0].item()
    return float(similarity)


def compute_hybrid_relevance_single(
    requirement, essay, sbert_model, alpha, tfidf_vectorizer=None
):
    tfidf_rel = compute_prompt_relevance_single(
        requirement, essay, tfidf_vectorizer=tfidf_vectorizer
    )
    sbert_rel = compute_cos_sim_relevance_single(requirement, essay, sbert_model)
    return float((1.0 - alpha) * tfidf_rel + alpha * sbert_rel)


def compute_cohesion_score(essay):
    sentences = sent_tokenize(essay)
    if len(sentences) <= 1:
        return 1.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
            sentences
        )
        sims = []
        for i in range(len(sentences) - 1):
            sims.append(cosine_similarity(vec[i], vec[i + 1])[0, 0])
        return float(np.mean(sims)) if sims else 0.0
    except ValueError:
        print(
            "[Warning] Cohesion score computation failed. Returning cohesion score of 0.0."
        )
        return 0.0


def compute_grammar_errors(essay, grammar_tool=None):
    try:
        return float(len(grammar_tool.check(essay)))
    except Exception:
        print("[Warning] Grammar checking failed. Returning error count of 0.0.")
        return 0.0

def build_handcrafted_features_single(
    requirement,
    essay,
    no_relevance=False,
    use_cos_sim_rel=False,
    use_hybrid_rel=False,
    alpha=0.7,
    sbert_model=None,
    grammar_tool=None,
    tfidf_vectorizer=None,
):
    if not tfidf_vectorizer and (
        not no_relevance and not use_cos_sim_rel and not use_hybrid_rel
    ):
        raise ValueError(
            "TF-IDF vectorizer is required for relevance feature computation."
        )

    words = [w for w in word_tokenize(essay) if re.match(r"^[A-Za-z']+$", w)]
    sentences = sent_tokenize(essay)
    pos = pos_tag(words)

    n_words = len(words)
    n_sents = max(1, len(sentences))
    sent_lengths = [len(word_tokenize(s)) for s in sentences] if sentences else [0]
    unique_words = len(set(w.lower() for w in words))

    noun = sum(1 for _, p in pos if p.startswith("NN"))
    verb = sum(1 for _, p in pos if p.startswith("VB"))
    adj = sum(1 for _, p in pos if p.startswith("JJ"))
    adv = sum(1 for _, p in pos if p.startswith("RB"))

    flesch, fk_grade = compute_readability_features(words, sentences)
    grammar_errors = compute_grammar_errors(essay, grammar_tool=grammar_tool)
    cohesion = compute_cohesion_score(essay)

    if no_relevance:
        prompt_rel = 0.0
    elif use_cos_sim_rel:
        prompt_rel = compute_cos_sim_relevance_single(requirement, essay, sbert_model)
    elif use_hybrid_rel:
        prompt_rel = compute_hybrid_relevance_single(
            requirement, essay, sbert_model, alpha, tfidf_vectorizer=tfidf_vectorizer
        )
    else:
        prompt_rel = compute_prompt_relevance_single(
            requirement, essay, tfidf_vectorizer=tfidf_vectorizer
        )

    denom_words = max(1, n_words)
    features = [
        float(n_words),
        float(len(sentences)),
        float(np.mean(sent_lengths)),
        float(np.std(sent_lengths)),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        float(unique_words / denom_words),
        float(noun / denom_words),
        float(verb / denom_words),
        float(adj / denom_words),
        float(adv / denom_words),
        float((noun + verb + adj + adv) / denom_words),
        float(grammar_errors / n_sents),
        float(flesch),
        float(fk_grade),
        float(prompt_rel),
        float(cohesion),
    ]

    if no_relevance:
        features = features[:-2] + features[-1:]

    return np.asarray([features], dtype=np.float32)


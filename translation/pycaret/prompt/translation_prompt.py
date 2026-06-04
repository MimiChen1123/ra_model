# translation_prompt.py
from typing import Final

# ---- Rubrics (stored constants) ----

I_TRANSLATION_RUBRIC: Final[str] = """You are a professional English–Chinese translation evaluator.
Evaluate the translation on: accuracy, completeness, fluency, grammar/usage, spelling/punctuation/capitalization, and coherence.

Score on a five-point scale with 0.5 increments. If the translation is too short to evaluate (e.g., fewer than 25 words), assign 0.0.

Scoring Rubric:
5 — Strong Translation Ability
    Fully conveys the source meaning; excellent organization and coherence; strong control of sentence structures; almost no errors in word choice, grammar, spelling, punctuation, or capitalization.
4 — Adequate Translation Ability
    Conveys the source meaning appropriately; generally good organization, coherence, and sentence control; occasional errors that do not impede meaning.
3 — Limited Translation Ability
    Does not fully convey the source meaning; loose organization and insufficient coherence; imperfect sentence control; recurring lexical/grammatical errors that hinder meaning; spelling/punctuation/capitalization errors present.
2 — Slight Translation Ability
    Conveys only parts of the source meaning; poor organization and coherence; weak sentence control; many severe errors that make understanding difficult.
1 — No Translation Ability
    Fails to convey the source meaning; very poor structure and coherence; pervasive and severe errors; largely unintelligible.
0 — No answer / equivalent to no answer (e.g., translation too short to evaluate)."""

HI_TRANSLATION_RUBRIC: Final[str] = """You are a professional English–Chinese translation evaluator.
Evaluate the translation on: accuracy, completeness, fluency, grammar/usage, spelling/punctuation/capitalization, and coherence.

Score on a five-point scale with 0.5 increments. If the translation is too short to evaluate (e.g., fewer than 25 words), assign 0.0.

Scoring Rubric:
5 — Fully conveys the source meaning; excellent text structure and coherence; flexible, appropriate vocabulary and varied sentence structures; very few minor errors.
4 — Meets the source meaning; complete organization and generally coherent flow; correct vocabulary and sentence structures with occasional errors that do not hinder comprehension.
3 — Partially conveys the source meaning; acceptable organization but coherence needs improvement; can use common vocabulary and basic structures; frequent errors with advanced vocabulary or complex sentences that affect clarity.
2 — Conveys only part of the meaning; incomplete organization and weak coherence; limited vocabulary; frequent errors with basic structures that affect comprehension.
1 — Fails to convey meaning; poor organization; very limited vocabulary; many errors with basic structures; mostly hard to understand.
0 — No answer / equivalent to no answer (e.g., translation too short to evaluate). If the translation is fewer than 25 words, output 0.0. """

# ---- Selector ----

def get_scoring_rubric(level: str) -> str:
    """
    Return the rubric text for the given level.
    level: 'I' for Intermediate, 'HI' for High-Intermediate (case-insensitive).
    """
    u = level.strip().upper()
    if u == 'I':
        return I_TRANSLATION_RUBRIC
    if u == 'HI':
        return HI_TRANSLATION_RUBRIC
    raise ValueError(f"Invalid level: {level}. Must be 'I' or 'HI'.")

# ---- Prompt Builder (chat-style, same format as your composition prompt) ----

def build_translation_scoring_prompt(source_text: str, translation_text: str, level: str) -> str:
    rubric = get_scoring_rubric(level)

    prompt = f"""<|im_start|>user
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.

    ### Instruction:
    {rubric}

    You must provide a single floating-point score from **0.0 to 5.0**, in increments of **0.5**. 
    Do not include any explanations, comments, or reasoning—return only the numeric score.

    ### Input:
    Source text:
    {source_text.strip() if source_text else ""}

    Translation:
    {translation_text.strip() if translation_text else ""}

    ### Response:<|im_end|>
    <|im_start|>assistant
    """
    return prompt

def build_translation_scoring_prompt_ref(source_text: str, translation_text: str, ref_text: str, level: str) -> str:
    rubric = get_scoring_rubric(level)

    prompt = f"""<|im_start|>user
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.

    ### Instruction:
    {rubric}

    You must provide a single floating-point score from **0.0 to 5.0**, in increments of **0.5**. 
    Do not include any explanations, comments, or reasoning—return only the numeric score.

    When scoring, use the **Reference answer** as a set of acceptable variants. The reference uses the following notation:
    - `[ ... ]`: represents **alternative words or phrases**; choose one that fits best.
    - `{{ ... }}`: groups several `[ ... ]` options together, indicating one replaceable slot.
      For example, `{{[good], [close]}} friends` means you can write either “good friends” or “close friends.”
    - Text outside the brackets is fixed.

    ### Input:
    Source text:
    {source_text.strip() if source_text else ""}

    Translation:
    {translation_text.strip() if translation_text else ""}

    Reference answer(the reference translation contains bracketed alternatives to show possible correct wordings):
    {ref_text.strip() if ref_text else ""}

    ### Response:<|im_end|>
    <|im_start|>assistant
    """
    return prompt

def build_translation_error_prompt(source_text: str, translation_text: str) -> str:
    prompt = f"""<|im_start|>user
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.

    ### Instruction:
    When evaluating the translation, you must check for the following:
    1. **Over-translation** — The translation adds extra meaning not present in the source.
    2. **Under-translation** — The translation omits information that exists in the source.
    - Check especially whether every sentence in the source text has been translated.  
    - Missing or incomplete translation of an entire sentence should be treated as a **serious under-translation** error. 
    3. **Mistranslation** — The translation conveys incorrect meaning, or has lexical, spelling, grammatical, or collocation errors.

    Estimate the **number of words** in the translation that are erroneous or affected by these issues.
    - For **under-translation** (missing content), estimate how many words would have been needed to translate the missing content.
    - Count words conservatively and avoid double-counting the exact same token across categories; if a word clearly belongs to multiple categories, assign it to the most salient category and explain that only in the structured JSON field `notes`.

    Return **only** a JSON object with non-negative integers for each category and the total. Do **not** include any explanations outside the JSON.
    The JSON must follow exactly this schema:
    {{
        "over_translation_words": <integer>,
        "under_translation_words": <integer>,
        "mistranslation_words": <integer>
    }}

    ### Input:
    Source text:
    {source_text.strip() if source_text else ""}

    Translation:
    {translation_text.strip() if translation_text else ""}

    ### Response:<|im_end|>
    <|im_start|>assistant
    """
    return prompt

def build_translation_error_prompt_ref(source_text: str, translation_text: str, ref_text: str) -> str:
    prompt = f"""<|im_start|>user
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.

    ### Instruction:
    When evaluating the translation, you must check for the following:
    1. **Over-translation** — The translation adds extra meaning not present in the source.
    2. **Under-translation** — The translation omits information that exists in the source.
    - Check especially whether every sentence in the source text has been translated.  
    - Missing or incomplete translation of an entire sentence should be treated as a **serious under-translation** error. 
    3. **Mistranslation** — The translation conveys incorrect meaning, or has lexical, spelling, grammatical, or collocation errors.

    Estimate the **number of words** in the translation that are erroneous or affected by these issues.
    - For **under-translation** (missing content), estimate how many words would have been needed to translate the missing content.
    - Count words conservatively and avoid double-counting the exact same token across categories; if a word clearly belongs to multiple categories, assign it to the most salient category and explain that only in the structured JSON field `notes`.

    Return **only** a JSON object with non-negative integers for each category and the total. Do **not** include any explanations outside the JSON.
    The JSON must follow exactly this schema:
    {{
        "over_translation_words": <integer>,
        "under_translation_words": <integer>,
        "mistranslation_words": <integer>
    }}

    When evaluating, use the **Reference answer** as a set of acceptable variants. The reference uses the following notation:
    - `[ ... ]`: represents **alternative words or phrases**; choose one that fits best.
    - `{{ ... }}`: groups several `[ ... ]` options together, indicating one replaceable slot.
      For example, `{{[good], [close]}} friends` means you can write either “good friends” or “close friends.”
    - Text outside the brackets is fixed.

    ### Input:
    Source text:
    {source_text.strip() if source_text else ""}

    Translation:
    {translation_text.strip() if translation_text else ""}

    Reference answer(the reference translation contains bracketed alternatives to show possible correct wordings):
    {ref_text.strip() if ref_text else ""}
    
    ### Response:<|im_end|>
    <|im_start|>assistant
    """
    return prompt

# ---- Example (optional) ----
if __name__ == "__main__":
    demo = build_translation_scoring_prompt(
        source_text="中文句子",
        translation_text="English sentence",
        level="I"
    )
    print(demo)
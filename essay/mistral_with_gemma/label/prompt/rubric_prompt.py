from typing import Final

SYSTEM_INSTRUCTIONS: Final[str] = (
    "You are a strict writing evaluator. "
    "You must score the given essay ONLY according to the rubric below. "
    "Return a valid JSON object and nothing else."
)

I_DOCUMENT_RUBRIC: Final[str] = """Give a score of five-point-scale in json format based on the following scoring rubrics in terms of relevance to the topic, coherence, and organization.

5 point Good Writing Ability Content appropriately addresses the topic requirements, clear and well-organized; excellent organization; demonstrates flexible use of vocabulary and sentence patterns; occasional errors in grammar, spelling, or punctuation.

4 point Adequate Writing Ability Content meets the topic requirements and is generally clear; organization is mostly complete; demonstrates correct use of vocabulary and sentence patterns; although there are errors in grammar, spelling, or punctuation, they do not affect comprehension.

3 point Limited Writing Ability Content generally meets the topic requirements but does not fully convey the intended meaning; organization is acceptable; poor grasp of vocabulary and sentence patterns; relatively many errors in grammar, spelling, and punctuation that affect comprehension.

2 point Slight Writing Ability Content partially meets the topic requirements, mostly difficult to understand; poor organization; limited vocabulary and sentence patterns available for use; many errors in grammar, spelling, and punctuation.

1 point No Writing Ability Content fails to meet topic requirements and is incomprehensible; lacks organization; very limited vocabulary and sentence patterns available for use; excessive errors in grammar, spelling, and punctuation.

0 point No answer/Equivalent to no answer For example, 1) Article too short (less than 40 characters) cannot be scored; 2) Content completely incomprehensible; 3) Completely off-topic."""


HI_DOCUMENT_RUBRIC: Final[str] = """Give a score of five-point-scale in json format based on the following scoring rubrics in terms of relevance to the topic, coherence, and organization.

5 point Content appropriately addresses the topic requirements, with complete organization and coherent flow throughout; demonstrates flexible and appropriate use of vocabulary and various sentence structures, with very few errors.

4 point Content meets the topic requirements, with complete organization and generally coherent flow; demonstrates correct use of vocabulary and sentence structures, but with occasional errors.

3 point Content generally meets the topic requirements, with acceptable organization but coherence needs improvement; able to use common vocabulary and basic sentence structures, but frequently makes errors when using more difficult vocabulary or complex sentences.

2 point Content only partially meets the topic requirements, with incomplete organization and lack of coherence; limited vocabulary, frequent errors in using basic sentence structures that affect comprehension.

1 point Content fails to meet topic requirements, with poor organization; limited vocabulary, many errors in using basic sentence structures, mostly difficult to understand.

0 point No answer/Equivalent to no answer Equivalent to no answer: For example, 1) Article too short (less than 40 characters) cannot be scored; 2) Content completely incomprehensible; 3) Completely off-topic."""


def get_scoring_rubric(level: str) -> str:

    if level.upper() == 'I':
        return I_DOCUMENT_RUBRIC
    elif level.upper() == 'HI':
        return HI_DOCUMENT_RUBRIC
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'I' or 'HI'")

def build_rubric_prompt(subject_text: str, content: str, level: str) -> str:

    rubric = get_scoring_rubric(level)
    
    prompt = f"""{rubric}

[Topic]
{subject_text.strip() if subject_text else "N/A"}

[Text to Score]
{content.strip()}

Return the result **only in JSON format** as follows:
{{
  "RELEVANCE": <integer 0-5>,
  "COHERENCE": <integer 0-5>,
  "ORGANIZATION": <integer 0-5>
}}
"""
    
    return prompt
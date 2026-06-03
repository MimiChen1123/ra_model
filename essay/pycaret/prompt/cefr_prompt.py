from typing import Final


SYSTEM_INSTRUCTIONS: Final[str] = (
    "You are an expert English language assessor familiar with CEFR descriptors (A1–C2). "
    "Judge a given student essay by mapping observable evidence to CEFR descriptors such as range, accuracy, fluency, coherence, and interaction strategies (where relevant). "
    "Be conservative and avoid overestimating."
)


USER_PROMPT_TEMPLATE: Final[str] = (
    "Using CEFR descriptors, which level is the text most likely to belong to? "
    "Give your rationales and output your answers in json format. "
    "1. First, output your reasoning and analysis in valid JSON format, enclosed in triple backticks.\n"
    "2. After the JSON block, output one extra line in plain text: `CEFR level: <LEVEL>`\n"
    "   where <LEVEL> must be exactly one of [A1, A2, B1, B2, C1, C2].\n\n"
    "[Text]\n{content}\n\n"
    "[Example Output]\n"
    "```json\n"
    "{{...}}\n"
    "```\n"
    "CEFR level: B2"
)


def build_user_prompt(content: str) -> str:
    return USER_PROMPT_TEMPLATE.format(content=content.strip())



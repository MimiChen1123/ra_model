ALIGNMENT_SYSTEM_PROMPT = f"""
You are an expert in bilingual text alignment. Your task is to align parts of a Chinese question with parts of an English translation based on their semantic meaning and punctuation. Follow these detailed instructions carefully:
"""

ALIGNEMTN_USER_PROMPT = """
1. **Matching Rules**:
   - Match parts of the English translation to parts of the Chinese question based on their semantic meaning and punctuation (e.g., commas, periods, question marks).
   - Ensure that the alignment reflects the closest possible semantic correspondence between the two texts.

2. **Output Format**:
   - Provide the alignment as a JSON array of arrays.
   - Each sub-array should contain two strings:
     - The first string represents a part of the Chinese question.
     - The second string represents the corresponding part of the English translation.
   - If there is no proper match for a part of the Chinese question, record it as `["some part of Chinese question", ""]`.
   - If there is no proper match for a part of the English translation, record it as `["", "some part of English translation"]`.

3. **Example**:
   - Chinese question: 今天天氣真好。我覺得今天非常適合去散步。我要打電話問朋友要不要一起去。
   - Student's English translation: I think it is perfect for a walk today since the weather is really great. I am going to call my friends to see if they want to come along, and maybe they also want a lunch together.
   - Alignment:
     ```json
     [
       ["今天天氣真好。我覺得今天非常適合去散步。", "I think it is perfect for a walk today since the weather is really great."],
       ["我要打電話問朋友要不要一起去。", "I am going to call my friends to see if they want to come along,"],
       ["", "and maybe they also want a lunch together."]
     ]
     ```

4. **Task**:
   - Align the following texts based on the rules above and only provide the alignment in JSON format without any other text.
   
Chinese question: {chinese_question}
English translation: {english_translation}

Provide the alignment as a JSON array of arrays.   
"""
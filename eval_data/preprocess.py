import pandas as pd
import json

input_xlsx = "1150328中高一日CBT寫作作答_.xlsx"

df = pd.read_excel(input_xlsx)

required_columns = ["座位號碼", "寫作卷別", "翻譯", "作文"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"缺少欄位：{missing_columns}")

translation_data = []
essay_data = []

for idx, row in df.iterrows():
    document_id = idx + 1
    subject = row["寫作卷別"]

    if not pd.isna(row["翻譯"]):
        translation_data.append({
            "document_id": document_id,
            "subject": subject,
            "content": str(row["翻譯"]),
            "level": "HI"
        })

    if not pd.isna(row["作文"]):
        essay_data.append({
            "document_id": document_id,
            "subject": subject,
            "content": str(row["作文"]),
            "level": "HI"
        })

with open("translation.json", "w", encoding="utf-8") as f:
    json.dump(translation_data, f, ensure_ascii=False, indent=4)

with open("essay.json", "w", encoding="utf-8") as f:
    json.dump(essay_data, f, ensure_ascii=False, indent=4)
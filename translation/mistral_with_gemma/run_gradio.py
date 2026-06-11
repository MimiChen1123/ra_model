from __future__ import annotations

import json
import os
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from functools import lru_cache

from prompt.translation_prompt import build_translation_scoring_prompt, build_translation_error_prompt

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from prompt.alignment_prompt import ALIGNMENT_SYSTEM_PROMPT, ALIGNEMTN_USER_PROMPT
from alignment_inference import parse_alignment_from_response

from prompt.pseudo_label_prompt import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT
from pseudo_label_inference import parse_labels_from_response

# ========== Config ==========
load_dotenv()

CKPT_BY_LEVEL = {
    "I": "model/I_translation.pt",   
    "HI": "model/HI_translation.pt"
}

# ========== Helpers ==========
logging.basicConfig(level=logging.INFO)

# ===== Trained Model (LoRA regression head) =====
class MistralWithRegression(nn.Module):
    def __init__(self, model_name, no_quantize=False, HF_token=None, pseudo_labels_dim=5):
        """Initialize model with LoRA and regression head."""
        super().__init__()
        if not no_quantize:
            try:
                import bitsandbytes  # noqa: F401
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logging.warning("bitsandbytes not installed. Falling back to full precision.")
                bnb_config = None
        else:
            bnb_config = None

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_token
        )
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base) if bnb_config else base
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora_config)

        hidden_size = self.model.config.hidden_size  # e.g., 4096

        F = pseudo_labels_dim  # Assuming 5-dimensional pseudo labels
        self.feature_extractor_score= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.feature_extractor_err= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.pseudo_labels_processor = nn.Sequential(
            nn.Linear(F, 16),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids_score, input_ids_err, attention_mask_score, attention_mask_err, pseudo_labels_score):
        """Forward pass to predict score."""
        # Ensure input tensors are located on the same device as the LLM's input embeddings
        try:
            llm_device = self.model.model.get_input_embeddings().weight.device
        except Exception:
            llm_device = next(self.model.model.parameters()).device

        # === 推 score prompt ===
        input_ids_score = input_ids_score.to(llm_device)
        attention_mask_score = attention_mask_score.to(llm_device)
        outputs_score = self.model.model(
            input_ids=input_ids_score,
            attention_mask=attention_mask_score,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_score = outputs_score.hidden_states[-1]  # (B, T, H)
        cls_score = last_hidden_score[:, -1, :]  # 取最後 token 的表示 (B, H)

        # === 推 error prompt ===
        input_ids_err = input_ids_err.to(llm_device)
        attention_mask_err = attention_mask_err.to(llm_device)
        outputs_err = self.model.model(
            input_ids=input_ids_err,
            attention_mask=attention_mask_err,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_err = outputs_err.hidden_states[-1]
        cls_err = last_hidden_err[:, -1, :]

        head_device = cls_err.device

        # Move small custom heads to the LLM output device if they're not already there
        for module in (
            self.feature_extractor_score,
            self.feature_extractor_err,
            self.pseudo_labels_processor,
            self.regressor,
        ):
            # Only move if the module has parameters (avoid unnecessary ops)
            try:
                params = next(module.parameters())
            except StopIteration:
                params = None
            if params is not None and params.device != head_device:
                module.to(head_device)


        feat_score = self.feature_extractor_score(cls_score)  # (B, 512)
        feat_err = self.feature_extractor_err(cls_err)

        pseudo_labels_score = pseudo_labels_score.to(head_device)
        pseudo_labels_score = self.pseudo_labels_processor(pseudo_labels_score)  # (B, F)

        print(f"feat_score: {feat_score.shape}, feat_err: {feat_err.shape}, pseudo_labels_score: {pseudo_labels_score.shape}")
        combined_feat = torch.cat([feat_score, feat_err, pseudo_labels_score], dim=-1)  # (B, 1024+F)
        out = self.regressor(combined_feat).squeeze(-1)  # (B,)

        return out

@lru_cache(maxsize=2)
def _load_trained_components(model_name: str, checkpoint_path: str, no_quantize: bool) -> tuple[AutoTokenizer, MistralWithRegression]:
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = MistralWithRegression(model_name=model_name, no_quantize=no_quantize, HF_token=token, pseudo_labels_dim=5)

    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            print(f"[PEFT] ❌ checkpoint not found: {checkpoint_path}")
        else:
            print(f"[PEFT] 🔄 loading adapter checkpoint: {checkpoint_path}")
            state = torch.load(
                checkpoint_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[PEFT] ✅ loaded. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[PEFT] ⚠️ no checkpoint_path provided; using randomly-initialized regression head")

    model.eval()
    return tokenizer, model

def is_chinese(text: str) -> bool:
    """
    Check if the given text contains Chinese characters.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text contains Chinese characters, False otherwise.
    """
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Unicode range for Chinese characters
            return True
    return False

def make_client(api_key: Optional[str]) -> OpenAI:
    """
    Create an OpenAI client that talks to DeepInfra's OpenAI-compatible endpoint.
    """
    key = api_key or os.getenv("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("Please set --api_key or environment variable DEEPINFRA_API_KEY")
    return OpenAI(
        api_key=key,
        base_url="https://api.deepinfra.com/v1/openai"
    )

def chat_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=60,
            )
            # print("Here is the raw output:\n",resp)
            return resp.choices[0].message.content.strip()
        
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            time.sleep(2 ** attempt)
            if attempt == 3:
                return f"Reached maximum retries. Last error: {e}"

def run_alignment(
    api_key: Optional[str],
    model: str,
    content: str,
    question: str,
):
    client = make_client(api_key)
    
    if not content:
        alignment = None
        resp_text = ""
    else:
        user_prompt = ALIGNEMTN_USER_PROMPT.format(
            chinese_question=question,
            english_translation=content,
        )

        resp_text = chat_once(
            client=client,
            model=model,
            system_prompt=ALIGNMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
        )
        alignment = parse_alignment_from_response(resp_text)

    return alignment

def run_pseudo_labeling(
    api_key: Optional[str],
    model: str,
    alignments: Optional[dict],
):
    client = make_client(api_key)
    pseudo_labels_list = []
    for alignment in alignments:
        try:
            chinese_sent = alignment[0].strip()
            english_sent = alignment[1].strip()
        except IndexError:
            if len(alignment) == 1:
                if is_chinese(alignment[0].strip()):
                    chinese_sent = alignment[0].strip()
                    english_sent = ""
                else:
                    chinese_sent = ""
                    english_sent = alignment[0].strip()
            else:
                continue
        
        if not chinese_sent and english_sent:
            pseudo_labels = {
                "missing translation": 0,
                "over-translation": len(english_sent.split(" ")),
                "mistranslation": 0,
                "grammar errors": 0,
                "spelling errors": 0,
                "explanation": "Over-translation detected."
            }
            resp_text = "Over-translation detected."
        elif not english_sent and chinese_sent:
            pseudo_labels = {
                "missing translation": len(chinese_sent.split(" ")),
                "over-translation": 0,
                "mistranslation": 0,
                "grammar errors": 0,
                "spelling errors": 0,
                "explanation": "Missing translation detected."
            }
            resp_text = "Missing translation detected."
        else:
            user_prompt = EVALUATION_USER_PROMPT.format(
                chinese_sentence=chinese_sent,
                english_sentence=english_sent,
            )

            resp_text = chat_once(
                client=client,
                model=model,
                system_prompt=EVALUATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=2048,
            )
            pseudo_labels = parse_labels_from_response(resp_text)
        
        pseudo_labels_list.append(pseudo_labels)
        
    return pseudo_labels_list
        
def score_with_trained_model(
    level: str,
    requirement: str,
    content: str,
    model_name: str,
    checkpoint_path: str,
    no_quantize: bool = False,
) -> float:
    """Single-example inference using the user's trained model."""
    tokenizer, model = _load_trained_components(model_name, checkpoint_path or "", no_quantize)
    
    deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
    alignments = None
    pseudo_labels_list = []
    retry = 10
    
    while not isinstance(alignments, list) and retry > 0:
        alignments = run_alignment(
            api_key=deepinfra_api_key,
            model="google/gemma-3-12b-it",
            content=content,
            question=requirement
        )
        retry -= 1
        time.sleep(1)
        print(f"alignments result info: {alignments}")

    while len(pseudo_labels_list) == 0 and retry > 0:
        pseudo_labels_list = run_pseudo_labeling(
            api_key=deepinfra_api_key,
            model="google/gemma-3-12b-it",
            alignments=alignments
        )
        retry -= 1
        time.sleep(1)
        print(f"pseudo labeling result info: {pseudo_labels_list}")
    
    assert isinstance(alignments, list), "alignments should be a list of tuples, LLM alignment failed. Please retry again."
    assert len(pseudo_labels_list) != 0, "pseudo label list should not be empty, LLM pseudo-labeling failed. Please retry again."
    
    pseudo_labels = {}
    num_labeled_sentences = sum([1 if "missing translation" in item else 0 for item in pseudo_labels_list])
    pseudo_labels["missing_translation"] = sum([item.get("missing translation", 0.0) for item in pseudo_labels_list])/num_labeled_sentences
    pseudo_labels["over-translation"] = sum([item.get("over-translation", 0.0)for item in pseudo_labels_list])/num_labeled_sentences
    pseudo_labels["mistranslation"] = sum([item.get("mistranslation", 0.0) for item in pseudo_labels_list])/num_labeled_sentences
    pseudo_labels["grammar_errors"] = sum([item.get("grammar errors", 0.0) for item in pseudo_labels_list])/num_labeled_sentences
    pseudo_labels["spelling_errors"] = sum([item.get("spelling errors", 0.0) for item in pseudo_labels_list])/num_labeled_sentences
    
    prompt_score = build_translation_scoring_prompt(
        source_text=requirement,
        translation_text=content,
        level=level
    )

    prompt_err = build_translation_error_prompt(
        source_text=requirement,
        translation_text=content,
    )
    
    max_length = 2048
    
    tokens_score = tokenizer(
        prompt_score,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    ) 

    tokens_err = tokenizer(
        prompt_err,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    missing_translation = pseudo_labels.get("missing_translation", 0.0)
    over_translation = pseudo_labels.get("over-translation", 0.0)
    mistranslation = pseudo_labels.get("mistranslation", 0.0)
    grammar_errors = pseudo_labels.get("grammar_errors", 0.0)
    spelling_errors = pseudo_labels.get("spelling_errors", 0.0)

    pseudo_labels_score = torch.tensor([float(missing_translation), float(over_translation), float(mistranslation), float(grammar_errors), float(spelling_errors)], dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        pred = model(
            input_ids_score=tokens_score["input_ids"],
            input_ids_err=tokens_err["input_ids"],
            attention_mask_score=tokens_score["attention_mask"],
            attention_mask_err=tokens_err["attention_mask"],
            pseudo_labels_score=pseudo_labels_score
        )
        val = float(pred.squeeze(0).cpu().numpy().item())
        # clamp + round to nearest 0.5
        print(val)
        val = max(0.0, min(5.0, val))
        val = round(val * 2) / 2.0
        return val


# ========== Pipeline ==========
@dataclass
class PipelineOutput:
    requirement: str
    content: str
    level: str
    final_score: float

def run_pipeline(
    level: str,
    requirement: str,
    content: str,
    trained_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    no_quantize: bool = False,
):
    level = (level or "I").strip().upper()

    # Final score via user's trained model
    checkpoint_path = CKPT_BY_LEVEL.get(level, "")
    final_score = score_with_trained_model(
        level=level,
        requirement=requirement,
        content=content,
        model_name=trained_model_name,
        checkpoint_path=checkpoint_path,
        no_quantize=no_quantize,
    )

    result = PipelineOutput(
        requirement=requirement,
        content=content,
        level=level,
        final_score=final_score,
    )

    # For Gradio outputs: pretty JSON & flat dict
    pretty = json.dumps({
        "requirement": requirement,
        "content": content,
        "level": level,
        "final_score": result.final_score,
    }, ensure_ascii=False, indent=2)

    flat = {
        "Level": level,
        "Final Score": result.final_score,
    }

    return pretty, flat


# ========== Gradio UI ==========
with gr.Blocks(title="GEPT / Translation Scoring") as demo:
    gr.Markdown("# GEPT / Translation Scoring (Trained Model)")

    with gr.Row():
        level = gr.Radio(["I", "HI"], value="I", label="難度 (Level)")
        # Removing API Key input since we use local model

    requirement = gr.Textbox(label="題目 / Requirement (Source Text)", placeholder="請輸入中文題目...", lines=3)
    content = gr.Textbox(label="翻譯 / Translation", placeholder="請貼上英文翻譯...", lines=5)

    with gr.Row():
        run_btn = gr.Button("Run Scoring", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        pretty_json = gr.Code(label="Pipeline Output (JSON)")
        table = gr.JSON(label="Summary")

    def _go(level_v, req_v, cont_v):
        try:
            if not req_v or not cont_v:
                raise gr.Error("Please fill in both the 'Requirement' and 'Translation' fields.")
            return run_pipeline(level_v, req_v, cont_v)
        except AssertionError as e:
            raise gr.Error(str(e))
        except Exception as e:
            logging.exception("An unexpected error occurred in the pipeline.")
            raise gr.Error(f"An unexpected error occurred: {e}")

    run_btn.click(_go, inputs=[level, requirement, content], outputs=[pretty_json, table])

    def _clear():
        return "I", "", "", "", {}

    clear_btn.click(_clear, outputs=[level, requirement, content, pretty_json, table])

if __name__ == "__main__":
    demo.launch()
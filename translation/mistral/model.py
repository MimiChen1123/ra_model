import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv
import os
import logging

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it to your Hugging Face API token.")

class MistralWithRegression(nn.Module):
    def __init__(self, model_name, no_quantize=False):
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
            token=HF_TOKEN
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

        self.feature_extractor_score= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.feature_extractor_err= nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids_score, input_ids_err, attention_mask_score, attention_mask_err):
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
        combined_feat = torch.cat([feat_score, feat_err], dim=-1)  # (B, 1024)
        out = self.regressor(combined_feat).squeeze(-1)  # (B,)

        return out

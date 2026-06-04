import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/home/mimi911123/ra_model/.env")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please set it to your Hugging Face API token.")

class MistralWithRegression(nn.Module):
    def __init__(self, model_name, other_feature_dim=5, no_quantize=False):
        """Initialize model with LoRA and regression head."""
        super().__init__()
        if not no_quantize:
            try:
                import bitsandbytes
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
        
        hidden_size = self.model.config.hidden_size  # Typically 4096
        F = other_feature_dim  # Dimension of other_feature, default 5
        
        # Feature extractor for LLM: Project the last token's hidden state to a lower dimension
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
        )

        
        # Other feature processor: Project auxiliary features to a lower dimension
        self.other_feature_processor = nn.Sequential(
            nn.Linear(F, 32),
            nn.ReLU(),
        )

        # Combine features and regress to a single score
        self.combined_regressor = nn.Sequential(
            nn.Linear(512 + 32, 64),  # 512 (LLM) + 32 (other) = 544
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, other_feature):
        """Forward pass to predict score."""
        # Ensure input tensors are located on the same device as the LLM's input embeddings
        try:
            llm_device = self.model.model.get_input_embeddings().weight.device
        except Exception:
            # Fallback: use first parameter of the base model
            llm_device = next(self.model.model.parameters()).device

        input_ids = input_ids.to(llm_device)
        attention_mask = attention_mask.to(llm_device)

        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        cls_hidden = last_hidden[:, -1, :]  # Take the last token (B, H)
        # Ensure regression heads and auxiliary tensors are on the same device as the LLM outputs
        head_device = cls_hidden.device

        # Move small custom heads to the LLM output device if they're not already there
        for module in (
            self.feature_extractor,
            self.other_feature_processor,
            self.combined_regressor,
        ):
            # Only move if the module has parameters (avoid unnecessary ops)
            try:
                params = next(module.parameters())
            except StopIteration:
                params = None
            if params is not None and params.device != head_device:
                module.to(head_device)

        # Move auxiliary tensors to the head device
        other_feature = other_feature.to(head_device)

        # Process LLM features
        llm_features = self.feature_extractor(cls_hidden)  # (B, 512)

        # Process other auxiliary features
        other_features = self.other_feature_processor(other_feature)  # (B, 32)

        # Concatenate and regress
        combined = torch.cat([llm_features, other_features], dim=-1)  # (B, 544)
        score = self.combined_regressor(combined).squeeze(-1)  # (B,)

        return score


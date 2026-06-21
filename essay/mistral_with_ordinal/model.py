import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

class MistralWithOrdinalRegression(nn.Module):
    def __init__(self, model_name, num_classes=11):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=bnb_config
        )
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base = prepare_model_for_kbit_training(base)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora_config)
        self.ordinal_head = nn.Linear(self.model.config.hidden_size, num_classes - 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        cls_hidden = last_hidden[:, -1, :]
        return self.ordinal_head(cls_hidden)

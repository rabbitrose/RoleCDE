import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# =========================
# Paths / Config
# =========================
DATA_PATH = "xxxx"

MODEL_NAME = "xxxx"
OUTPUT_DIR = "xxxx"
MAX_SEQ_LEN = 2048

NUM_EPOCHS = 1
PER_DEVICE_BS = 2
GRAD_ACC = 8
LR = 5e-5  
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200


# =========================
# Utils
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    # -------- tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048
    # -------- dataset --------
    rows = load_jsonl(DATA_PATH)
    ds = Dataset.from_list(rows)

    def to_text(ex):
        ex["text"] = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return ex

    ds = ds.map(to_text, num_proc=8)

    # -------- 4bit quant config--------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_use_double_quant=True,
    )

    # -------- model --------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # close cache + open gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # kbit training 
    model = prepare_model_for_kbit_training(model)

    # -------- LoRA config --------
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ]

    lora_config = LoraConfig(
        r=16,                
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------- TRL SFT config --------
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        bf16=True,                 
        fp16=False,

        optim="paged_adamw_8bit",  
        lr_scheduler_type="cosine",

        # 训练更稳
        max_grad_norm=1.0,
        gradient_checkpointing=True,

        report_to="none",
    )

    # -------- trainer --------
    import inspect
    from trl import SFTTrainer

    trainer_kwargs = dict(
        model=model,
        train_dataset=ds,
        # dataset_text_field="text",
        args=cfg,
    )

    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    if "processing_class" in params:

        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "Your TRL SFTTrainer doesn't accept tokenizer/processing_class. "
            "Use the fallback tokenization+Trainer path."
        )

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()


    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[OK] Saved LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

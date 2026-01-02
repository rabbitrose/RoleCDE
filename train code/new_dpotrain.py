import json
import inspect
import importlib.util
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model



DATA_PATH = "xxxxx"
MODEL_NAME = "xxxx"
OUTPUT_DIR = "xxxx"

MAX_LEN = 2048
NUM_EPOCHS = 1

PER_DEVICE_BS = 1
GRAD_ACC = 16

LR = 1e-5
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
BETA = 0.1

def has_bitsandbytes() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    # --------- dataset ---------
    rows = load_jsonl(DATA_PATH)
    ds = Dataset.from_list(rows)
    for col in ("prompt", "chosen", "rejected"):
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column: {col}. Have: {ds.column_names}")

    # --------- tokenizer ---------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = MAX_LEN

    # --------- models ---------
    use_4bit = has_bitsandbytes()
    print(f"[Info] bitsandbytes found: {use_4bit}")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        policy = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    ref_model.config.use_cache = False
    ref_model.eval()

    # --------- LoRA on policy ---------
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()

    # --------- TRL DPOTrainer + DPOConfig ---------
    from trl import DPOTrainer, DPOConfig

    # 你的 TRL 版本期望 args 是 DPOConfig（有 model_init_kwargs）
    # 注意：不同版本字段名可能不同，但 DPOConfig 一般兼容面更好
    args = DPOConfig(
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
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="none",
    )

    # --------- build kwargs with signature introspection ---------
    sig = inspect.signature(DPOTrainer.__init__)
    params = sig.parameters

    trainer_kwargs = {}

    # required core
    if "model" in params:
        trainer_kwargs["model"] = policy
    if "ref_model" in params:
        trainer_kwargs["ref_model"] = ref_model

    # args
    if "args" in params:
        trainer_kwargs["args"] = args
    elif "training_args" in params:
        trainer_kwargs["training_args"] = args

    # tokenizer / processing_class
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tok

    # dataset
    if "train_dataset" in params:
        trainer_kwargs["train_dataset"] = ds
    elif "dataset" in params:
        trainer_kwargs["dataset"] = ds

    # beta
    if "beta" in params:
        trainer_kwargs["beta"] = BETA

    # column mapping (some TRL versions need these)
    if "prompt_column" in params:
        trainer_kwargs["prompt_column"] = "prompt"
    if "chosen_column" in params:
        trainer_kwargs["chosen_column"] = "chosen"
    if "rejected_column" in params:
        trainer_kwargs["rejected_column"] = "rejected"

    # lengths
    if "max_length" in params:
        trainer_kwargs["max_length"] = MAX_LEN
    if "max_prompt_length" in params:
        trainer_kwargs["max_prompt_length"] = min(1024, MAX_LEN // 2)

    print("[Info] DPOTrainer init kwargs keys:", list(trainer_kwargs.keys()))

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[OK] Saved LoRA adapter to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

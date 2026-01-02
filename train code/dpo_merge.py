import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


base_model_path = "xxxx"  
dpo_lora_model_path = ("xxxx")
dpo_output_path = ("xxxx")

os.makedirs(dpo_output_path, exist_ok=True)

print("ADD MODEL...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,     
    device_map="cpu",               
    trust_remote_code=True
)

print("ADD Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

print("ADD DPO LoRA...")
model = PeftModel.from_pretrained(
    model,
    dpo_lora_model_path,
    device_map="cpu"                
)

print(" (Merge)...")
merged_model = model.merge_and_unload()

print(f"SAVE  {dpo_output_path} ...")
merged_model.save_pretrained(dpo_output_path, safe_serialization=True)

print("SAVE Tokenizer ...")
tokenizer.save_pretrained(dpo_output_path)

print("✅ DPO LoRA FINISH！")
print(f"- base_model_path: {base_model_path}")
print(f"- dpo_lora_model_path: {dpo_lora_model_path}")
print(f"- merged_output_path: {dpo_output_path}")

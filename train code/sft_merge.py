import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 设置路径

DEFAULT_MODEL_NAME = "xxxx"
base_model_path="xxxx"
lora_model_path="xxxxx"
output_path="xxxxx"

print("load base model...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cpu",              
    trust_remote_code=True
)

print("load lora adapter...")
model = PeftModel.from_pretrained(model, lora_model_path)

print("merge model...")
merged_model = model.merge_and_unload()

print(f"save model {output_path}...")
merged_model.save_pretrained(output_path, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("finish")
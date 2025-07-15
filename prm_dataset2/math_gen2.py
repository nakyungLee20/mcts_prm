import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict, deque
import json
import wandb
import os
import random

# Project‑level helpers 
from config import PRMConfig
from mc_shaped_reward import MCRewardShaped
from prm_trainer import PRMTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/QwQ-32B"             # "Qwen/Qwen2.5-Math-7B-Instruct"  #"Qwen/Qwen2.5-Math-7B" "Qwen/QwQ-32B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,            # enable 4-bit (QLoRA-style) weights
        bnb_4bit_quant_type="nf4",    # NF4 gives the best accuracy for most LLMs
        bnb_4bit_use_double_quant=True, # optional: second quantisation pass to save ~0.4 bits/param
        bnb_4bit_compute_dtype=torch.bfloat16  # faster matmuls on recent GPUs; fall back to float16 if needed
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # let Accelerate split layers across all visible GPUs
        quantization_config=bnb_config,
        torch_dtype="auto",     # keeps non-linear layers in their original dtype
        trust_remote_code=True  # Qwen models need their custom code
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    cfg = PRMConfig()
    print("Finish load model and config!")

    mcrs = MCRewardShaped(config=cfg , model=model, tokenizer=tokenizer)
    math_raw = mcrs.math_mi_reward_dataset(split="train", start=0, take=1000)
    with open("/home/leena/ccc_eval/mcts_prm/samples/math_mi_1000.json", "w", encoding="utf-8") as f2:
        json.dump(math_raw, f2, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
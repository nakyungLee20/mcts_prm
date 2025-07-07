import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict, deque
import json
import wandb
import os
import random

# Project‑level helpers 
from config import PRMConfig
from mc_reward import MCReward
from prm_trainer import PRMTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def main():
    model_name =  "Qwen/Qwen2.5-Math-7B" # "Qwen/Qwen2.5-Math-7B-Instruct"  #"Qwen/Qwen2.5-Math-7B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = PRMConfig()
    print("Finish load model and config!")
    
    mcr = MCReward(config=cfg , model=model, tokenizer=tokenizer)
    gsm8k_raw= mcr.build_datasets_gsm8k(split="train", start=300, take=200)
    # gsm8k_val = mcr.build_datasets_gsm8k(split="test")
    with open("gsm8k_train_0704_500.json", "w", encoding="utf-8") as f:
        json.dump(gsm8k_raw, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
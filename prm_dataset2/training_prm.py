import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict, deque
import json
import wandb
import os
import random
from torch.utils.data import DataLoader
import wandb

# Project‑level helpers 
from config import PRMConfig
from mc_shaped_reward import MCRewardShaped
from prm_trainer import PRMTrainer
from prm_dataset import StepwisePRMDataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct" # PRM training용 작은 모델 사용 (dataset generation과 독립적)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    cfg = PRMConfig()
    print(f"Using model: {model_name} for PRM training")

    with open("/home/leena/ccc_eval/mcts_prm/samples/math_gsm8k_400.json", "r") as file:
        gsm8k_raw = json.load(file)
    
    random.shuffle(gsm8k_raw)
    split_idx       = int(0.9 * len(gsm8k_raw)) if len(gsm8k_raw) > 1 else 1
    gsm8k_train   = gsm8k_raw[:split_idx]
    gsm8k_val     = gsm8k_raw[split_idx:] or gsm8k_raw[:1]   # 최소 1개 확보

    train_ds = StepwisePRMDataset(gsm8k_train, tokenizer, cfg.max_new_tokens, reward_type="contri")
    val_ds   = StepwisePRMDataset(gsm8k_val, tokenizer, cfg.max_new_tokens, reward_type="contri")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,num_workers=cfg.num_workers, pin_memory=True,)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,num_workers=cfg.num_workers, pin_memory=True,)
    print("Train/Val size:", len(train_ds), len(val_ds))
    print("Finish Loadng PRM Datasets!")

    trainer = PRMTrainer(cfg, model=model, tokenizer=tokenizer)
    history = trainer.fit(train_loader, val_loader)
    print("PRM Training complete. Loss history:", history)


if __name__ == "__main__":
    main()
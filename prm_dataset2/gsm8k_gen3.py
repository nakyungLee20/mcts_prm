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
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def read_jsonl(file_path):
    """JSONL 파일을 읽어서 리스트로 반환"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jsonl_to_json(jsonl_path, json_path):
    """JSONL 파일을 일반 JSON 파일로 변환"""
    data = read_jsonl(jsonl_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Converted {jsonl_path} to {json_path}")


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

    output_file = "/home/leena/ccc_eval/mcts_prm/samples/gsm8k_2000.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in mcrs.gsm8k_reward_dataset_streaming(split="train", start=1000, take=1000):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # 즉시 디스크에 쓰기
    print(f"Data saved to {output_file}")

    jsonl_to_json(output_file, "/home/leena/ccc_eval/mcts_prm/samples/gsm8k_2000_converted.json")


if __name__ == "__main__":
    main()
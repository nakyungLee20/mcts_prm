import json
import time
import os
from tqdm import tqdm
import torch

# Project-level helpers
from config import PRMConfig
from mc_shaped_reward_vllm2 import MCRewardShapedVLLM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    cfg = PRMConfig()
    model_name = "Qwen/QwQ-32B"  # 또는 더 작은 모델 사용 가능
    mcrs = MCRewardShapedVLLM(config=cfg, model_name=model_name)

    output_file = "/home/leena/ccc_eval/mcts_prm/samples/math_1000_vllm2.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(mcrs.math_reward_dataset_vllm(split="train", start=1000, take=1000)):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush() 

    print(f"Data saved to {output_file}")
    jsonl_to_json(output_file, "/home/leena/ccc_eval/mcts_prm/samples/math_1000_vllm2_converted.json")

if __name__ == "__main__":
    main() 
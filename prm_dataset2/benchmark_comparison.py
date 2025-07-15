import time
import json
import os
from tqdm import tqdm

# Project-level helpers
from config import PRMConfig
from mc_shaped_reward import MCRewardShaped
from mc_shaped_reward_vllm import MCRewardShapedVLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def benchmark_original_method():
    """기존 방식의 성능 측정"""
    print("=== Benchmarking Original Method ===")
    
    # 모델 로딩
    start_time = time.time()
    model_name = "Qwen/QwQ-32B"
    
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
    
    load_time = time.time() - start_time
    print(f"Original model loaded in {load_time:.2f} seconds")
    
    # 설정
    cfg = PRMConfig()
    cfg.num_rollouts = 5  # 테스트용으로 줄임
    cfg.max_new_tokens = 64  # 테스트용으로 줄임
    
    mcrs = MCRewardShaped(config=cfg, model=model, tokenizer=tokenizer)
    
    # 테스트 데이터
    test_question = "Janet's dogs eat 2 pounds of food each day. How many pounds of food do her dogs eat in a week?"
    test_steps = [
        "Step 1: Janet's dogs eat 2 pounds of food each day.",
        "Step 2: There are 7 days in a week.",
        "Step 3: To find the total food eaten in a week, multiply 2 by 7.",
        "Step 4: 2 × 7 = 14"
    ]
    test_gold_answer = "14"
    
    # 성능 측정
    start_time = time.time()
    
    # 10번 반복하여 평균 측정
    for i in range(10):
        ori = mcrs.compute_step_rewards(test_question, "Solve step by step:", test_steps, test_gold_answer)
        ptb = mcrs.perturb_step_rewards(test_question, "Solve step by step:", test_steps, test_gold_answer, True)
        mi = mcrs.compute_step_mi(test_question, test_steps, test_gold_answer)
    
    total_time = time.time() - start_time
    avg_time = total_time / 10
    
    print(f"Original method average time per sample: {avg_time:.2f} seconds")
    return avg_time

def benchmark_vllm_method():
    """vLLM 방식의 성능 측정"""
    print("=== Benchmarking vLLM Method ===")
    
    # 모델 로딩
    start_time = time.time()
    model_name = "Qwen/QwQ-32B"
    
    cfg = PRMConfig()
    cfg.num_rollouts = 5  # 테스트용으로 줄임
    cfg.max_new_tokens = 64  # 테스트용으로 줄임
    
    mcrs = MCRewardShapedVLLM(config=cfg, model_name=model_name)
    
    load_time = time.time() - start_time
    print(f"vLLM model loaded in {load_time:.2f} seconds")
    
    # 테스트 데이터
    test_question = "Janet's dogs eat 2 pounds of food each day. How many pounds of food do her dogs eat in a week?"
    test_steps = [
        "Step 1: Janet's dogs eat 2 pounds of food each day.",
        "Step 2: There are 7 days in a week.",
        "Step 3: To find the total food eaten in a week, multiply 2 by 7.",
        "Step 4: 2 × 7 = 14"
    ]
    test_gold_answer = "14"
    
    # 성능 측정
    start_time = time.time()
    
    # 10번 반복하여 평균 측정
    for i in range(10):
        ori = mcrs.compute_step_rewards_batch(test_question, "Solve step by step:", test_steps, test_gold_answer)
        ptb = mcrs.perturb_step_rewards_batch(test_question, "Solve step by step:", test_steps, test_gold_answer, True)
        mi = mcrs.compute_step_mi_batch(test_question, test_steps, test_gold_answer)
    
    total_time = time.time() - start_time
    avg_time = total_time / 10
    
    print(f"vLLM method average time per sample: {avg_time:.2f} seconds")
    return avg_time

def main():
    print("=== Performance Comparison: Original vs vLLM ===")
    
    try:
        # 기존 방식 벤치마크
        original_time = benchmark_original_method()
        
        print("\n" + "="*50 + "\n")
        
        # vLLM 방식 벤치마크
        vllm_time = benchmark_vllm_method()
        
        print("\n" + "="*50)
        print("=== RESULTS ===")
        print(f"Original method: {original_time:.2f} seconds per sample")
        print(f"vLLM method: {vllm_time:.2f} seconds per sample")
        
        if vllm_time < original_time:
            speedup = original_time / vllm_time
            print(f"vLLM is {speedup:.2f}x faster!")
        else:
            slowdown = vllm_time / original_time
            print(f"vLLM is {slowdown:.2f}x slower (unexpected)")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        print("Make sure vLLM is properly installed: pip install vllm")

if __name__ == "__main__":
    main() 
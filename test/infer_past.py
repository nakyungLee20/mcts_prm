import json, random, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

# # ---------------- 사용자 정의 유틸 ------------------------------
from prm_trainer import ProcessRewardModel        # PRM head class
from config import PRMConfig
from mc_reward import MCReward
from utils import _sanitize, _numeric_equiv, _extract_answer  # 앞서 만든 함수

LLM_NAME      = "Qwen/Qwen2.5-Math-7B"
PRM_CKPT_PATH = "./checkpoints/gsm8k/ori_mse/best_prm.pt"
N_ROLLOUTS    = 5                         # Best-of-N
MAX_NEW_TOK   = PRMConfig.max_new_tokens
SEED          = PRMConfig.seed
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM    = PRMConfig.hidden_size 

SYSTEM_PROMPT_SAMPLE = (
        "Problem: (sample)\n"
        "Please provide a short and precise step-by-step solution, and a numerical answer in the end, for the question above in the following format, without any extra wording:\n"
        "Step 1: (logical step 1)\n"
        "Step 2: (logical step 2)\n"
        "...\n"
        "Step n: (logical last step)\n"
        "Answer: (Final result)"
        "Please strictly stick to the format above."
)

# Load 
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load base model
config = PRMConfig()
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
base  = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE).eval()
for p in base.parameters():
    p.requires_grad_(False)
mcr = MCReward(config=config, tokenizer=tokenizer, model=base)

# Load prm model
feat_dim = base.config.hidden_size
prm = ProcessRewardModel(feat_dim, cfg=config)
ckpt = torch.load(PRM_CKPT_PATH, map_location="cpu", weights_only=False)
prm.load_state_dict(ckpt["prm_state"])
prm.to(DEVICE).eval()

# Load Datasets
problems = []
gsm8k_test = load_dataset("openai/gsm8k", "main")["test"]
small_gsm8k = gsm8k_test.select(range(2,4))
for obj in small_gsm8k:
    problems.append({"q": obj["question"], "gold": obj["answer"]})
print("Finish Loading model and dataset!")

# Evaluation utils
def generate_solutions(
    backbone,
    tokenizer,
    question: str,
    n: int,
) -> List[str]:
    prompt = (
        f"{SYSTEM_PROMPT_SAMPLE}\n\n"
        f"Problem: {question}\n"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOK,
        do_sample=True, temperature=0.8, top_p=0.9,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = backbone.generate(input_ids.repeat(n, 1), **gen_cfg.to_dict())
    return [
        tokenizer.decode(seq[input_ids.shape[-1]:], skip_special_tokens=True)
        for seq in out
    ]

def parse_steps(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().lower().startswith("step")]

@torch.no_grad()
def prm_score(
    prm: ProcessRewardModel,
    backbone,
    tokenizer,
    question: str,
    steps: List[str],
) -> float:
    """
    각 prefix(Problem+Step1…i)에 대해 PRM 예측 → 평균 점수
    점수가 클수록 '좋은' reasoning.
    """
    prefix_lines = [f"Problem: {question}"]
    scores = []
    for step in steps:
        prefix_lines.append(step)
        txt = "\n".join(prefix_lines)
        ids = tokenizer(
            txt,
            max_length=384,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        feats = backbone(
            input_ids=ids,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1][:, 0, :]           # CLS
        # score = prm(feats).sigmoid().item()    # 0~1 확률
        score = prm(feats).item()
        scores.append(score)
    print("Scores by PRM:", scores, "\n")
    return sum(scores) / len(scores)

# Evaluation
correct = 0
total   = len(problems)
for idx, item in enumerate(problems, 1):
    q, gold = item["q"], _sanitize(item["gold"])
    # ① N개의 솔루션 생성
    sols = generate_solutions(base, tokenizer, q, N_ROLLOUTS)
    # ② PRM 스코어 계산
    scored: List[Tuple[float, str]] = []
    for sol in sols:
        steps = parse_steps(sol)
        if not steps:
            continue
        s = prm_score(prm, base, tokenizer, q, steps)
        scored.append((s, sol))

    if not scored:
        pred_answer = "N/A"
    else:
        best_sol = max(scored, key=lambda t: t[0])[1]
        pred_answer = mcr._extract_answer(text=best_sol) or "N/A"

    if _numeric_equiv(pred_answer, gold):
        correct += 1

    print(f"[{idx}/{total}] pred={pred_answer} | gold={gold} | {'✓' if _numeric_equiv(pred_answer, gold) else '✗'}")

accuracy = correct / total * 100
print(f"\n=== GSM8K Test Accuracy (Best-of-{N_ROLLOUTS} w/ PRM) : {accuracy:.2f}% ===")

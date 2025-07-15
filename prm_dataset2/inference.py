import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

from config import PRMConfig
from utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer
from mc_shaped_reward import MCRewardShaped
from prm_dataset import StepwisePRMDataset
from prm_model import ProcessRewardModel
from prm_trainer import PRMTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# utils for inference
ANSWER_PATTERN = re.compile(
    r"""^[\s>#*\-]*          # optional markdown/bullet symbols
        answer               # the word 'answer' (case‑insensitive)
        \s*[:.\-]\s*         # separator ( :, . or ‑ )
        (.+?)                # capture group
        (?:\s*[.!?])?\s*$   # optional trailing punctuation
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
# "Step 1:" or "step‑2‑" etc.
STEP_PATTERN = re.compile(
    r"(?i)step\s*\d+\s*[:\-]\s*(.*?)(?=\n\s*step\s*\d+|\n\s*answer\s*[:\-]|$)",
    re.S,
)

def build_prompt(question: str) -> str:
    return f"""<|im_start|>system
You are a helpful math tutor. 
<|im_end|>
<|im_start|>user
You are a helpful math tutor. You must solve problems step-by-step using the exact format:
Step 1: [first step]
Step 2: [second step]
...
Answer: [final answer]

Example:
Problem: What is 5 + 3?
Step 1: Add 5 and 3
Step 2: 5 + 3 = 8
Answer: 8

Now solve the given problem using the same format.
Problem: {question}
<|im_end|>
<|im_start|>assistant
"""

def parse_steps_and_answer(text: str) -> Tuple[List[str], str]:
    """Extract step list and answer string from a generated trajectory."""
    answer = ""
    ans_match = ANSWER_PATTERN.search(text)
    if ans_match:
        answer = ans_match.group(1).strip()
    else:
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
        if boxed_match:
            answer = boxed_match.group(1).strip()
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                last_line = lines[-1]
                if re.search(r'\d', last_line):  # contains digit
                    answer = last_line
    
    # Extract steps - handle both regular and LaTeX formats
    steps = []
    step_matches = list(STEP_PATTERN.finditer(text))
    if step_matches:
        steps = [m.group(1).strip() for m in step_matches]
    else:
        lines = text.split('\n') # If no steps found, try to split by lines and look for step-like content
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if (re.search(r'\d', line) and  # contains numbers
                not line.startswith('\\') and  # not LaTeX command
                not line.startswith('**') and  # not markdown
                not line.startswith('Therefore') and  # not conclusion
                len(line) > 10):  # reasonable length
                # Clean up LaTeX formatting from the line
                clean_line = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', line)  # Remove LaTeX commands
                clean_line = re.sub(r'\*\*[^*]*\*\*', '', clean_line)  # Remove markdown
                clean_line = re.sub(r'\[[^\]]*\]', '', clean_line)  # Remove brackets
                clean_line = clean_line.strip()
                if clean_line and len(clean_line) > 5:
                    steps.append(clean_line)
    
    return steps, answer

def generate_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    num_candidates: int,
    gen_cfg: GenerationConfig,
    device: torch.device,
) -> List[str]:
    """Generate *num_candidates* reasoning trajectories for the prompt."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = prompt_ids["input_ids"].repeat(num_candidates, 1)
    attention_mask = prompt_ids["attention_mask"].repeat(num_candidates, 1)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_cfg.to_dict())
    
    gen_only = outputs[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return texts

def compute_step_rewards(
    baseline: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prm: ProcessRewardModel,
    prm_device: torch.device,
    prompt: str,
    steps: List[str],
) -> List[float]:
    """Return a list of scalar rewards (float) for each *completed* step."""
    rewards: List[float] = []
    # We will iteratively feed *prompt + completed steps* through baseline.
    cumulative_text = prompt
    for i, step_txt in enumerate(steps):
        # Clean up the step text - remove LaTeX and markdown formatting
        clean_step = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', step_txt)  # Remove LaTeX commands
        clean_step = re.sub(r'\*\*[^*]*\*\*', '', clean_step)  # Remove markdown
        clean_step = re.sub(r'\[[^\]]*\]', '', clean_step)  # Remove brackets
        clean_step = clean_step.strip()
        
        if clean_step:  # Only add non-empty steps
            cumulative_text += f"Step {i + 1}: {clean_step}\n"
            tokens = tokenizer(cumulative_text, return_tensors="pt").to(prm_device)
            with torch.no_grad():
                outputs = baseline(**tokens, output_hidden_states=True)
            # Use hidden states of the last token (or pool as needed)
            last_hidden = outputs.hidden_states[-1][0, -1, :]  # (hidden_dim,)
            last_hidden = last_hidden.float() 
            reward = prm(last_hidden.unsqueeze(0)).item()  # type: ignore
            rewards.append(reward)
    
    return rewards

def main():
    # ------------------- Load baseline LM -------------------
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct" 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    config = PRMConfig()

    # ------------------- Load PRM ---------------------------
    prm_ckpt_path = "/home/leena/ccc_eval/mcts_prm/prm_dataset2/checkpoints/0715/contri/best_prm.pt"
    prm_ckpt = torch.load(prm_ckpt_path, map_location="cpu", weights_only=False)
    prm_cfg = PRMConfig(**prm_ckpt.get("cfg", {}))
    prm = ProcessRewardModel(baseline.config.hidden_size, cfg=prm_cfg)
    prm.load_state_dict(prm_ckpt["prm_state"])
    prm = prm.float()  # 명시적으로 Float32로 설정
    prm = prm.to(device).eval()
    print("Finish Loading Baseline and PRM!")

    # ------------------- Load Dataset ---------------------------
    dataset = "gsm8k"
    ds = load_dataset(dataset, "main", split="test")
    max_samples = 5
    if max_samples:
        ds = ds.select(range(max_samples))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Finish Loading Dataset!")
    
    # ------------------- Load Generation Config ---------------------------
    gen_cfg = GenerationConfig(
        temperature=0.1,  # 낮은 temperature로 일관된 출력
        top_p=0.9,        # 높은 top_p로 다양성 유지
        top_k=50,         # top_k 추가로 품질 향상
        max_new_tokens=config.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1,
        # repetition_penalty=1.1,  # 반복 방지
        do_sample=True,
    )
    
    results = []
    n_correct = 0
    for idx, sample in tqdm(enumerate(loader)):
        question = sample["question"][0]
        gold = sample.get("answer", [""])[0]
        prompt = build_prompt(question)

        # 1) Generate candidate CoTs
        cand_texts = generate_candidates(baseline, tokenizer, prompt, config.num_candidates, gen_cfg, device)

        # 2) Score each candidate via PRM
        cand_scores: List[float] = []
        cand_answers: List[str] = []
        best_chain = ""
        for text in cand_texts:
            steps, answer = parse_steps_and_answer(text)
            print(f"Steps: {steps}")
            print(f"Answer: {answer}")
            
            # If no steps or answer found, use a default low score
            if not steps or not answer:
                cand_scores.append(0.0)
                cand_answers.append("")
                continue
                
            step_rewards = compute_step_rewards(baseline, tokenizer, prm, device, prompt, steps)
            print(f"Step rewards: {step_rewards}")
            total_r = sum(step_rewards)
            cand_scores.append(total_r)
            cand_answers.append(answer)
            # Keep full chain for printing if it wins
            if total_r == max(cand_scores):
                best_chain = text

        # Handle case where all candidates failed
        if not cand_scores or max(cand_scores) == 0:
            best_idx = 0
            best_answer = ""
            best_score = 0.0
        else:
            best_idx = int(torch.tensor(cand_scores).argmax().item())
            best_answer = cand_answers[best_idx]
            best_score = cand_scores[best_idx]
        
        # Use enhanced numeric comparison for correctness
        correct = False
        if best_answer and gold:
            correct = _numeric_equiv_enhanced(best_answer.strip(), gold.strip())
        n_correct += int(correct)

        # 3) Save result
        results.append(
            {
                "id": sample.get("id", [idx])[0] if isinstance(sample.get("id", [idx]), list) else idx,
                "question": question,
                "gold": gold,
                "pred": best_answer,
                "chain": best_chain,
                "score": best_score,
                "correct": correct,
            }
        )

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(loader)} samples…")
    
    
    print(f"Accuracy: {n_correct}/{len(results)} = {n_correct / len(results):.2%}")

    # ------------------- Save Results ---------------------------
    results_path = f"/home/leena/ccc_eval/mcts_prm/prm_dataset2/results/{dataset}_{config.reward_type}_infer_test.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
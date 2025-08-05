import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple
import os, sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer
from prm_training.config import PRMConfig
from prm_training.train_prm_ft import FTPRM
from prm_training.prm_dataset import StepwisePRMDataset
from prm_training.train_lm_ft import FTLM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
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
BOXED_ANSWER_PATTERN = re.compile(
    r'\\boxed\{([^}]*)\}',
    re.IGNORECASE
)
STEP_PATTERN = re.compile(
    r"""(?i)                    # case insensitive
        step\s*(\d+)\s*[:\-]\s* # "Step 1:", "step 2-", etc.
        (.*?)                   # step content
        (?=\n\s*step\s*\d+|\n\s*answer\s*[:\-]|\n\s*therefore|\n\s*thus|\n\s*hence|$)  # lookahead
    """,
    re.S | re.IGNORECASE
)

def build_chat_messages(question: str) -> List[dict]:
    messages = [
        {"role": "system", "content": """You are an expert mathematical reasoning assistant. Solve the given problem step-by-step using clear mathematical logic.

Format Requirements:
- Start each step with "Step k: " (where k is the step number)
- Use precise mathematical notation and clear reasoning
- End with "Answer: [final numerical result]" and stop immediately
- Keep steps focused and mathematically rigorous
- Do not generate any other steps or text after reaching the answer

Mathematical Guidelines:
- Show all calculations clearly
- Use proper mathematical symbols (x, ÷, ±, etc.)

Now solve the given problem using the same format."""
        },
        {"role": "user", "content": "Problem: What is the next number in the sequence 2, 4, 8, 16?"},
        {"role": "assistant", "content": "Step 1: Analyze the pattern: each term is multiplied by 2.\nStep 2: 16 × 2 = 32\nAnswer: 32"},
        {"role": "user", "content": "Problem: Solve for x: 3x + 7 = 22"},
        {"role": "assistant", "content": "Step 1: Subtract 7 from both sides: 3x = 22 - 7 = 15\nStep 2: Divide both sides by 3: x = 15 ÷ 3 = 5\nAnswer: 5"},
        {"role": "user", "content": f"Problem: {question}\n"}
    ]
    return messages

STEP_BLOCK_RE = re.compile(
    r"^Step\s*(\d+)\s*:\s*(.+?)(?=\nStep\s*\d+\s*:|\nAnswer\s*:|$)",
    re.I | re.M | re.S,
)
ANSWER_LINE_RE = re.compile(r"^Answer\s*[:\-]\s*(.+)$", re.I | re.M)
BOXED_RE       = re.compile(r"\\boxed\{([^}]*)\}")

def _clean(txt: str) -> str:
    txt = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", txt)   # 일반 LaTeX 명령
    txt = re.sub(r"\*\*([^*]*)\*\*", r"\1", txt)     # **bold**
    txt = re.sub(r"\[[^\]]*\]", "", txt)             # [link]
    return txt.strip()

def parse_steps_and_answer(text: str) -> Tuple[List[str], str]:
    # (A) Answer 추출
    answer = ""
    m = BOXED_RE.search(text)
    if m:
        answer = _clean(m.group(1))
    else:
        m = ANSWER_LINE_RE.search(text)
        if m:
            answer = _clean(m.group(1))
        else:                          # fallback: 마지막 비어 있지 않은 줄
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line:
                    answer = _clean(line)
                    break
    # (B) Step 추출
    steps = []
    for step_no, body in STEP_BLOCK_RE.findall(text):
        cleaned = _clean(body)
        if cleaned:
            steps.append(cleaned)

    return steps, answer

def generate_candidates_with_chat_template(llm: LLM, tokenizer: PreTrainedTokenizerBase, question: str, num_candidates: int, gen_cfg: GenerationConfig, device: torch.device) -> List[str]:
    messages = build_chat_messages(question)
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )

    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=256,
        n=num_candidates,  # Best-of-N sampling
        stop=["<|endoftext|>", "<|im_end|>"]  # vLLM에서 사용하는 stop tokens
    )
    
    outputs = llm.generate([conversation_str], sampling_params)
    generated_texts = []
    for output in outputs:
        for completion in output.outputs:
            generated_texts.append(completion.text)
    
    return generated_texts

def compute_step_rewards_with_chat_template(baseline_tok: AutoTokenizer, prm: FTPRM, prm_tokenizer: AutoTokenizer, prm_device: torch.device, question: str, steps: List[str]) -> List[float]:
    rewards: List[float] = []
    messages = build_chat_messages(question)
    
    for i, step_txt in enumerate(steps):
        clean_step = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', step_txt)
        clean_step = re.sub(r'\*\*[^*]*\*\*', '', clean_step)
        clean_step = re.sub(r'\[[^\]]*\]', '', clean_step)
        clean_step = clean_step.strip()
        if clean_step:
            assistant_content = ""
            for j in range(i + 1):
                if j < len(steps):
                    clean_prev_step = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', steps[j])
                    clean_prev_step = re.sub(r'\*\*[^*]*\*\*', '', clean_prev_step)
                    clean_prev_step = re.sub(r'\[[^\]]*\]', '', clean_prev_step)
                    clean_prev_step = clean_prev_step.strip()
                    if clean_prev_step:
                        assistant_content += f"Step {j + 1}: {clean_prev_step}\n"
            
            # No additional generation, only inspect already generated steps with forward pass
            current_messages = messages + [{"role": "assistant", "content": assistant_content}]
            conversation_str = baseline_tok.apply_chat_template(
                current_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

            tokens = prm_tokenizer(conversation_str, return_tensors="pt").to(prm_device)
            with torch.no_grad():
                outputs = prm(**tokens)
                reward = outputs.item()
            rewards.append(reward)
    
    return rewards

def evaluate_candidate_with_prm(baseline_tok: AutoTokenizer, prm: FTPRM, prm_tokenizer: AutoTokenizer, prm_device: torch.device, question: str, candidate_text: str) -> dict:
    steps, answer = parse_steps_and_answer(candidate_text)
    # print("Candidate text:",candidate_text)
    # print(f"Steps: {steps}, Answer: {answer}")

    if not steps or not answer:
        return {
            "text": candidate_text,
            "steps": [],
            "answer": "",
            "step_rewards": [],
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "num_steps": 0,
            "valid": False
        }
    
    # Compute step-wise rewards
    step_rewards = compute_step_rewards_with_chat_template(baseline_tok, prm, prm_tokenizer, prm_device, question, steps)
    
    total_reward = sum(step_rewards)
    avg_reward = total_reward / len(step_rewards) if step_rewards else 0.0
    return {
        "text": candidate_text,
        "steps": steps,
        "answer": answer,
        "step_rewards": step_rewards,
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "num_steps": len(steps),
        "valid": True
    }

def select_best_candidate(candidates: List[dict], selection_criterion: str = "avg_reward") -> dict:
    valid_candidates = [c for c in candidates if c["valid"]]
    if not valid_candidates:
        return candidates[0] if candidates else None
    
    if selection_criterion == "avg_reward":
        best_candidate = max(valid_candidates, key=lambda x: x["avg_reward"])
    elif selection_criterion == "total_reward":
        best_candidate = max(valid_candidates, key=lambda x: x["total_reward"])
    else:
        best_candidate = max(valid_candidates, key=lambda x: x["avg_reward"])
    
    return best_candidate

def load_prm_full_model(prm_ckpt_path: str, base_model_name: str) -> Tuple[FTPRM, AutoTokenizer]:
    model = FTPRM(base_model_name)
    safetensors_path = os.path.join(prm_ckpt_path, "model.safetensors")
    state_dict = load_file(safetensors_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(prm_ckpt_path, trust_remote_code=True)
    return model, tokenizer

def load_lm_full_model(prm_ckpt_path: str, base_model_name: str) -> Tuple[FTPRM, AutoTokenizer]:
    model = FTLM(base_model_name)
    safetensors_path = os.path.join(prm_ckpt_path, "model.safetensors")
    state_dict = load_file(safetensors_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(prm_ckpt_path, trust_remote_code=True)
    return model, tokenizer

def main():
    # ------------------- Load baseline LM -------------------
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"  # "mistralai/Mathstral-7B-v0.1", "Qwen/Qwen2.5-Math-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            quantization="bitsandbytes",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    config = PRMConfig()

    # ------------------- Load PRM ---------------------------
    prm_ckpt_path = "/home/leena/ccc_eval/mcts_prm/prm_training/checkpoints/pt_prm/contri/final_model"
    base_model_name = "Qwen/Qwen2.5-Math-7B-Instruct"  # base model 이름
    prm_model, prm_tokenizer = load_prm_full_model(prm_ckpt_path, base_model_name)
    # prm_model, prm_tokenizer = load_lm_full_model(prm_ckpt_path, base_model_name)
    print("Finish Loading Baseline and PRM!")

    # ------------------- Load Dataset ---------------------------
    dataset = "gsm8k"
    ds = load_dataset(dataset, "main", split="test")
    start = 0
    max_samples = 30
    if max_samples:
        ds = ds.select(range(start, start+max_samples))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Finish Loading Dataset!")
    
    # ------------------- Load Generation Config ---------------------------
    gen_cfg = GenerationConfig(
        temperature=0.4,
        top_p=0.9,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1,
    )
    
    selection_criterion = "avg_reward" 
    results = []
    n_correct = 0

    for idx, sample in tqdm(enumerate(loader)):
        question, g_sol = sample["question"][0], sample["answer"][0]
        lines, gold_ans = [], None
        if "####" in g_sol:
            parts = g_sol.split("####")
            if len(parts) >= 2:
                steps_text = parts[0].strip()
                gold_ans = parts[1].strip()
                for ln in steps_text.splitlines():
                    ln = ln.strip()
                    if ln and not ln.startswith("####"):  # ####로 시작하지 않는 라인만
                        lines.append(ln)
            else:
                gold_ans = g_sol.strip()
        else:
            for ln in g_sol.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                m = _ANSWER_RE.match(ln)
                if m:
                    gold_ans = _sanitize_enhanced(m.group(1))
                    break
                lines.append(ln)
        
        if gold_ans is None:
            raise ValueError("gold answer not found for sample")
        
        gold_steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines)]
        print(f"Extracted Gold {len(gold_steps)} steps and answer: {gold_ans}")

        # 1) Generate candidate CoTs
        candidate_texts = generate_candidates_with_chat_template(llm, tokenizer, question, config.num_candidates, gen_cfg, device)

        # 2) Score each candidate via PRM
        candidates_evaluated = []
        cand_scores: List[float] = []
        cand_answers: List[str] = []
        best_chain = ""

        for i, candidate_text in enumerate(candidate_texts):
            print(f"\n--- Candidate {i + 1} ---")
            evaluation = evaluate_candidate_with_prm(tokenizer, prm_model, prm_tokenizer, device, question, candidate_text)
            candidates_evaluated.append(evaluation)

            if evaluation["valid"]:
                print(f"Steps: {len(evaluation['steps'])}")
                print(f"Answer: {evaluation['answer']}")
                print(f"Step rewards: {[f'{r:.3f}' for r in evaluation['step_rewards']]}")
                print(f"Total reward: {evaluation['total_reward']:.3f}")
                print(f"Average reward: {evaluation['avg_reward']:.3f}")
            else:
                print("Invalid candidate (no steps or answer found)")
        
        print(f"\n--- Selecting Best Candidate (criterion: {selection_criterion}) ---")
        best_candidate = select_best_candidate(candidates_evaluated, selection_criterion)
        if best_candidate and best_candidate["valid"]:
            print(f"Selected candidate with:")
            print(f"  Answer: {best_candidate['answer']}")
            print(f"  Total reward: {best_candidate['total_reward']:.3f}")
            print(f"  Average reward: {best_candidate['avg_reward']:.3f}")
            print(f"  Number of steps: {best_candidate['num_steps']}")
        else:
            print("No valid candidate found!")
            best_candidate = {
                "text": "",
                "steps": [],
                "answer": "",
                "step_rewards": [],
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "num_steps": 0,
                "valid": False
            }
        
        # Use enhanced numeric comparison for correctness
        correct = False
        if best_candidate["answer"] and gold_ans:
            correct = _numeric_equiv_enhanced(best_candidate["answer"].strip(), gold_ans.strip())
        n_correct += int(correct)
        print(f"Correct: {correct}")

        # 3) Save result
        result = {
            "id": sample.get("id", [idx])[0] if isinstance(sample.get("id", [idx]), list) else idx,
            "question": question,
            "gold": gold_ans,
            "pred": best_candidate["answer"],
            "chain": best_candidate["text"],
            "total_reward": best_candidate["total_reward"],
            "avg_reward": best_candidate["avg_reward"],
            "num_steps": best_candidate["num_steps"],
            "correct": correct,
            "selection_criterion": selection_criterion,
            "num_candidates": config.num_candidates,
            "all_candidates": [
                {
                    "text": c["text"],
                    "answer": c["answer"],
                    "total_reward": c["total_reward"],
                    "avg_reward": c["avg_reward"],
                    "num_steps": c["num_steps"],
                    "valid": c["valid"]
                }
                for c in candidates_evaluated
            ]
        }
        results.append(result)

        if (idx + 1) % 10 == 0:
            print(f"\nProcessed {idx + 1}/{len(loader)} samples...")
            print(f"Current accuracy: {n_correct}/{idx + 1} = {n_correct / (idx + 1):.2%}")
    
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {n_correct}/{len(results)} = {n_correct / len(results):.2%}")
    print(f"Selection criterion: {selection_criterion}")
    print(f"Number of candidates per question: {config.num_candidates}")

    # ------------------- Save Results ---------------------------
    results_path = f"/home/leena/ccc_eval/mcts_prm/test/gsm8k_qwen7b_prm_contri_bon_vllm.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    # ------------------- Additional Analysis ---------------------------
    print(f"\n=== Analysis ===")
    valid_results = [r for r in results if r["pred"]]
    if valid_results:
        avg_total_reward = sum(r["total_reward"] for r in valid_results) / len(valid_results)
        avg_avg_reward = sum(r["avg_reward"] for r in valid_results) / len(valid_results)
        avg_steps = sum(r["num_steps"] for r in valid_results) / len(valid_results)
        
        print(f"Average total reward: {avg_total_reward:.3f}")
        print(f"Average average reward: {avg_avg_reward:.3f}")
        print(f"Average number of steps: {avg_steps:.1f}")
        
        total_rewards = [r["total_reward"] for r in valid_results]
        avg_rewards = [r["avg_reward"] for r in valid_results]
        
        print(f"Total reward range: {min(total_rewards):.3f} - {max(total_rewards):.3f}")
        print(f"Average reward range: {min(avg_rewards):.3f} - {max(avg_rewards):.3f}")


if __name__ == "__main__":
    main()
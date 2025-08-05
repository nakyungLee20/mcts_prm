import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple
import os
import torch
import torch.nn as nn 
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
import random
import sys
from pathlib import Path
from safetensors.torch import load_file
from peft import PeftModel, PeftConfig, get_peft_model, prepare_model_for_kbit_training, LoraConfig

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from data_generation.utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer
from prm_training.config import PRMConfig
from prm_training.train_prm_ft import FTPRM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
STEP_PATTERN = re.compile(
    r"(?i)step\s*\d+\s*[:\-]\s*(.*?)(?=\n\s*step\s*\d+|\n\s*answer\s*[:\-]|$)",
    re.S,
)
_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")

# class FTPRM(nn.Module):
#     def __init__(self, base_model_name: str, adapter_path: str | None = None, lora_rank: int = 16, lora_alpha: int = 32, trainable: bool = False):
#         super().__init__()

#         self.backbone = AutoModel.from_pretrained(
#             base_model_name,
#             device_map="auto",
#             trust_remote_code=True,
#             quantization_config=BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_compute_dtype=torch.bfloat16
#             ),
#         )

#         if hasattr(self.backbone, "score"):
#             # For Qwen2.5-Math-PRM-7B
#             in_feat = self.backbone.score[0].in_features
#             self.backbone.score = nn.Sequential(
#                 nn.Linear(in_feat, in_feat),
#                 nn.ReLU(),
#                 nn.Linear(in_feat, 1, bias=True)  # 2 → 1
#             )
#             self.reg_head = None
#         else:
#             # Other AutoModel(Causal LM)
#             hidden = self.backbone.config.hidden_size
#             self.reg_head = nn.Sequential(
#                 nn.Linear(hidden, hidden // 4),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden // 4, 1)
#             )

#         # Add Lora Adapter
#         self.backbone = prepare_model_for_kbit_training(self.backbone)
#         lora_cfg = LoraConfig(
#             r=lora_rank,
#             lora_alpha=lora_alpha,
#             target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#             lora_dropout=0.05,
#             bias="none",
#         )

#         if adapter_path is None:
#             self.backbone = get_peft_model(self.backbone, lora_cfg)
#         else:
#             self.backbone = PeftModel.from_pretrained(self.backbone, adapter_path, is_trainable=trainable)
        
#         self._activate_head_params()   

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
#         out = self.backbone(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             output_hidden_states=True,
#                             return_dict=True)
#         hidden = out.hidden_states[-1]                     # (B, L, H)

#         # Last token vector
#         if attention_mask is None:   
#             rep = hidden[:, -1, :]
#         else:
#             seq_len = attention_mask.sum(1) - 1           # (B,)
#             rep = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_len, :]

#         # head 통과
#         if self.reg_head is None:
#             pred = self.backbone.score(rep).squeeze(-1)
#         else:
#             pred = self.reg_head(rep).squeeze(-1)

#         if labels is not None:              # training / eval
#             loss = F.mse_loss(pred, labels.float())
#             return loss, pred   
#         else:                               # pure inference
#             return pred

#     def _activate_head_params(self):
#         if self.reg_head is not None:
#             for p in self.reg_head.parameters():
#                 p.requires_grad_(True)
#         else:
#             for p in self.backbone.score.parameters():
#                 p.requires_grad_(True)

#     def get_trainable_parameters(self):
#         return [p for p in self.parameters() if p.requires_grad]
    
#     def get_parameter_stats(self):
#         trainable_params = 0
#         all_param = 0
#         module_stats = {}
        
#         for name, param in self.named_parameters():
#             all_param += param.numel()
#             if param.requires_grad:
#                 trainable_params += param.numel()
                
#                 module_name = name.split('.')[0]
#                 if module_name not in module_stats:
#                     module_stats[module_name] = {'trainable': 0, 'total': 0}
#                 module_stats[module_name]['trainable'] += param.numel()
#                 module_stats[module_name]['total'] += param.numel()
#             else:
#                 module_name = name.split('.')[0]
#                 if module_name not in module_stats:
#                     module_stats[module_name] = {'trainable': 0, 'total': 0}
#                 module_stats[module_name]['total'] += param.numel()
        
#         return {
#             'total_params': all_param,
#             'trainable_params': trainable_params,
#             'trainable_ratio': trainable_params / all_param * 100,
#             'module_stats': module_stats
#         }

# def load_prm_model_with_peft(prm_ckpt_path: str, base_model_name: str):
#     adapter_dir = os.path.join(prm_ckpt_path, "adapter")
    
#     if not os.path.exists(adapter_dir):
#         tmp = FTPRM(base_model_name).to(device)
#         missing, unexpected = tmp.load_state_dict(load_file(os.path.join(prm_ckpt_path,"model.safetensors")), strict=False)
#         os.makedirs(adapter_dir, exist_ok=True)
#         tmp.backbone.save_pretrained(adapter_dir, safe_serialization=True)  # ← ‘base_model’ 기준으로 저장
#         AutoTokenizer.from_pretrained(prm_ckpt_path, trust_remote_code=True).save_pretrained(adapter_dir)
#         print(f"Loaded weights - missing: {len(missing)}, unexpected: {len(unexpected)}")
#         print(f"Adapter saved to {adapter_dir}")

#     model = FTPRM(base_model_name, adapter_path=adapter_dir).eval().to(device)
#     tok   = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
#     return model, tok

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

def parse_steps(text: str) -> Tuple[List[str], str]:
    steps = []
    step_matches = list(STEP_PATTERN.finditer(text))
    if step_matches:
        steps = [m.group(1).strip() for m in step_matches]
    else:
        lines = text.split('\n')
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
    return steps

def simple_gsm8k_infer(sample):
    q_txt, g_sol = sample["question"][0], sample["answer"][0]
    lines, gold_ans = [], None
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
    steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines)]
    return steps, gold_ans

def create_perturbed_steps(gold_steps: List[str], perturbation_type: str = "irrelevant") -> List[str]:
    if not gold_steps:
        return gold_steps
    perturbed_steps = gold_steps.copy()
    
    if perturbation_type == "self_reflection":
        # insert_position = random.randint(1, len(gold_steps))
        insert_position = 2
        irrelevant_steps = [
            "Let me think about this problem carefully.",
            "I need to check my calculations.",
            "This step seems important for the solution.",
            "Let me verify the previous step.",
            "I should double-check my work.",
            "This is a crucial part of the solution.",
            "Let me organize my thoughts.",
            "I need to be careful with the math.",
        ]
        insert_step = f"Step {insert_position + 1}: {random.choice(irrelevant_steps)}"
        perturbed_steps.insert(insert_position, insert_step)

        for i in range(insert_position + 1, len(perturbed_steps)):
            if perturbed_steps[i].startswith("Step "):
                step_num = i + 1
                step_content = perturbed_steps[i].split(":", 1)[1] if ":" in perturbed_steps[i] else ""
                perturbed_steps[i] = f"Step {step_num}:{step_content}"
    
    elif perturbation_type == "wrong_step":
        # insert_position = random.randint(1, len(gold_steps))
        insert_position = 2
        wrong_steps = [
            f"Step {insert_position + 1}: 5 + 3 = 9",
            f"Step {insert_position + 1}: 10 * 2 = 15",
            f"Step {insert_position + 1}: 20 / 4 = 6",
            f"Step {insert_position + 1}: 7 - 3 = 5",
            f"Step {insert_position + 1}: 2^2 = 5",
            f"Step {insert_position + 1}: \sqrt(16) = 3",
        ]
        insert_step = random.choice(wrong_steps)
        perturbed_steps.insert(insert_position, insert_step)
        
        for i in range(insert_position + 1, len(perturbed_steps)):
            if perturbed_steps[i].startswith("Step "):
                step_num = i + 1
                step_content = perturbed_steps[i].split(":", 1)[1] if ":" in perturbed_steps[i] else ""
                perturbed_steps[i] = f"Step {step_num}:{step_content}"
    
    elif perturbation_type == "irrelevant":
        # insert_position = random.randint(1, len(gold_steps))
        insert_position = 2
        irrelevant_steps = [
            f"Step {insert_position + 1}: The weather is nice today.",
            f"Step {insert_position + 1}: I like mathematics very much.",
            f"Step {insert_position + 1}: This reminds me of my school days.",
            f"Step {insert_position + 1}: The sky is blue and beautiful.",
            f"Step {insert_position + 1}: I should drink more water.",
            f"Step {insert_position + 1}: Mathematics is the language of the universe.",
        ]
        
        insert_step = random.choice(irrelevant_steps)
        perturbed_steps.insert(insert_position, insert_step)
        
        for i in range(insert_position + 1, len(perturbed_steps)):
            if perturbed_steps[i].startswith("Step "):
                step_num = i + 1
                step_content = perturbed_steps[i].split(":", 1)[1] if ":" in perturbed_steps[i] else ""
                perturbed_steps[i] = f"Step {step_num}:{step_content}"
    
    elif perturbation_type == "repetition":
        if len(gold_steps) > 1:
            # insert_position = random.randint(1, len(gold_steps))
            insert_position = 2
            repeat_step = gold_steps[insert_position - 1]  # 이전 step
            step_num = insert_position + 1
            step_content = repeat_step.split(":", 1)[1] if ":" in repeat_step else ""
            insert_step = f"Step {step_num}:{step_content}"
            perturbed_steps.insert(insert_position, insert_step)
            
            for i in range(insert_position + 1, len(perturbed_steps)):
                if perturbed_steps[i].startswith("Step "):
                    step_num = i + 1
                    step_content = perturbed_steps[i].split(":", 1)[1] if ":" in perturbed_steps[i] else ""
                    perturbed_steps[i] = f"Step {step_num}:{step_content}"
    
    return perturbed_steps

def analyze_step_rewards_with_perturbations(
    prm_model: PreTrainedModel,
    prm_tokenizer: PreTrainedTokenizer,
    prm_device: torch.device,
    prompt: str,
    gold_steps: List[str],
    gold_ans: str,
    perturbed_type: str = "random_insert"
) -> dict:
    results = {}
    gold_rewards = compute_step_rewards(prm_model, prm_tokenizer, prm_device, prompt, gold_steps)
    results["gold"] = {
        "steps": gold_steps,
        "rewards": gold_rewards,
        "total_reward": sum(gold_rewards),
        "avg_reward": sum(gold_rewards) / len(gold_rewards) if gold_rewards else 0.0,
        "step_count": len(gold_steps)
    }

    random_insert_steps = create_perturbed_steps(gold_steps, "self_reflection")
    random_insert_rewards = compute_step_rewards(prm_model, prm_tokenizer, prm_device, prompt, random_insert_steps)
    results["self_reflection"] = {
        "steps": random_insert_steps,
        "rewards": random_insert_rewards,
        "total_reward": sum(random_insert_rewards),
        "avg_reward": sum(random_insert_rewards) / len(random_insert_rewards) if random_insert_rewards else 0.0,
        "step_count": len(random_insert_steps)
    }

    wrong_step_steps = create_perturbed_steps(gold_steps, "wrong_step")
    wrong_step_rewards = compute_step_rewards(prm_model, prm_tokenizer, prm_device, prompt, wrong_step_steps)
    results["wrong_step"] = {
        "steps": wrong_step_steps,
        "rewards": wrong_step_rewards,
        "total_reward": sum(wrong_step_rewards),
        "avg_reward": sum(wrong_step_rewards) / len(wrong_step_rewards) if wrong_step_rewards else 0.0,
        "step_count": len(wrong_step_steps)
    }

    irrelevant_steps = create_perturbed_steps(gold_steps, "irrelevant")
    irrelevant_rewards = compute_step_rewards(prm_model, prm_tokenizer, prm_device, prompt, irrelevant_steps)
    results["irrelevant"] = {
        "steps": irrelevant_steps,
        "rewards": irrelevant_rewards,
        "total_reward": sum(irrelevant_rewards),
        "avg_reward": sum(irrelevant_rewards) / len(irrelevant_rewards) if irrelevant_rewards else 0.0,
        "step_count": len(irrelevant_steps)
    }

    repetition_steps = create_perturbed_steps(gold_steps, "repetition")
    repetition_rewards = compute_step_rewards(prm_model, prm_tokenizer, prm_device, prompt, repetition_steps)
    results["repetition"] = {
        "steps": repetition_steps,
        "rewards": repetition_rewards,
        "total_reward": sum(repetition_rewards),
        "avg_reward": sum(repetition_rewards) / len(repetition_rewards) if repetition_rewards else 0.0,
        "step_count": len(repetition_steps)
    }
    return results

def compute_step_rewards(
    prm_model: PreTrainedModel,
    prm_tokenizer: PreTrainedTokenizer,
    prm_device: torch.device,
    prompt: str,
    steps: List[str],
) -> List[float]:
    rewards: List[float] = []
    cumulative_text = prompt
    for i, step_txt in enumerate(steps):
        # Clean up the step text - remove LaTeX and markdown formatting
        clean_step = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', step_txt)  # Remove LaTeX commands
        clean_step = re.sub(r'\*\*[^*]*\*\*', '', clean_step)  # Remove markdown
        clean_step = re.sub(r'\[[^\]]*\]', '', clean_step)  # Remove brackets
        clean_step = clean_step.strip()
        
        if clean_step:  # Only add non-empty steps
            cumulative_text += f"Step {i + 1}: {clean_step}\n"
            tokens = prm_tokenizer(cumulative_text, return_tensors="pt").to(prm_device)
            with torch.no_grad():
                outputs = prm_model(**tokens)
                reward = outputs.item()  # (hidden_dim,)
            rewards.append(reward)
    
    return rewards

def main():
    config = PRMConfig()
    # ------------------- Load PRM ---------------------------
    prm_ckpt_path = "/home/leena/ccc_eval/mcts_prm/prm_training/checkpoints/pt_prm/cmi/final_model"
    base_model_name = "Qwen/Qwen2.5-Math-PRM-7B"  # base model 이름

    prm_model, prm_tokenizer = load_prm_full_model(prm_ckpt_path, base_model_name)
    print("Finish Loading QLoRA model and PRM!")

    # ------------------- Load Dataset ---------------------------
    dataset = "openai/gsm8k"
    ds = load_dataset(dataset, "main", split="test")
    max_samples = 20
    if max_samples:
        ds = ds.select(range(max_samples))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Finish Loading Dataset!")
    
    results = []
    perturbation_analysis = []
    n_correct = 0
    for idx, sample in tqdm(enumerate(loader)):
        gold_steps, gold_ans = simple_gsm8k_infer(sample)
        prompt = "You are a math expert.Let's solve the following problem step by step.\n" + sample["question"][0]
        gold_rewards = compute_step_rewards(prm_model, prm_tokenizer, device, prompt, gold_steps)
        perturbation_results = analyze_step_rewards_with_perturbations(prm_model, prm_tokenizer, device, prompt, gold_steps, gold_ans)
        
        print(f"\n=== Sample {idx + 1} ===")
        print(f"Question: {sample['question'][0][:100]}...")
        print(f"Gold Answer: {gold_ans}")
        print(f"Gold Steps Count: {len(gold_steps)}")
        
        gold_avg = sum(gold_rewards) / len(gold_rewards) if gold_rewards else 0.0
        print(f"Gold Steps - Total: {sum(gold_rewards):.4f}, Avg: {gold_avg:.4f}")
        
        for perturbation_type in ["self_reflection", "wrong_step", "irrelevant", "repetition"]:
            result = perturbation_results[perturbation_type]
            avg_reward = result["avg_reward"]
            step_count = result["step_count"]
            reward_drop = gold_avg - avg_reward
            print(f"{perturbation_type.title()} - Steps: {step_count}, Total: {result['total_reward']:.4f}, Avg: {avg_reward:.4f}, Drop: {reward_drop:.4f}")
        
        results.append({
            "id": idx,
            "question": sample["question"][0],
            "gold_answer": gold_ans,
            "gold_steps": gold_steps,
            "gold_rewards": gold_rewards,
            "gold_total": sum(gold_rewards),
            "gold_avg": gold_avg,
            "gold_step_count": len(gold_steps),
        })
        
        perturbation_analysis.append({
            "id": idx,
            "question": sample["question"][0],
            "gold_answer": gold_ans,
            "gold_avg_reward": gold_avg,
            "perturbation_results": perturbation_results,
        })

    print(f"\n=== Overall Statistics (Average Reward Comparison) ===")
    gold_avgs = [r["gold_avg"] for r in results]
    overall_gold_avg = sum(gold_avgs) / len(gold_avgs)
    print(f"Overall Gold Average Reward: {overall_gold_avg:.4f}")
    # Perturbation 효과 분석 (평균값 기준)
    perturbation_summary = {}
    for perturbation_type in ["self_reflection", "wrong_step", "irrelevant", "repetition"]:
        perturbation_avgs = [r["perturbation_results"][perturbation_type]["avg_reward"] for r in perturbation_analysis]
        perturbation_step_counts = [r["perturbation_results"][perturbation_type]["step_count"] for r in perturbation_analysis]
        
        avg_perturbation_reward = sum(perturbation_avgs) / len(perturbation_avgs)
        avg_step_count = sum(perturbation_step_counts) / len(perturbation_step_counts)
        reward_drop = overall_gold_avg - avg_perturbation_reward
        drop_percentage = (reward_drop / overall_gold_avg) * 100 if overall_gold_avg > 0 else 0
        
        perturbation_summary[perturbation_type] = {
            "avg_reward": avg_perturbation_reward,
            "avg_step_count": avg_step_count,
            "reward_drop": reward_drop,
            "drop_percentage": drop_percentage
        }
        print(f"\n{perturbation_type.title()}:")
        print(f"  Average Reward: {avg_perturbation_reward:.4f}")
        print(f"  Average Step Count: {avg_step_count:.1f}")
        print(f"  Reward Drop: {reward_drop:.4f} ({drop_percentage:.1f}%)")
    
    detailed_results = {
        "summary": {
            "total_samples": len(results),
            "overall_gold_avg_reward": overall_gold_avg,
            "perturbation_summary": perturbation_summary
        },
        "sample_results": results,
        "perturbation_analysis": perturbation_analysis
    }
    
    results_path = f"/home/leena/ccc_eval/mcts_prm/inference/test/analysis_gsm8k_cmi_prm_ptr.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(detailed_results, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    print(f"\n=== Quick Summary ===")
    print(f"Gold steps average reward: {overall_gold_avg:.4f}")
    for perturbation_type in ["self_reflection", "wrong_step", "irrelevant", "repetition"]:
        summary = perturbation_summary[perturbation_type]
        print(f"{perturbation_type}: {summary['avg_reward']:.4f} (drop: {summary['drop_percentage']:.1f}%)")


if __name__ == "__main__":
    main()
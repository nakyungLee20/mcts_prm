import math as pymath
import json, os, re, sys, math, argparse
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from pathlib import Path
from vllm import LLM, SamplingParams
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 1. Prompt builder & parser & extractor & matcher                           #
##############################################################################
from answer_extractor import AnswerExtractor
from step_parser import StepParser, build_chat_messages, build_chat_messages2
from answer_matcher import MathAnswerScorer

##############################################################################
# 2. Dataset loaders & field mapping                                         #
##############################################################################
FIELD_MAP: Dict[str, Tuple[str, str]] = { # dataset name  : (question_field, answer_field)
    "gsm8k": ("question", "answer"),
    "math": ("problem", "solution"),
    "omni": ("problem", "answer"),
    "olympiad": ("question", "final_answer"),
    "aime": ("problem", "answer"),
}

def load_olympiadbench_english(split: str = "train"):
    all_cfgs = get_dataset_config_names("Hothan/OlympiadBench")
    en_cfgs = [cfg for cfg in all_cfgs if "_en_" in cfg or cfg.endswith("_en")]
    ds_list = []
    for cfg in en_cfgs:
        try:
            ds = load_dataset("Hothan/OlympiadBench", cfg, split=split)
            ds_list.append(ds)
        except Exception as e:
            print(f"⚠️  {cfg} load failed: {e}")
    if len(ds_list) == 0:
        raise ValueError("Fail to load English configs")
    full_ds = concatenate_datasets(ds_list)
    return full_ds

def get_loader(ds_name: str, split: str, batch_size: int):
    """Return torch DataLoader yielding (idx, question, gold_answer)."""
    if ds_name == "math":
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
    elif ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
    elif ds_name == "omni":
        ds = load_dataset("KbsdJames/Omni-MATH", split=split)
    elif ds_name == "olympiad":
        ds = load_olympiadbench_english(split)
    elif ds_name =="aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")

    q_key, a_key = FIELD_MAP[ds_name]
    def collate(indices):
        items = [ds[i] for i in indices]
        idxs = indices
        qs = [item[q_key] for item in items]
        golds = [item[a_key] for item in items]
        return idxs, qs, golds

    return DataLoader(range(len(ds)), batch_size=batch_size, shuffle=False, collate_fn=collate), len(ds)

##############################################################################
# 3. Inference utilities                                                     #
##############################################################################
def batched_generate_vllm(prompts: List[str], llm: "LLM", max_tokens: int, N: int=1) -> List[str]:
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.9, n=N) # naive one trajectory decoding
    outs = llm.generate(prompts, sp)
    results = []
    for o in outs:
        texts = [cand.text for cand in o.outputs]
        results.append(texts)
    return results

# 3‑A. Best‑of‑N **path‑level** search
def best_of_n_paths_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser,
    prm, prm_tok, device: torch.device, N: int = 8, max_tokens: int = 2048) -> List[str]:
    prompts = [build_chat_messages(q, tokenizer, ds_name) for q in qs]
    decoded_lists = batched_generate_vllm(prompts, llm, max_tokens, n=N)
    diagnostics: List[Dict[str, Any]] = []
    chosen_texts: List[str] = []
    for decodes in decoded_lists:
        cand_info = []
        best_body, best_score = "", -1e9
        for d in decodes:
            body = d.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            raw_steps = parser.parse(body)
            steps = parser.parse_clean(raw_steps)
            if len(steps) == 0:
                continue
            s = score_path(steps, prm, prm_tok, device)
            cand_info.append({"body": body, "score": s, "n_steps": len(steps)})
            if s > best_score:
                best_score, best_body = s, body
        # fallback if all decodes invalid
        chosen_texts.append(best_body if best_body else decodes[0])
        diagnostics.append({"chosen": best_body, "chosen_score": best_score, "candidates": cand_info})
    return chosen_texts, diagnostics

# 3‑B. Best‑of‑N **step‑level** search (Monte‑Carlo greedy)
ANSWER_KEYWORDS = ("answer", "final answer", "therefore", "result", "the final answer is",)
def generate_stepwise_with_prm(question: str, tokenizer, llm: "LLM", ds_name: str, parser: StepParser, prm, 
    prm_tok, device: torch.device, N: int = 8, max_tokens_per_step: int = 128, max_total_steps: int = 30) -> str:
    steps: List[str] = []
    for step_idx in range(max_total_steps):
        messages = []
        # system prompt & few‑shot & problem statement
        prompt_base = build_chat_messages2(question, tokenizer, ds_name)
        # Remove the problematic line that tries to apply chat template to empty list
        base_text = prompt_base + "\n" + "\n".join(steps)
        prompt = base_text + "\n"  # generation prompt after last newline
        print("prompt: ", prompt)
        # Generate N continuations of at most max_tokens_per_step tokens
        cand_list = batched_generate_vllm([prompt], llm, max_tokens_per_step, N=N)[0]
        best_cand, best_score = None, -1e9
        for cand in cand_list:
            line = cand.split("\n")[0].strip()
            if not line:
                continue
            s = score_step(line, prm, prm_tok, device)
            if s > best_score:
                best_score, best_cand = s, line
        if best_cand is None:
            break  # unable to proceed
        steps.append(best_cand)
        lower_line = best_cand.lower()
        if any(lower_line.startswith(k) for k in ANSWER_KEYWORDS):
            break
    return "\n".join(steps)

def best_of_n_steps_batch(qs: List[str], tokenizer, llm: "LLM", ds_name: str, parser: StepParser,
    prm, prm_tok, device: torch.device, N: int = 8, max_tokens_per_step: int = 384) -> List[str]:
    return [
        generate_stepwise_with_prm(q, tokenizer, llm, ds_name, parser, prm, prm_tok, device, N, max_tokens_per_step)
        for q in qs
    ]

##############################################################################
# 4. PRM utilities                                                           #
##############################################################################
# PRM Custom Class
class FTLM(nn.Module):
    def __init__(self, base_model_name: str, lora_rank: int = 16, lora_alpha: int = 32, mlp_ratio: int = 4, value_head_prefix: str = "value_head", 
                 normalize_reward: bool = False):
        super().__init__()
        
        # Use AutoModelForCausalLM for pure CausalLM fine-tuning
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
        )
        # Add Lora Adapter for efficient fine-tuning
        self.backbone = prepare_model_for_kbit_training(self.backbone)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        # Value head for reward prediction
        hidden = self.backbone.config.hidden_size
        mlp_hidden = hidden // mlp_ratio
        head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1, bias=False),
        )
        self.value_head_prefix = value_head_prefix
        setattr(self, value_head_prefix, head)
        # head 가중치 학습 가능하도록 보장
        for p in head.parameters():
            p.requires_grad_(True)
        self.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std",  torch.ones(1),  persistent=False)
        # 캐시 비활성 (gradient checkpointing 호환)
        self.backbone.config.use_cache = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None, return_hidden: bool = False):
        if attention_mask is None:
            attention_mask = (input_ids != self.backbone.config.pad_token_id).long()

        # position_ids = cumulative mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]        # (B, T, H)
        # Index of last non-pad token  → (B, 1)
        eos_idx = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(-1, keepdim=True)

        values = getattr(self, self.value_head_prefix)(last_hidden).squeeze(-1)   # (B, T)
        reward = values.gather(1, eos_idx).squeeze(1)                             # (B,)

        if labels is not None:
            loss = F.mse_loss(reward, labels.float())
            return loss, reward
        else:
            # if (not self.training) and self.normalize_reward:
            #     reward = (reward - self.mean) / (self.std + 1e-8)
            return (reward, last_hidden) if return_hidden else reward

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_stats(self):
        trainable_params = 0
        all_param = 0
        module_stats = {}
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = {'trainable': 0, 'total': 0}
                module_stats[module_name]['trainable'] += param.numel()
                module_stats[module_name]['total'] += param.numel()
            else:
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = {'trainable': 0, 'total': 0}
                module_stats[module_name]['total'] += param.numel()
        
        return {
            'total_params': all_param,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / all_param * 100,
            'module_stats': module_stats
        }

def load_prm(prm_ckpt_path: str, base_model_name: str, device: torch.device) -> Tuple[FTLM, AutoTokenizer]:
    from safetensors.torch import load_file as safe_load

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Load LoRA configuration and adapters
    adapter_path = os.path.join(prm_ckpt_path, "adapter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        base_model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"[PRM] LoRA adapters loaded from: {adapter_path}")
    else:
        print("[PRM] Warning: No LoRA adapters found")
    
    # Create FTLM wrapper with the base model
    model = FTLM(base_model_name=base_model_name, lora_rank=16, lora_alpha=32, mlp_ratio=4, value_head_prefix="value_head")
    
    # Replace the backbone with the loaded model
    model.backbone = base_model
    
    # Load value head weights from the final model
    final_model_path = os.path.join(prm_ckpt_path, "final_model", "model.safetensors")
    if os.path.exists(final_model_path):
        state_dict = safe_load(final_model_path)
        # Filter only value head weights
        value_head_keys = {k: v for k, v in state_dict.items() if k.startswith("value_head.")}
        if value_head_keys:
            missing, unexpected = model.value_head.load_state_dict(value_head_keys, strict=False)
            print(f"[PRM] Value head loaded from final_model; missing={missing}, unexpected={unexpected}")
        else:
            print("[PRM] Warning: No value head weights found in final_model")
    else:
        print("[PRM] Warning: final_model/model.safetensors not found")
    
    model.eval().to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(prm_ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def load_prm_case_b(prm_root: str, base_model_name: str, device: torch.device, *, mlp_ratio: int = 4, value_head_prefix: str = "value_head", adapter_subdir: str = "adapter",):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
    )
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    adapter_dir = os.path.join(prm_root, adapter_subdir)
    adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        raise FileNotFoundError(f"Case B expects LoRA adapters at {adapter_dir} (adapter_config.json not found)")

    backbone = PeftModel.from_pretrained(backbone, adapter_dir)
    print(f"[PRM/CaseB] LoRA adapters attached from: {adapter_dir}")

    class MinimalPRM(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone
            hidden = self.backbone.config.hidden_size
            mlp_hidden = max(1, hidden // mlp_ratio)
            head = nn.Sequential(
                nn.Linear(hidden, mlp_hidden, bias=False, dtype=torch.bfloat16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, 1, bias=False, dtype=torch.bfloat16),
            )
            setattr(self, value_head_prefix, head)
            self.value_head_prefix = value_head_prefix

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if attention_mask is None:
                pad_id = getattr(self.backbone.config, "pad_token_id", 0)
                attention_mask = (input_ids != pad_id).long()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]
            eos_idx = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(-1, keepdim=True)
            last_hidden = last_hidden.float()
            # values = getattr(self, self.value_head_prefix)(last_hidden).squeeze(-1)
            value_head = getattr(self, self.value_head_prefix)
            if last_hidden.dtype != value_head[0].weight.dtype:
                last_hidden = last_hidden.to(value_head[0].weight.dtype)
            
            values = value_head(last_hidden).squeeze(-1)
            reward = values.gather(1, eos_idx).squeeze(1)
            return reward

    model = MinimalPRM(backbone).to(device).eval()

    # Load only value head weights from likely files
    candidates = [
        os.path.join(prm_root, "final_model", "model.safetensors"),
        os.path.join(prm_root, "final_model", "pytorch_model.bin"),
        os.path.join(prm_root, "value_head.safetensors"),
        os.path.join(prm_root, "value_head.bin"),
        os.path.join(prm_root, "model.safetensors"),
        os.path.join(prm_root, "pytorch_model.bin"),
    ]
    loaded = False
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load
                sd = safe_load(p)
            else:
                sd = torch.load(p, map_location="cpu")
            # head_keys = {k: v for k, v in sd.items() if k.startswith(f"{value_head_prefix}.")}

            head_keys = {}
            for k, v in sd.items():
                if k.startswith(f"{value_head_prefix}."):
                    # "value_head.0.weight" -> "0.weight"
                    new_key = k.replace(f"{value_head_prefix}.", "")
                    head_keys[new_key] = v

            if head_keys:
                missing, unexpected = getattr(model, value_head_prefix).load_state_dict(head_keys, strict=False)
                print(f"[PRM/CaseB] value head loaded from {os.path.basename(p)}; missing={missing}, unexpected={unexpected}")
                loaded = True
                break
        except Exception as e:
            print(f"[PRM/CaseB] reading {p} failed: {e}")
    if not loaded:
        print("[PRM/CaseB] ⚠️  value head weights not found; using randomly initialized head.")

    return model, tok

def _find_first(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def load_prm_test(checkpoint_path: str, base_model_name: str, device: torch.device, *, head_prefix: str = "value_head",mlp_ratio: int = 4,):
    # Try Case A if a unified file exists
    # if _find_first([
    #     os.path.join(checkpoint_path, "model.safetensors"),
    #     os.path.join(checkpoint_path, "pytorch_model.bin"),
    #     os.path.join(checkpoint_path, "final_model", "model.safetensors"),
    #     os.path.join(checkpoint_path, "final_model", "pytorch_model.bin"),
    # ]):
    #     if base_model_name is None:
    #         raise ValueError("Case A requires the base model name used at training.")
    #     try:
    #         return load_prm_case_a(
    #             checkpoint_path, base_model_name, device,
    #             lora_rank=16, lora_alpha=32, mlp_ratio=mlp_ratio, value_head_prefix=head_prefix,
    #         )
    #     except Exception as e:
    #         print("[PRM] Case A load failed:", e)

    # Try Case B if adapter exists
    adapter_dir = os.path.join(checkpoint_path, "adapter")
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        if base_model_name is None:
            raise ValueError("Case B requires --prm_base_model to rebuild the backbone.")
        try:
            return load_prm_case_b(
                checkpoint_path, base_model_name, device,
                mlp_ratio=mlp_ratio, value_head_prefix=head_prefix,
            )
        except Exception as e:
            print("[PRM] Case B load failed:", e)

    raise RuntimeError(
        "No compatible PRM checkpoint layout found. Expected either a unified state_dict (Case A) "
        "or an adapter directory with adapter_config.json (Case B)."
    )

@torch.no_grad()
def score_step(step: str, prm, prm_tok, device: torch.device) -> float:
    inputs = prm_tok(step, truncation=True, return_tensors="pt").to(device)
    logits = prm(**inputs)[0]  # supports both (loss, reward) or reward tensor
    if isinstance(logits, tuple):
        logits = logits[-1]
    if logits.ndim > 1:
        logits = logits.squeeze()
    return float(logits.item())

def score_path(steps: List[str], prm, prm_tok, device: torch.device) -> float:
    if len(steps) == 0:
        return -1e9
    scores = [score_step(s, prm, prm_tok, device) for s in steps]
    return sum(scores) / len(scores)

##############################################################################
# 5. Main – end‑to‑end evaluation                                            #
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "gsm8k", "omni", "olympiad"]) 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--max_samples", type=int, default=200, help="0 = all")
    parser.add_argument("--search_mode", type=str, default="step_bon", choices=["single", "path_bon", "step_bon"])
    parser.add_argument("--N", type=int, default=4, help="Number of candidates per search")
    args = parser.parse_args()

    ## Load Policy Model Using vllm ##
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.6, quantization="bitsandbytes", max_model_len=4096,)

    ## Load Math Dataset ##
    loader, total = get_loader(args.dataset, "test" if args.dataset != "olympiad" else "train", args.batch_size)
    max_n = total if args.max_samples == 0 else min(args.max_samples, total)

    ## Load PRM Model ##
    prm_path ="/home/leena/ccc_eval/mcts_prm/prm_training/checkpoints/pt_lm/contri/final_model"
    # prm, prm_tok = load_prm(prm_path, device)
    prm, prm_tok = load_prm_test(prm_path, args.model, device)
    print("Finish Loading Model and Dataset!")

    ## Helpers ##
    extractor = AnswerExtractor()
    scorer = MathAnswerScorer()
    parser_obj = StepParser()
    
    ## Evaluation loop ##
    correct = 0
    seen = 0
    mode = args.search_mode
    diagnostics_all = []  # Initialize diagnostics list
    pbar = tqdm(loader, total = math.ceil(max_n / args.batch_size))
    for idxs, qs, golds in pbar:
        if seen >= max_n:
            break
        take = min(len(qs), max_n - seen) # trim overflow inside batch
        qs, golds = qs[:take], golds[:take]

        if mode == "single":
            raw_batches = [build_chat_messages(q, tokenizer, args.dataset) for q in qs]
            decoded = batched_generate_vllm(raw_batches, llm, 3096, n=1)
            raw_texts = [d[0] for d in decoded]
        elif args.search_mode == "path_bon":
            raw_texts, diags = best_of_n_paths_batch(qs, tokenizer, llm, args.dataset, parser_obj, prm, prm_tok, device, args.N, 3096)
            diagnostics_all.extend(diags)
        elif args.search_mode == "step_bon":
            raw_texts = best_of_n_steps_batch(qs, tokenizer, llm, args.dataset, parser_obj, prm, prm_tok, device, args.N)
        else:
            raise ValueError(f"Unknown search_mode {mode}")

        # Extract answers & score
        pred_answers = [extractor.extract_pred_answer(rt.split("Answer:")[-1]) for rt in raw_texts]
        gold_answers = [extractor.extract_gold_answer(g, args.dataset) for g in golds]
        batch_corr = [scorer.answers_match(p, g) for p, g in zip(pred_answers, gold_answers)]
        correct += sum(batch_corr)
        seen += take
        pbar.set_postfix(acc=f"{correct/seen:.3%}")

    print("\n====================== Summary ======================")
    print(f"Dataset      : {args.dataset}")
    print(f"Search mode  : {args.search_mode} (N={args.N})")
    print(f"Correct      : {correct}")
    print(f"Samples seen : {seen}")
    print(f"Accuracy     : {correct/seen:.3%}")

    # Optionally dump diagnostics for analysis
    if diagnostics_all:
        with open("bon_path_analysis_test.json", "w", encoding="utf8") as f:
            json.dump(diagnostics_all, f, ensure_ascii=False, indent=2)
        print("Diagnostics saved to file")


if __name__ == "__main__":
    main()

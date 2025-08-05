import argparse
import math as pymath
import json, os, re, sys
from typing import Dict, List, Optional, Tuple
import torch
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
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 1. Prompt builder & parser & extractor & matcher                           #
##############################################################################
from answer_extractor import AnswerExtractor
from step_parser import StepParser, build_chat_messages
from answer_matcher import MathAnswerScorer

##############################################################################
# 2. Dataset loaders & field mapping                                         #
##############################################################################
FIELD_MAP: Dict[str, Tuple[str, str]] = { # dataset name  : (question_field, answer_field)
    "gsm8k": ("question", "answer"),
    "math": ("problem", "solution"),
    "omni": ("problem", "answer"),
    "olympiad": ("question", "final_answer"),
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
def batched_generate_vllm(prompts: List[str], llm: "LLM", max_tokens: int, n: int=1) -> List[str]:
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.9, n=n) # naive one trajectory decoding
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]

def generate_with_retry_batch_vllm(qs: List[str],tokenizer,llm: "LLM",ds_name: str, parser: StepParser, max_tokens: int = 512,max_retries: int = 2,) -> List[str]:
    remaining = list(range(len(qs)))
    texts: List[str] = ["" for _ in qs]

    for _ in range(max_retries + 1):  # initial + retries
        if not remaining:
            break
        prompts = [build_chat_messages(qs[i], tokenizer, ds_name) for i in remaining]
        decoded = batched_generate_vllm(prompts, llm, max_tokens)
        next_remaining = []
        for idx_local, full in enumerate(decoded):
            gidx = remaining[idx_local]
            body = full.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            raw_steps = parser.parse(body)
            steps = parser.parse_clean(raw_steps)
            if len(steps) >= 2 and steps[-1].lower().startswith("answer"):
                texts[gidx] = body
            else:
                next_remaining.append(gidx)
        remaining = next_remaining
    # whatever still empty → fill with last decoded raw (best we have)
    for i in remaining:
        texts[i] = body  # last attempt's body
    return texts

##############################################################################
# 4. Main – end‑to‑end evaluation                                            #
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "gsm8k", "omni", "olympiad"]) 
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--max_samples", type=int, default=200, help="0 = all")
    args = parser.parse_args()

    ## Load Policy Model Using vllm ##
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.8, quantization="bitsandbytes", max_model_len=4096,)

    ## Load Math Dataset ##
    loader, total = get_loader(args.dataset, "test" if args.dataset != "olympiad" else "train", args.batch_size)
    max_n = total if args.max_samples == 0 else min(args.max_samples, total)
    print("Finish Loading Model and Dataset!")

    ## Helpers ##
    extractor = AnswerExtractor()
    scorer = MathAnswerScorer()
    parser_obj = StepParser()
    
    ## Evaluation loop ##
    correct = 0
    seen = 0
    pbar = tqdm(loader, total=math.ceil(max_n / args.batch_size))
    for idxs, qs, golds in pbar:
        if seen >= max_n:
            break
        take = min(len(qs), max_n - seen) # trim overflow inside batch
        qs, golds, idxs = qs[:take], golds[:take], idxs[:take]

        raw_texts = generate_with_retry_batch_vllm(qs, tokenizer, llm, args.dataset, parser_obj, max_tokens=3096,)

        # extract structured answers
        pred_answers = [extractor.extract_pred_answer(rt.split("Answer:")[-1]) for rt in raw_texts]
        gold_answers = [extractor.extract_gold_answer(g, args.dataset) for g in golds]

        batch_corr = [scorer.answers_match(p, g) for p, g in zip(pred_answers, gold_answers)]
        correct += sum(batch_corr)
        seen += take
        pbar.set_postfix(acc=f"{correct/seen:.3%}")

    print("\n===========================================")
    print(f"Dataset       : {args.dataset}")
    print(f"Samples seen  : {seen}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {correct/seen:.3%}")


if __name__ == "__main__":
    main()

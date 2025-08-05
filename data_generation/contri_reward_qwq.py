import math
import re, sys
from typing import List, Optional, Tuple, Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path

# Project-level helpers
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from config import PRMConfig
from utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer, system_prompt
from inference.answer_extractor import AnswerExtractor
from inference.answer_matcher import MathAnswerScorer

def _format_system_prompt() -> str:
    return (
        "You are an **expert mathematical-reasoning assistant**.\n\n"
        "## Format rules\n"
        "1. Begin *every* reasoning line with the exact prefix `Step k:` where `k = 1, 2, …`. No other prefix is allowed.\n"
        "2. Show *all* intermediate calculations using standard symbols (×, ÷, ±, √).\n"
        "3. Put your final answer within `Answer: \boxed{}`. and **stop immediately** — no extra text after the answer.\n"
        "4. Each step must be concise *yet mathematically rigorous*.\n"
        "5. Do not generate any text or reflection if you reach the final answer.\n\n"
        "Follow these rules exactly — evaluations are case- and format‑sensitive.\n"
        "Respond *only* in the specified format."
    )

def build_chat_messages_qwq(*, question: str, tokenizer, dataset: str, shots: Optional[List[Tuple[str, str, str]]] = None,
    prefix_context: Optional[str] = None, next_label: Optional[str] = None,) -> str:
    
    system_prompt = _format_system_prompt()
    default_shots: List[Tuple[str, str, str]] = [
        (
            "gsm8k, math, olympiad, omni",
            "Problem: What is the next number in the sequence 2, 4, 8, 16?",
            "Step 1: Identify the pattern; each term is multiplied by 2.\n"
            "Step 2: 16 × 2 = 32\n"
            "Answer: 32",
        ),
        (
            "gsm8k, math",
            "Problem: Solve for x: 3x + 7 = 22",
            "Step 1: Subtract 7 from both sides: 3x = 15\n"
            "Step 2: Divide by 3: x = 5\n"
            "Answer: 5",
        ),
        (
            "olympiad, omni",
            "Problem: Determine whether v₁ = [1,2] and v₂ = [3,6] are linearly independent.",
            "Step 1: Observe v₂ = 3 · v₁, so v₂ is a scalar multiple of v₁.\n"
            "Step 2: Therefore the vectors are linearly dependent.\n"
            "Answer: Dependent",
        ),
    ]

    if shots is None:
        shots = default_shots
    user_lines: List[str] = []
    
    if prefix_context:
        user_lines.append(prefix_context.rstrip())
    
    user_lines.append(f"Problem: {question}".rstrip())
    if next_label:
        user_lines.append(next_label.rstrip())
    user_content = "\n".join([ln for ln in user_lines if ln])

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for tag, q, a in shots:
        if dataset.lower() in tag.lower():
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def build_masking_chat_messages_qwq(tokenizer, sentence: str) -> str:
    masking_system = (
        "In the sentence below, mask any word or expression that seems crucial "
        "(such as a variable, a number or an operator, etc.) for solving the math problem "
        "by replacing it with '[MASKED]'."
    )
    user_content = f"Sentence: \"{sentence}\"\nRewritten:"
    messages = [
        {"role": "system", "content": masking_system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

class ContriRewardvLLM:
    def __init__(self, config: "PRMConfig", model_name: str = "mistralai/Mathstral-7B-v0.1"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            quantization="bitsandbytes",
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.scorer = MathAnswerScorer()
        self.extractor = AnswerExtractor()
        
        self.rollout_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_rollouts,
            repetition_penalty=1.1,
        )
        self.masking_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_rollouts,
            repetition_penalty=1.1,
        )
        print(f"vLLM model loaded: {model_name}")

    def _batched_generate(self, prompts: List[str], params: SamplingParams):
        return self.llm.generate(prompts, params)

    def _score_batch(self, outputs, gold_answer: str) -> List[float]:
        rewards = []
        for result in outputs:
            correct = 0
            for comp in result.outputs:
                text = comp.text or ""
                tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text
                pred = self.extractor.extract_pred_answer(tail)
                print("Prediction Answer:", pred)
                if self.scorer.answers_match(pred, gold_answer):
                    correct += 1
            rewards.append(correct / float(self.config.num_rollouts))
        return rewards

    def _make_prompt(self, *, question: str, staged_steps: List[str], next_label: str, dataset: str) -> str:
        prefix_context = "\n".join(staged_steps)
        return build_chat_messages_qwq(question=question, tokenizer=self.tokenizer, dataset=dataset, prefix_context=prefix_context, next_label=next_label)
    
    def compute_step_rewards_batch(self, question: str, dataset: str, steps: List[str], gold_answer: str) -> List[float]:
        prompts: List[str] = []
        for i in range(len(steps)):
            next_label = f"Step {i + 2}:" if i < len(steps) - 1 else "Answer:"
            staged_steps = steps[: i + 1]
            prompts.append(self._make_prompt(question=question, staged_steps=staged_steps, next_label=next_label, dataset=dataset))
        outputs = self._batched_generate(prompts, self.rollout_params)
        print("Generated rollout outputs:")
        for i, output in enumerate(outputs):
            print(f"Output {i}: {output.outputs[0].text}")
        return self._score_batch(outputs, gold_answer)
        
    def model_masking_batch(self, texts: List[str]) -> List[str]:
        mask_prompts = [build_masking_chat_messages_qwq(self.tokenizer, t) for t in texts]
        outputs = self._batched_generate(mask_prompts, self.masking_params)
        print("Mask Generation:")
        for i, output in enumerate(outputs):
            print(f"Output {i}: {output.outputs[0].text}")
        return [out.outputs[0].text.strip() for out in outputs]

    def perturb_step_rewards_batch(self, question: str, dataset: str, steps: List[str], gold_answer: str, use_llm: bool = True) -> List[float]:
        bodies = []
        prefixes = []
        for step in steps:
            m = re.match(r"^[\s>#*\-]*Step\s*\d+\s*[:.\-]\s*", step, flags=re.I)
            prefixes.append(m.group(0) if m else "")
            bodies.append(step[len(prefixes[-1]):])

        if use_llm:
            masked_bodies = self.model_masking_batch(bodies)
        else:
            masked_bodies = [self._MASK_PATTERN.sub("[MASKED]", b) for b in bodies]
            
        prompts = []
        for i in range(len(steps)):
            masked_step = prefixes[i] + masked_bodies[i]
            staged_steps = steps[:i] + [masked_step]
            next_label = f"Step {i + 2}:" if i < len(steps) - 1 else "Answer:"
            prompts.append(self._make_prompt(question=question, staged_steps=staged_steps, next_label=next_label, dataset=dataset))
        
        outputs = self._batched_generate(prompts, self.rollout_params)
        return self._score_batch(outputs, gold_answer)

    def gsm8k_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: Optional[int] = None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        ds = ds.select(range(start, start + take)) if take else ds
        # ds = ds.select(range(start, len(ds)))
        print("Generated dataset size: ", len(ds))

        for sample in tqdm(ds, desc="Building GSM8K contri reward-dataset"):
            q_txt, g_sol = sample["question"], sample["answer"]
            lines, gold_ans = [], None
            
            gold_ans = self.extractor.extract_gold_answer(g_sol, "gsm8k")
            print("Gold Answer:", gold_ans)
            if gold_ans is None:
                raise ValueError("gold answer not found for sample")
            
            lines = [ln.strip() for ln in g_sol.splitlines() if ln.strip()]
            steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines)]
            # steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines) if not t.lower().startswith("answer")] 
            print("Steps Split:", steps)

            ori = self.compute_step_rewards_batch(q_txt, "gsm8k", steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(q_txt, "gsm8k", steps, gold_ans, self.config.use_llm)
            print("Original Rewards:", ori)
            print("Masked Rewards:", ptb)
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                "question": q_txt,
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "gold_answer": gold_ans,
            }
            yield entry

    def math_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: Optional[int] = None):
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        ds = ds.select(range(start, start + take)) if take else ds
        # ds = ds.select(range(start, len(ds)))
        print("Generated dataset size: ", len(ds))
        
        for sample in tqdm(ds, desc="Building MATH contri reward-dataset"):
            full_sol = sample["solution"]
            
            gold_ans = self.extractor.extract_gold_answer(full_sol, "math")
            print("Gold Answer:", gold_ans)
            if gold_ans is None:
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]
            print("Steps Split:", steps)

            ori = self.compute_step_rewards_batch(sample["problem"],"math", steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(sample["problem"], "math", steps, gold_ans, self.config.use_llm)
            print("Original Rewards:", ori)
            print("Masked Rewards:", ptb)
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]

            entry = {
                "question": sample["problem"],
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "gold_answer": gold_ans,
            }
            yield entry

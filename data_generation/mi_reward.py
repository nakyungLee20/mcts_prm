import math
import sympy as sp
import re
from typing import Optional, List
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import system_prompt, _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer

class MIReward:
    ANSWER_PATTERN = re.compile(
        r"""^[\s>#*\-]*          # optional markdown/bullet symbols
            Answer               # word 'Answer'
            \s*[:.\-]\s*         # separator
            (.+?)\s*$            # capture everything after
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    _ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")
    
    def __init__(self, config: "PRMConfig", model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    # Function to parse a solution text into steps and final answer.
    def _extract_answer(self, text: str) -> Optional[str]:
        """Try multiple heuristics / regexes to pull out an answer string."""
        match = self.ANSWER_PATTERN.search(text)
        if match:
            return _sanitize_enhanced(match.group(1))
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            candidate = lines[-1]
            if re.search(r"\d", candidate):  # contains digit
                return _sanitize_enhanced(candidate)
        for line in reversed(text.splitlines()):
            if line.strip().lower().startswith("answer"):
                return _sanitize_enhanced(line.split("Answer", 1)[-1])
        return None
    
    def entropy_bits_exact(self, prompt: str, target: str) -> float:
        """True H(A|prompt) in bits/token, by ∑_t H(p_t). Memory-intensive: stores full probs tensor."""
        LOG2E = 1 / math.log(2)
        full   = prompt + target
        inputs = self.tokenizer(full, return_tensors="pt").to(self.device)
        Lp     = len(self.tokenizer(prompt)["input_ids"])

        with torch.no_grad():
            logits = self.model(**inputs).logits.float()      # [1,L,V]

        probs = logits.softmax(-1)                      # [...,V]
        token_H = -(probs * probs.log()).sum(-1) * LOG2E  # bits/token

        mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        mask[:, Lp:] = True                             # answer tokens
        return token_H[mask].sum().item() / mask.sum().item()
    
    def compute_step_mi(self, question: str, steps: List[str], gold_answer: str):
        # I(S_i ; A | context, S_1,...,S_{i-1}) = H(A|prev) - H(A|prev,S_i)
        sys_prompt = """Solve the given problem with step by step reasoning in the format of "Step k: <k-th rationale>" and write final answer in the format of "Answer: <answer>".\nProblem: """
        question = re.sub(r' +', ' ', question) 
        gold_answer = "Answer: " + gold_answer
        context = sys_prompt + question + "\n\n"

        mi_incremental = []
        cumulative_prompt = context
        for i, step in enumerate(steps):
            h_before = self.entropy_bits_exact(cumulative_prompt, gold_answer)
            cumulative_prompt += step+"\n"
            h_after = self.entropy_bits_exact(cumulative_prompt, gold_answer)
            incremental_mi = h_before - h_after
            mi_incremental.append(incremental_mi)
        return mi_incremental
    
    # Streaming versions for memory-efficient processing
    def gsm8k_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: int | None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        # ds = ds.select(range(start, start + take)) if take else ds
        fin = len(ds)
        ds = ds.select(range(start, fin))
        
        for sample in tqdm(ds, desc="Building GSM8K MI reward-dataset"):
            q_txt   = sample["question"]
            g_sol   = sample["answer"]
            lines, gold_ans = [], None
            for ln in g_sol.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                m = self._ANSWER_RE.match(ln)
                if m:
                    gold_ans = _sanitize_enhanced(m.group(1))
                    break
                lines.append(ln)
            if gold_ans is None:
                raise ValueError("gold answer not found for sample")
            steps = [f"Step {i+1}: {t}" for i, t in enumerate(lines)]

            mi = self.compute_step_mi(q_txt, steps, gold_ans)
            mi_filtered = [round(max(m, 0), 4) for m in mi]

            entry = {
                    "question":      q_txt,
                    "completion":    steps,
                    "mi_rewards":   mi,
                    "mi_filtered":   mi_filtered,
                    "gold_answer":   gold_ans,
                }
            yield entry

    def math_reward_dataset_streaming(self, *, split: str = "train", start: int = 0, take: int | None):
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')   # 소수점·수식 내부 마침표 무시
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        # ds = ds.select(range(start, start + take)) if take else ds
        fin = len(ds)
        ds = ds.select(range(start, fin))

        for sample in tqdm(ds, desc="Building MATH MI reward-dataset"):
            full_sol   = sample["solution"]
            boxed_content = _extract_boxed_answer(full_sol)
            gold_ans = _sanitize_enhanced(boxed_content) if boxed_content else None
            if gold_ans is None:
                lines = [line.strip() for line in full_sol.splitlines() if line.strip()]
                for line in reversed(lines):
                    if re.search(r'[\d\-+*/()=]', line):
                        gold_ans = _sanitize_enhanced(line)
                        break
            
            sol_wo_box = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', full_sol)
            raw_steps = [s.strip() for s in sent_split.split(sol_wo_box) if s.strip()]
            steps = [f"Step {i+1}: {s}" for i, s in enumerate(raw_steps)]

            mi = self.compute_step_mi(sample["problem"], steps, gold_ans)
            mi_filtered = [round(max(m, 0), 4) for m in mi]

            entry = {
                "question":      sample["problem"],
                "completion":    steps,
                "mi_rewards":   mi,
                "mi_filtered":   mi_filtered,
                "gold_answer":   gold_ans,
            }
            yield entry

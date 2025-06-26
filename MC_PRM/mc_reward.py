import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from typing import List, Optional
from tqdm import tqdm
import random

# Project‑level helpers 
from utils import _sanitize, _numeric_equiv, _strip_markup, _to_float, system_prompt
from config import PRMConfig

class MCReward:
    STEP_PATTERN = re.compile(
    r"""^[\s>#*\-]*          # optional markdown/bullet symbols
        Step\s*              # word 'Step' (case-insensitive)
        (\d+)                # capture step number
        \s*[:.\-]            # separator (: . or -)
    """,
    re.IGNORECASE | re.VERBOSE,
    )
    ANSWER_PATTERN = re.compile(
        r"""^[\s>#*\-]*          # optional markdown/bullet symbols
            Answer               # word 'Answer'
            \s*[:.\-]\s*         # separator
            (.+?)\s*$            # capture everything after
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    ## Masked rewards ##
    OP_TOKENS = ["add", "plus", "sum", "subtract", "minus",
             "multiply", "times", "product", "divide", "quotient"]
    _MASK_PATTERN = re.compile(
        r"""
        (?:
        # {ops_pattern}|                # operator patterns
            \b\d+(?:\.\d+)?\b         # integers / decimals
          | \b\d+/\d+\b                 # simple fractions
        #   | \b[a-zA-Z]\b                 # single‑letter variables
        )
        """,
        re.VERBOSE,
    )

    def __init__(self, config: "PRMConfig", model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to generate one or more step-by-step solutions for a given question.
    def generate_solutions(self, question: str, sys_prompt: str, num_solutions: int):
        prompt = f"{sys_prompt}\n\n{question}\n"  # Prompt the model to start the step-by-step solution
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        # Generate multiple solutions via sampling
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            num_return_sequences=num_solutions,
            temperature=0.8,         # sampling temperature for diversity (adjust as needed)
            top_p=0.8,               # top-p sampling for diversity
            pad_token_id=self.tokenizer.eos_token_id  # pad token ID to avoid warning for some models
        )
        solutions = []
        prompt_len = input_ids.shape[-1]
        for i in range(num_solutions):
            # Each output is the concatenation of the prompt and the generated completion.
            generated_ids = outputs[i]
            # Extract only the newly generated tokens (skip the prompt tokens).
            gen_ids = generated_ids[prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            solutions.append(text)
            # print(f"{i}-th Sampled Solutions:",text)
        return solutions
    
    def gsm8k_solutions(self, question: str, gold_solution: str):
        # 1. Split lines *before* the final answer marker (#### …)
        lines: List[str] = []
        gold_answer: str = ""
        _ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")

        for raw_ln in gold_solution.splitlines():
            ln = raw_ln.strip()
            if not ln:
                continue  # skip empty
            ans_match = _ANSWER_RE.match(ln)
            if ans_match:
                gold_answer = ans_match.group(1).strip()
                break  # everything after #### is ignored
            lines.append(ln)

        if not gold_answer:
            raise ValueError("Could not find final answer marker '#### <answer>' in gold_solution.")

        # 2. Prefix each explanatory line with "Step i:"
        solution_steps = [f"Step {i + 1}: {txt}" for i, txt in enumerate(lines)]
        return {
            "question": question,
            "solution": solution_steps,
            "gold_answer": gold_answer,
        }

    # Function to parse a solution text into steps and final answer.
    def _extract_answer(self, text: str) -> Optional[str]:
        """Try multiple heuristics / regexes to pull out an answer string."""
        # Primary regex (robust to Answer:, Answer ‑, etc.)
        match = self.ANSWER_PATTERN.search(text)
        if match:
            return _sanitize(match.group(1))
        
        # Fallback 1: last non‑empty line if it looks simple / numeric
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            candidate = lines[-1]
            if re.search(r"\d", candidate):  # contains digit
                return _sanitize(candidate)

        # Fallback 2: look for last line that starts with 'Answer'
        for line in reversed(text.splitlines()):
            if line.strip().lower().startswith("answer"):
                return _sanitize(line.split("Answer", 1)[-1])
        
        return None

    def parse_solution(self, solution_text: str):
        """Split each step to start with 'Step X:' and the answer to start with 'Answer:'."""
        steps = []
        # Split by lines to identify steps and answer
        for line in solution_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if self.STEP_PATTERN.match(line):
                cleaned = re.sub(r'^[\s>#*\-]+', '', line)
                steps.append(cleaned)
            answer = self._extract_answer(solution_text)
        return steps, answer
    
    # Function to estimate intermediate rewards for each step via rollouts.
    def compute_step_rewards(self, question, sys_prompt, steps, gold_answer):
        """
        For each prefix ending at a given step in 'steps', generate rollouts and compute the reward 
        (fraction of rollouts ending in the correct answer). Returns a list of reward values corresponding to each step.
        """
        rewards = []
        total_steps = len(steps)

        # Pre‑encode static prefix (sys_prompt + question) once for efficiency
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        base_ids = self.tokenizer.encode(base_prompt, return_tensors="pt").to(self.device)

        for i in range(total_steps):
            prefix_tokens = self.tokenizer.encode("\n".join(steps[: i + 1]) + "\n", return_tensors="pt").to(self.device) # steps up to current step i (0-indexed)
            # Decide how to prompt the next part:
            if i < total_steps - 1:
                next_label = f"Step {i + 2}:"
            else:
                next_label = "Answer:"
            cont_ids = self.tokenizer.encode(next_label, return_tensors="pt").to(self.device)
            # Build full prefix ids (avoid Python concat inefficiency by cat)
            prefix_ids = torch.cat([base_ids, prefix_tokens, cont_ids], dim=-1)
            rollout_outputs = self.model.generate(
                prefix_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_return_sequences=self.config.num_rollouts,
                temperature=0.8,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
            new_token_start = prefix_ids.shape[-1] 
            # Check each rollout's final answer against the gold answer
            correct_count = 0
            for idx, seq in enumerate(rollout_outputs):
                completion = self.tokenizer.decode(seq[new_token_start:], skip_special_tokens=True)
                pred_answer = self._extract_answer(completion)
                # print(f"[{i+1}-th Step, {idx}-th Original Rollout]", completion, "Pred Answer", pred_answer)
                if pred_answer is not None and _numeric_equiv(pred_answer, gold_answer):
                    correct_count += 1
            reward = correct_count / float(self.config.num_rollouts)
            rewards.append(reward)
        return rewards
    
    # Masked solution paths
    def model_masking(self, text: str, *, max_new_tokens: int = 64) -> str:
        prompt = "In the sentence below, mask any word or expression that seems crucial for solving the math step. This may include key numbers, variables, or action words (like operations), but you should decide what matters. Replace each important item with '[MASKED]'. Keep everything else unchanged. Return ONE line.\n\nSentence: \"{sent}\"\nRewritten:".format(sent=text)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        out_ids   = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0, top_p=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out_ids[0][input_ids.shape[-1]:],
                                     skip_special_tokens=True).strip()

    def perturbed_step_rewards(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str, use_llm: bool = True) -> List[float]:
        """Compute MC correctness rates *after masking* the current step.
        Each step `i` is replaced with a *perturbed* version where important
        tokens (numbers, fractions, single‑letter variables) are substituted by
        the literal string ``[MASKED]``. All preceding steps remain intact.
        """
        ptb_rewards: List[float] = []
        total_steps = len(steps)
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        base_ids = self.tokenizer.encode(base_prompt, return_tensors="pt").to(self.device)

        for i in range(total_steps):
            # 1. Perturb *only* step i
            orig_step = steps[i] 
            step_match = re.match(r"^[\s>#*\-]*Step\s*\d+\s*[:.\-]\s*", orig_step, flags=re.I)
            prefix = step_match.group(0) if step_match else ""
            # ② 나머지 부분(body)만 마스킹
            body   = steps[i][len(prefix):]                       # 접두사 뒷부분
            if use_llm:
                masked_body = self.model_masking(body)
            else:
                masked_body = self._MASK_PATTERN.sub("[MASKED]", body)
            # ③ 접두사 + 마스킹된 body
            masked_step = prefix + masked_body    
            ptb_prefix_steps = steps[:i] + [masked_step]
            # print("perturbed step:", ptb_prefix_steps)

            prefix_tokens = self.tokenizer.encode("\n".join(ptb_prefix_steps) + "\n", return_tensors="pt").to(self.device)
            next_label = f"Step {i + 2}:" if i < total_steps - 1 else "Answer:"
            cont_ids = self.tokenizer.encode(next_label, return_tensors="pt").to(self.device)
            prefix_ids = torch.cat([base_ids, prefix_tokens, cont_ids], dim=-1)

            rollout_outputs = self.model.generate(
                prefix_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_return_sequences=self.config.num_rollouts,
                temperature=0.8,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_token_start = prefix_ids.shape[-1]
            correct_count = 0
            for idx, seq in enumerate(rollout_outputs):
                completion = self.tokenizer.decode(seq[new_token_start:], skip_special_tokens=True)
                pred_answer = self._extract_answer(completion)
                # print(f"[{i+1}-th Step, {idx}-th Rollout]", completion, "Pred Answer", pred_answer)
                if pred_answer is not None and _numeric_equiv(pred_answer, gold_answer):
                    correct_count += 1
            ptb_rewards.append(correct_count / float(self.config.num_rollouts))
        return ptb_rewards

    # Build datasets based on input datas
    def build_datasets(self, problems: List):
        dataset = []  # will hold the output list of dicts
        for problem in problems:
            question = problem["question"]
            # gold_answer = problem["gold_answer"]
            gold_answer = _sanitize(problem["gold_answer"])
            # Generate one or more solutions for this question
            sample_prompt = system_prompt("sample")
            rollout_prompt = system_prompt("rollout")
            solutions = self.generate_solutions(question, sys_prompt=sample_prompt, num_solutions=self.config.samples_per_question)
            
            for sol_text in solutions:
                steps, answer = self.parse_solution(sol_text)
                # print("Parsed solution:", steps, answer)
                if answer is None: # If no answer was found in the solution (edge case), skip this solution
                    continue
                # 2. Compute *original* & *perturbed* per‑step rewards
                # ----------------------------------------------------------
                ori_rewards = self.compute_step_rewards(
                    question=question,
                    sys_prompt=rollout_prompt,
                    steps=steps,
                    gold_answer=gold_answer,
                )
                ptb_rewards = self.perturbed_step_rewards(
                    question=question,
                    sys_prompt=rollout_prompt,
                    steps=steps,
                    gold_answer=gold_answer,
                )
                # Align lengths (robustness)
                if len(ptb_rewards) != len(ori_rewards):
                    ptb_rewards = ptb_rewards[: len(ori_rewards)]
                # contributions = [max(0, o - p) for o, p in zip(ori_rewards, ptb_rewards)]
                contributions = [o - p for o, p in zip(ori_rewards, ptb_rewards)]
                entry = {
                    "question": question,
                    "completion": steps,          # list[str] (Step i: ...)
                    "ori_rewards": ori_rewards,    # list[float]
                    "ptb_rewards": ptb_rewards,    # list[float]
                    "contributions": contributions,  # ori − ptb
                    "answer": answer,
                    "gold_answer": gold_answer,
                }
                dataset.append(entry)
        return dataset
    
    # Build datasets based on input datas
    def build_datasets_gsm8k(self, split: Optional[str] = None):
        dataset = []  # will hold the output list of dicts
        rollout_prompt = system_prompt("rollout")
        ds_full = load_dataset("openai/gsm8k", "main")[split]
        ds_split = ds_full.shuffle(seed=self.config.seed)
        problems  = ds_split.select(range(100, 300))

        for problem in tqdm(problems):
            parsed = self.gsm8k_solutions(problem["question"], problem["answer"])
            question = parsed["question"]
            steps = parsed["solution"]
            gold_answer = _sanitize(parsed["gold_answer"])
            # print("Parsed:", question, "\n", steps, "\nGold:", gold_answer)
            
            ori_rewards = self.compute_step_rewards(question, rollout_prompt, steps, gold_answer)
            ptb_rewards = self.perturbed_step_rewards(question, rollout_prompt, steps, gold_answer, self.config.use_llm)
            # print("original rewards:", ori_rewards)
            # print("perturbed rewards:", ptb_rewards)
            # Align lengths (robustness)
            if len(ptb_rewards) != len(ori_rewards):
                ptb_rewards = ptb_rewards[: len(ori_rewards)]
            contributions = [round(o - p, 4) for o, p in zip(ori_rewards, ptb_rewards)]
            # print("contributions:", contributions)

            entry = {
                "question": question,
                "completion": steps,          # list[str] (Step i: ...)
                "ori_rewards": ori_rewards,    # list[float]
                "ptb_rewards": ptb_rewards,    # list[float]
                "contributions": contributions,  # ori − ptb
                "answer": gold_answer,
                "gold_answer": gold_answer,
            }
            dataset.append(entry)
        return dataset

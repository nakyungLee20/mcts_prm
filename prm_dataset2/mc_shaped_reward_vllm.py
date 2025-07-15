import re
import math
from typing import List, Optional
from tqdm import tqdm
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Project-level helpers
from config import PRMConfig
from utils import _sanitize_enhanced, _numeric_equiv_enhanced, _extract_boxed_answer, system_prompt

class MCRewardShapedVLLM:
    ANSWER_PATTERN = re.compile(
        r"""^[\s>#*\-]*          # optional markdown/bullet symbols
            Answer               # word 'Answer'
            \s*[:.\-]\s*         # separator
            (.+?)\s*$            # capture everything after
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    _ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")
    _MASK_PATTERN = re.compile(
        r"""
        (?:
            \b\d+(?:\.\d+)?\b         # integers / decimals
          | \b\d+/\d+\b                 # simple fractions
        )
        """,
        re.VERBOSE,
    )
    
    def __init__(self, config: "PRMConfig", model_name: str = "Qwen/QwQ-32B"):
        self.config = config
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",  # 메모리 절약
            gpu_memory_utilization=0.9,  # GPU 메모리 활용도
            max_model_len=4096,  # 전체 시퀀스 최대 길이 (prompt + generation)
            quantization="bitsandbytes",
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # Sampling parameters for Qwen/QwQ-32B
        self.rollout_params = SamplingParams(
            temperature=0.1,        # 낮은 temperature로 일관된 수학 문제 해결
            top_p=0.9,              # 높은 top_p로 품질 유지
            max_tokens=self.config.max_new_tokens, # 생성할 최대 토큰 수 (prompt 이후)
            n=self.config.num_rollouts,
            repetition_penalty=1.1, # 반복 방지 강화
            stop=["\n\n", "Therefore", "Thus", "Hence", "So", "Finally", "In conclusion"], # Answer 이후 추가 생성 방지
        )
        
        self.masking_params = SamplingParams(
            temperature=0.1,        # 마스킹도 일관성 있게
            top_p=0.8,              # 적당한 다양성
            max_tokens=64,          # 마스킹은 짧게
            n=self.config.num_rollouts,
            repetition_penalty=1.1, # 약간의 반복 방지
        )
        
        print(f"vLLM model loaded: {model_name}")

    def _clean_generated_text(self, text: str) -> str:
        """생성된 텍스트에서 Answer 이후의 불필요한 내용 제거하되 Answer는 보존"""
        # 여러 줄에 걸친 Answer 패턴 처리
        answer_patterns = [
            r"(Answer:\s*[^\n]*(?:\n[^\n]*)*)",  # Answer: 다음 한 줄 또는 여러 줄
            r"(Answer\s*[^\n]*(?:\n[^\n]*)*)",   # Answer 다음 한 줄 또는 여러 줄
            r"(####\s*[^\n]*(?:\n[^\n]*)*)",     # #### 다음 한 줄 또는 여러 줄
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer_part = match.group(1)
                # Answer 부분에서 불필요한 후속 내용 제거
                lines = answer_part.split('\n')
                cleaned_lines = []
                answer_found = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Answer 패턴이 발견되면 포함
                    if re.search(r'answer|####', line, re.IGNORECASE):
                        answer_found = True
                        cleaned_lines.append(line)
                    elif answer_found:
                        # Answer 이후의 줄이 숫자나 수식이면 포함 (정답의 일부일 수 있음)
                        if re.search(r'[\d\-+*/()=]', line):
                            cleaned_lines.append(line)
                        else:
                            # 설명적인 텍스트는 제거
                            break
                
                if cleaned_lines:
                    return '\n'.join(cleaned_lines).strip()
        
        # Answer가 없는 경우 원본 반환
        return text.strip()

    def _test_text_cleaning(self):
        """텍스트 정리 기능 테스트"""
        test_cases = [
            # 정상적인 경우
            ("Step 1: Calculate 2+3\nStep 2: 2+3=5\nAnswer: 5\nTherefore, the answer is 5.", 
             "Step 1: Calculate 2+3\nStep 2: 2+3=5\nAnswer: 5"),
            
            # 여러 줄 답변
            ("Step 1: Solve equation\nAnswer: x = 5\ny = 10\nTherefore, x=5 and y=10.", 
             "Step 1: Solve equation\nAnswer: x = 5\ny = 10"),
            
            # #### 패턴
            ("Step 1: Find value\n#### 42\nThis is the final answer.", 
             "Step 1: Find value\n#### 42"),
            
            # 설명적인 텍스트 제거
            ("Answer: 15\nThis means the result is 15. Let me explain why...", 
             "Answer: 15"),
        ]
        
        for input_text, expected in test_cases:
            result = self._clean_generated_text(input_text)
            print(f"Input: {input_text}")
            print(f"Expected: {expected}")
            print(f"Result: {result}")
            print(f"Match: {result == expected}")
            print("-" * 50)

    def _extract_answer(self, text: str) -> Optional[str]:
        """Try multiple heuristics / regexes to pull out an answer string."""
        # 먼저 텍스트 정리
        cleaned_text = self._clean_generated_text(text)
        
        match = self.ANSWER_PATTERN.search(cleaned_text)
        if match:
            return _sanitize_enhanced(match.group(1))
        lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
        if lines:
            candidate = lines[-1]
            if re.search(r"\d", candidate):  # contains digit
                return _sanitize_enhanced(candidate)
        for line in reversed(cleaned_text.splitlines()):
            if line.strip().lower().startswith("answer"):
                return _sanitize_enhanced(line.split("Answer", 1)[-1])
        return None

    def compute_step_rewards_batch(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str) -> List[float]:
        """최적화된 배치 처리로 모든 step의 rewards를 한 번에 계산"""
        rewards = []
        total_steps = len(steps)
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        
        # 더 큰 배치로 처리 (메모리 허용 범위에서)
        batch_size = min(32, total_steps)  # 배치 크기 조정
        
        for batch_start in range(0, total_steps, batch_size):
            batch_end = min(batch_start + batch_size, total_steps)
            prompts = []
            step_indices = []
            
            for i in range(batch_start, batch_end):
                prefix_steps = "\n".join(steps[:i + 1]) + "\n"
                if i < total_steps - 1:
                    next_label = f"Step {i + 2}:"
                else:
                    next_label = "Answer: "
                
                full_prompt = base_prompt + prefix_steps + next_label
                prompts.append(full_prompt)
                step_indices.append(i)
            
            # 배치 처리
            outputs = self.llm.generate(prompts, self.rollout_params)

            for i, output in enumerate(outputs):
                step_idx = step_indices[i]
                correct_count = 0
                # num_rollouts개의 결과 확인
                for completion in output.outputs:
                    # print(f"Completion: {completion.text}")  # 디버깅용
                    pred_answer = self._extract_answer(completion.text)
                    if pred_answer is not None and _numeric_equiv_enhanced(pred_answer, gold_answer):
                        correct_count += 1
                
                reward = correct_count / float(self.config.num_rollouts)
                rewards.append(reward)
        
        return rewards

    def model_masking_batch(self, texts: List[str]) -> List[str]:
        prompts = []
        for text in texts:
            prompt = f"""In the sentence below, mask any word or expression that seems crucial for solving the math step. This may include key numbers, variables, or action words (like operations), but you should decide what matters. Replace each important item with '[MASKED]'. Keep everything else unchanged. Return ONE line.

Sentence: "{text}"
Rewritten:"""
            prompts.append(prompt)
        
        outputs = self.llm.generate(prompts, self.masking_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def perturb_step_rewards_batch(self, question: str, sys_prompt: str, steps: List[str], gold_answer: str, use_llm: bool = True) -> List[float]:
        ptb_rewards = []
        total_steps = len(steps)
        base_prompt = f"{sys_prompt}\n\nProblem: {question}\n"
        
        all_prompts = []
        step_indices = []
        for i in range(total_steps):
            orig_step = steps[i]    # Step i 마스킹
            step_match = re.match(r"^[\s>#*\-]*Step\s*\d+\s*[:.\-]\s*", orig_step, flags=re.I)
            prefix = step_match.group(0) if step_match else ""
            body = steps[i][len(prefix):]
            
            if use_llm:
                masked_body = self.model_masking_batch([body])[0]
            else:
                masked_body = self._MASK_PATTERN.sub("[MASKED]", body)
            
            masked_step = prefix + masked_body
            ptb_prefix_steps = steps[:i] + [masked_step]
            
            prefix_steps = "\n".join(ptb_prefix_steps) + "\n"
            next_label = f"Step {i + 2}:" if i < total_steps - 1 else "Answer:"
            full_prompt = base_prompt + prefix_steps + next_label
            
            all_prompts.append(full_prompt)
            step_indices.append(i)
        
        outputs = self.llm.generate(all_prompts, self.rollout_params)
        
        for i, output in enumerate(outputs):
            step_idx = step_indices[i]
            correct_count = 0
            for completion in output.outputs:
                print(f"Perturbed Completion: {completion.text}")
                pred_answer = self._extract_answer(completion.text)
                if pred_answer is not None and _numeric_equiv_enhanced(pred_answer, gold_answer):
                    correct_count += 1
            
            reward = correct_count / float(self.config.num_rollouts)
            ptb_rewards.append(reward)
        return ptb_rewards

    # def surprisal_bits_batch(self, prompts: List[str], target: str) -> List[float]:
    #     LOG2E = 1.0 / math.log(2)
    #     full_seqs = [p + target for p in prompts]
    #     pref_lens = [len(self.tokenizer(p)["input_ids"]) for p in prompts]
        
    #     # vLLM에서 logprobs 계산
    #     outs = self.llm.generate(full_seqs, self.logprob_params)
    #     bits_list = []
        
    #     for pref_len, out in zip(pref_lens, outs):
    #         if hasattr(out, 'prompt_logprobs') and out.prompt_logprobs:
    #             # target 부분의 logprobs (prompt 이후)
    #             target_logprobs = out.prompt_logprobs[pref_len:]
    #             if target_logprobs:
    #                 surprisals = [-lp * LOG2E for lp in target_logprobs]
    #                 # 평균 surprisal을 entropy 근사값으로 사용
    #                 bits_list.append(sum(surprisals) / len(surprisals))
    #             else:
    #                 bits_list.append(0.0)
    #         else:
    #             # Fallback: 고정값 사용
    #             bits_list.append(0.5)
                
    #     return bits_list
    
    # def compute_step_mi_batch(self, question: str, steps: list[str], gold: str):
    #     sys_prompt = """Solve the given problem with step by step reasoning in the format of "Step k: <k-th rationale>" and write final answer in the format of "Answer: <answer>".\nProblem: """
    #     question = re.sub(r' +', ' ', question)
    #     gold = "Answer: " + gold.strip()
    #     ctx  = sys_prompt + question + "\n\n"

    #     mi_inc = []
    #     cumul_prompts = [ctx]
    #     # prompt 미리 누적해서 생성
    #     for st in steps:
    #         cumul_prompts.append(cumul_prompts[-1] + st + "\n")
    #     # 한번에 배치로 계산
    #     surprisal_bits = self.surprisal_bits_batch(cumul_prompts, gold)
    #     # MI 증분 = H_before − H_after
    #     mi_incremental = [round(a - b, 5) for a, b in zip(surprisal_bits[:-1], surprisal_bits[1:])]
    #     return mi_incremental
        
    def compute_step_mi(self, question: str, steps: List[str], gold_answer: str) -> List[float]:
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

    def entropy_bits_exact(self, prompt: str, target: str) -> float:
        """vLLM을 사용한 entropy 계산"""
        try:
            # vLLM 0.9.1에서 내부 모델에 접근
            model_runner = self.llm.llm_engine.driver_worker.model_runner
            model = model_runner.model  # 실제 PyTorch 모델
            
            LOG2E = 1 / math.log(2)
            full = prompt + target
            inputs = self.tokenizer(full, return_tensors="pt").to(self.device)
            Lp = len(self.tokenizer(prompt)["input_ids"])

            with torch.no_grad():
                logits = model(**inputs).logits.float()

            probs = logits.softmax(-1)
            token_H = -(probs * probs.log()).sum(-1) * LOG2E

            mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
            mask[:, Lp:] = True
            return token_H[mask].sum().item() / mask.sum().item()
            
        except Exception as e:
            print(f"Entropy calculation failed: {e}")
            # Fallback: 근사값 반환
            target_tokens = len(self.tokenizer(target)["input_ids"])
            return 2.0 + target_tokens * 0.1

    def gsm8k_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: int | None):
        """최적화된 streaming dataset 생성"""
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        for sample in tqdm(ds, desc="Building GSM-8K reward-dataset (vLLM optimized)"):
            q_txt = sample["question"]
            g_sol = sample["answer"]
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

            # Batch processing으로 모든 rewards 계산
            ori = self.compute_step_rewards_batch(q_txt, system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(q_txt, system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(q_txt, steps, gold_ans)

            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]
            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]
            print("ori: ", ori)
            print("ptb: ", ptb)
            print("mi: ", mi)
            print("naive: ", naive)
            print("contrib: ", contrib)

            entry = {
                "question": q_txt,
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "mi_rewards": mi,
                "naive_rewards": naive,
                "gold_answer": gold_ans,
            }
            yield entry

    def math_reward_dataset_vllm(self, *, split: str = "train", start: int = 0, take: int | None):
        """최적화된 MATH dataset streaming"""
        sent_split = re.compile(r'\.(?!\d)(?=\s|$)')
        ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        if take is not None:
            ds = ds.select(range(start, start+take))
        else:
            ds = ds.select(range(start, len(ds)))

        for sample in tqdm(ds, desc="Building MATH reward-dataset (vLLM optimized)"):
            full_sol = sample["solution"]

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

            # Batch processing
            ori = self.compute_step_rewards_batch(sample["problem"], system_prompt("rollout"), steps, gold_ans)
            ptb = self.perturb_step_rewards_batch(sample["problem"], system_prompt("rollout"), steps, gold_ans, self.config.use_llm)
            mi = self.compute_step_mi(sample["problem"], steps, gold_ans)

            contrib = [round(o - p, 4) for o, p in zip(ori, ptb)]
            naive = [round(o + max(m, 0), 4) for o, m in zip(ori, mi)]

            entry = {
                "question": sample["problem"],
                "completion": steps,
                "ori_rewards": ori,
                "ptb_rewards": ptb,
                "contributions": contrib,
                "mi_rewards": mi,
                "naive_rewards": naive,
                "gold_answer": gold_ans,
            }
            yield entry

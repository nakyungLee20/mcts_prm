import random, re
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class StepwisePRMDataset(Dataset):
    """mcr rewards가 반환한 entries(list[dict])를 (input_ids, scalar_reward) 샘플들로 변환한다.
    한 entry = {question, completion[steps], rewards[float], …} →  (Problem + Step1, r1), (Problem + Step1 \nStep2, r2) …"""
    def __init__(
        self,
        entries: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        reward_type: str = "cmi",
        *,
        cache_encodings: bool = True,
        preprocess: bool = True,
    ):
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.reward_type = reward_type
        self.cache       = {} if cache_encodings else None
        self.samples: List[Tuple[str, float]] = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        print(f"reward_type: {self.reward_type}")
        for e in entries:
            q_txt   = e["question"]
            steps   = e["completion"]
            ans = e["gold_answer"]
            o_rewards = e["ori_rewards"]
            assert len(steps) == len(o_rewards)

            if self.reward_type == "contri":
                rewards = e["contributions"]
            elif self.reward_type == "mi":
                rewards = e["mi_filtered"]
            elif self.reward_type == "ori":
                rewards = o_rewards
            elif self.reward_type == "cmi":
                rewards = [c + m for c, m in zip(e["contributions"], e["mi_filtered"])]
            elif self.reward_type == "orimi":
                rewards = [o + m for o, m in zip(o_rewards, e["mi_filtered"])]
            else:
                rewards = o_rewards

            prefix_lines = [f"Problem: {q_txt}"]
            for step_txt, r in zip(steps, rewards):
                prefix_lines.append(step_txt)
                full_txt = "\n".join(prefix_lines)
                if preprocess:
                    full_txt = self._clean(full_txt)
                self.samples.append((full_txt, float(r)))   # (text, reward)

    # --------------------------------------------------------------------- utils
    @staticmethod
    def _clean(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # --------------------------------------------------------------------- dunder
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, reward = self.samples[idx]

        if self.cache is not None and text in self.cache:
            ids = self.cache[text]
        else:
            ids = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            if self.cache is not None:
                self.cache[text] = ids

        return ids, torch.tensor(reward, dtype=torch.float32)

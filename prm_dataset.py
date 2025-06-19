import torch
from transformers import PreTrainedTokenizer
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from collections import Counter
import random
import string
import re

class PRMDataset(Dataset):
    """
    • 최소 설정(토큰화만)으로도 바로 학습 가능  
    • 옵션으로 whitespace-정규화 / 소문자화 / 간단한 텍스트 증강 / 인코딩 캐시 지원
    """
    def __init__(
        self,
        solutions: List[str],
        rewards  : List[float],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        *,
        preprocess: bool = True,
        augment: bool = False,
        augment_prob: float = 0.1,
        cache_encodings: bool = True,
    ):
        assert len(solutions) == len(rewards)
        self.sol, self.r = solutions, rewards                    # 원본 보존
        self.tok  = tokenizer
        self.max  = max_length
        self.preprocess = preprocess
        self.augment    = augment
        self.augment_prob = augment_prob
        self.cache = {} if cache_encodings else None

        # ─ preprocessing ────────────────────────────────────────────────
        self.proc = [self._clean(s) if preprocess else s for s in solutions]

        # ─ augmentation vocabulary (길이별 단어 집합) ────────────────────
        self.vocab_by_len = self._build_vocab() if augment else {}

    # ------------------------------------------------------------------ core
    def __len__(self): return len(self.sol)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        txt = self.proc[idx]
        if self.augment and random.random() < self.augment_prob:
            txt = self._augment(txt)

        if self.cache is not None and txt in self.cache:
            ids = self.cache[txt]
        else:
            ids = self.tok(
                txt,
                max_length=self.max,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)
            if self.cache is not None:
                self.cache[txt] = ids

        return ids, torch.tensor(self.r[idx], dtype=torch.float32)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _clean(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    def _build_vocab(self) -> Dict[int, List[str]]:
        cnt = Counter(w for s in self.proc for w in s.split())
        v = {}
        for w in cnt:
            v.setdefault(len(w), []).append(w)
        return v

    # － augmentation (세 가지만 간단히) －
    def _augment(self, txt: str) -> str:
        return random.choice([self._swap, self._delete_char, self._insert_char])(txt)

    def _swap(self, t: str) -> str:
        w = t.split(); n = len(w)
        if n < 2: return t
        i = random.randint(0, n-2)
        w[i], w[i+1] = w[i+1], w[i]
        return " ".join(w)

    def _delete_char(self, t: str) -> str:
        if len(t) == 0: return t
        i = random.randint(0, len(t)-1)
        return t[:i] + t[i+1:]

    def _insert_char(self, t: str) -> str:
        i = random.randint(0, len(t))
        c = random.choice(string.ascii_lowercase)
        return t[:i] + c + t[i:]

    # optional helpers --------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        rl = np.array(self.r)
        return {
            "n": len(self),
            "avg_r": rl.mean(),
            "std_r": rl.std(),
            "min_r": rl.min(),
            "max_r": rl.max(),
            "avg_len": np.mean([len(s) for s in self.proc]),
        }

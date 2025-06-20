import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import wandb
import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from prm_config import OmegaPRMConfig, logger
from prm_model import ProcessRewardModel
from prm_dataset import PRMDataset
from mcts_core import MCTS

class PRMTrainer:
    """
    ①  MCTS 로부터 (solution, reward) 쌍을 수집하고
    ②  Process-Reward Model(PRM)을 학습한다.
    """
    def __init__(self, mcts: MCTS, config: OmegaPRMConfig):
        self.mcts   = mcts
        self.cfg    = config
        self.device = mcts.device
        self.tok    = mcts.tokenizer

        # PRM 자체 초기화
        feat_dim = mcts.model.config.hidden_size        # LLM hidden size
        self.prm = ProcessRewardModel(feat_dim, self.cfg.hidden_size, output_size=1).to(self.device)
        self.opt = optim.AdamW(self.prm.parameters(), lr=self.cfg.learning_rate, weight_decay=0.01)
        # self.crit = nn.BCELoss()
        self.crit = nn.MSELoss()

        # Initialize wandb if enabled
        if self.cfg.use_wandb:
            wandb.init(project="omega-prm", name="prm-train", config=vars(self.cfg))
        # Create checkpoint directory
        Path(self.cfg.checkpoint_dir).mkdir(exist_ok=True)

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _encode_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        LLM 의 hidden state → [CLS-pooling] 식 임베딩.
        input_ids : [B, T]
        return    : [B, feat_dim]
        """
        # emb_layer = self.model.get_input_embeddings()          # shared embedding matrix
        # emb = (emb_layer(token_ids)).mean(dim=1) 
        out = self.mcts.model(input_ids=input_ids,
                              output_hidden_states=True,
                              return_dict=True)
        # 마지막 hidden-state의 0-번 토큰(CLS) 임베딩 사용
        return out.hidden_states[-1][:, 0, :]

    # ------------------------------------------------------ data preparation
    def build_dataset(
        self,
        questions: List[str],
        samples_per_q: int = 1,
        add_question: bool = True          # 프롬프트에 문제문 포함 여부
    ) -> Tuple[Dataset, List[Dict]]:
        """
        Step-wise 데이터셋을 만든다.
        반환: (torch Dataset, [entry…])  entry 는 질문 하나에 대한 원본 구조
        """
        texts, lbls = [], []
        structured  = []

        for q in tqdm(questions, desc="Collecting MCTS data"):
            paths = self.mcts.mcts_for_prm(q, samples=samples_per_q)[q]

            for path in paths:         # path = {"question", "completion", "rewards", …}
                steps   = path["completion"]
                rewards = path["rewards"]

                assert len(steps) == len(rewards)

                # step-wise 분해
                prefix_lines = [f"Problem: {q}"] if add_question else []
                for i in range(len(steps)):
                    prefix_lines.append(steps[i])
                    txt  = "\n".join(prefix_lines)          # 문제+현재까지 스텝
                    score = rewards[i]
                    texts.append(txt)
                    lbls.append(score)

                structured.append(path)     # 진단용

        if len(texts)==0:
            dummy = [{'question': 'What is (5+7)/2 - 3?', 'completion': ['Calculate the contents inside the parenthesis, 5+7 = 12.', 'Divide by 2, which is 12/2=6.', 'Subtract 3 from 6, 6-3=3.'], 'rewards': [0.7, 0.6,0.75], 'answer': '3.'}]
            return dummy, structured
        
        ds = PRMDataset(texts, lbls,
                        tokenizer=self.tok,
                        max_length=self.cfg.max_length)
        
        print("PRMDataset(size={}, avg_len={:.1f})".format(len(ds), sum(len(t.split()) for t in texts)/len(texts)))
        return ds, structured

    # ---------------------------------------------------------- train / valid
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.prm.train() if train else self.prm.eval()
        tot = 0.0
        for step, (ids, r) in enumerate(loader):
            ids, r = ids.to(self.device), r.to(self.device)
            with torch.set_grad_enabled(train):
                feats = self._encode_features(ids)
                out   = self.prm(feats)
                loss  = self.crit(out, r)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
                    self.opt.step()
            tot += loss.item()

            if self.cfg.use_wandb and train:
                wandb.log({
                    "batch_loss": loss.item(),
                    # "epoch"     : self.cur_epoch,   # train_prm 에서 설정
                    "step"      : step
                })
        return tot / len(loader)

    def train_prm(
        self,
        train_questions: List[str],
        val_questions  : List[str],
        num_epochs: int = 5,
    ) -> Dict[str, List[float]]:
        # 1) 데이터 수집
        train_ds, _ = self.build_dataset(train_questions)
        val_ds,   _ = self.build_dataset(val_questions)
        print("train ds:", train_ds)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size,shuffle=False, num_workers=self.cfg.num_workers)

        # 2) 학습 loop
        hist = {"train": [], "val": []}
        best = float("inf")
        for ep in range(num_epochs):
            tr = self._run_epoch(train_loader, train=True)
            vl = self._run_epoch(val_loader,   train=False)
            hist["train"].append(tr)
            hist["val"].append(vl)
            print(f"[EP {ep}] train {tr:.4f} | val {vl:.4f}")

            if self.cfg.use_wandb:
                wandb.log({
                    "epoch"     : ep,
                    "train_loss": tr,
                    "val_loss"  : vl,
                })
            if vl < best:
                best = vl
                self.save_checkpoint(ep, vl)
                # torch.save(self.prm.state_dict(), Path(self.cfg.checkpoint_dir) / "best_prm.pt")
        return hist

    # -------------------------------------------------------------- metrics
    def get_metrics(self) -> Dict[str, float]:
        return {
            "params": sum(p.numel() for p in self.prm.parameters()),
        }
    
    # -------------------------------------------------------------- save checkpoints
    def save_checkpoint(self, epoch: int, validation_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.prm.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'validation_loss': validation_loss,
            'config': self.cfg.__dict__
        }
        path = Path(self.cfg.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.prm.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return checkpoint['epoch'], checkpoint['validation_loss']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

import json
from pathlib import Path
from typing import Dict, List, Optional
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import math
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Project‑level helpers 
from prm_dataset import StepwisePRMDataset  
from prm_model import ProcessRewardModel  
from config import PRMConfig
# from prm_training.prm_dataset import StepwisePRMDataset  
# from prm_training.prm_model import ProcessRewardModel  
# from prm_training.config import PRMConfig

class PRMTrainer:
    """
    (1) entries(list[dict]) → StepwisePRMDataset
    (2) LLM encoder + PRM head fine-tuning
    """
    def __init__(self, cfg: PRMConfig, model, tokenizer):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # ----------------------------- Backbone model LLM (frozen or fine-tuned)
        self.tokenizer = tokenizer
        self.model  = model
        self.model.eval()       # LLM은 feature extractor로 freeze
        for p in self.model.parameters():
            p.requires_grad_(False)

        feat_dim = self.model.config.hidden_size
        self.prm = ProcessRewardModel(feat_dim, cfg=cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.prm.to(self.device)

        self.opt  = optim.AdamW(self.prm.parameters(), lr=cfg.learning_rate, weight_decay = cfg.weight_decay)
        self.crit = nn.MSELoss()
        # self.crit = nn.BCELoss()

        self.scheduler = None
        if cfg.lr_scheduler == "cosine":                   
            # total steps = (#batches per epoch) × epochs
            self.total_steps = math.ceil(cfg.epochs * cfg.dataset_size / cfg.batch_size)
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                progress = (step - cfg.warmup_steps) / max(1, self.total_steps - cfg.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        elif cfg.lr_scheduler == "linear":
            # Linear warmup + decay
            self.total_steps = math.ceil(cfg.epochs * cfg.dataset_size / cfg.batch_size)
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return step / max(1, cfg.warmup_steps)
                return max(0.0, (self.total_steps - step) / (self.total_steps - cfg.warmup_steps))
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        elif cfg.lr_scheduler == "step":
            # Step decay
            self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.5)

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)

        self.wandb_run = None
        if cfg.use_wandb:                                  # <-- config에 플래그
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,                 # e.g. "omega-prm"
                name=cfg.run_name,                         # e.g. "qwen7b-prm"
                config=vars(cfg),                          # 모든 하이퍼파라미터 로깅
            )

    # ----------------------------------------------------------------- features
    @torch.no_grad()
    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B,T] → [B, feat_dim] using 마지막 hidden state의 CLS-like 첫 토큰"""
        out = self.model(input_ids=ids, return_dict=True,output_hidden_states=True)
        features = out.hidden_states[-1][:, 0, :]     # CLS embedding
        return features.float()

    # ----------------------------------------------------------------- loop util
    def _run_epoch(self, loader: DataLoader, train: bool, epoch_idx: int) -> float:
        self.prm.train(train)
        total = 0.0
        num_batches = len(loader)
        
        for step, (ids, reward) in enumerate(loader):
            ids, reward = ids.to(self.device), reward.to(self.device)

            with torch.set_grad_enabled(train):
                feats  = self._encode(ids)
                pred   = self.prm(feats).squeeze(-1)
                loss   = self.crit(pred, reward)
                
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.prm.parameters(), self.cfg.grad_clip)
                    # Gradient accumulation (optional)
                    if hasattr(self.cfg, 'grad_accum_steps') and self.cfg.grad_accum_steps > 1:
                        if (step + 1) % self.cfg.grad_accum_steps == 0:
                            self.opt.step()
                            if self.scheduler: self.scheduler.step()
                    else:
                        self.opt.step()
                        if self.scheduler: self.scheduler.step()

            total += loss.item()

            # -------- minibatch logging --------
            if self.wandb_run and train:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch_idx + step / num_batches,
                    "lr": self.opt.param_groups[0]["lr"],
                    "grad_norm": sum(p.grad.data.norm(2).item() for p in self.prm.parameters() if p.grad is not None),
                    "pred_mean": pred.mean().item(),
                    "pred_std": pred.std().item(),
                    "reward_mean": reward.mean().item(),
                    "reward_std": reward.std().item(),
                })

        return total / len(loader)

    # ----------------------------------------------------------------- public
    def fit(self, train_loader, val_loader) -> Dict[str, List[float]]:
        self.cfg.dataset_size = len(train_loader) 

        history = {"train": [], "val": []}
        best_val, bad_epochs, patience = float("inf"), 0, 8  # patience 증가

        for ep in range(self.cfg.epochs):
            tr_loss = self._run_epoch(train_loader, train=True,  epoch_idx=ep)
            vl_loss = self._run_epoch(val_loader,   train=False, epoch_idx=ep)

            history["train"].append(tr_loss)
            history["val"].append(vl_loss)
            print(f"[Epoch {ep+1}/{self.cfg.epochs}] train={tr_loss:.4f}  val={vl_loss:.4f}")

            # -------- epoch logging --------
            if self.wandb_run:
                wandb.log({
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                    "epoch": ep,
                    "lr": self.opt.param_groups[0]["lr"],
                })

            # Save checkpoints
            if vl_loss < best_val:
                best_val = vl_loss
                bad_epochs = 0
                self._save_checkpoint("best_prm.pt", epoch=ep, val_loss=vl_loss)
                print(f"[Best] New best validation loss: {vl_loss:.4f}")
            else:
                bad_epochs += 1
                print(f"[Early-Stopping] No improvement for {bad_epochs}/{patience} epochs")
                if bad_epochs >= patience:
                    print(f"[Early-Stopping] Stopping training after {patience} epochs without improvement")
                    break
        
        self._save_checkpoint("last_prm.pt", epoch=self.cfg.epochs - 1, val_loss=vl_loss)
        return history
    
    # Checkpoint helpers
    def _save_checkpoint(self, filename: str, *, epoch: int, val_loss: float) -> None:
        path = self.ckpt_dir / filename
        save_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "prm_state": self.prm.state_dict(),
            "scheduler_state": (self.scheduler.state_dict() if self.scheduler else None),
            "optimizer_state": self.opt.state_dict(),
            "config": vars(self.cfg),              # hyper‑params for reproducibility
            "model_name_or_path": getattr(self.model, "name_or_path", None),
            "tokenizer_config": self.tokenizer.__dict__.get("init_kwargs", {}),
        }
        torch.save(save_dict, path)
        print(f"[CKPT] Saved ⇒ {path}")

    # Simple inference helper
    @torch.no_grad()
    def predict_reward(self, text: str) -> float:
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        feat = self._encode(ids)
        return float(torch.sigmoid(self.prm(feat)).item())

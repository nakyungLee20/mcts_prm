import json
from pathlib import Path
from typing import Dict, List, Optional
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

# Project‑level helpers 
from dataset import StepwisePRMDataset  # noqa: F401 – custom Dataset wrapper
from prm_model import ProcessRewardModel    # noqa: F401 – small MLP «PRM» head
from config import PRMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('omega_prm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        self.prm = ProcessRewardModel(feat_dim, hidden_size=cfg.hidden_size, output_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.prm.to(self.device)

        self.opt  = optim.AdamW(self.prm.parameters(), lr=cfg.learning_rate)
        # self.crit = nn.MSELoss()
        self.crit = nn.BCELoss()

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
        """
        input_ids [B,T] → [B, feat_dim] using 마지막 hidden state의 CLS-like 첫 토큰
        """
        out = self.model(
            input_ids=ids,
            return_dict=True,
            output_hidden_states=True,
        )
        return out.hidden_states[-1][:, 0, :]     # CLS embedding

    # ----------------------------------------------------------------- loop util
    def _run_epoch(self, loader: DataLoader, train: bool, epoch_idx: int) -> float:
        self.prm.train(train)
        total = 0.0
        for step, (ids, reward) in enumerate(loader):
            ids, reward = ids.to(self.device), reward.to(self.device)

            with torch.set_grad_enabled(train):
                feats  = self._encode(ids)
                pred   = self.prm(feats).squeeze(-1)
                loss   = self.crit(pred, reward)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
                    self.opt.step()

            total += loss.item()

            # -------- minibatch logging --------
            if self.wandb_run and train:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch_idx + step / len(loader),
                    "lr": self.opt.param_groups[0]["lr"],
                    "grad_norm": sum(p.grad.data.norm(2).item()
                                     for p in self.prm.parameters()
                                     if p.grad is not None),
                })

        return total / len(loader)

    # ----------------------------------------------------------------- public
    def fit(self, train_entries: List[dict], val_entries: List[dict]) -> Dict[str, List[float]]:
        train_ds = StepwisePRMDataset(train_entries, self.tokenizer, self.cfg.max_new_tokens, self.cfg.use_contri)
        val_ds   = StepwisePRMDataset(val_entries,   self.tokenizer, self.cfg.max_new_tokens, self.cfg.use_contri)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

        history = {"train": [], "val": []}
        best_val = float("inf")

        for ep in range(self.cfg.epochs):
            tr_loss = self._run_epoch(train_loader, train=True,  epoch_idx=ep)
            vl_loss = self._run_epoch(val_loader,   train=False, epoch_idx=ep)

            # -------- epoch logging --------
            if self.wandb_run:
                wandb.log({"train_loss": tr_loss,"val_loss": vl_loss,"epoch": ep})

            history["train"].append(tr_loss)
            history["val"].append(vl_loss)
            print(f"[Epoch {ep+1}/{self.cfg.epochs}] train={tr_loss:.4f}  val={vl_loss:.4f}")

            # 체크포인트 저장
            if vl_loss < best_val:
                best_val = vl_loss
                self._save_checkpoint("best_prm.pt", epoch=ep, val_loss=vl_loss)
        
        self._save_checkpoint("last_prm.pt", epoch=self.cfg.epochs - 1, val_loss=vl_loss)
        return history
    
    # ------------------------------------------------------------------
    # Checkpoint helpers
    def _save_checkpoint(self, filename: str, *, epoch: int, val_loss: float) -> None:
        path = self.ckpt_dir / filename
        save_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "prm_state": self.prm.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "config": vars(self.cfg),              # hyper‑params for reproducibility
            "model_name_or_path": getattr(self.model, "name_or_path", None),
            "tokenizer_config": self.tokenizer.__dict__.get("init_kwargs", {}),
        }
        torch.save(save_dict, path)
        print(f"[CKPT] Saved ⇒ {path}")

    # ------------------------------------------------------------------
    # Simple inference helper
    @torch.no_grad()
    def predict_reward(self, text: str) -> float:
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        feat = self._encode(ids)
        return float(torch.sigmoid(self.prm(feat)).item())

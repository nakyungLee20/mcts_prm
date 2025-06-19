from prm_config import OmegaPRMConfig
from mcts_core import MCTS
from prm_trainer import PRMTrainer

from torch.utils.data import DataLoader

# Simple toy example
cfg = OmegaPRMConfig(
        use_wandb=False,         # 예시이므로 off
        batch_size=8,
        num_workers=4,
    )
golden = {
    "What is (5+7)/2 - 3?": "3",
    "What is 2 + 2?": "4",
}

mcts = MCTS(cfg, golden)
trainer = PRMTrainer(mcts, cfg)

train_q = ["What is (5+7)/2 - 3?"]
val_q = ["What is 2 + 2?"]
tr_ds, tr_st = trainer.build_dataset(train_q)
print(tr_ds)
print("MCTS train print tree")
mcts.print_tree(mcts.root)

val_ds, val_st = trainer.build_dataset(val_q)
print(val_ds)
print("MCTS val print tree")
mcts.print_tree(mcts.root)

train_loader = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=4)

tr = trainer._run_epoch(train_loader, train=True)
vl = trainer._run_epoch(val_loader,   train=False)
print(f"[EP {1}] train {tr:.4f} | val {vl:.4f}")


# 완성 이후에 다시 돌려보기
# def main():
#     cfg = OmegaPRMConfig(
#         use_wandb=True,         # 예시이므로 off
#         batch_size=12,
#         num_workers=4,
#     )

#     golden = {
#         "What is 2 + 2?": "4",
#         "What is (5+7)/2 - 3?": "3",
#     }

#     # 1) MCTS 초기화
#     mcts = MCTS(cfg, golden)

#     # 2) PRM trainer
#     trainer = PRMTrainer(mcts, cfg)

#     # 3) 학습 파이프라인
#     train_q = ["What is 2 + 2?"]
#     val_q   = ["What is (5+7)/2 - 3?"]
#     trainer.train_prm(train_q, val_q, num_epochs=1)

#     print("MCTS print tree")
#     mcts.print_tree(mcts.root)

#     # (선택) MCTS 메트릭·결과 저장
#     mcts.export_results("results.json")
#     print("MCTS metrics:", mcts.get_metrics())
#     print("PRM metrics:",  trainer.get_metrics())

# if __name__ == "__main__":
#     main()

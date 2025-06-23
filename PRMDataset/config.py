class PRMConfig:
    """Configuration class for PRM hyperparameters and settings"""
    # MC config
    max_new_tokens: int = 384
    num_rollouts: int = 8
    reward_threshold: float = 0.5
    samples_per_question: int = 1
    use_llm: bool = True
    use_contri: bool = True
    # PRMTrainer config
    batch_size: int = 32
    learning_rate: float = 5e-4
    hidden_size: int = 256
    num_workers: int = 4
    epochs: int = 4
    # Misc config
    use_wandb: bool = True
    wandb_project: str = "mc_prm"
    run_name: str = "test_gsm8k_0623"
    checkpoint_dir: str = "./checkpoints/gsm8k"
    seed: int = 42
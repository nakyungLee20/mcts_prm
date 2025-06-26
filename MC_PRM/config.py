class PRMConfig:
    """Configuration class for PRM hyperparameters and settings"""
    # MC config
    max_new_tokens:         int = 384
    num_rollouts:           int = 8      
    reward_threshold:       float = 0.5
    samples_per_question:   int = 1
    use_llm:                bool = True  # Use llm for masking
    use_contri:             bool = True  # If true, use "contributions" as step rewards else use ori_rewards
    # PRM Model config
    hidden_size:        int = 512
    num_layers:         int = 3
    dropout:            float = 0.2
    # PRMTrainer config
    batch_size:         int = 12
    learning_rate:      float = 5e-4
    num_workers:        int = 4
    weight_decay:       float = 1e-2
    lr_scheduler:       str   = "cosine"
    dataset_size:       int = 0
    warmup_steps:       int   = 22
    grad_clip:          float = 1.0
    epochs:             int = 25
    # Misc config
    use_wandb:          bool = True
    wandb_project:      str = "mc_prm"
    run_name:           str = "test_gsm8k_100_contri_mse"
    checkpoint_dir:     str = "./checkpoints/gsm8k/contri_mse"
    seed:               int = 42
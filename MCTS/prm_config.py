import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizer
from collections import defaultdict, deque
import math
import logging
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
from tqdm import tqdm, trange
import wandb
import os
from pathlib import Path
from collections import deque
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import random
import string
import re

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

@dataclass
class OmegaPRMConfig:
    """Configuration class for OmegaPRM hyperparameters and settings"""
    # MCTS config
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"     # "meta-llama/Meta-Llama-3-8B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    search_limit: int = 5 # 100
    max_rollout_tokens: int = 384
    rollout_width: int = 5
    alpha: float = 0.5
    beta: float = 0.9
    L: int = 300 # 500
    cpuct: float = 0.125
    use_mc_reward: bool = True
    reward_threshold: float = 0.7
    # PRM config
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 256
    max_length: int = 256
    num_workers: int = 4
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"
    
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"[{device}] {message}", flush=True)


def steps_per_epoch(num_batches: int, accum: int) -> int:
    return max(1, (num_batches + accum - 1) // accum)

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = ["CFG", "Config"]

from .config import CFG, Config

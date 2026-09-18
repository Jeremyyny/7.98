"""Seed all random sources."""
from __future__ import annotations

import random

def set_seed(seed: int) -> None:
    # Data preparation and scoring can import utils without loading a GPU stack.
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

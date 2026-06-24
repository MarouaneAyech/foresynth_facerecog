"""Reproductibilité."""
import os, random
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

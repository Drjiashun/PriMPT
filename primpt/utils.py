from __future__ import annotations
import os
import random
import numpy as np
import torch
def build_amp_grad_scaler(use_cuda_amp: bool):
    """
    Build a CUDA AMP GradScaler with backward compatibility across PyTorch versions.

    Newer PyTorch versions recommend torch.amp.GradScaler("cuda", ...),
    while older versions only provide torch.cuda.amp.GradScaler(...).
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    return torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

def cuda_autocast_context(use_cuda_amp: bool):
    """
    Return a CUDA autocast context with backward compatibility across PyTorch versions.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_cuda_amp)
    return torch.cuda.amp.autocast(enabled=use_cuda_amp)

def seed_everything(seed: int = 90) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import random

import numpy as np
import torch


def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # noqa: NPY002 - intentionally seeds NumPy's global RNG.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        return batch


def make_groups(lengths: torch.Tensor) -> torch.Tensor:
    if lengths.dim() != 1 or lengths.is_floating_point():
        raise ValueError("Invalid lengths tensor")
    if lengths.size(0) == 0:
        return lengths.clone()
    offset = lengths.cumsum(dim=0)
    addition = torch.zeros(int(offset[-1]) + 1, dtype=offset.dtype, device=offset.device)
    addition = addition.scatter_add_(dim=0, index=offset, src=torch.ones_like(offset))
    return addition[:-1].cumsum(dim=0)


def make_pairs(lengths: torch.Tensor) -> torch.Tensor:
    repeat = lengths[make_groups(lengths)]
    first = make_groups(repeat)
    addition = torch.ones(first.size(0) + 1, dtype=first.dtype, device=first.device)
    addition.scatter_add_(dim=0, index=lengths.square().cumsum(dim=0), src=lengths)
    addition.scatter_add_(dim=0, index=repeat.cumsum(dim=0), src=-repeat)
    addition[0].add_(-1)
    second = addition[:-1].cumsum(dim=0)
    return torch.stack([first, second])

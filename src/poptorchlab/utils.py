import poptorch
import torch
import numpy as np


def count_parameters(model: torch.nn.Module, trainable_only=False):
    return sum([p.numel() for p in model.parameters() if not trainable_only or p.requires_grad])


def set_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def print_nan_count(tensor: torch.Tensor, title: str = ""):
    tensor += 0 * poptorch.ipu_print_tensor(torch.sum(torch.isnan(tensor)), f'{title}_nan_count')

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + self.scaling * lora_out


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_modules(model: nn.Module, target_suffixes: Iterable[str], rank: int, alpha: float, dropout: float):
    target_suffixes = tuple(target_suffixes)
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not module_name.endswith(target_suffixes):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    return replaced

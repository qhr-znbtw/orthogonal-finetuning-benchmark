
"""LoRA layer used by the ViT adaptation experiment."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Compute ``Wx + (alpha / rank) BAx`` while preserving the base layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank必须为正整数。")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scaling * self.lora_b(self.lora_a(inputs))

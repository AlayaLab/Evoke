

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

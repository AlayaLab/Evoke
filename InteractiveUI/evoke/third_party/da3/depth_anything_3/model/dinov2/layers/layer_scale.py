

from typing import Union
import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def extra_repr(self) -> str:
        return f"{self.dim}, init_values={self.init_values}, inplace={self.inplace}"



from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch


@dataclass
class Gaussians:


    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor


@dataclass
class Prediction:
    depth: np.ndarray
    is_metric: int
    sky: np.ndarray | None = None
    conf: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    processed_images: np.ndarray | None = None
    gaussians: Gaussians | None = None
    aux: dict[str, Any] = None
    scale_factor: Optional[float] = None



from typing import Tuple
import torch


def least_squares_scale_scalar(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:


    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.device != b.device:
        raise ValueError(f"Device mismatch: {a.device} vs {b.device}")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Tensors must be floating point type")


    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.dot(b.reshape(-1), b.reshape(-1)).clamp_min(eps)
    return num / den


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:


    return sky_prediction < threshold


def compute_alignment_mask(
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    depth: torch.Tensor,
    metric_depth: torch.Tensor,
    median_conf: torch.Tensor,
    min_depth_threshold: float = 1e-3,
    min_metric_depth_threshold: float = 1e-2,
) -> torch.Tensor:


    return (
        (depth_conf >= median_conf)
        & non_sky_mask
        & (metric_depth > min_metric_depth_threshold)
        & (depth > min_depth_threshold)
    )


def sample_tensor_for_quantile(tensor: torch.Tensor, max_samples: int = 100000) -> torch.Tensor:


    if tensor.numel() <= max_samples:
        return tensor

    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor.flatten()[idx]


def apply_metric_scaling(
    depth: torch.Tensor, intrinsics: torch.Tensor, scale_factor: float = 300.0
) -> torch.Tensor:


    focal_length = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2
    return depth * (focal_length[:, :, None, None] / scale_factor)


def set_sky_regions_to_max_depth(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    max_depth: float = 200.0,
) -> Tuple[torch.Tensor, torch.Tensor]:


    depth = depth.clone()


    depth[~non_sky_mask] = max_depth
    if depth_conf is not None:
        depth_conf = depth_conf.clone()
        depth_conf[~non_sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None

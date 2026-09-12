import torch

def normalized_uv(height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=-1)

@torch.no_grad()
def recover_focal_from_xy(xy: torch.Tensor, sample_size: tuple[int, int] = (64, 64)):
    B, T, channels, H, W = xy.shape
    if channels != 2:
        raise ValueError(f"xy must have shape [B, T, 2, H, W], got {tuple(xy.shape)}")

    sample_h = min(H, sample_size[0])
    sample_w = min(W, sample_size[1])
    y_idx = torch.linspace(0, H - 1, sample_h, device=xy.device).round().long()
    x_idx = torch.linspace(0, W - 1, sample_w, device=xy.device).round().long()

    xy_sampled = xy.index_select(-2, y_idx).index_select(-1, x_idx)
    xy_sampled = xy_sampled.permute(0, 1, 3, 4, 2).reshape(B, T, -1, 2)

    uv_sampled = normalized_uv(H, W, xy.dtype, xy.device)
    uv_sampled = uv_sampled.index_select(0, y_idx).index_select(1, x_idx).reshape(1, 1, -1, 2)

    numerator = (xy_sampled * uv_sampled).sum(dim=(-2, -1))
    denominator = xy_sampled.square().sum(dim=(-2, -1))
    return numerator / denominator.clamp(min=1e-8)

def depth_to_pointmap(depth: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    B, T, channels, H, W = depth.shape
    if channels != 1:
        raise ValueError(f"depth must have shape [B, T, 1, H, W], got {tuple(depth.shape)}")
    if focal.shape != (B, T):
        raise ValueError(f"focal must have shape [B, T], got {tuple(focal.shape)}")

    uv_grid = normalized_uv(H, W, depth.dtype, depth.device).view(1, 1, H, W, 2)
    z_grid = depth.permute(0, 1, 3, 4, 2)
    xy_grid = (uv_grid / focal.view(B, T, 1, 1, 1)) * z_grid
    return torch.cat([xy_grid, z_grid], dim=-1)

def pose_encoding_to_extri(pose_encoding):
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]

    R = quat_to_mat(quat)
    return torch.cat([R, T[..., None]], dim=-1)

def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def median(x: torch.Tensor, default_value: float = 1.0) -> torch.Tensor:


    x_masked = torch.where(x > 0.0, x, torch.nan)

    x_flat = x_masked.flatten(start_dim=1)
    medians = torch.nanmedian(x_flat, dim=1).values
    medians = torch.where(torch.isnan(medians),
                          torch.tensor(default_value, device=x.device, dtype=x.dtype),
                          medians)
    ndim = x.ndim
    return medians.view(-1, *(1,) * (ndim - 1))

def log(x: torch.Tensor) -> torch.Tensor:


    return torch.where(x > 0, torch.log(x), torch.zeros_like(x))

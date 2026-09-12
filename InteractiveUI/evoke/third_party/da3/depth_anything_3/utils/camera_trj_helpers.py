

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    from depth_anything_3.utils.logger import logger

    logger.warn("Dependency 'scipy' not found. Required for interpolating camera trajectory.")

from depth_anything_3.utils.geometry import as_homogeneous


@torch.no_grad()
def render_stabilization_path(poses, k_size=45):


    num_frames = poses.shape[0]
    device = poses.device
    dtype = poses.dtype


    if num_frames <= 1:
        return as_homogeneous(poses)


    if k_size < 1:
        k_size = 1
    if k_size % 2 == 0:
        k_size += 1

    max_odd = num_frames if (num_frames % 2 == 1) else (num_frames - 1)
    if max_odd < 1:
        max_odd = 1
    k_size = min(k_size, max_odd)

    if num_frames >= 3 and k_size < 3:
        k_size = 3

    input_poses = []
    for i in range(num_frames):
        input_poses.append(
            torch.cat([poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], dim=-1)
        )
    input_poses = torch.stack(input_poses)


    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1).astype(np.float32).squeeze()
    gaussian_kernel = torch.tensor(gaussian_kernel, dtype=dtype, device=device).view(1, 1, -1)
    pad = k_size // 2

    output_vectors = []
    for idx in range(3):
        vec = (
            input_poses[:, :, idx].T.unsqueeze(0).unsqueeze(0)
        )


        vec = input_poses[:, :, idx].T.unsqueeze(1)
        vec_padded = F.pad(vec, (pad, pad), mode="reflect")
        filtered = F.conv1d(vec_padded, gaussian_kernel)
        output_vectors.append(filtered.squeeze(1).T)

    output_r1, output_r2, output_t = output_vectors


    output_r1 = output_r1 / output_r1.norm(dim=-1, keepdim=True)
    output_r2 = output_r2 / output_r2.norm(dim=-1, keepdim=True)

    output_poses = []
    for i in range(num_frames):
        output_r3 = torch.linalg.cross(output_r1[i], output_r2[i])
        render_pose = torch.cat(
            [
                output_r1[i].unsqueeze(-1),
                output_r2[i].unsqueeze(-1),
                output_r3.unsqueeze(-1),
                output_t[i].unsqueeze(-1),
            ],
            dim=-1,
        )
        output_poses.append(render_pose[:3, :])
    output_poses = as_homogeneous(torch.stack(output_poses, dim=0))

    return output_poses


@torch.no_grad()
def render_wander_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 48.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx = intrinsic[0, 0] * w
    r = max_disp / fx
    th = torch.linspace(0, 2.0 * torch.pi, steps=num_frames, device=device, dtype=dtype)
    x = r * torch.sin(th)
    yz = r * torch.cos(th) / 3.0
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, :3, 3] = torch.stack([x, yz, yz], dim=-1) * -1.0
    c2ws = cam2world.unsqueeze(0) @ T

    c2ws = torch.cat([cam2world.unsqueeze(0), c2ws, cam2world.unsqueeze(0)], dim=0)
    Ks = intrinsic.unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
    return c2ws, Ks


@torch.no_grad()
def render_dolly_zoom_path(
    cam2world: torch.Tensor,
    intrinsic: torch.Tensor,
    h: int,
    w: int,
    num_frames: int = 120,
    max_disp: float = 0.1,
    D_focus: float = 10.0,
):
    device, dtype = cam2world.device, cam2world.dtype
    fx0, fy0 = intrinsic[0, 0] * w, intrinsic[1, 1] * h
    t = torch.linspace(0.0, 2.0, steps=num_frames, device=device, dtype=dtype)
    z = 0.5 * (1.0 - torch.cos(torch.pi * t)) * max_disp
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_frames, 1, 1)
    T[:, 2, 3] = -z
    c2ws = cam2world.unsqueeze(0) @ T
    Df = torch.as_tensor(D_focus, device=device, dtype=dtype)
    scale = (Df / (Df + z)).clamp(min=1e-6)
    Ks = intrinsic.unsqueeze(0).repeat(num_frames, 1, 1)
    Ks[:, 0, 0] = (fx0 * scale) / w
    Ks[:, 1, 1] = (fy0 * scale) / h
    return c2ws, Ks


@torch.no_grad()
def interpolate_intrinsics(
    initial: torch.Tensor,
    final: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: torch.Tensor,
    a_directions: torch.Tensor,
    b_origins: torch.Tensor,
    b_directions: torch.Tensor,
) -> torch.Tensor:


    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)


    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)


    lhs = reduce(n, "... n i j -> ... i j", "sum")


    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")


    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: torch.Tensor) -> torch.Tensor:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:

    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z, dim=-1), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:


    device = a.device


    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)


    return generate_coordinate_frame(normalize(torch.linalg.cross(a, b)), a)


def matrix_to_euler(
    rotations: torch.Tensor,
    pattern: str,
) -> torch.Tensor:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: torch.Tensor,
    pattern: str,
) -> torch.Tensor:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: torch.Tensor,
    pivot_coordinate_frame: torch.Tensor,
    pivot_point: torch.Tensor,
) -> torch.Tensor:


    pivot_axis = pivot_coordinate_frame[..., :, 1]


    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")


    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: torch.Tensor,
    pivot_coordinate_frame: torch.Tensor,
    pivot_point: torch.Tensor,
) -> torch.Tensor:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")


    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau


    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: torch.Tensor,
    final: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: torch.Tensor,
    final: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:


    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)


    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps


    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )


    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)


    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)


    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


@torch.no_grad()
def generate_wobble_transformation(
    radius: torch.Tensor,
    t: torch.Tensor,
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> torch.Tensor:

    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf


@torch.no_grad()
def render_wobble_inter_path(
    cam2world: torch.Tensor, intr_normed: torch.Tensor, inter_len: int, n_skip: int = 3
):


    frame_per_round = n_skip * inter_len
    num_rotations = 1

    t = torch.linspace(0, 1, frame_per_round, dtype=torch.float32, device=cam2world.device)

    tgt_c2w_b = []
    tgt_intr_b = []
    for b_idx in range(cam2world.shape[0]):
        tgt_c2w = []
        tgt_intr = []
        for cur_idx in range(0, cam2world.shape[1] - n_skip, n_skip):
            origin_a = cam2world[b_idx, cur_idx, :3, 3]
            origin_b = cam2world[b_idx, cur_idx + n_skip, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            if cur_idx == 0:
                delta_prev = delta
            else:
                delta = (delta_prev + delta) / 2
                delta_prev = delta
            tf = generate_wobble_transformation(
                radius=delta * 0.5,
                t=t,
                num_rotations=num_rotations,
                scale_radius_with_t=False,
            )
            cur_extrs = (
                interpolate_extrinsics(
                    cam2world[b_idx, cur_idx],
                    cam2world[b_idx, cur_idx + n_skip],
                    t,
                )
                @ tf
            )
            tgt_c2w.append(cur_extrs[(0 if cur_idx == 0 else 1) :])
            tgt_intr.append(
                interpolate_intrinsics(
                    intr_normed[b_idx, cur_idx],
                    intr_normed[b_idx, cur_idx + n_skip],
                    t,
                )[(0 if cur_idx == 0 else 1) :]
            )
        tgt_c2w_b.append(torch.cat(tgt_c2w))
        tgt_intr_b.append(torch.cat(tgt_intr))
    tgt_c2w = torch.stack(tgt_c2w_b)
    tgt_intr = torch.stack(tgt_intr_b)
    return tgt_c2w, tgt_intr

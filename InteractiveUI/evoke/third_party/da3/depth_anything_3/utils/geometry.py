

from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum


def as_homogeneous(ext):


    if isinstance(ext, torch.Tensor):

        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):

            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def transpose_last_two_axes(arr):


    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))

    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )


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


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:


    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )


    quat_by_rijk = torch.stack(
        [


            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),


            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),


            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),


            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )


    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))


    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:


    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:


    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:


    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)


    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(points: torch.Tensor) -> torch.Tensor:

    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:

    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: torch.Tensor,
    transformation: torch.Tensor,
) -> torch.Tensor:

    return einsum(
        transformation,
        homogeneous_coordinates.to(transformation.dtype),
        "... i j, ... j -> ... i",
    )


def transform_cam2world(
    homogeneous_coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:

    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates: torch.Tensor,
    z: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:


    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.float().inverse().to(intrinsics),
        coordinates.to(intrinsics.dtype),
        "... i j, ... j -> ... i",
    )


    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)


    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]


    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    intrinsics_inv = intrinsics.float().inverse().to(intrinsics)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=intrinsics.dtype, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def map_pdf_to_opacity(
    pdf: torch.Tensor,
    global_step: int = 0,
    opacity_mapping: Optional[dict] = None,
) -> torch.Tensor:


    if opacity_mapping is not None:
        cfg = SimpleNamespace(**opacity_mapping)
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    else:
        x = 0.0
    exponent = 2**x


    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

def normalize_homogenous_points(points):

    return points / points[..., -1:]

def inverse_intrinsic_matrix(ixts):

    return torch.inverse(ixts)

def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):


    pixel_space_points = homogenize_points(pixel_space_points)


    camera_space_points = torch.einsum(
        "b v i j , h w j -> b v h w i", inverse_intrinsic_matrix(intrinsics), pixel_space_points
    )
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):


    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum("b v i j , b v h w j -> b v h w i", c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):


    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum(
        "b u i j , b v u h w j -> b v u h w i", intrinsics, camera_space_points
    )
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):


    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum(
        "b u i j , b v h w j -> b v u h w i", c2w.inverse(), world_space_points
    )
    return camera_space_points[..., :3]


def unproject_depth(
    depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None
):


    if c2w is None:
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        c2w = c2w[None, None].repeat(depth.shape[0], depth.shape[1], 1, 1)

    if not ixt_normalized:

        h, w = depth.shape[-3], depth.shape[-2]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )
    else:

        assert num_patches_x is not None and num_patches_y is not None
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x

        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(
                min_y + grid_shift,
                max_y + grid_shift,
                num_patches_y,
                dtype=torch.float32,
                device=depth.device,
            ),
            torch.linspace(
                min_x + grid_shift,
                max_x + grid_shift,
                num_patches_x,
                dtype=torch.float32,
                device=depth.device,
            ),
            indexing="ij",
        )


    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)
    camera_points = pixel_space_to_camera_space(
        pixel_space_points, depth, intrinsics
    )


    world_points = camera_space_to_world_space(camera_points, c2w)

    return world_points

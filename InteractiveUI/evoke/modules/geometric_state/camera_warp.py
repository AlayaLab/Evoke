from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import cv2
except Exception:
    cv2 = None


CAMERA_CONTROL_PROMPT_TRIGGER = "camctl23x."
CAMERA_CONTROL_NUM_FRAMES = 33
CAMERA_CONTROL_PI3_PIXEL_LIMIT = 255000
CAMERA_CONTROL_CONF_THRESHOLD = 0.1
CAMERA_CONTROL_EMPTY_DEPTH_FALLBACK_TOP_PERCENT = 5.0
CAMERA_CONTROL_DEPTH_EDGE_RTOL = 0.03
CAMERA_CONTROL_MESH_SAMPLES_PER_AXIS = 4
CAMERA_CONTROL_FILL_TOP_MAX_Y_FRAC = 1.0
CAMERA_CONTROL_FILL_MIN_COMPONENT_AREA = 1024
CAMERA_CONTROL_FILL_BOUNDARY_KERNEL = 17
CAMERA_CONTROL_FILL_BOUNDARY_MIN_SAMPLES = 128
CAMERA_CONTROL_FILL_BOUNDARY_DEPTH_QUANTILE = 0.95
CAMERA_CONTROL_FILL_GLOBAL_DEPTH_QUANTILE = 0.98
CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE = 0.1
CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH = True
CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL = "mean_first_frame"
CAMERA_CONTROL_WARP_RENDER_MODES = frozenset({"splat", "target_fill"})
CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE = "target_fill"
CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS = 1
CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS = 4
CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE = "depth_normal"
CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL = 0.03
CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG = 5.0
CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY = True
CAMERA_CONTROL_PI3X_KEYFRAME_PREVIOUS_MESH_SAMPLES_PER_AXIS = 1
CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_HEIGHT = 384
CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_WIDTH = 768
CAMERA_CONTROL_KEYFRAME_BACKGROUND_DEPTH_QUANTILE = 0.65
CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_FILL_RADIUS = 3
CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_CLAMP_MIN = 0.65
CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_CLAMP_MAX = 1.35
CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_SMOOTH_ALPHA = 0.5
CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MIN = 0.1
CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MAX = 10.0


def _repo_root() -> Path:

    return Path(__file__).resolve().parents[3]


def default_pi3_repo() -> Path:


    env = os.environ.get("EVOKE_PI3_SRC")
    if env:
        return Path(env)
    return _repo_root() / "evoke" / "third_party"


def default_pi3x_ckpt() -> Path:

    return _repo_root() / "models" / "Pi3X" / "model.safetensors"


def center_crop_resize_first_frame(image: Any, height: int, width: int) -> Any:
    target_size = (int(width), int(height))

    def fit_one(item: Any) -> Any:
        if not isinstance(item, Image.Image):
            return item
        return ImageOps.fit(
            item.convert("RGB"),
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    if isinstance(image, list):
        return [fit_one(item) for item in image]
    return fit_one(image)


def as_pose4x4(poses: torch.Tensor | np.ndarray) -> torch.Tensor:
    poses = torch.as_tensor(poses, dtype=torch.float32)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3:
        raise ValueError(f"Unsupported camera pose shape: {tuple(poses.shape)}")
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] == (3, 4):
        bottom = torch.zeros(*poses.shape[:-2], 1, 4, device=poses.device, dtype=poses.dtype)
        bottom[..., 0, 3] = 1.0
        return torch.cat([poses, bottom], dim=-2)
    raise ValueError(f"Unsupported camera pose shape: {tuple(poses.shape)}")


def se3_inverse(T: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:

    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        bottom_row = torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
        top_part = torch.cat([R_inv, t_inv], dim=-1)
        return torch.cat([top_part, bottom_row], dim=-2)

    R = T[..., :3, :3]
    t = T[..., :3, 3, np.newaxis]
    R_inv = np.swapaxes(R, -2, -1)
    t_inv = -R_inv @ t
    bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
    bottom_row[..., :, 3] = 1
    top_part = np.concatenate([R_inv, t_inv], axis=-1)
    return np.concatenate([top_part, bottom_row], axis=-2)


def prepare_camera_pose_rollout(camera_poses: torch.Tensor | np.ndarray, total_target_frames: int) -> torch.Tensor:
    poses = as_pose4x4(camera_poses)
    inv0 = se3_inverse(poses[0:1])
    poses = inv0 @ poses
    if poses.shape[0] < total_target_frames:
        pad = poses[-1:].repeat(total_target_frames - poses.shape[0], 1, 1)
        poses = torch.cat([poses, pad], dim=0)
    elif poses.shape[0] > total_target_frames:
        poses = poses[:total_target_frames]
    return poses


def scale_camera_pose_translations(
    camera_poses: torch.Tensor | np.ndarray,
    translation_scale: float,
) -> torch.Tensor:
    poses = as_pose4x4(camera_poses).clone()
    poses[:, :3, 3] *= float(translation_scale)
    return poses


def fuse2_apply_T(keyframe: dict[str, Any], T: np.ndarray, intrinsic0: np.ndarray) -> dict[str, Any]:

    T = np.asarray(T, np.float32)
    keyframe = dict(keyframe)
    pts = np.asarray(keyframe["point_map_world"], np.float32)
    keyframe["point_map_world"] = (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    keyframe["source_pose"] = (T @ np.asarray(keyframe["source_pose"], np.float32)).astype(np.float32)
    keyframe["intrinsic"] = np.asarray(intrinsic0, np.float32)
    return keyframe


def fuse2_build_fused_geom(
    cloud: list[dict[str, Any]],
    origin_pose: np.ndarray,
    intrinsic0: np.ndarray,
    render_height: int,
    render_width: int,
) -> dict[str, Any]:


    latest = cloud[-1]
    return {
        "geometry_backend": "pi3x_keyframes",
        "preserve_pi3x_keyframe_points": True,
        "intrinsic": np.asarray(intrinsic0, np.float32),
        "render_height": int(render_height),
        "render_width": int(render_width),
        "keyframe_count": len(cloud),
        "keyframe_geometries": cloud,
        "pose_source": "pi3x_estimated",
        "scale_source": "fuse2",
        "source_pose": np.asarray(origin_pose, np.float32),
        "source_rgb_u8": latest["source_rgb_u8"],
        "depth_map": latest["depth_map"],
        "valid_mask": latest.get("valid_mask"),
    }


def fuse2_add_keyframe_incremental(
    renderer: Any,
    state: dict[str, Any],
    frame_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:


    pair = renderer.estimate_keyframe_geometry(
        image_tensors=[state["prev_frame_cpu"], frame_cpu],
        device=device,
        scale_reference_geometry=state["prev_geom"],
    )
    kg = pair["keyframe_geometries"]
    p0 = np.asarray(kg[0]["source_pose"], np.float32)
    p1 = np.asarray(kg[1]["source_pose"], np.float32)
    rel = np.linalg.inv(p0) @ p1
    src_pose_c0 = np.asarray(state["prev_pose_c0"], np.float32) @ rel
    T = src_pose_c0 @ np.linalg.inv(p1)
    state["cloud"].append(fuse2_apply_T(kg[1], T, state["intrinsic0"]))
    state["prev_frame_cpu"] = frame_cpu
    state["prev_pose_c0"] = src_pose_c0
    state["prev_geom"] = pair
    step_t = float(np.linalg.norm(rel[:3, 3]))
    cum_t = float(np.linalg.norm(src_pose_c0[:3, 3] - np.asarray(state["origin_pose"], np.float32)[:3, 3]))
    return step_t, cum_t, step_t


def normalize_camera_poses_anchor_max_norm(
    camera_poses: torch.Tensor | np.ndarray,
    eps: float = 1e-6,
) -> torch.Tensor:

    poses = as_pose4x4(camera_poses).clone()
    t = poses[:, :3, 3]
    t_rel = t - t[0:1]
    max_norm = t_rel.norm(dim=-1).max().clamp(min=float(eps))
    poses[:, :3, 3] = t / max_norm
    return poses


def _pi3_target_size(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    target_w = width * scale
    target_h = height * scale
    k = round(target_w / 14)
    m = round(target_h / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > target_w / max(target_h, 1):
            k -= 1
        else:
            m -= 1
    return max(1, m) * 14, max(1, k) * 14


def _import_pi3(pi3_repo: Path):


    repo = Path(pi3_repo).expanduser().resolve()
    if repo == (_repo_root() / "evoke" / "third_party").resolve():
        from evoke.third_party.pi3.models.pi3x import Pi3X
        from evoke.third_party.pi3.utils.geometry import recover_intrinsic_from_rays_d

        return Pi3X, recover_intrinsic_from_rays_d

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    return Pi3X, recover_intrinsic_from_rays_d


def _load_pi3x_model(Pi3X: Any, device: torch.device):
    ckpt = default_pi3x_ckpt().expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing Pi3X checkpoint: {ckpt}. Download the Pi3X checkpoint and place it at "
            "checkpoints/pi3x/model.safetensors before running inference, dryrun, or training."
        )
    model = Pi3X(use_multimodal=False).eval()
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        weight = load_file(str(ckpt))
    else:
        weight = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(weight, strict=False)

    model.disable_multimodal()
    return model.to(device)


def _force_pi3x_float_heads(model: Any) -> None:
    for name in ("point_head", "conf_head", "camera_head", "metric_head"):
        module = getattr(model, name, None)
        if module is not None:
            module.to(device=next(module.parameters()).device, dtype=torch.float32)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return torch.amp.autocast("cpu", enabled=False)
    major, _minor = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def _normalize_intrinsics_shape(intrinsics: torch.Tensor, num_frames: int) -> torch.Tensor:
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None].repeat(num_frames, 1, 1)
    if intrinsics.shape[0] == 1 and num_frames > 1:
        intrinsics = intrinsics.repeat(num_frames, 1, 1)
    if intrinsics.shape[0] != num_frames:
        raise ValueError(f"Recovered {intrinsics.shape[0]} intrinsics for {num_frames} frames.")
    return intrinsics.detach().float().cpu()


def _smooth_pi3x_keyframe_intrinsics(
    intrinsics: np.ndarray,
    *,
    render_height: int,
    render_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    recovered = np.asarray(intrinsics, dtype=np.float32)
    if recovered.ndim != 3 or recovered.shape[-2:] != (3, 3):
        raise ValueError(f"Expected Pi3X keyframe intrinsics with shape [N, 3, 3], got {recovered.shape}.")

    reference = recovered[0].astype(np.float32, copy=True)
    focal_min = float(CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_CLAMP_MIN)
    focal_max = float(CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_CLAMP_MAX)
    smooth_alpha = float(CAMERA_CONTROL_PI3X_KEYFRAME_FOCAL_SMOOTH_ALPHA)
    center_x = (float(render_width) - 1.0) * 0.5
    center_y = (float(render_height) - 1.0) * 0.5
    reference_fx = max(abs(float(reference[0, 0])), 1e-6)
    reference_fy = max(abs(float(reference[1, 1])), 1e-6)

    smoothed = np.zeros_like(recovered, dtype=np.float32)
    records: list[dict[str, Any]] = []
    previous_fx = reference_fx
    previous_fy = reference_fy
    for index, current in enumerate(recovered):
        raw_fx = max(abs(float(current[0, 0])), 1e-6)
        raw_fy = max(abs(float(current[1, 1])), 1e-6)
        clamped_fx = float(np.clip(raw_fx / reference_fx, focal_min, focal_max) * reference_fx)
        clamped_fy = float(np.clip(raw_fy / reference_fy, focal_min, focal_max) * reference_fy)
        if index == 0:
            final_fx = clamped_fx
            final_fy = clamped_fy
        else:
            final_fx = smooth_alpha * clamped_fx + (1.0 - smooth_alpha) * previous_fx
            final_fy = smooth_alpha * clamped_fy + (1.0 - smooth_alpha) * previous_fy
        previous_fx = float(final_fx)
        previous_fy = float(final_fy)

        smoothed[index] = np.array(
            [
                [final_fx, 0.0, center_x],
                [0.0, final_fy, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        records.append(
            {
                "keyframe_index": int(index),
                "raw_fx": float(raw_fx),
                "raw_fy": float(raw_fy),
                "raw_fx_ratio": float(raw_fx / reference_fx),
                "raw_fy_ratio": float(raw_fy / reference_fy),
                "clamped_fx": float(clamped_fx),
                "clamped_fy": float(clamped_fy),
                "smoothed_fx": float(final_fx),
                "smoothed_fy": float(final_fy),
            }
        )

    return smoothed, {
        "clamp_min": focal_min,
        "clamp_max": focal_max,
        "center_x": center_x,
        "center_y": center_y,
        "reference_fx": reference_fx,
        "reference_fy": reference_fy,
        "smooth_alpha": smooth_alpha,
        "keyframes": records,
    }


def _intrinsic_stats(intrinsic: np.ndarray) -> dict[str, float]:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    return {
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
    }


def _resize_numpy_map(value: np.ndarray, shape: tuple[int, int], *, mode: str) -> np.ndarray:
    value_np = np.asarray(value)
    if value_np.shape == shape:
        return value_np
    if value_np.ndim != 2:
        raise ValueError(f"Expected a 2D map for resize, got {value_np.shape}.")
    value_t = torch.from_numpy(value_np.astype(np.float32, copy=False)).view(1, 1, *value_np.shape)
    if mode == "nearest":
        resized = F.interpolate(value_t, size=shape, mode=mode)
    else:
        resized = F.interpolate(value_t, size=shape, mode=mode, align_corners=False)
    return resized[0, 0].cpu().numpy()


def _depth_scale_to_reference(
    *,
    reference_depth: np.ndarray,
    current_depth: np.ndarray,
    reference_valid_mask: np.ndarray | None = None,
    current_valid_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    reference = np.asarray(reference_depth, dtype=np.float32)
    current = np.asarray(current_depth, dtype=np.float32)
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError(f"Depth scale alignment expects 2D maps, got {reference.shape} and {current.shape}.")
    if reference.shape != current.shape:
        reference = _resize_numpy_map(reference, current.shape, mode="bilinear").astype(np.float32, copy=False)
        if reference_valid_mask is not None:
            reference_valid_mask = _resize_numpy_map(
                np.asarray(reference_valid_mask, dtype=np.float32),
                current.shape,
                mode="nearest",
            ) > 0.5

    valid = np.isfinite(reference) & np.isfinite(current) & (reference > 0.0) & (current > 0.0)
    if reference_valid_mask is not None:
        valid &= np.asarray(reference_valid_mask, dtype=bool)
    if current_valid_mask is not None:
        valid &= np.asarray(current_valid_mask, dtype=bool)

    ratios = reference[valid].astype(np.float64) / current[valid].astype(np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    stats: dict[str, Any] = {
        "clamp_max": float(CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MAX),
        "clamp_min": float(CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MIN),
        "current_shape": [int(current.shape[0]), int(current.shape[1])],
        "policy": "first_frame_depth_median_ratio",
        "reference_shape": [int(reference.shape[0]), int(reference.shape[1])],
        "used_pixels": int(ratios.size),
    }
    if ratios.size == 0:
        stats.update({"applied_scale": 1.0, "raw_scale": 1.0, "status": "no_valid_overlap"})
        return 1.0, stats

    if ratios.size >= 32:
        low, high = np.percentile(ratios, [10.0, 90.0])
        trimmed = ratios[(ratios >= low) & (ratios <= high)]
        ratios_for_scale = trimmed if trimmed.size > 0 else ratios
        stats["ratio_p10"] = float(low)
        stats["ratio_p90"] = float(high)
    else:
        ratios_for_scale = ratios

    raw_scale = float(np.median(ratios_for_scale))
    applied_scale = float(
        np.clip(
            raw_scale,
            CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MIN,
            CAMERA_CONTROL_PI3X_KEYFRAME_DEPTH_SCALE_CLAMP_MAX,
        )
    )
    stats.update(
        {
            "applied_scale": applied_scale,
            "current_median_depth": float(np.median(current[valid])),
            "raw_scale": raw_scale,
            "reference_median_depth": float(np.median(reference[valid])),
            "status": "ok" if applied_scale == raw_scale else "clamped",
            "trimmed_pixels": int(ratios_for_scale.size),
        }
    )
    return applied_scale, stats


def _world_rays_from_intrinsic(height: int, width: int, intrinsic: np.ndarray, pose: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    ys, xs = np.mgrid[0:int(height), 0:int(width)].astype(np.float32)
    fx = max(abs(float(intrinsic[0, 0])), 1e-6)
    fy = max(abs(float(intrinsic[1, 1])), 1e-6)
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    dirs_cam = np.stack(((xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)), axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True).clip(min=1e-6)
    dirs_world = dirs_cam @ pose[:3, :3].T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True).clip(min=1e-6)
    return dirs_world.astype(np.float32, copy=False)


def _spherical_uv_from_world_rays(
    rays_world: np.ndarray,
    *,
    atlas_height: int,
    atlas_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    rays_world = np.asarray(rays_world, dtype=np.float32)
    x = rays_world[..., 0]
    y = rays_world[..., 1]
    z = rays_world[..., 2]
    yaw = np.arctan2(x, z)
    pitch = np.arctan2(y, np.sqrt(np.maximum(x * x + z * z, 1e-12)))
    u = np.floor((yaw + np.pi) * (float(atlas_width) / (2.0 * np.pi))).astype(np.int64) % int(atlas_width)
    v = np.floor((pitch + np.pi * 0.5) * (float(atlas_height) / np.pi)).astype(np.int64)
    v = np.clip(v, 0, int(atlas_height) - 1)
    return u, v


def _keyframe_background_mask(keyframe_geometry: dict[str, Any]) -> np.ndarray:
    rgb = np.asarray(keyframe_geometry["source_rgb_u8"])
    height, width = rgb.shape[:2]
    valid_mask = np.asarray(keyframe_geometry.get("valid_mask", np.ones((height, width), dtype=bool)), dtype=bool)
    depth_map = np.asarray(keyframe_geometry.get("depth_map", np.zeros((height, width))), dtype=np.float32)
    conf_map = np.asarray(keyframe_geometry.get("conf_map", np.ones((height, width))), dtype=np.float32)

    finite_depth = np.isfinite(depth_map) & (depth_map > 0.0)
    depth_values = depth_map[finite_depth & valid_mask]
    background_mask = (~valid_mask) | (conf_map < CAMERA_CONTROL_CONF_THRESHOLD)
    if depth_values.size >= 128:
        depth_threshold = float(np.nanquantile(depth_values, CAMERA_CONTROL_KEYFRAME_BACKGROUND_DEPTH_QUANTILE))
        background_mask |= finite_depth & valid_mask & (depth_map >= depth_threshold)

    min_pixels = max(1, int(0.1 * height * width))
    if int(background_mask.sum()) < min_pixels:
        background_mask = np.ones((height, width), dtype=bool)
    return background_mask.astype(bool, copy=False)


def _fill_atlas_rows_nearest(atlas_rgb: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atlas_rgb = np.asarray(atlas_rgb, dtype=np.uint8)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    filled = atlas_rgb.copy()
    filled_valid = valid_mask.copy()
    height, width = valid_mask.shape
    cols = np.arange(width, dtype=np.int64)
    fill_radius = int(CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_FILL_RADIUS)
    for row_idx in range(height):
        row_valid = np.flatnonzero(valid_mask[row_idx])
        if row_valid.size == 0:
            continue
        if row_valid.size == width:
            filled_valid[row_idx] = True
            continue
        extended = np.concatenate((row_valid - width, row_valid, row_valid + width))
        positions = np.searchsorted(extended, cols)
        left_positions = np.maximum(positions - 1, 0)
        right_positions = np.minimum(positions, extended.size - 1)
        left = extended[left_positions]
        right = extended[right_positions]
        left_dist = np.abs(cols - left)
        right_dist = np.abs(right - cols)
        use_left = left_dist <= right_dist
        nearest = np.where(use_left, left, right) % width
        nearest_dist = np.where(use_left, left_dist, right_dist)
        fill_cols = nearest_dist <= fill_radius
        filled[row_idx, fill_cols] = atlas_rgb[row_idx, nearest[fill_cols]]
        filled_valid[row_idx, fill_cols] = True
    return filled, filled_valid


def _build_keyframe_background_atlas(keyframe_geometries: list[dict[str, Any]]) -> dict[str, Any] | None:
    atlas_height = int(CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_HEIGHT)
    atlas_width = int(CAMERA_CONTROL_KEYFRAME_BACKGROUND_ATLAS_WIDTH)
    sum_rgb = np.zeros((atlas_height, atlas_width, 3), dtype=np.float32)
    count = np.zeros((atlas_height, atlas_width), dtype=np.float32)
    records: list[dict[str, Any]] = []

    for keyframe_index, keyframe_geometry in enumerate(keyframe_geometries):
        rgb = np.asarray(keyframe_geometry["source_rgb_u8"], dtype=np.uint8)
        height, width = rgb.shape[:2]
        rays_world = _world_rays_from_intrinsic(
            height,
            width,
            keyframe_geometry["intrinsic"],
            keyframe_geometry["source_pose"],
        )
        u, v = _spherical_uv_from_world_rays(
            rays_world,
            atlas_height=atlas_height,
            atlas_width=atlas_width,
        )
        background_mask = _keyframe_background_mask(keyframe_geometry)
        flat_mask = background_mask.reshape(-1)
        flat_u = u.reshape(-1)[flat_mask]
        flat_v = v.reshape(-1)[flat_mask]
        flat_rgb = rgb.reshape(-1, 3)[flat_mask].astype(np.float32, copy=False)
        if flat_rgb.size:
            np.add.at(sum_rgb, (flat_v, flat_u), flat_rgb)
            np.add.at(count, (flat_v, flat_u), 1.0)
        records.append(
            {
                "background_pixels": int(flat_mask.sum()),
                "height": int(height),
                "keyframe_index": int(keyframe_index),
                "width": int(width),
            }
        )

    valid = count > 0.0
    if not bool(valid.any()):
        return None

    atlas = np.zeros_like(sum_rgb, dtype=np.uint8)
    atlas[valid] = np.clip(np.rint(sum_rgb[valid] / count[valid, None]), 0, 255).astype(np.uint8)
    sample_atlas, sample_valid = _fill_atlas_rows_nearest(atlas, valid)
    return {
        "atlas": sample_atlas,
        "stats": {
            "atlas_height": atlas_height,
            "atlas_width": atlas_width,
            "filled_texels": int(sample_valid.sum()),
            "keyframes": records,
            "raw_texels": int(valid.sum()),
            "raw_texel_fraction": float(valid.sum() / max(valid.size, 1)),
            "sample_texel_fraction": float(sample_valid.sum() / max(sample_valid.size, 1)),
        },
        "valid": sample_valid,
    }


def _render_background_atlas(
    background_atlas: dict[str, Any],
    target_pose: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    fill_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    atlas = np.asarray(background_atlas["atlas"], dtype=np.uint8)
    valid = np.asarray(background_atlas["valid"], dtype=bool)
    rays_world = _world_rays_from_intrinsic(height, width, intrinsic, target_pose)
    u, v = _spherical_uv_from_world_rays(rays_world, atlas_height=atlas.shape[0], atlas_width=atlas.shape[1])
    frame = atlas[v, u].copy()
    frame_valid = valid[v, u].copy()
    frame[~frame_valid] = np.asarray(fill_rgb, dtype=np.uint8)
    return frame, frame_valid


def _camera_to_world(points_cam: np.ndarray, source_c2w: np.ndarray) -> np.ndarray:
    rotation = source_c2w[:3, :3].astype(np.float32, copy=False)
    translation = source_c2w[:3, 3].astype(np.float32, copy=False)
    return points_cam @ rotation.T + translation


def _points_from_depth_map(depth_map: np.ndarray, intrinsic: np.ndarray, source_pose: np.ndarray) -> np.ndarray:
    height, width = depth_map.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    z = depth_map.astype(np.float32, copy=False)
    x = (xs - float(intrinsic[0, 2])) / float(intrinsic[0, 0]) * z
    y = (ys - float(intrinsic[1, 2])) / float(intrinsic[1, 1]) * z
    points_cam = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    return _camera_to_world(points_cam.reshape(-1, 3), source_pose).reshape(height, width, 3)


def _geometry_with_intrinsic(geometry: dict[str, Any], intrinsic: np.ndarray) -> dict[str, Any]:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"target intrinsic must have shape (3, 3), got {intrinsic.shape}.")

    current_intrinsic = np.asarray(geometry["intrinsic"], dtype=np.float32)
    if current_intrinsic.shape == (3, 3) and np.allclose(current_intrinsic, intrinsic, rtol=1e-5, atol=1e-4):
        return geometry

    updated = dict(geometry)
    updated["intrinsic"] = intrinsic.astype(np.float32, copy=True)
    source_pose = np.asarray(updated.get("source_pose", np.eye(4, dtype=np.float32)), dtype=np.float32)
    point_depth = np.asarray(updated.get("dense_depth_map", updated["depth_map"]), dtype=np.float32)
    updated["point_map_world"] = _points_from_depth_map(point_depth, updated["intrinsic"], source_pose)
    return updated


def _build_camera_points_from_depth(mask: np.ndarray, depth_value: float, intrinsic: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    z = np.full((ys.shape[0],), float(depth_value), dtype=np.float32)
    x = (xs.astype(np.float32) - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (ys.astype(np.float32) - intrinsic[1, 2]) / intrinsic[1, 1] * z
    return np.stack([x, y, z], axis=1)


def _estimate_component_fill_depth(
    component_mask: np.ndarray,
    valid_mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    boundary_kernel: int,
    boundary_min_samples: int,
    boundary_depth_quantile: float,
    global_far_depth: float,
) -> float | None:
    if cv2 is None:
        raise ImportError("opencv-python is required for camera-control top-fill rendering.")
    kernel_size = max(3, int(boundary_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1) > 0
    boundary_ring = dilated & valid_mask & ~component_mask
    boundary_depths = depth_map[boundary_ring]
    boundary_depths = boundary_depths[np.isfinite(boundary_depths) & (boundary_depths > 0)]

    if boundary_depths.size >= int(boundary_min_samples):
        boundary_far = float(np.quantile(boundary_depths, boundary_depth_quantile))
        return max(boundary_far, float(global_far_depth))
    if np.isfinite(global_far_depth) and global_far_depth > 0:
        return float(global_far_depth)
    return None


def _fill_top_invalid_with_far_plane(
    *,
    point_map_world: np.ndarray,
    depth_map: np.ndarray,
    conf_map: np.ndarray,
    source_pose: np.ndarray,
    intrinsic: np.ndarray,
    conf_threshold: float,
    fill_top_max_y_frac: float,
    fill_min_component_area: int,
    fill_boundary_kernel: int,
    fill_boundary_min_samples: int,
    fill_boundary_depth_quantile: float,
    fill_global_depth_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    if cv2 is None:
        raise ImportError("opencv-python is required for camera-control top-fill rendering.")

    base_valid = (conf_map > conf_threshold) & np.isfinite(point_map_world).all(axis=-1)
    depth_valid = np.isfinite(depth_map) & (depth_map > 0)
    valid_mask = base_valid & depth_valid

    augmented_points = point_map_world.astype(np.float32, copy=True)
    fill_mask = np.zeros(valid_mask.shape, dtype=bool)
    top_fill_limit = int(round(valid_mask.shape[0] * float(fill_top_max_y_frac)))
    top_fill_limit = min(max(top_fill_limit, 0), valid_mask.shape[0] - 1)

    global_depths = depth_map[valid_mask]
    global_depths = global_depths[np.isfinite(global_depths) & (global_depths > 0)]
    global_far_depth = (
        float(np.quantile(global_depths, fill_global_depth_quantile)) if global_depths.size > 0 else float("nan")
    )

    invalid_mask = ~valid_mask
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        invalid_mask.astype(np.uint8),
        connectivity=8,
    )
    fill_stats: list[dict[str, float]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(fill_min_component_area):
            continue

        component_mask = labels == label_idx
        ys, _xs = np.nonzero(component_mask)
        if ys.size == 0 or ys.min() != 0:
            continue

        component_fill_mask = component_mask.copy()
        component_fill_mask[top_fill_limit + 1 :] = False
        component_fill_area = int(component_fill_mask.sum())
        if component_fill_area < int(fill_min_component_area):
            continue

        fill_depth = _estimate_component_fill_depth(
            component_fill_mask,
            valid_mask,
            depth_map,
            boundary_kernel=fill_boundary_kernel,
            boundary_min_samples=fill_boundary_min_samples,
            boundary_depth_quantile=fill_boundary_depth_quantile,
            global_far_depth=global_far_depth,
        )
        if fill_depth is None or not np.isfinite(fill_depth) or fill_depth <= 0:
            continue

        points_cam = _build_camera_points_from_depth(component_fill_mask, fill_depth, intrinsic)
        points_world = _camera_to_world(points_cam, source_pose)
        fill_ys, fill_xs = np.nonzero(component_fill_mask)
        augmented_points[fill_ys, fill_xs] = points_world
        fill_mask[component_fill_mask] = True
        valid_mask[component_fill_mask] = True

        fill_stats.append(
            {
                "label": int(label_idx),
                "fill_area": float(component_fill_area),
                "fill_depth": float(fill_depth),
                "component_area": float(area),
            }
        )

    return augmented_points, valid_mask, fill_mask, fill_stats


def _sample_mesh_quads(
    point_map: np.ndarray,
    color_map: np.ndarray,
    valid_mask: np.ndarray,
    samples_per_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples_per_axis = max(1, int(samples_per_axis))
    quad_mask = (
        valid_mask[:-1, :-1]
        & valid_mask[1:, :-1]
        & valid_mask[1:, 1:]
        & valid_mask[:-1, 1:]
    )
    if not np.any(quad_mask):
        raise ValueError("No valid first-frame mesh quads were found.")

    p00 = point_map[:-1, :-1][quad_mask]
    p10 = point_map[1:, :-1][quad_mask]
    p11 = point_map[1:, 1:][quad_mask]
    p01 = point_map[:-1, 1:][quad_mask]
    c00 = color_map[:-1, :-1][quad_mask].astype(np.float32)
    c10 = color_map[1:, :-1][quad_mask].astype(np.float32)
    c11 = color_map[1:, 1:][quad_mask].astype(np.float32)
    c01 = color_map[:-1, 1:][quad_mask].astype(np.float32)

    points = []
    colors = []
    source_xy = []
    ys, xs = np.nonzero(quad_mask)
    offsets = (np.arange(samples_per_axis, dtype=np.float32) + 0.5) / samples_per_axis
    for sv in offsets:
        for su in offsets:
            w00 = (1.0 - sv) * (1.0 - su)
            w10 = sv * (1.0 - su)
            w11 = sv * su
            w01 = (1.0 - sv) * su
            points.append(w00 * p00 + w10 * p10 + w11 * p11 + w01 * p01)
            colors.append(w00 * c00 + w10 * c10 + w11 * c11 + w01 * c01)
            source_xy.append(
                np.stack(
                    [
                        xs.astype(np.float32, copy=False) + su,
                        ys.astype(np.float32, copy=False) + sv,
                    ],
                    axis=1,
                )
            )

    sampled_points = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    sampled_colors = np.clip(np.concatenate(colors, axis=0), 0, 255).astype(np.uint8)
    sampled_source_xy = np.concatenate(source_xy, axis=0).astype(np.float32, copy=False)
    return sampled_points, sampled_colors, sampled_source_xy


def _sample_mesh_quads_rejecting_mixed_fill(
    point_map: np.ndarray,
    color_map: np.ndarray,
    valid_mask: np.ndarray,
    fill_mask: np.ndarray,
    samples_per_axis: int,
    *,
    allow_mixed_fill_quads: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if allow_mixed_fill_quads or not np.any(fill_mask):
        return _sample_mesh_quads(
            point_map=point_map,
            color_map=color_map,
            valid_mask=valid_mask,
            samples_per_axis=samples_per_axis,
        )

    quad_mask = (
        valid_mask[:-1, :-1]
        & valid_mask[1:, :-1]
        & valid_mask[1:, 1:]
        & valid_mask[:-1, 1:]
    )
    fill00 = fill_mask[:-1, :-1]
    fill10 = fill_mask[1:, :-1]
    fill11 = fill_mask[1:, 1:]
    fill01 = fill_mask[:-1, 1:]
    quad_all_filled = fill00 & fill10 & fill11 & fill01
    quad_any_filled = fill00 | fill10 | fill11 | fill01
    quad_mask = quad_mask & (~quad_any_filled | quad_all_filled)
    if not np.any(quad_mask):
        raise ValueError("No valid first-frame mesh quads remain after rejecting mixed fill quads.")

    samples_per_axis = max(1, int(samples_per_axis))
    p00 = point_map[:-1, :-1][quad_mask]
    p10 = point_map[1:, :-1][quad_mask]
    p11 = point_map[1:, 1:][quad_mask]
    p01 = point_map[:-1, 1:][quad_mask]
    c00 = color_map[:-1, :-1][quad_mask].astype(np.float32)
    c10 = color_map[1:, :-1][quad_mask].astype(np.float32)
    c11 = color_map[1:, 1:][quad_mask].astype(np.float32)
    c01 = color_map[:-1, 1:][quad_mask].astype(np.float32)

    points = []
    colors = []
    source_xy = []
    ys, xs = np.nonzero(quad_mask)
    offsets = (np.arange(samples_per_axis, dtype=np.float32) + 0.5) / samples_per_axis
    for sv in offsets:
        for su in offsets:
            w00 = (1.0 - sv) * (1.0 - su)
            w10 = sv * (1.0 - su)
            w11 = sv * su
            w01 = (1.0 - sv) * su
            points.append(w00 * p00 + w10 * p10 + w11 * p11 + w01 * p01)
            colors.append(w00 * c00 + w10 * c10 + w11 * c11 + w01 * c01)
            source_xy.append(
                np.stack(
                    [
                        xs.astype(np.float32, copy=False) + su,
                        ys.astype(np.float32, copy=False) + sv,
                    ],
                    axis=1,
                )
            )

    sampled_points = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    sampled_colors = np.clip(np.concatenate(colors, axis=0), 0, 255).astype(np.uint8)
    sampled_source_xy = np.concatenate(source_xy, axis=0).astype(np.float32, copy=False)
    return sampled_points, sampled_colors, sampled_source_xy


def _max_pool_2d_np(x: np.ndarray, kernel_size: int, padding: int = 0) -> np.ndarray:
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        x = np.pad(x, ((padding, padding), (padding, padding)), mode="constant", constant_values=fill_value)
    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel_size, kernel_size))
    return np.nanmax(windows, axis=(-2, -1))


def _depth_edge_np(depth: np.ndarray, *, rtol: float, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    padding = kernel_size // 2
    max_valid = _max_pool_2d_np(np.where(mask, depth, -np.inf), kernel_size, padding=padding)
    min_valid = -_max_pool_2d_np(np.where(mask, -depth, -np.inf), kernel_size, padding=padding)
    diff = max_valid - min_valid
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = (diff / depth) > float(rtol)
    return edge & mask


def _points_to_normals_np(point: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(point).all(axis=-1)
    height, width = point.shape[:2]
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1] = np.where(mask[..., None], point, 0.0)

    up = pts[:-2, 1:-1] - pts[1:-1, 1:-1]
    left = pts[1:-1, :-2] - pts[1:-1, 1:-1]
    down = pts[2:, 1:-1] - pts[1:-1, 1:-1]
    right = pts[1:-1, 2:] - pts[1:-1, 1:-1]

    normals = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ],
        axis=0,
    )
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1.0e-12)
    valid = (
        np.stack(
            [
                mask_pad[:-2, 1:-1] & mask_pad[1:-1, :-2],
                mask_pad[1:-1, :-2] & mask_pad[2:, 1:-1],
                mask_pad[2:, 1:-1] & mask_pad[1:-1, 2:],
                mask_pad[1:-1, 2:] & mask_pad[:-2, 1:-1],
            ],
            axis=0,
        )
        & mask_pad[None, 1:-1, 1:-1]
    )
    normal = np.where(valid[..., None], normals, 0.0).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1.0e-12)
    normal_mask = valid.any(axis=0)
    normal = np.where(normal_mask[..., None], normal, 0.0)
    normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    return normal.astype(np.float32, copy=False), normal_mask


def _normal_edge_np(
    normals: np.ndarray,
    *,
    tol_deg: float,
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1.0e-12)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    padding = kernel_size // 2
    normals_pad = np.pad(normals, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    mask_pad = np.pad(mask, ((padding, padding), (padding, padding)), mode="edge")
    normal_windows = np.lib.stride_tricks.sliding_window_view(normals_pad, (kernel_size, kernel_size), axis=(0, 1))
    mask_windows = np.lib.stride_tricks.sliding_window_view(mask_pad, (kernel_size, kernel_size), axis=(0, 1))
    center = normals[..., None, None, :]
    dot = (center * np.moveaxis(normal_windows, 2, -1)).sum(axis=-1).clip(-1.0, 1.0)
    angle = np.where(mask_windows, np.arccos(dot), 0.0).max(axis=(-2, -1))
    angle = _max_pool_2d_np(angle, kernel_size, padding=padding)
    return (angle > np.deg2rad(float(tol_deg))) & mask


def _depth_normal_break_mask_np(
    *,
    point_map_world: np.ndarray,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_rtol: float,
    normal_tol_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    normals, normal_mask = _points_to_normals_np(point_map_world, valid_mask)
    depth_edge = _depth_edge_np(depth_map, rtol=float(depth_rtol), mask=valid_mask)
    normal_edge = _normal_edge_np(normals, tol_deg=float(normal_tol_deg), mask=normal_mask)
    break_mask = depth_edge & normal_edge
    valid_count = int(valid_mask.sum())
    return break_mask, {
        "depth_rtol": float(depth_rtol),
        "normal_tol_deg": float(normal_tol_deg),
        "valid_pixels": valid_count,
        "depth_edge_pixels": int(depth_edge.sum()),
        "normal_edge_pixels": int(normal_edge.sum()),
        "break_pixels": int(break_mask.sum()),
        "break_fraction_of_valid": float(break_mask.sum() / max(valid_count, 1)),
    }


def _apply_mesh_break(
    *,
    point_map_world: np.ndarray,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
    depth_rtol: float,
    normal_tol_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode or "none")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if mode == "none":
        return valid_mask, {"mode": "none", "valid_pixels": int(valid_mask.sum())}
    if mode != "depth_normal":
        raise ValueError(f"Unsupported camera-control mesh break mode: {mode!r}.")

    break_mask, edge_stats = _depth_normal_break_mask_np(
        point_map_world=point_map_world,
        depth_map=depth_map,
        valid_mask=valid_mask,
        depth_rtol=float(depth_rtol),
        normal_tol_deg=float(normal_tol_deg),
    )
    mesh_valid = valid_mask & ~break_mask
    return mesh_valid, {
        **edge_stats,
        "mode": mode,
        "mesh_depth_rtol": float(depth_rtol),
        "mesh_normal_tol_deg": float(normal_tol_deg),
        "mesh_valid_pixels": int(mesh_valid.sum()),
    }


def _splat_mesh_samples_to_view_fast(
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    target_c2w: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    *,
    source_xy: np.ndarray | None = None,
    return_source_xy: bool = False,
    z_eps: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2c = se3_inverse(target_c2w.astype(np.float32, copy=False))
    points_cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = points_cam[:, 2].astype(np.float32, copy=False)
    valid = np.isfinite(points_cam).all(axis=1) & (z > float(z_eps))
    if not np.any(valid):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        visible = np.zeros((height, width), dtype=bool)
        if return_source_xy:
            coords = np.full((height, width, 2), -1.0, dtype=np.float32)
            return frame, visible, coords
        return frame, visible

    points_cam = points_cam[valid]
    z = z[valid]
    colors = colors_u8[valid]
    source_xy_valid = None if source_xy is None else np.asarray(source_xy, dtype=np.float32)[valid]

    u = intrinsic[0, 0] * (points_cam[:, 0] / z) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (points_cam[:, 1] / z) + intrinsic[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        visible = np.zeros((height, width), dtype=bool)
        if return_source_xy:
            coords = np.full((height, width, 2), -1.0, dtype=np.float32)
            return frame, visible, coords
        return frame, visible

    flat_idx = (vi[inside] * width + ui[inside]).astype(np.int64, copy=False)
    z = z[inside]
    colors = colors[inside]
    if source_xy_valid is not None:
        source_xy_valid = source_xy_valid[inside]

    depth_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, flat_idx, z)
    keep = z == depth_buffer[flat_idx]
    kept_idx = flat_idx[keep]

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    visible = np.zeros(height * width, dtype=bool)
    frame.reshape(-1, 3)[kept_idx] = colors[keep]
    visible[kept_idx] = True
    visible = visible.reshape(height, width)
    if return_source_xy:
        coords = np.full((height * width, 2), -1.0, dtype=np.float32)
        if source_xy_valid is not None:
            coords[kept_idx] = source_xy_valid[keep]
        return frame, visible, coords.reshape(height, width, 2)
    return frame, visible


def _splat_mesh_samples_to_view_torch(
    points_world: torch.Tensor,
    colors01: torch.Tensor,
    target_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    fill_rgb01: torch.Tensor,
    *,
    source_xy: torch.Tensor | None = None,
    return_source_xy: bool = False,
    z_eps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w2c = se3_inverse(target_c2w.to(dtype=torch.float32))
    points_cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = points_cam[:, 2]
    valid = torch.isfinite(points_cam).all(dim=1) & (z > float(z_eps))
    points_cam = points_cam[valid]
    z = z[valid]
    colors = colors01[valid]
    source_xy_valid = None if source_xy is None else source_xy.to(device=points_world.device, dtype=torch.float32)[valid]

    u = intrinsic[0, 0] * (points_cam[:, 0] / z) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (points_cam[:, 1] / z) + intrinsic[1, 2]
    ui = torch.round(u).to(torch.long)
    vi = torch.round(v).to(torch.long)
    inside = (ui >= 0) & (ui < int(width)) & (vi >= 0) & (vi < int(height))
    flat_idx = (vi[inside] * int(width) + ui[inside]).to(torch.long)
    z = z[inside]
    colors = colors[inside]
    if source_xy_valid is not None:
        source_xy_valid = source_xy_valid[inside]

    depth_buffer = torch.full(
        (int(height) * int(width),),
        torch.inf,
        device=points_world.device,
        dtype=torch.float32,
    )
    depth_buffer.scatter_reduce_(0, flat_idx, z.to(torch.float32), reduce="amin", include_self=True)
    keep = z <= depth_buffer[flat_idx] + 1.0e-6
    kept_idx = flat_idx[keep]

    frame = fill_rgb01.view(1, 3).expand(int(height) * int(width), 3).clone()
    visible = torch.zeros(int(height) * int(width), device=points_world.device, dtype=torch.bool)
    frame[kept_idx] = colors[keep].to(torch.float32)
    visible[kept_idx] = True
    if return_source_xy:
        coords = torch.full(
            (int(height) * int(width), 2),
            -1.0,
            device=points_world.device,
            dtype=torch.float32,
        )
        if source_xy_valid is not None:
            coords[kept_idx] = source_xy_valid[keep].to(torch.float32)
        return frame.view(height, width, 3), visible.view(height, width), coords.view(height, width, 2)
    return frame.view(height, width, 3), visible.view(height, width)


def _target_fill_from_source_xy_torch(
    frame01: torch.Tensor,
    visible: torch.Tensor,
    source_xy: torch.Tensor,
    source_rgb01: torch.Tensor,
    radius: int,
    min_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radius = max(0, int(radius))
    if radius <= 0:
        fill_mask = torch.zeros_like(visible, dtype=torch.bool)
        return frame01, visible, fill_mask
    kernel_size = radius * 2 + 1
    min_neighbors = max(1, int(min_neighbors))
    area = float(kernel_size * kernel_size)

    visible_bool = visible.to(dtype=torch.bool)
    visible_f = visible_bool.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    neighbor_count = F.avg_pool2d(visible_f, kernel_size, stride=1, padding=radius) * area
    neighbor_count_2d = neighbor_count[0, 0]
    fill_mask = (~visible_bool) & (neighbor_count_2d >= float(min_neighbors))

    source_xy_chw = source_xy.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    source_xy_sum = F.avg_pool2d(source_xy_chw * visible_f, kernel_size, stride=1, padding=radius) * area
    source_xy_avg = (source_xy_sum / neighbor_count.clamp_min(1.0))[0].permute(1, 2, 0)

    src_h, src_w = int(source_rgb01.shape[0]), int(source_rgb01.shape[1])
    fill_mask = (
        fill_mask
        & torch.isfinite(source_xy_avg).all(dim=-1)
        & (source_xy_avg[..., 0] >= 0.0)
        & (source_xy_avg[..., 0] <= float(max(src_w - 1, 0)))
        & (source_xy_avg[..., 1] >= 0.0)
        & (source_xy_avg[..., 1] <= float(max(src_h - 1, 0)))
    )

    if src_w > 1:
        grid_x = source_xy_avg[..., 0] / float(src_w - 1) * 2.0 - 1.0
    else:
        grid_x = torch.zeros_like(source_xy_avg[..., 0])
    if src_h > 1:
        grid_y = source_xy_avg[..., 1] / float(src_h - 1) * 2.0 - 1.0
    else:
        grid_y = torch.zeros_like(source_xy_avg[..., 1])
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        source_rgb01.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0].permute(1, 2, 0)
    frame01 = torch.where(fill_mask[..., None], sampled, frame01)
    visible_bool = visible_bool | fill_mask
    return frame01, visible_bool, fill_mask


def _target_fill_from_source_xy_numpy(
    frame_u8: np.ndarray,
    visible: np.ndarray,
    source_xy: np.ndarray,
    source_rgb_u8: np.ndarray,
    radius: int,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_t = torch.from_numpy(np.asarray(frame_u8, dtype=np.float32) / 255.0)
    visible_t = torch.from_numpy(np.asarray(visible, dtype=bool))
    source_xy_t = torch.from_numpy(np.asarray(source_xy, dtype=np.float32))
    source_rgb_t = torch.from_numpy(np.asarray(source_rgb_u8, dtype=np.float32) / 255.0)
    filled_frame_t, filled_visible_t, fill_mask_t = _target_fill_from_source_xy_torch(
        frame_t,
        visible_t,
        source_xy_t,
        source_rgb_t,
        radius,
        min_neighbors,
    )
    filled_frame = np.clip(filled_frame_t.numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return filled_frame, filled_visible_t.numpy().astype(bool, copy=False), int(fill_mask_t.sum().item())


def _frames_uint8_to_video_tensor(frames_uint8: list[np.ndarray], height: int, width: int) -> torch.Tensor:
    video = torch.from_numpy(np.stack(frames_uint8, axis=0)).float() / 255.0
    video = video.permute(0, 3, 1, 2)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video * 2.0 - 1.0


def _frames01_to_video_tensor(frames01: torch.Tensor, height: int, width: int) -> torch.Tensor:
    video = frames01.to(dtype=torch.float32).permute(0, 3, 1, 2)
    if video.shape[-2:] != (int(height), int(width)):
        video = F.interpolate(video, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video * 2.0 - 1.0


def _visibility_frames_to_tensor(
    visibility_frames: list[np.ndarray] | torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if isinstance(visibility_frames, torch.Tensor):
        mask = visibility_frames.to(dtype=torch.float32)
    else:
        mask = torch.from_numpy(np.stack(visibility_frames, axis=0)).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.shape[-2:] != (int(height), int(width)):
        mask = F.interpolate(mask, size=(mask.shape[2], int(height), int(width)), mode="trilinear", align_corners=False)
    return mask.clamp_(0.0, 1.0)


@dataclass(frozen=True)
class Pi3XWarpRendererConfig:
    pi3_pixel_limit: int = CAMERA_CONTROL_PI3_PIXEL_LIMIT
    conf_threshold: float = CAMERA_CONTROL_CONF_THRESHOLD
    depth_edge_rtol: float = CAMERA_CONTROL_DEPTH_EDGE_RTOL
    mesh_samples_per_axis: int = CAMERA_CONTROL_MESH_SAMPLES_PER_AXIS
    render_mode: str = CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE
    target_fill_radius: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS
    target_fill_min_neighbors: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS
    mesh_break_mode: str = CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE
    mesh_depth_rtol: float = CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL
    mesh_normal_tol_deg: float = CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG


class Pi3XWarpRenderer:


    def __init__(self, config: Pi3XWarpRendererConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = config if config is not None else Pi3XWarpRendererConfig(**kwargs)
        self._pi3x_runtime: dict[str, Any] | None = None

    def _runtime_key(self) -> tuple[str, str | None]:
        repo = default_pi3_repo().expanduser().resolve()
        ckpt_key = str(default_pi3x_ckpt().expanduser().resolve())
        return str(repo), ckpt_key

    def _get_pi3x_runtime(self, device: torch.device) -> dict[str, Any]:
        repo_key, ckpt_key = self._runtime_key()
        runtime = self._pi3x_runtime
        if runtime is None or runtime.get("repo_key") != repo_key or runtime.get("ckpt_key") != ckpt_key:
            Pi3X, recover_intrinsic_from_rays_d = _import_pi3(Path(repo_key))
            model = _load_pi3x_model(Pi3X, device)
            runtime = {
                "ckpt_key": ckpt_key,
                "model": model,
                "recover_intrinsic_from_rays_d": recover_intrinsic_from_rays_d,
                "repo_key": repo_key,
            }
            self._pi3x_runtime = runtime

        model = runtime["model"]
        if next(model.parameters()).device != device:
            model = model.to(device)
            runtime["model"] = model
        return runtime

    def estimate_first_frame_geometry(self, image_tensor: torch.Tensor, device: torch.device | None = None) -> dict[str, Any]:
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
            raise ValueError(f"image_tensor must be [1, 3, H, W] in [-1, 1], got {tuple(image_tensor.shape)}.")
        device = torch.device(device or image_tensor.device)
        runtime = self._get_pi3x_runtime(device)
        model = runtime["model"]
        recover_intrinsic_from_rays_d = runtime["recover_intrinsic_from_rays_d"]

        src_height = int(image_tensor.shape[-2])
        src_width = int(image_tensor.shape[-1])
        render_height, render_width = _pi3_target_size(src_width, src_height, int(self.config.pi3_pixel_limit))

        pi3_imgs = ((image_tensor.to(device=device, dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(1)
        pi3_imgs = F.interpolate(
            pi3_imgs.reshape(-1, pi3_imgs.shape[2], pi3_imgs.shape[3], pi3_imgs.shape[4]),
            size=(render_height, render_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(1, 1, 3, render_height, render_width)

        _force_pi3x_float_heads(model)
        with torch.no_grad(), _autocast_context(device):
            res = model(
                imgs=pi3_imgs,
                depths=None,
                intrinsics=None,
                rays=None,
                poses=None,
                with_prior=False,
            )

        rays_d = F.normalize(res["local_points"][0].detach(), dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
        intrinsics = _normalize_intrinsics_shape(intrinsics, 1)[0].numpy()
        conf = torch.sigmoid(res["conf"][..., 0])
        local_finite = torch.isfinite(res["local_points"]).all(dim=-1) & (res["local_points"][..., 2] > 0.0)
        valid = (conf > float(self.config.conf_threshold)) & local_finite
        conf_map = conf[0, 0].detach().float().cpu().numpy()
        depth_map = res["local_points"][0, 0, ..., 2].detach().float().cpu().numpy()
        point_map_world = res["points"][0, 0].detach().float().cpu().numpy()
        valid_mask = valid[0, 0].detach().cpu().numpy()
        break_mask, edge_filter_stats = _depth_normal_break_mask_np(
            point_map_world=point_map_world,
            depth_map=depth_map,
            valid_mask=valid_mask,
            depth_rtol=float(self.config.depth_edge_rtol),
            normal_tol_deg=float(self.config.mesh_normal_tol_deg),
        )
        valid_mask = valid_mask & ~break_mask

        geometry = {
            "conf_map": conf_map,
            "depth_map": depth_map,
            "depth_normal_edge_filter_stats": edge_filter_stats,
            "intrinsic": intrinsics.astype(np.float32, copy=False),
            "point_map_world": point_map_world,
            "render_height": render_height,
            "render_width": render_width,
            "source_pose": res["camera_poses"][0, 0].detach().float().cpu().numpy(),
            "source_rgb_u8": np.clip(
                pi3_imgs[0, 0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0 + 0.5,
                0,
                255,
            ).astype(np.uint8),
            "valid_mask": valid_mask,
        }
        return self.ensure_valid_depth_geometry(geometry, reason="pi3x_runtime")

    def estimate_keyframe_geometry(
        self,
        image_tensors: list[torch.Tensor],
        device: torch.device | None = None,
        scale_reference_geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keyframe_tensors: list[torch.Tensor] = []
        for image_tensor in image_tensors:
            if image_tensor is None:
                continue
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.ndim != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
                raise ValueError(
                    "Pi3X keyframe geometry expects each image tensor to have shape [1, 3, H, W] "
                    f"or [3, H, W], got {tuple(image_tensor.shape)}."
                )
            keyframe_tensors.append(image_tensor.detach().float())
        if not keyframe_tensors:
            raise ValueError("Pi3X keyframe geometry requires at least one keyframe image.")

        device = torch.device(device or keyframe_tensors[-1].device)
        keyframe_tensors = [tensor.to(device=device, dtype=torch.float32) for tensor in keyframe_tensors]
        runtime = self._get_pi3x_runtime(device)
        model = runtime["model"]
        recover_intrinsic_from_rays_d = runtime["recover_intrinsic_from_rays_d"]

        src_height = int(keyframe_tensors[-1].shape[-2])
        src_width = int(keyframe_tensors[-1].shape[-1])
        render_height, render_width = _pi3_target_size(src_width, src_height, int(self.config.pi3_pixel_limit))

        pi3_frame_batch = []
        for keyframe_tensor in keyframe_tensors:
            frame01 = ((keyframe_tensor + 1.0) * 0.5).clamp(0.0, 1.0)
            frame01 = F.interpolate(
                frame01,
                size=(render_height, render_width),
                mode="bilinear",
                align_corners=False,
            )
            pi3_frame_batch.append(frame01[0])
        pi3_imgs = torch.stack(pi3_frame_batch, dim=0).unsqueeze(0)
        num_keyframes = int(pi3_imgs.shape[1])

        _force_pi3x_float_heads(model)
        with torch.no_grad(), _autocast_context(device):
            res = model(
                imgs=pi3_imgs,
                depths=None,
                intrinsics=None,
                rays=None,
                poses=None,
                with_prior=True,
            )

        rays_d = F.normalize(res["local_points"][0].detach(), dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
        recovered_intrinsics_np = _normalize_intrinsics_shape(intrinsics, num_keyframes).numpy()
        intrinsics_np, intrinsic_smoothing_stats = _smooth_pi3x_keyframe_intrinsics(
            recovered_intrinsics_np,
            render_height=render_height,
            render_width=render_width,
        )
        if intrinsics_np.shape != (num_keyframes, 3, 3):
            raise ValueError(
                "Recovered Pi3X keyframe intrinsics have unexpected shape "
                f"{intrinsics_np.shape}, expected {(num_keyframes, 3, 3)}."
            )

        conf = torch.sigmoid(res["conf"][..., 0])
        local_finite = torch.isfinite(res["local_points"]).all(dim=-1) & (res["local_points"][..., 2] > 0.0)
        valid = (conf > float(self.config.conf_threshold)) & local_finite

        conf_maps = conf[0].detach().float().cpu().numpy()
        depth_maps = res["local_points"][0, ..., 2].detach().float().cpu().numpy()
        point_maps_world = res["points"][0].detach().float().cpu().numpy()
        source_poses = res["camera_poses"][0].detach().float().cpu().numpy()
        valid_masks = valid[0].detach().cpu().numpy()
        edge_filter_stats = []
        for keyframe_index in range(num_keyframes):
            break_mask, stats = _depth_normal_break_mask_np(
                point_map_world=point_maps_world[keyframe_index],
                depth_map=depth_maps[keyframe_index],
                valid_mask=valid_masks[keyframe_index],
                depth_rtol=float(self.config.depth_edge_rtol),
                normal_tol_deg=float(self.config.mesh_normal_tol_deg),
            )
            valid_masks[keyframe_index] = valid_masks[keyframe_index] & ~break_mask
            stats["keyframe_index"] = int(keyframe_index)
            edge_filter_stats.append(stats)
        scale_alignment_stats: dict[str, Any] = {
            "applied_scale": 1.0,
            "policy": "none",
            "status": "disabled",
        }
        if scale_reference_geometry is not None:
            reference_valid = scale_reference_geometry.get("valid_mask")
            scale, scale_alignment_stats = _depth_scale_to_reference(
                reference_depth=np.asarray(scale_reference_geometry["depth_map"], dtype=np.float32),
                reference_valid_mask=None
                if reference_valid is None
                else np.asarray(reference_valid, dtype=bool),
                current_depth=depth_maps[0],
                current_valid_mask=valid_masks[0],
            )
            depth_maps = depth_maps * float(scale)
            point_maps_world = point_maps_world * float(scale)
            source_poses = source_poses.copy()
            source_poses[:, :3, 3] *= float(scale)

        source_rgb_u8 = np.clip(
            pi3_imgs[0].detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)

        keyframe_geometries = []
        for keyframe_index in range(num_keyframes):
            keyframe_geometry = {
                "conf_map": conf_maps[keyframe_index],
                "depth_map": depth_maps[keyframe_index],
                "depth_normal_edge_filter_stats": edge_filter_stats[keyframe_index],
                "geometry_backend": "pi3x_keyframe",
                "intrinsic": intrinsics_np[keyframe_index].astype(np.float32, copy=False),
                "intrinsic_recovered": recovered_intrinsics_np[keyframe_index].astype(np.float32, copy=False),
                "keyframe_index": keyframe_index,
                "keyframe_count": num_keyframes,
                "point_map_world": point_maps_world[keyframe_index],
                "pose_source": "pi3x_estimated",
                "render_height": render_height,
                "render_width": render_width,
                "scale_alignment": scale_alignment_stats,
                "source_pose": source_poses[keyframe_index],
                "source_rgb_u8": source_rgb_u8[keyframe_index],
                "valid_mask": valid_masks[keyframe_index],
            }
            keyframe_geometries.append(
                self.ensure_valid_depth_geometry(keyframe_geometry, reason=f"pi3x_keyframe_{keyframe_index}")
            )

        latest_geometry = keyframe_geometries[-1]

        return {
            "geometry_backend": "pi3x_keyframes",
            "intrinsic": latest_geometry["intrinsic"],
            "intrinsic_smoothing_stats": intrinsic_smoothing_stats,
            "intrinsics_source": "pi3x_recovered_focal_clamped_smoothed",
            "keyframe_count": num_keyframes,
            "keyframe_geometries": keyframe_geometries,
            "pose_source": "pi3x_estimated",
            "preserve_pi3x_keyframe_points": True,
            "render_height": render_height,
            "render_width": render_width,
            "scale_alignment_stats": scale_alignment_stats,
            "scale_source": str(scale_alignment_stats.get("policy", "none")),
            "source_pose": latest_geometry["source_pose"],
            "source_rgb_u8": latest_geometry["source_rgb_u8"],
            "depth_map": latest_geometry["depth_map"],
            "valid_mask": latest_geometry.get("valid_mask"),
        }

    def ensure_valid_depth_geometry(
        self,
        geometry: dict[str, Any],
        *,
        reason: str,
        top_percent: float | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        depth_map = np.asarray(geometry["depth_map"], dtype=np.float32)
        valid_mask = np.asarray(geometry["valid_mask"], dtype=bool)
        positive_depth = np.isfinite(depth_map) & (depth_map > 0.0)
        if not force and bool((valid_mask & positive_depth).any()):
            return geometry

        conf_map = np.asarray(geometry.get("conf_map", np.ones_like(depth_map, dtype=np.float32)), dtype=np.float32)
        if conf_map.shape != depth_map.shape:
            conf_map = np.ones_like(depth_map, dtype=np.float32)
        point_map_world = geometry.get("point_map_world")
        if point_map_world is not None:
            point_valid = np.isfinite(np.asarray(point_map_world)).all(axis=-1)
        else:
            point_valid = np.ones_like(depth_map, dtype=bool)
        candidate_mask = positive_depth & point_valid & np.isfinite(conf_map)
        if not bool(candidate_mask.any()):
            candidate_mask = positive_depth
        if not bool(candidate_mask.any()):
            geometry["valid_mask"] = candidate_mask
            geometry["empty_depth_fallback"] = {"reason": reason, "mode": "none_no_positive_depth", "kept_pixels": 0}
            return geometry

        candidate_conf = conf_map[candidate_mask]
        if candidate_conf.size > 0 and np.isfinite(candidate_conf).any():
            top_percent_value = (
                float(CAMERA_CONTROL_EMPTY_DEPTH_FALLBACK_TOP_PERCENT)
                if top_percent is None
                else float(top_percent)
            )
            keep_count = max(1, int(math.ceil(candidate_conf.size * top_percent_value / 100.0)))
            threshold = np.partition(candidate_conf, max(0, candidate_conf.size - keep_count))[
                max(0, candidate_conf.size - keep_count)
            ]
            fallback_mask = candidate_mask & (conf_map >= threshold)
            mode = "top_confidence_percent"
        else:
            fallback_mask = candidate_mask
            top_percent_value = 100.0
            mode = "all_positive_depth_no_confidence"

        fallback_mask = fallback_mask.astype(bool, copy=False)
        relaxed_conf_map = np.asarray(conf_map, dtype=np.float32).copy()
        relaxed_conf_map[fallback_mask] = np.maximum(
            relaxed_conf_map[fallback_mask],
            np.float32(float(self.config.conf_threshold) + 1.0e-3),
        )
        geometry["valid_mask"] = fallback_mask
        geometry["conf_map"] = relaxed_conf_map
        geometry["empty_depth_fallback"] = {
            "reason": reason,
            "mode": mode,
            "top_percent": float(top_percent_value),
            "candidate_pixels": int(candidate_mask.sum()),
            "kept_pixels": int(fallback_mask.sum()),
        }
        return geometry

    def estimate_first_frame_depth_scale(self, geometry: dict[str, Any]) -> float:
        geometry = self.ensure_valid_depth_geometry(geometry, reason="depth_scale")
        depth_map = np.asarray(geometry["depth_map"], dtype=np.float32)
        valid_mask = np.asarray(geometry["valid_mask"], dtype=bool)
        valid_depths = depth_map[valid_mask & np.isfinite(depth_map) & (depth_map > 0.0)]
        if valid_depths.size == 0:
            valid_depths = depth_map[np.isfinite(depth_map) & (depth_map > 0.0)]
        if valid_depths.size == 0:
            return 1.0
        return float(np.median(valid_depths))

    def _sample_geometry_mesh_with_retry(
        self,
        geometry: dict[str, Any],
        mesh_break_mode: str,
        *,
        samples_per_axis: int,
        use_top_fill: bool,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]:
        def sample(current_geometry: dict[str, Any], current_mesh_break_mode: str):
            if use_top_fill:
                augmented_points, augmented_valid_mask, fill_mask, fill_stats = _fill_top_invalid_with_far_plane(
                    point_map_world=current_geometry["point_map_world"],
                    depth_map=current_geometry["depth_map"],
                    conf_map=current_geometry["conf_map"],
                    source_pose=current_geometry["source_pose"],
                    intrinsic=current_geometry["intrinsic"],
                    conf_threshold=float(self.config.conf_threshold),
                    fill_top_max_y_frac=CAMERA_CONTROL_FILL_TOP_MAX_Y_FRAC,
                    fill_min_component_area=CAMERA_CONTROL_FILL_MIN_COMPONENT_AREA,
                    fill_boundary_kernel=CAMERA_CONTROL_FILL_BOUNDARY_KERNEL,
                    fill_boundary_min_samples=CAMERA_CONTROL_FILL_BOUNDARY_MIN_SAMPLES,
                    fill_boundary_depth_quantile=CAMERA_CONTROL_FILL_BOUNDARY_DEPTH_QUANTILE,
                    fill_global_depth_quantile=CAMERA_CONTROL_FILL_GLOBAL_DEPTH_QUANTILE,
                )
                mesh_depth_map = current_geometry["depth_map"]
            else:
                augmented_points = np.asarray(current_geometry["point_map_world"], dtype=np.float32)
                augmented_valid_mask = np.asarray(current_geometry["valid_mask"], dtype=bool)
                augmented_valid_mask = augmented_valid_mask & np.isfinite(augmented_points).all(axis=-1)
                fill_mask = np.zeros_like(augmented_valid_mask, dtype=bool)
                fill_stats = []
                mesh_depth_map = current_geometry["depth_map"]

            mesh_valid_mask, mesh_break_stats = _apply_mesh_break(
                point_map_world=augmented_points,
                depth_map=mesh_depth_map,
                valid_mask=augmented_valid_mask,
                mode=current_mesh_break_mode,
                depth_rtol=float(self.config.mesh_depth_rtol),
                normal_tol_deg=float(self.config.mesh_normal_tol_deg),
            )
            mesh_break_stats["fill_pixels"] = int(np.asarray(fill_mask, dtype=bool).sum())
            mesh_break_stats["fill_stats"] = fill_stats
            mesh_break_stats["geometry_backend"] = str(current_geometry.get("geometry_backend", "pi3x"))
            mesh_fill_mask = fill_mask & mesh_valid_mask
            sampled = _sample_mesh_quads_rejecting_mixed_fill(
                point_map=augmented_points,
                color_map=current_geometry["source_rgb_u8"],
                valid_mask=mesh_valid_mask,
                fill_mask=mesh_fill_mask,
                samples_per_axis=samples_per_axis,
                allow_mixed_fill_quads=False,
            )
            return sampled, mesh_break_stats

        try:
            sampled, mesh_break_stats = sample(geometry, mesh_break_mode)
            return sampled, mesh_break_stats, geometry
        except ValueError as exc:
            if "No valid first-frame mesh quads" not in str(exc):
                raise
            last_error = exc
            for top_percent in (20.0, 50.0, 100.0):
                retry_geometry = self.ensure_valid_depth_geometry(
                    dict(geometry),
                    reason="mesh_quad_retry",
                    top_percent=top_percent,
                    force=True,
                )
                for retry_mesh_break_mode in (mesh_break_mode, "none"):
                    try:
                        sampled, mesh_break_stats = sample(retry_geometry, retry_mesh_break_mode)
                        mesh_break_stats["retry_top_percent"] = float(top_percent)
                        mesh_break_stats["retry_mesh_break_mode"] = retry_mesh_break_mode
                        return sampled, mesh_break_stats, retry_geometry
                    except ValueError as retry_exc:
                        if "No valid first-frame mesh quads" not in str(retry_exc):
                            raise
                        last_error = retry_exc
            raise last_error

    def render_from_geometry(
        self,
        geometry: dict[str, Any],
        target_relative_poses: torch.Tensor | np.ndarray,
        *,
        height: int,
        width: int,
        device: torch.device | None = None,
        target_intrinsics: torch.Tensor | np.ndarray | None = None,
        chunk_index: int | None = None,
        invisible_fill_mode: str = CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
        render_mode: str | None = None,
        target_fill_radius: int | None = None,
        target_fill_min_neighbors: int | None = None,
        mesh_break_mode: str | None = None,
    ) -> dict[str, Any]:
        device = torch.device(device or "cpu")
        render_height = int(geometry["render_height"])
        render_width = int(geometry["render_width"])
        relative_target_poses = as_pose4x4(target_relative_poses).cpu().numpy().astype(np.float32, copy=False)

        if target_intrinsics is None:
            render_intrinsics_np = np.repeat(
                np.asarray(geometry["intrinsic"], dtype=np.float32)[None],
                relative_target_poses.shape[0],
                axis=0,
            )
        elif isinstance(target_intrinsics, torch.Tensor):
            render_intrinsics_np = target_intrinsics.detach().float().cpu().numpy().astype(np.float32, copy=False)
        else:
            render_intrinsics_np = np.asarray(target_intrinsics, dtype=np.float32)
        if render_intrinsics_np.shape != (relative_target_poses.shape[0], 3, 3):
            raise ValueError(
                "target_intrinsics must match target_relative_poses: "
                f"expected {(relative_target_poses.shape[0], 3, 3)}, got {render_intrinsics_np.shape}."
            )
        if target_intrinsics is not None:
            if "keyframe_geometries" in geometry:
                if not bool(geometry.get("preserve_pi3x_keyframe_points", False)):
                    updated_keyframes = [
                        _geometry_with_intrinsic(keyframe_geometry, render_intrinsics_np[0])
                        for keyframe_geometry in geometry["keyframe_geometries"]
                    ]
                    latest_geometry = updated_keyframes[-1]
                    geometry = dict(geometry)
                    geometry["intrinsic"] = latest_geometry["intrinsic"]
                    geometry["intrinsics_source"] = "fixed_render_intrinsic_reproject_fallback"
                    geometry["keyframe_geometries"] = updated_keyframes
                    geometry["source_pose"] = latest_geometry["source_pose"]
                    geometry["source_rgb_u8"] = latest_geometry["source_rgb_u8"]
            else:
                geometry = _geometry_with_intrinsic(geometry, render_intrinsics_np[0])

        keyframe_geometries = geometry.get("keyframe_geometries")
        if keyframe_geometries is not None:
            latest_keyframe_intrinsic = np.asarray(keyframe_geometries[-1]["intrinsic"], dtype=np.float32)
            render_intrinsics_np = np.repeat(
                latest_keyframe_intrinsic[None],
                relative_target_poses.shape[0],
                axis=0,
            ).astype(np.float32, copy=False)
            geometry = dict(geometry)
            geometry["target_intrinsics_stats"] = {
                "chunk_index": None if chunk_index is None else int(chunk_index),
                "end": _intrinsic_stats(render_intrinsics_np[-1]),
                "policy": "prev_pi3x_source",
                "source": _intrinsic_stats(latest_keyframe_intrinsic),
                "start": _intrinsic_stats(render_intrinsics_np[0]),
            }
            if render_intrinsics_np.shape[0] > 1:
                geometry["target_intrinsics_stats"]["first_target"] = _intrinsic_stats(render_intrinsics_np[1])

        background_atlas = _build_keyframe_background_atlas(list(keyframe_geometries)) if keyframe_geometries is not None else None

        mesh_break_stats_list: list[dict[str, Any]] = []
        if keyframe_geometries is not None:
            sampled_points_parts: list[np.ndarray] = []
            sampled_color_parts: list[np.ndarray] = []
            sampled_source_xy_parts: list[np.ndarray] = []
            keyframe_count = len(keyframe_geometries)
            for keyframe_index, keyframe_geometry in enumerate(keyframe_geometries):
                is_latest_keyframe = keyframe_index == keyframe_count - 1
                samples_per_axis = (
                    int(self.config.mesh_samples_per_axis)
                    if is_latest_keyframe
                    else int(CAMERA_CONTROL_PI3X_KEYFRAME_PREVIOUS_MESH_SAMPLES_PER_AXIS)
                )
                use_top_fill = bool(is_latest_keyframe and background_atlas is None)
                try:
                    (keyframe_points, keyframe_colors, keyframe_source_xy), mesh_break_stats, _ = (
                        self._sample_geometry_mesh_with_retry(
                            keyframe_geometry,
                            str(mesh_break_mode or self.config.mesh_break_mode),
                            samples_per_axis=samples_per_axis,
                            use_top_fill=use_top_fill,
                        )
                    )
                except ValueError as exc:
                    if is_latest_keyframe or "No valid first-frame mesh quads" not in str(exc):
                        raise
                    continue
                mesh_break_stats["keyframe_index"] = int(keyframe_index)
                mesh_break_stats["keyframe_count"] = int(keyframe_count)
                mesh_break_stats["samples_per_axis"] = int(samples_per_axis)
                mesh_break_stats["use_top_fill"] = bool(use_top_fill)
                mesh_break_stats["background_atlas"] = background_atlas is not None
                mesh_break_stats_list.append(dict(mesh_break_stats))
                sampled_points_parts.append(keyframe_points)
                sampled_color_parts.append(keyframe_colors)
                sampled_source_xy_parts.append(keyframe_source_xy)
            if not sampled_points_parts:
                raise ValueError("No Pi3X keyframe mesh samples were available for camera warp rendering.")
            sampled_points_world = np.concatenate(sampled_points_parts, axis=0)
            sampled_colors_u8 = np.concatenate(sampled_color_parts, axis=0)
            sampled_source_xy = np.concatenate(sampled_source_xy_parts, axis=0)
        else:
            (sampled_points_world, sampled_colors_u8, sampled_source_xy), mesh_break_stats, geometry = (
                self._sample_geometry_mesh_with_retry(
                    geometry,
                    str(mesh_break_mode or self.config.mesh_break_mode),
                    samples_per_axis=int(self.config.mesh_samples_per_axis),
                    use_top_fill=True,
                )
            )
            mesh_break_stats["samples_per_axis"] = int(self.config.mesh_samples_per_axis)
            mesh_break_stats["use_top_fill"] = True
            mesh_break_stats_list.append(dict(mesh_break_stats))

        source_pose = np.asarray(geometry["source_pose"], dtype=np.float32)
        target_poses_world = np.einsum("ij,tjk->tik", source_pose, relative_target_poses)
        if invisible_fill_mode == "mean_first_frame":
            invisible_fill_rgb = np.rint(geometry["source_rgb_u8"].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
        elif invisible_fill_mode == "black":
            invisible_fill_rgb = np.zeros((3,), dtype=np.uint8)
        else:
            raise ValueError(
                "invisible_fill_mode must be one of {'mean_first_frame', 'black'}, "
                f"got {invisible_fill_mode!r}."
            )

        background_frames_uint8: list[np.ndarray] | None = None
        background_valid_frames: list[np.ndarray] | None = None
        if background_atlas is not None:
            background_frames_uint8 = []
            background_valid_frames = []
            for frame_idx in range(relative_target_poses.shape[0]):
                background_frame, background_valid = _render_background_atlas(
                    background_atlas,
                    target_poses_world[frame_idx],
                    render_intrinsics_np[frame_idx],
                    render_height,
                    render_width,
                    invisible_fill_rgb,
                )
                background_frames_uint8.append(background_frame)
                background_valid_frames.append(background_valid)

        warp_render_mode = str(render_mode or self.config.render_mode)
        if warp_render_mode not in CAMERA_CONTROL_WARP_RENDER_MODES:
            raise ValueError(
                f"render_mode must be one of {sorted(CAMERA_CONTROL_WARP_RENDER_MODES)}, got {warp_render_mode!r}."
            )
        fill_radius = int(self.config.target_fill_radius if target_fill_radius is None else target_fill_radius)
        fill_min_neighbors = int(
            self.config.target_fill_min_neighbors
            if target_fill_min_neighbors is None
            else target_fill_min_neighbors
        )
        target_fill_enabled = warp_render_mode == "target_fill" and fill_radius > 0
        warp_render_stats: dict[str, Any] = {
            "frame_count": int(relative_target_poses.shape[0]),
            "mode": warp_render_mode,
            "target_fill_min_neighbors": int(fill_min_neighbors),
            "target_fill_radius": int(fill_radius),
        }
        if background_atlas is not None:
            warp_render_stats["background_atlas"] = background_atlas["stats"]

        if device.type == "cuda":
            sampled_points_world_t = torch.from_numpy(sampled_points_world).to(device=device, dtype=torch.float32)
            sampled_colors01_t = torch.from_numpy(sampled_colors_u8).to(device=device, dtype=torch.float32) / 255.0
            sampled_source_xy_t = None
            source_rgb01_t = None
            if target_fill_enabled:
                sampled_source_xy_t = torch.from_numpy(sampled_source_xy).to(device=device, dtype=torch.float32)
                source_rgb01_t = (
                    torch.from_numpy(geometry["source_rgb_u8"]).to(device=device, dtype=torch.float32) / 255.0
                )
            target_poses_world_t = torch.from_numpy(target_poses_world).to(device=device, dtype=torch.float32)
            render_intrinsics_t = torch.from_numpy(render_intrinsics_np).to(device=device, dtype=torch.float32)
            fill_rgb01_t = torch.from_numpy(invisible_fill_rgb).to(device=device, dtype=torch.float32) / 255.0
            background_frames01_t = None
            background_valid_t = None
            if background_frames_uint8 is not None and background_valid_frames is not None:
                background_frames01_t = (
                    torch.from_numpy(np.stack(background_frames_uint8, axis=0)).to(device=device, dtype=torch.float32)
                    / 255.0
                )
                background_valid_t = torch.from_numpy(np.stack(background_valid_frames, axis=0)).to(
                    device=device,
                    dtype=torch.bool,
                )
            warp_frames01: list[torch.Tensor] = [
                torch.from_numpy(geometry["source_rgb_u8"]).to(device=device, dtype=torch.float32) / 255.0
            ]
            visibility_frames_t: list[torch.Tensor] = [
                torch.ones((render_height, render_width), device=device, dtype=torch.float32)
            ]
            target_fill_counts_t: list[torch.Tensor] = []
            for frame_idx in range(1, relative_target_poses.shape[0]):
                if target_fill_enabled:
                    frame, visible, source_xy_frame = _splat_mesh_samples_to_view_torch(
                        sampled_points_world_t,
                        sampled_colors01_t,
                        target_poses_world_t[frame_idx],
                        render_intrinsics_t[frame_idx],
                        render_height,
                        render_width,
                        fill_rgb01_t,
                        source_xy=sampled_source_xy_t,
                        return_source_xy=True,
                    )
                    frame, visible, fill_mask = _target_fill_from_source_xy_torch(
                        frame,
                        visible,
                        source_xy_frame,
                        source_rgb01_t,
                        fill_radius,
                        fill_min_neighbors,
                    )
                    target_fill_counts_t.append(fill_mask.sum().to(dtype=torch.float32))
                else:
                    frame, visible = _splat_mesh_samples_to_view_torch(
                        sampled_points_world_t,
                        sampled_colors01_t,
                        target_poses_world_t[frame_idx],
                        render_intrinsics_t[frame_idx],
                        render_height,
                        render_width,
                        fill_rgb01_t,
                    )
                visible = visible.to(dtype=torch.bool)
                if background_frames01_t is not None and background_valid_t is not None:
                    frame = torch.where(visible[..., None], frame, background_frames01_t[frame_idx])
                    visible = visible | background_valid_t[frame_idx]
                warp_frames01.append(frame)
                visibility_frames_t.append(visible.to(dtype=torch.float32))

            if target_fill_counts_t:
                target_fill_counts = torch.stack(target_fill_counts_t)
                warp_render_stats["target_fill_pixels"] = int(target_fill_counts.sum().detach().cpu().item())
                warp_render_stats["target_fill_mean_pixels_per_frame"] = float(
                    target_fill_counts.mean().detach().cpu().item()
                )
            visibility_frames = torch.stack(visibility_frames_t, dim=0)
            return {
                "geometry": geometry,
                "mesh_break_stats": mesh_break_stats_list,
                "visibility_frames": visibility_frames,
                "warp_render_stats": warp_render_stats,
                "warp_video": _frames01_to_video_tensor(torch.stack(warp_frames01, dim=0), height=height, width=width),
                "warp_visibility_mask": _visibility_frames_to_tensor(visibility_frames, height=height, width=width),
            }

        warp_frames_uint8: list[np.ndarray] = [geometry["source_rgb_u8"]]
        visibility_frames_np: list[np.ndarray] = [np.ones((render_height, render_width), dtype=np.float32)]
        target_fill_pixels = 0
        for frame_idx in range(1, relative_target_poses.shape[0]):
            if target_fill_enabled:
                frame, visible, source_xy_frame = _splat_mesh_samples_to_view_fast(
                    sampled_points_world,
                    sampled_colors_u8,
                    target_poses_world[frame_idx],
                    render_intrinsics_np[frame_idx],
                    render_height,
                    render_width,
                    source_xy=sampled_source_xy,
                    return_source_xy=True,
                )
                frame, visible, fill_count = _target_fill_from_source_xy_numpy(
                    frame,
                    visible,
                    source_xy_frame,
                    geometry["source_rgb_u8"],
                    fill_radius,
                    fill_min_neighbors,
                )
                target_fill_pixels += int(fill_count)
            else:
                frame, visible = _splat_mesh_samples_to_view_fast(
                    sampled_points_world,
                    sampled_colors_u8,
                    target_poses_world[frame_idx],
                    render_intrinsics_np[frame_idx],
                    render_height,
                    render_width,
                )
            if background_frames_uint8 is not None and background_valid_frames is not None:
                background_frame = background_frames_uint8[frame_idx]
                background_valid = background_valid_frames[frame_idx]
                frame[~visible] = background_frame[~visible]
                visible = visible | background_valid
            else:
                frame[~visible] = invisible_fill_rgb
            warp_frames_uint8.append(frame)
            visibility_frames_np.append(visible.astype(np.float32, copy=False))

        if target_fill_enabled:
            warp_render_stats["target_fill_pixels"] = int(target_fill_pixels)
            warp_render_stats["target_fill_mean_pixels_per_frame"] = float(
                target_fill_pixels / max(int(relative_target_poses.shape[0]) - 1, 1)
            )
        return {
            "geometry": geometry,
            "mesh_break_stats": mesh_break_stats_list,
            "visibility_frames": visibility_frames_np,
            "warp_render_stats": warp_render_stats,
            "warp_video": _frames_uint8_to_video_tensor(warp_frames_uint8, height=height, width=width),
            "warp_visibility_mask": _visibility_frames_to_tensor(visibility_frames_np, height=height, width=width),
        }

    def render(
        self,
        image_tensor: torch.Tensor,
        camera_poses: torch.Tensor | np.ndarray,
        *,
        height: int,
        width: int,
        num_frames: int = CAMERA_CONTROL_NUM_FRAMES,
        device: torch.device | None = None,
        geometry: dict[str, Any] | None = None,
        target_intrinsics: torch.Tensor | np.ndarray | None = None,
        chunk_index: int | None = None,
        translation_scale: float = CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
        translation_scale_use_first_frame_depth: bool = CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH,
        invisible_fill_mode: str = CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
        render_mode: str | None = None,
        target_fill_radius: int | None = None,
        target_fill_min_neighbors: int | None = None,
        mesh_break_mode: str | None = None,
    ) -> dict[str, Any]:
        device = torch.device(device or image_tensor.device)
        if geometry is None:
            geometry = self.estimate_first_frame_geometry(image_tensor=image_tensor, device=device)
        effective_scale = float(translation_scale)
        if bool(translation_scale_use_first_frame_depth):
            effective_scale *= self.estimate_first_frame_depth_scale(geometry)
        pose_rollout = prepare_camera_pose_rollout(
            scale_camera_pose_translations(camera_poses, effective_scale),
            int(num_frames),
        )
        rendered = self.render_from_geometry(
            geometry,
            pose_rollout,
            height=int(height),
            width=int(width),
            device=device,
            target_intrinsics=target_intrinsics,
            chunk_index=chunk_index,
            invisible_fill_mode=invisible_fill_mode,
            render_mode=render_mode,
            target_fill_radius=target_fill_radius,
            target_fill_min_neighbors=target_fill_min_neighbors,
            mesh_break_mode=mesh_break_mode,
        )
        rendered["camera_pose_rollout"] = pose_rollout
        rendered["camera_translation_effective_scale"] = float(effective_scale)
        return rendered


def render_pi3x_camera_warp(
    image_tensor: torch.Tensor,
    camera_poses: torch.Tensor | np.ndarray,
    *,
    height: int,
    width: int,
    num_frames: int = CAMERA_CONTROL_NUM_FRAMES,
    device: torch.device | None = None,
    renderer: Pi3XWarpRenderer | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    renderer = renderer or Pi3XWarpRenderer()
    return renderer.render(
        image_tensor=image_tensor,
        camera_poses=camera_poses,
        height=height,
        width=width,
        num_frames=num_frames,
        device=device,
        **kwargs,
    )

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


POSE_FPS = 30
SOURCE_HEIGHT = 720
SOURCE_WIDTH = 1280
SECONDS_PER_CHUNK = 1.5
OUTPUT_FPS = 24


def validate_camera_samples(samples: list[dict], chunks: int) -> np.ndarray:

    if not isinstance(samples, list) or len(samples) != int(chunks * SECONDS_PER_CHUNK * POSE_FPS):
        raise ValueError("cameraSamples must contain exactly 45 samples per chunk")
    try:
        values = np.asarray([[sample[k] for k in ("x", "y", "z", "yaw", "pitch")] for sample in samples], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each camera sample needs numeric x/y/z/yaw/pitch") from error
    if not np.isfinite(values).all() or np.abs(values).max() > 1e6:
        raise ValueError("Camera samples must be finite and within +/- 1e6")
    return values


def _recorded_poses(samples: list[dict], chunks: int) -> np.ndarray:
    values = validate_camera_samples(samples, chunks)
    yaw, pitch = values[:, 3], values[:, 4]
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    poses = np.repeat(np.eye(4)[None], len(values), axis=0)
    poses[:, :3, 0] = np.stack((cy, np.zeros_like(cy), -sy), axis=1)
    poses[:, :3, 1] = np.stack((sy * sp, cp, cy * sp), axis=1)
    poses[:, :3, 2] = np.stack((sy * cp, -sp, cy * cp), axis=1)
    poses[:, :3, 3] = values[:, :3]

    return (np.linalg.inv(poses[0])[None] @ poses).astype(np.float32)


def _sanitize_points(points: list[dict[str, float]]) -> np.ndarray:
    clean: list[tuple[float, float]] = []
    for point in points:
        x = min(1.0, max(0.0, float(point.get("x", 0.5))))
        y = min(1.0, max(0.0, float(point.get("y", 0.5))))
        if not clean or math.dist(clean[-1], (x, y)) > 1e-4:
            clean.append((x, y))
    if not clean:
        clean = [(0.5, 0.82), (0.5, 0.18)]
    elif len(clean) == 1:
        clean.append(clean[0])
    return np.asarray(clean, dtype=np.float32)


def _resample_path(points: np.ndarray, frame_count: int) -> np.ndarray:

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total < 1e-7:
        return np.repeat(points[:1], frame_count, axis=0)

    cumulative = np.concatenate(([0.0], np.cumsum(lengths))) / total
    t = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    t = t * t * (3.0 - 2.0 * t)
    x = np.interp(t, cumulative, points[:, 0])
    y = np.interp(t, cumulative, points[:, 1])
    return np.stack((x, y), axis=1).astype(np.float32)


def _smooth(values: np.ndarray, radius: int = 5) -> np.ndarray:
    if len(values) < radius * 2 + 1:
        return values
    kernel = np.hanning(radius * 2 + 1).astype(np.float32)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def build_pose_npz(
    destination: Path,
    points: list[dict[str, float]],
    chunks: int,
    motion_scale: float = 24.0,
    vertical_lift: float = 0.0,
    horizontal_fov: float = 70.0,
    follow_path: bool = True,
    look_points: list[dict[str, float]] | None = None,
    look_yaw_degrees: float = 90.0,
    look_pitch_degrees: float = 45.0,
    camera_samples: list[dict] | None = None,
) -> dict[str, float | int]:


    chunks = min(40, max(1, int(chunks)))
    frame_count = int(round(chunks * SECONDS_PER_CHUNK * POSE_FPS))
    scale = min(80.0, max(0.0, float(motion_scale)))


    control_scale = scale / 24.0
    lift = min(20.0, max(-20.0, float(vertical_lift))) * control_scale
    fov = min(110.0, max(30.0, float(horizontal_fov)))

    sampled = _resample_path(_sanitize_points(points), frame_count)
    centered = sampled - sampled[0]
    translation = np.zeros((frame_count, 3), dtype=np.float32)
    translation[:, 0] = centered[:, 0] * scale
    translation[:, 2] = -centered[:, 1] * scale
    translation[:, 1] = np.linspace(0.0, lift, frame_count, dtype=np.float32)

    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    c2w[:, :3, 3] = translation

    yaw = np.zeros(frame_count, dtype=np.float32)
    if follow_path and np.linalg.norm(translation[-1] - translation[0]) > 1e-6:
        dx = np.gradient(translation[:, 0])
        dz = np.gradient(translation[:, 2])
        yaw = np.unwrap(np.arctan2(dx, dz + 1e-8))
        yaw -= yaw[0]
        yaw = _smooth(yaw.astype(np.float32))
        yaw *= control_scale

    pitch = np.zeros(frame_count, dtype=np.float32)
    if look_points:
        look = _resample_path(_sanitize_points(look_points), frame_count)
        look_yaw_limit = math.radians(min(180.0, max(0.0, float(look_yaw_degrees) * control_scale)))
        look_pitch_limit = math.radians(min(89.0, max(0.0, float(look_pitch_degrees) * control_scale)))
        look_yaw = (look[:, 0] - 0.5) * 2.0 * look_yaw_limit
        pitch = (0.5 - look[:, 1]) * 2.0 * look_pitch_limit


        look_yaw -= look_yaw[0]
        pitch -= pitch[0]
        look_yaw = _smooth(look_yaw.astype(np.float32))
        pitch = _smooth(pitch.astype(np.float32))
        look_yaw[0] = 0.0
        pitch[0] = 0.0
        yaw += look_yaw

    yaw_rotation = np.zeros((frame_count, 3, 3), dtype=np.float32)
    yaw_cosine, yaw_sine = np.cos(yaw), np.sin(yaw)
    yaw_rotation[:, 0, 0] = yaw_cosine
    yaw_rotation[:, 0, 2] = yaw_sine
    yaw_rotation[:, 1, 1] = 1.0
    yaw_rotation[:, 2, 0] = -yaw_sine
    yaw_rotation[:, 2, 2] = yaw_cosine
    pitch_rotation = np.zeros((frame_count, 3, 3), dtype=np.float32)
    pitch_cosine, pitch_sine = np.cos(pitch), np.sin(pitch)
    pitch_rotation[:, 0, 0] = 1.0
    pitch_rotation[:, 1, 1] = pitch_cosine
    pitch_rotation[:, 1, 2] = -pitch_sine
    pitch_rotation[:, 2, 1] = pitch_sine
    pitch_rotation[:, 2, 2] = pitch_cosine
    c2w[:, :3, :3] = np.einsum("tij,tjk->tik", yaw_rotation, pitch_rotation)

    if camera_samples is not None:
        c2w = _recorded_poses(camera_samples, chunks)
        translation = c2w[:, :3, 3]
        recorded = validate_camera_samples(camera_samples, chunks)
        yaw, pitch = recorded[:, 3], recorded[:, 4]

    focal = (SOURCE_WIDTH / 2.0) / math.tan(math.radians(fov) / 2.0)
    intrinsic = np.array(
        [[focal, 0.0, SOURCE_WIDTH / 2.0], [0.0, focal, SOURCE_HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsics = np.repeat(intrinsic[None], frame_count, axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, cam_c2w=c2w, intrinsics=intrinsics)
    return {
        "frames": frame_count,
        "fps": POSE_FPS,
        "duration": chunks * SECONDS_PER_CHUNK,
        "path_length": float(np.linalg.norm(np.diff(translation, axis=0), axis=1).sum()),
        "yaw_range_degrees": float(np.degrees(yaw.max() - yaw.min())),
        "pitch_range_degrees": float(np.degrees(pitch.max() - pitch.min())),
    }


def resample_pose_npz(source: Path, destination: Path, source_fps: int = POSE_FPS, target_fps: int = OUTPUT_FPS) -> dict[str, float | int]:


    with np.load(source, allow_pickle=True) as data:
        c2w_key = "cam_c2w" if "cam_c2w" in data else "data"
        intrinsic_key = next((key for key in ("intrinsics", "intrinsic", "K") if key in data), None)
        if intrinsic_key is None:
            raise KeyError(f"pose npz has no intrinsics/intrinsic/K key: {source}")
        c2w = np.asarray(data[c2w_key])
        intrinsics = np.asarray(data[intrinsic_key])

    target_frames = max(1, int(round(len(c2w) * target_fps / source_fps)))
    indices = np.asarray(
        [min(len(c2w) - 1, round(i * source_fps / target_fps)) for i in range(target_frames)],
        dtype=np.int64,
    )
    sampled_c2w = c2w[indices]
    sampled_intrinsics = intrinsics[indices] if intrinsics.ndim == 3 else intrinsics
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, cam_c2w=sampled_c2w, intrinsics=sampled_intrinsics)
    return {
        "frames": target_frames,
        "fps": target_fps,
        "duration": target_frames / target_fps,
        "sourceFrames": int(len(c2w)),
        "sourceFps": int(source_fps),
    }


def append_pose_npz(
    parent: Path,
    destination: Path,
    points: list[dict[str, float]],
    chunks: int,
    motion_scale: float = 24.0,
    vertical_lift: float = 0.0,
    horizontal_fov: float = 70.0,
    follow_path: bool = True,
    look_points: list[dict[str, float]] | None = None,
    look_yaw_degrees: float = 90.0,
    look_pitch_degrees: float = 45.0,
    camera_samples: list[dict] | None = None,
) -> dict[str, float | int]:

    temporary = destination.with_name(f".{destination.stem}.suffix.npz")
    suffix_info = build_pose_npz(
        temporary,
        points,
        chunks,
        motion_scale,
        vertical_lift,
        horizontal_fov,
        follow_path,
        look_points,
        look_yaw_degrees,
        look_pitch_degrees,
        camera_samples,
    )
    try:
        with np.load(parent, allow_pickle=True) as data:
            parent_c2w = np.asarray(data["cam_c2w"], dtype=np.float32)
            parent_intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        with np.load(temporary, allow_pickle=True) as data:
            suffix_c2w = np.asarray(data["cam_c2w"], dtype=np.float32)
            suffix_intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        anchored_suffix = np.einsum("ij,tjk->tik", parent_c2w[-1], suffix_c2w).astype(np.float32)
        anchored_suffix[0] = parent_c2w[-1]
        combined_c2w = np.concatenate((parent_c2w, anchored_suffix), axis=0)
        if parent_intrinsics.ndim == 3:
            combined_intrinsics = np.concatenate((parent_intrinsics, suffix_intrinsics), axis=0)
        else:
            combined_intrinsics = suffix_intrinsics
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, cam_c2w=combined_c2w, intrinsics=combined_intrinsics)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": int(len(combined_c2w)),
        "fps": POSE_FPS,
        "duration": len(combined_c2w) / POSE_FPS,
        "parentFrames": int(len(parent_c2w)),
        "appendedFrames": int(suffix_info["frames"]),
        "appendedPathLength": float(suffix_info["path_length"]),
        "appendedYawRangeDegrees": float(suffix_info["yaw_range_degrees"]),
        "appendedPitchRangeDegrees": float(suffix_info["pitch_range_degrees"]),
    }

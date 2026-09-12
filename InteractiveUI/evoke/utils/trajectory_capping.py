

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _rot_angle_deg(R: np.ndarray) -> float:

    tr = float(np.trace(R[..., :3, :3]))
    cos_a = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_a)))


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:

    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _quat_slerp_seq(quats: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:


    N = len(t_orig)
    assert N >= 2 and quats.shape[0] == N

    j_arr = np.searchsorted(t_orig, t_new, side="right") - 1
    j_arr = np.clip(j_arr, 0, N - 2)

    out = np.zeros((len(t_new), 4), dtype=np.float64)
    for i, tn in enumerate(t_new):
        j = int(j_arr[i])
        t0, t1 = t_orig[j], t_orig[j + 1]
        if t1 <= t0:
            out[i] = quats[j]
            continue
        u = float(np.clip((tn - t0) / (t1 - t0), 0.0, 1.0))
        q0, q1 = quats[j], quats[j + 1]

        if np.dot(q0, q1) < 0:
            q1 = -q1
        cos_theta = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if cos_theta > 0.9995:
            q = (1 - u) * q0 + u * q1
        else:
            theta = np.arccos(cos_theta)
            sin_t = np.sin(theta)
            q = (np.sin((1 - u) * theta) / sin_t) * q0 + (np.sin(u * theta) / sin_t) * q1
        out[i] = q / (np.linalg.norm(q) + 1e-12)
    return out


def _slerp_rotmat(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:

    q0 = _rotmat_to_quat(R0)
    q1 = _rotmat_to_quat(R1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    cos_theta = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if cos_theta > 0.9995:
        q = (1 - t) * q0 + t * q1
    else:
        theta = np.arccos(cos_theta)
        sin_t = np.sin(theta)
        q = (np.sin((1 - t) * theta) / sin_t) * q0 + (np.sin(t * theta) / sin_t) * q1
    q = q / (np.linalg.norm(q) + 1e-12)
    return _quat_to_rotmat(q)


def _resample_c2w_path(c2w_path: np.ndarray, n_new: int, by_arc_length: bool = True) -> np.ndarray:


    n_orig = len(c2w_path)
    if n_new <= 1 or n_orig <= 1:
        return c2w_path[:n_new] if n_orig > 0 else np.zeros((0, 4, 4), dtype=np.float32)


    quats = np.stack([_rotmat_to_quat(c2w_path[i, :3, :3]) for i in range(n_orig)], axis=0)
    t_orig = np.linspace(0.0, 1.0, n_orig)

    if by_arc_length:

        arc = np.zeros(n_orig, dtype=np.float64)
        for i in range(1, n_orig):
            R_inc = c2w_path[i, :3, :3] @ c2w_path[i - 1, :3, :3].T
            arc[i] = arc[i - 1] + _rot_angle_deg(R_inc)
        total_arc = float(arc[-1])
        if total_arc < 1e-6:

            t_new = np.linspace(0.0, 1.0, n_new)
        else:

            arc_new_targets = np.linspace(0.0, total_arc, n_new)

            t_new = np.interp(arc_new_targets, arc, t_orig)
    else:
        t_new = np.linspace(0.0, 1.0, n_new)

    quats_new = _quat_slerp_seq(quats, t_orig, t_new)


    trans = c2w_path[:, :3, 3].astype(np.float64)
    trans_new = np.stack([np.interp(t_new, t_orig, trans[:, axis]) for axis in range(3)], axis=-1)


    out = np.tile(np.eye(4, dtype=np.float64), (n_new, 1, 1))
    for i in range(n_new):
        out[i, :3, :3] = _quat_to_rotmat(quats_new[i])
        out[i, :3, 3] = trans_new[i]
    return out.astype(c2w_path.dtype if c2w_path.dtype != np.float64 else np.float32)


def chunk_bounds(n: int, frames_per_chunk: int = 33,
                 first_chunk_frames: int | None = None) -> list[tuple[int, int]]:


    w0 = int(frames_per_chunk if first_chunk_frames is None else first_chunk_frames)
    w = int(frames_per_chunk)
    if w0 < 1 or w < 1:
        raise ValueError(f"chunk widths must be >= 1, got first={w0} rest={w}")
    out, s = [], 0
    while s < n:
        e = min(n, s + (w0 if not out else w))
        out.append((s, e))
        s = e
    return out


def analyze_per_chunk(c2w: np.ndarray, frames_per_chunk: int = 33,
                      first_chunk_frames: int | None = None) -> dict[str, Any]:

    c2w = np.asarray(c2w, dtype=np.float32)
    n = len(c2w)
    bounds = chunk_bounds(n, frames_per_chunk, first_chunk_frames)
    n_chunks = len(bounds)
    rots, trans = [], []
    for s, e in bounds:
        if e - s < 2:
            continue
        R_rel = c2w[e - 1, :3, :3] @ c2w[s, :3, :3].T
        rots.append(_rot_angle_deg(R_rel))
        trans.append(float(np.linalg.norm(c2w[e - 1, :3, 3] - c2w[s, :3, 3])))
    rots = np.asarray(rots, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    return {
        "n_chunks": int(n_chunks),
        "rot_per_chunk_deg": rots,
        "trans_per_chunk": trans,
        "rot_mean": float(rots.mean()) if len(rots) else 0.0,
        "rot_max": float(rots.max()) if len(rots) else 0.0,
        "rot_p95": float(np.percentile(rots, 95)) if len(rots) else 0.0,
        "trans_mean": float(trans.mean()) if len(trans) else 0.0,
    }


def cap_target_per_chunk_rotation(
    c2ws: torch.Tensor | np.ndarray,
    ref_pix: int,
    frames_per_chunk: int = 33,
    max_deg: float = 17.0,
    max_trans: float = 0.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, int, dict[str, Any]]:


    if isinstance(c2ws, torch.Tensor):
        c2w_np = c2ws.detach().cpu().numpy()
        was_tensor = True
        out_dtype = c2ws.dtype
        out_device = c2ws.device
    else:
        c2w_np = np.asarray(c2ws)
        was_tensor = False
        out_dtype = None
        out_device = None

    squeeze_batch = False
    if c2w_np.ndim == 4:
        assert c2w_np.shape[0] == 1, f"only B=1 supported, got {c2w_np.shape}"
        c2w_np = c2w_np[0]
        squeeze_batch = True

    n_total = c2w_np.shape[0]
    if ref_pix >= n_total:
        if verbose:
            print(f"[traj-cap] ref_pix={ref_pix} >= total={n_total}, no target, skip cap")
        return c2ws, max(0, n_total - ref_pix), {"n_subdivided": 0, "skipped": True}

    target = c2w_np[ref_pix:]
    n_target_old = len(target)
    if n_target_old < frames_per_chunk:
        if verbose:
            print(f"[traj-cap] target {n_target_old} < 1 chunk, skip cap")
        return c2ws, n_target_old, {"n_subdivided": 0, "skipped": True}

    n_chunks_old = (n_target_old + frames_per_chunk - 1) // frames_per_chunk
    out_chunks = []
    n_subdivided = 0
    chunk_log: list[str] = []
    for k in range(n_chunks_old):
        s, e = k * frames_per_chunk, min(n_target_old, (k + 1) * frames_per_chunk)
        chunk = target[s:e]
        if e - s < 2:
            out_chunks.append(chunk)
            continue
        R_rel = chunk[-1, :3, :3] @ chunk[0, :3, :3].T
        angle = _rot_angle_deg(R_rel)
        trans = float(np.linalg.norm(chunk[-1, :3, 3] - chunk[0, :3, 3]))

        K_rot = int(np.ceil(angle / max_deg)) if (max_deg > 0 and angle > max_deg) else 1
        K_trans = int(np.ceil(trans / max_trans)) if (max_trans > 0 and trans > max_trans) else 1
        K = max(K_rot, K_trans)
        if K <= 1:
            out_chunks.append(chunk)
            chunk_log.append(f"  chunk {k}: rot={angle:5.1f}° trans={trans:5.2f} ✓")
        else:
            n_sub_frames = K * frames_per_chunk
            resampled = _resample_c2w_path(chunk, n_sub_frames)
            out_chunks.append(resampled)
            n_subdivided += 1
            _bound_tag = []
            if K_rot >= K_trans and K_rot > 1:
                _bound_tag.append(f"rot:{angle:.1f}°→{K_rot}×")
            if K_trans >= K_rot and K_trans > 1:
                _bound_tag.append(f"trans:{trans:.2f}→{K_trans}×")
            chunk_log.append(
                f"  chunk {k}: rot={angle:5.1f}° trans={trans:5.2f} → split into {K} sub-chunks "
                f"({n_sub_frames} frames; bound by {', '.join(_bound_tag)})"
            )

    target_new = np.concatenate(out_chunks, axis=0)
    n_target_new = len(target_new)
    c2w_new = np.concatenate([c2w_np[:ref_pix], target_new], axis=0)

    if squeeze_batch:
        c2w_new = c2w_new[None]
    if was_tensor:
        c2w_new = torch.as_tensor(c2w_new, dtype=out_dtype, device=out_device)

    if verbose:
        _cap_desc = f"rot≤{max_deg:.1f}°"
        if max_trans > 0:
            _cap_desc += f", trans≤{max_trans:.2f}"
        print(
            f"[traj-cap] cap={_cap_desc} per chunk: target frames {n_target_old} → {n_target_new}, "
            f"chunks {n_chunks_old} → {(n_target_new + frames_per_chunk - 1) // frames_per_chunk}, "
            f"{n_subdivided}/{n_chunks_old} subdivided",
            flush=True,
        )
        if n_subdivided > 0:
            for line in chunk_log:
                print(line, flush=True)

    info = {
        "n_subdivided": n_subdivided,
        "n_target_old": n_target_old,
        "n_target_new": n_target_new,
        "n_chunks_old": n_chunks_old,
        "n_chunks_new": (n_target_new + frames_per_chunk - 1) // frames_per_chunk,
        "max_deg": float(max_deg),
        "max_trans": float(max_trans),
    }
    return c2w_new, n_target_new, info


def clamp_target_per_chunk_motion(
    c2ws: torch.Tensor | np.ndarray,
    ref_pix: int,
    frames_per_chunk: int = 33,
    max_deg: float = 17.0,
    max_trans: float = 0.0,
    verbose: bool = True,
    first_chunk_frames: int | None = None,
) -> tuple[torch.Tensor, int, dict[str, Any]]:


    if isinstance(c2ws, torch.Tensor):
        c2w_np = c2ws.detach().cpu().numpy()
        was_tensor = True
        out_dtype = c2ws.dtype
        out_device = c2ws.device
    else:
        c2w_np = np.asarray(c2ws)
        was_tensor = False
        out_dtype = None
        out_device = None

    squeeze_batch = False
    if c2w_np.ndim == 4:
        assert c2w_np.shape[0] == 1, f"only B=1 supported, got {c2w_np.shape}"
        c2w_np = c2w_np[0]
        squeeze_batch = True

    n_total = c2w_np.shape[0]
    if ref_pix >= n_total:
        if verbose:
            print(f"[traj-clamp] ref_pix={ref_pix} >= total={n_total}, no target, skip", flush=True)
        return c2ws, max(0, n_total - ref_pix), {"mode": "clamp", "n_clamped": 0, "skipped": True}

    target = c2w_np[ref_pix:].astype(np.float64)
    n_target = len(target)
    if n_target < 2:
        return c2ws, n_target, {"mode": "clamp", "n_clamped": 0, "skipped": True}

    bounds = chunk_bounds(n_target, frames_per_chunk, first_chunk_frames)
    n_chunks = len(bounds)
    out_target = np.zeros_like(target)
    cur_pose = target[0].copy()

    n_clamped = 0
    chunk_log: list[str] = []

    for k, (s, e) in enumerate(bounds):
        n_in_chunk = e - s
        if n_in_chunk < 1:
            continue

        orig_end = target[e - 1]
        cur_R, cur_T = cur_pose[:3, :3], cur_pose[:3, 3]
        orig_R, orig_T = orig_end[:3, :3], orig_end[:3, 3]


        R_delta = orig_R @ cur_R.T
        angle = _rot_angle_deg(R_delta)
        T_delta = orig_T - cur_T
        trans = float(np.linalg.norm(T_delta))


        rot_alpha = min(1.0, max_deg / angle) if (max_deg > 0 and angle > 0) else 1.0
        trans_alpha = min(1.0, max_trans / trans) if (max_trans > 0 and trans > 0) else 1.0

        was_clamped = (
            (max_deg > 0 and angle > max_deg + 1e-6)
            or (max_trans > 0 and trans > max_trans + 1e-6)
        )
        if was_clamped:
            n_clamped += 1


        new_R = _slerp_rotmat(cur_R, orig_R, rot_alpha) if rot_alpha < 1.0 else orig_R.copy()
        new_T = cur_T + T_delta * trans_alpha
        new_end = np.eye(4, dtype=np.float64)
        new_end[:3, :3] = new_R
        new_end[:3, 3] = new_T


        for i in range(n_in_chunk):
            if n_in_chunk == 1:
                out_target[s + i] = new_end
                continue
            frac = i / (n_in_chunk - 1)
            if frac <= 0:
                out_target[s + i] = cur_pose
            elif frac >= 1:
                out_target[s + i] = new_end
            else:
                f_R = _slerp_rotmat(cur_R, new_R, frac)
                f_T = cur_T + (new_T - cur_T) * frac
                f = np.eye(4, dtype=np.float64)
                f[:3, :3] = f_R
                f[:3, 3] = f_T
                out_target[s + i] = f

        if verbose:
            line = f"  chunk {k}: rot={angle:5.1f}° trans={trans:5.2f}"
            if was_clamped:
                tags = []
                if max_deg > 0 and angle > max_deg + 1e-6:
                    tags.append(f"rot ×{rot_alpha:.2f} → {angle*rot_alpha:.1f}°")
                if max_trans > 0 and trans > max_trans + 1e-6:
                    tags.append(f"trans ×{trans_alpha:.2f} → {trans*trans_alpha:.2f}")
                line += f" → CLAMPED ({', '.join(tags)})"
            else:
                line += " ✓"
            chunk_log.append(line)

        cur_pose = new_end

    c2w_new = np.concatenate([c2w_np[:ref_pix], out_target], axis=0)

    if squeeze_batch:
        c2w_new = c2w_new[None]
    if was_tensor:
        c2w_new = torch.as_tensor(c2w_new, dtype=out_dtype, device=out_device)
    else:
        c2w_new = c2w_new.astype(np.asarray(c2ws).dtype if hasattr(c2ws, "dtype") else np.float32)

    if verbose:
        _cap_desc = f"rot≤{max_deg:.1f}°" if max_deg > 0 else "rot UNLIM"
        if max_trans > 0:
            _cap_desc += f", trans≤{max_trans:.2f}"
        else:
            _cap_desc += ", trans UNLIM"
        print(
            f"[traj-clamp] mode=clamp (frame count UNCHANGED): cap={_cap_desc}, "
            f"{n_clamped}/{n_chunks} chunks clamped (target {n_target} frames unchanged)",
            flush=True,
        )
        if n_clamped > 0:
            for line in chunk_log:
                print(line, flush=True)

    info = {
        "mode": "clamp",
        "n_clamped": n_clamped,
        "n_chunks": n_chunks,
        "n_target": n_target,
        "max_deg": float(max_deg),
        "max_trans": float(max_trans),
    }
    return c2w_new, n_target, info


def smooth_c2w_trajectory(
    c2ws: torch.Tensor | np.ndarray,
    ref_pix: int = 0,
    win: int = 5,
    verbose: bool = True,
) -> tuple[torch.Tensor | np.ndarray, dict[str, Any]]:


    if win is None or int(win) <= 1:
        return c2ws, {"win": int(win or 0), "skipped": True}
    win = int(win)
    if win % 2 == 0:
        win += 1

    if isinstance(c2ws, torch.Tensor):
        c2w_np = c2ws.detach().cpu().numpy().astype(np.float64)
        was_tensor, out_dtype, out_device = True, c2ws.dtype, c2ws.device
    else:
        c2w_np = np.asarray(c2ws, dtype=np.float64)
        was_tensor, out_dtype, out_device = False, None, None

    squeeze_batch = False
    if c2w_np.ndim == 4:
        assert c2w_np.shape[0] == 1, f"only B=1 supported, got {c2w_np.shape}"
        c2w_np = c2w_np[0]
        squeeze_batch = True

    N = c2w_np.shape[0]
    if N < 3 or ref_pix >= N - 1:
        if verbose:
            print(f"[pose-smooth] N={N} ref_pix={ref_pix} no smoothable target, skip", flush=True)
        return c2ws, {"win": win, "skipped": True}

    r = win // 2
    sig = max(win / 3.0, 0.5)
    kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    kern /= kern.sum()

    t = c2w_np[:, :3, 3].copy()
    q = np.stack([_rotmat_to_quat(c2w_np[i, :3, :3]) for i in range(N)])
    for i in range(1, N):
        if float(np.dot(q[i], q[i - 1])) < 0:
            q[i] = -q[i]

    out = c2w_np.copy()
    n_smoothed = 0
    for i in range(N):
        if i < ref_pix:
            continue
        lo, hi = max(0, i - r), min(N, i + r + 1)
        kk = kern[(lo - (i - r)):(win - ((i + r + 1) - hi))]
        kk = kk / kk.sum()
        ts = (t[lo:hi] * kk[:, None]).sum(0)
        qs = (q[lo:hi] * kk[:, None]).sum(0)
        qs /= (np.linalg.norm(qs) + 1e-12)
        out[i, :3, :3] = _quat_to_rotmat(qs)
        out[i, :3, 3] = ts
        n_smoothed += 1


    disp = float(np.linalg.norm(out[ref_pix:, :3, 3] - c2w_np[ref_pix:, :3, 3], axis=1).mean())
    if verbose:
        print(f"[pose-smooth] win={win} smoothed {n_smoothed} target frames ({ref_pix} ref frames kept); "
              f"mean camera-center displacement={disp:.4f}", flush=True)

    if squeeze_batch:
        out = out[None]
    if was_tensor:
        out = torch.from_numpy(out).to(device=out_device, dtype=out_dtype)
    info = {"win": win, "n_smoothed": n_smoothed, "mean_disp": disp, "skipped": False}
    return out, info

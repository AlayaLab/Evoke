

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .depth_backend import reset_stream as _reset_depth_stream


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DA3_SRC = Path(os.environ.get("EVOKE_DA3_SRC", str(_REPO_ROOT / "evoke" / "third_party" / "da3")))
_DA3_WEIGHTS = Path(os.environ.get("EVOKE_DA3_WEIGHTS", str(_REPO_ROOT / "models" / "DA3")))


def unproject_depth_torch(depth: torch.Tensor, intr: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:

    h, w = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ys, xs = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32),
                            torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
    z = depth.float()
    cam = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=-1)
    R = c2w[:3, :3].float(); t = c2w[:3, 3].float()
    return cam @ R.T + t


def _affine_inv_np(M):

    M = np.asarray(M, np.float64)
    if M.shape[-2:] == (3, 4):
        pad = np.zeros(M.shape[:-2] + (4, 4), np.float64)
        pad[..., :3, :4] = M; pad[..., 3, 3] = 1.0
        M = pad
    R = M[..., :3, :3]; t = M[..., :3, 3]
    Ri = np.swapaxes(R, -1, -2)
    out = np.zeros_like(M)
    out[..., :3, :3] = Ri
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Ri, t)
    out[..., 3, 3] = 1.0
    return out


def _umeyama_scale_collinear_safe(src_centers, dst_centers):


    src = np.asarray(src_centers, np.float64); dst = np.asarray(dst_centers, np.float64)
    n = int(src.shape[0])
    if n < 2:
        return None
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s; dc = dst - mu_d
    var_s = float((sc ** 2).sum() / n)
    if not np.isfinite(var_s) or var_s < 1e-12:
        return None
    Sigma = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    c = float((D * np.diag(S)).sum() / var_s)
    if not np.isfinite(c) or c <= 1e-9:
        return None
    return c


def _align_pred_robust(model, ext_raw, ix, pred):


    try:
        return model._align_to_input_extrinsics_intrinsics(ext_raw, ix, pred, True)
    except Exception as _e:
        gt_w2c = ext_raw.numpy() if hasattr(ext_raw, "numpy") else np.asarray(ext_raw)
        pred_ext = np.asarray(pred.extrinsics)
        c = _umeyama_scale_collinear_safe(
            _affine_inv_np(gt_w2c)[..., :3, 3], _affine_inv_np(pred_ext)[..., :3, 3])
        if c is None:
            raise
        pred.intrinsics = (ix.numpy() if hasattr(ix, "numpy") else np.asarray(ix))
        pred.extrinsics = (ext_raw[..., :3, :].numpy() if hasattr(ext_raw, "numpy")
                           else np.asarray(ext_raw)[..., :3, :])
        pred.depth = np.asarray(pred.depth) / c
        print(f"[da3-degen-fallback] collinear/straight-line degeneracy -> SVD scale fallback scale={c:.4g} "
              f"({type(_e).__name__})", flush=True)
        return pred


class DA3DepthEstimator:
    def __init__(self, device="cuda", process_res: int = 504,
                 src: Path = _DA3_SRC, weights: Path = _DA3_WEIGHTS):
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.src = Path(src); self.weights = Path(weights)
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        if not (self.src / "depth_anything_3").is_dir():
            raise FileNotFoundError(
                f"DA3 source not found: {self.src / 'depth_anything_3'} does not exist. the vendored in-repo copy should be at "
                "evoke/third_party/da3 (see its PROVENANCE.md), or set EVOKE_DA3_SRC to point at an external checkout.")
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
        import types
        if "depth_anything_3.utils.export" not in sys.modules:
            _exp = types.ModuleType("depth_anything_3.utils.export")
            _exp.export = lambda *a, **k: None
            sys.modules["depth_anything_3.utils.export"] = _exp
        from depth_anything_3.api import DepthAnything3
        self._model = DepthAnything3.from_pretrained(str(self.weights)).to(self.device).eval()

    @torch.no_grad()
    def depth_window(self, frames_rgb, c2w_gt, K_gt):


        frames_rgb = np.asarray(frames_rgb); K = frames_rgb.shape[0]
        if K < 3:
            raise ValueError(f"DA3 GT-pose needs >=3 frames per call (align_to_input_ext_scale solves the scale), got {K}")
        deps, intrs, confs, rgbs = self.depth_windows_batched(
            [frames_rgb], [np.asarray(c2w_gt)], [np.asarray(K_gt)])
        if deps[0] is None:
            raise RuntimeError("DA3 depth_window degenerate (zero baseline, no solvable scale)")
        return (np.asarray(deps[0], dtype=np.float32), np.asarray(intrs[0], dtype=np.float32),
                None if confs[0] is None else np.asarray(confs[0], dtype=np.float32), rgbs[0])

    @staticmethod
    def _rgb_from_processed(pred):
        proc = pred.processed_images
        if proc is None:
            return None
        rgb = np.asarray(proc, dtype=np.float32)
        return rgb / 255.0 if rgb.max() > 1.5 else rgb

    @torch.no_grad()
    def depth_windows_batched(self, windows_rgb, windows_c2w, windows_K):


        self._lazy()
        m = self._model
        B = len(windows_rgb)
        if B == 0:
            return [], [], [], []
        dev = m._get_model_device()
        imgs_cpu_list, ex_raw_list, ix_list, exn_list = [], [], [], []
        for b in range(B):
            fr = np.asarray(windows_rgb[b]); N = fr.shape[0]
            if N < 3:
                raise ValueError(f"DA3 GT-pose needs >=3 frames per window, window {b} got {N}")
            imgs = [(np.clip(fr[i], 0, 1) * 255).astype(np.uint8) for i in range(N)]
            w2c = np.linalg.inv(np.asarray(windows_c2w[b], np.float32)).astype(np.float32)
            ic, ex, ix = m.input_processor(
                imgs, w2c, np.asarray(windows_K[b], np.float32),
                self.process_res, "upper_bound_resize")

            exn = m._normalize_extrinsics(ex[None].clone())[0]
            imgs_cpu_list.append(ic); ex_raw_list.append(ex); ix_list.append(ix); exn_list.append(exn)

        imgs_t = torch.stack(imgs_cpu_list, 0).to(dev).float()
        exn_t = torch.stack(exn_list, 0).to(dev).float()
        in_t = torch.stack(ix_list, 0).to(dev).float()
        raw = m._run_model_forward(imgs_t, exn_t, in_t, [], False, False, "saddle_balanced")
        depths, intrs, confs, rgbs = [], [], [], []
        for b in range(B):
            try:
                raw_b = {k: (v[b:b + 1] if torch.is_tensor(v) else v) for k, v in raw.items()}
                pred = m._convert_to_prediction(raw_b)


                pred = _align_pred_robust(m, ex_raw_list[b], ix_list[b], pred)
                pred = m._add_processed_images(pred, imgs_cpu_list[b])
                depths.append(np.asarray(pred.depth, dtype=np.float32))
                intrs.append(np.asarray(pred.intrinsics, dtype=np.float32))
                confs.append(None if pred.conf is None else np.asarray(pred.conf, dtype=np.float32))
                rgbs.append(self._rgb_from_processed(pred))
            except Exception as _e:
                print(f"[da3-batched] WARN window {b} align/convert failed ({type(_e).__name__}: {_e}); skipping", flush=True)
                depths.append(None); intrs.append(None); confs.append(None); rgbs.append(None)
        return depths, intrs, confs, rgbs


class PersistentCloud:

    def __init__(self, device="cuda", point_stride: int = 1,
                 conf_percentile: float = 30.0, voxel_size: Optional[float] = None,
                 max_points: Optional[int] = None):
        self.device = torch.device(device)
        self.point_stride = int(point_stride)
        self.conf_percentile = float(conf_percentile)
        self.voxel_size = voxel_size
        self.max_points = int(max_points) if max_points else None
        self.xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.rgb = torch.zeros((0, 3), dtype=torch.float32, device=self.device)

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def add_depth(self, depth, intr_proc, c2w_gt, frames_rgb, conf=None, rgb_proc=None):


        K = depth.shape[0]; st = self.point_stride
        depth_t = torch.as_tensor(depth, device=self.device)
        intr_t = torch.as_tensor(intr_proc, device=self.device)
        c2w_t = torch.as_tensor(np.asarray(c2w_gt, np.float32), device=self.device)
        h, w = depth.shape[1], depth.shape[2]
        if rgb_proc is not None:
            rgb_t = torch.as_tensor(np.asarray(rgb_proc, np.float32), device=self.device)
            if rgb_t.shape[1] != h or rgb_t.shape[2] != w:
                rgb_t = torch.nn.functional.interpolate(rgb_t.permute(0, 3, 1, 2), size=(h, w),
                                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        else:
            rgb_t = torch.as_tensor(np.asarray(frames_rgb, np.float32), device=self.device).permute(0, 3, 1, 2)
            rgb_t = torch.nn.functional.interpolate(rgb_t, size=(h, w), mode="bilinear", align_corners=False)
            rgb_t = rgb_t.permute(0, 2, 3, 1)
        thr = None
        if conf is not None and self.conf_percentile > 0:
            thr = float(np.percentile(conf.reshape(-1), self.conf_percentile))
        new_xyz, new_rgb = [], []
        for i in range(K):
            wp = unproject_depth_torch(depth_t[i], intr_t[i], c2w_t[i])
            m = torch.isfinite(wp).all(-1) & (depth_t[i] > 1e-4)
            if thr is not None:
                m &= torch.as_tensor(conf[i], device=self.device) >= thr
            wp_s = wp[::st, ::st][m[::st, ::st]]
            cl_s = rgb_t[i][::st, ::st][m[::st, ::st]]
            new_xyz.append(wp_s); new_rgb.append(cl_s)
        if new_xyz:
            self.xyz = torch.cat([self.xyz] + new_xyz, 0)
            self.rgb = torch.cat([self.rgb] + new_rgb, 0)
        if self.voxel_size or self.max_points:
            self.bound(voxel_size=self.voxel_size, max_points=self.max_points)

    def bound(self, voxel_size: Optional[float] = None, max_points: Optional[int] = None):


        if self.num_points == 0:
            return
        if voxel_size:
            keys = torch.floor(self.xyz / float(voxel_size)).to(torch.int64)
            _, inv = torch.unique(keys, dim=0, return_inverse=True)
            nv = int(inv.max().item()) + 1
            order = torch.arange(self.num_points, device=self.device)
            first = torch.full((nv,), self.num_points, dtype=torch.long, device=self.device)
            first = first.scatter_reduce(0, inv, order, reduce="amin", include_self=True)
            sel = first[first < self.num_points]
            self.xyz = self.xyz[sel]; self.rgb = self.rgb[sel]
        if max_points and self.num_points > int(max_points):
            sel = torch.randperm(self.num_points, device=self.device)[:int(max_points)]
            self.xyz = self.xyz[sel]; self.rgb = self.rgb[sel]


@torch.no_grad()
def render_cloud_batched(xyz, rgb, c2w_targets, K_render, height: int, width: int, *,
                         device="cuda", splat_radius: int = 2, invisible_fill: str = "black"):


    device = torch.device(device)
    P = torch.as_tensor(np.asarray(xyz) if not torch.is_tensor(xyz) else xyz,
                        dtype=torch.float32, device=device).reshape(-1, 3)
    C = torch.as_tensor(np.asarray(rgb) if not torch.is_tensor(rgb) else rgb,
                        dtype=torch.float32, device=device).reshape(-1, 3)
    npt = P.shape[0]


    assert npt < (1 << 24), (
        f"cloud has {npt} points >= 2^24, over the render int-pack limit -> bound the point count with voxel bound (see R6)")
    c2w = torch.as_tensor(np.asarray(c2w_targets) if not torch.is_tensor(c2w_targets) else c2w_targets,
                          dtype=torch.float32, device=device)
    Kt = torch.as_tensor(np.asarray(K_render) if not torch.is_tensor(K_render) else K_render,
                         dtype=torch.float32, device=device)
    V = c2w.shape[0]; HW = height * width

    if invisible_fill == "mean" and npt > 0:
        fill = C.mean(0)
    else:
        fill = torch.zeros(3, device=device)
    if npt == 0:
        warp = (fill[None, :, None, None, None].expand(1, 3, V, height, width) * 2 - 1)
        return warp.contiguous(), torch.zeros((1, 1, V, height, width), device=device)
    R = c2w[:, :3, :3]; t = c2w[:, :3, 3]
    cam = torch.einsum("vij,vpj->vpi", R.transpose(1, 2), P[None] - t[:, None])
    z = cam[..., 2]
    fx = Kt[:, 0, 0][:, None]; fy = Kt[:, 1, 1][:, None]
    cx = Kt[:, 0, 2][:, None]; cy = Kt[:, 1, 2][:, None]
    px = torch.round((cam[..., 0] / z) * fx + cx).long()
    py = torch.round((cam[..., 1] / z) * fy + cy).long()
    front = z > 1e-4
    zq = (z.clamp(0, 1e6) * 1000.0).long().clamp(0, (1 << 38) - 1)
    idx = torch.arange(npt, device=device)[None].expand(V, npt)
    vidx = torch.arange(V, device=device)[:, None].expand(V, npt)
    INF = torch.full((V * HW,), (1 << 62), dtype=torch.long, device=device)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            qx = px + dx; qy = py + dy
            ok = front & (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            key = (vidx * HW + qy * width + qx)[ok]
            packed = (zq[ok] << 24) | idx[ok]
            INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
    valid = INF < (1 << 62)
    win = (INF & ((1 << 24) - 1)).clamp(max=npt - 1)
    img01 = fill[None].expand(V * HW, 3).clone()
    img01[valid] = C[win[valid]]
    img = img01.reshape(V, height, width, 3).clamp(0, 1)
    warp = (img.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
    vis = valid.reshape(V, height, width).float()[None, None]
    return warp.contiguous(), vis.contiguous()


@torch.no_grad()
def build_training_cloud_warp(estimator, raw_video_b, lingbot_c2ws_b, K_pix, target_c2ws,
                              *, pix_start, pix_stride, window_pix, height, width,
                              update_n=16, history_chunks=8, lag=1, splat_radius=2,
                              voxel_size=None, max_points=None, device="cuda"):


    T = int(raw_video_b.shape[1])
    kc = int(pix_start) // int(pix_stride)
    newest = kc - 1 - int(lag)
    oldest = max(0, newest - int(history_chunks) + 1)
    K_pix = np.asarray(K_pix, np.float32)
    windows_rgb, windows_c2w, windows_K = [], [], []
    for j in range(oldest, newest + 1):
        s = j * int(pix_stride)
        if s < 0 or s >= T:
            continue
        e = min(s + int(window_pix), T)
        if e - s < 3:
            continue
        idxs = torch.linspace(s, e - 1, int(update_n)).round().long().clamp(0, T - 1)
        idxs = torch.unique(idxs)
        if idxs.numel() < 3:
            continue
        frames = (raw_video_b[:, idxs].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
        c2w = lingbot_c2ws_b[idxs].float().cpu().numpy()
        windows_rgb.append(frames); windows_c2w.append(c2w)
        windows_K.append(np.stack([K_pix] * int(idxs.numel())))


    cloud = PersistentCloud(device=device, voxel_size=(float(voxel_size) if voxel_size else None),
                            max_points=(int(max_points) if max_points else None))
    if windows_rgb:
        depths, intrs, confs, rgbs = estimator.depth_windows_batched(windows_rgb, windows_c2w, windows_K)
        for w in range(len(windows_rgb)):
            if depths[w] is None:
                continue
            cloud.add_depth(depths[w], intrs[w], windows_c2w[w], windows_rgb[w], confs[w], rgbs[w])
    K_render = np.stack([K_pix] * int(target_c2ws.shape[0]))
    warp, vis = render_cloud_batched(
        cloud.xyz, cloud.rgb, target_c2ws.float().cpu().numpy(), K_render,
        int(height), int(width), device=device, splat_radius=int(splat_radius))
    return warp, vis


def scale_intrinsics(K_src, src_hw, dst_hw):

    sh, sw = src_hw; dh, dw = dst_hw
    sx, sy = dw / float(sw), dh / float(sh)
    K = np.array(K_src, np.float32).copy()
    K[..., 0, 0] *= sx; K[..., 0, 2] *= sx
    K[..., 1, 1] *= sy; K[..., 1, 2] *= sy
    return K


def _project_inframe(xyz, c2w_t, K_t, height, width):

    R = c2w_t[:3, :3]; t = c2w_t[:3, 3]
    cam = (xyz - t[None]) @ R
    z = cam[:, 2]
    px = cam[:, 0] / z.clamp(min=1e-6) * K_t[0, 0] + K_t[0, 2]
    py = cam[:, 1] / z.clamp(min=1e-6) * K_t[1, 1] + K_t[1, 2]
    ok = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px, py, ok


def _covis_frac(xyz, c2w_t, K_t, height, width):

    if xyz.shape[0] == 0:
        return 0.0
    _, _, ok = _project_inframe(xyz, c2w_t, K_t, height, width)
    return float(ok.float().mean())


def _covis_mask(xyz, tframes_c2w, K_t, gh, gw, height, width):

    sx = width / gw; sy = height / gh; out = []
    for c2w in tframes_c2w:
        px, py, ok = _project_inframe(xyz, c2w, K_t, height, width)
        m = torch.zeros(gh * gw, dtype=torch.bool, device=xyz.device)
        if ok.any():
            gx = (px[ok] / sx).long().clamp(0, gw - 1); gy = (py[ok] / sy).long().clamp(0, gh - 1)
            m[gy * gw + gx] = True
        out.append(m)
    return torch.cat(out)


@torch.no_grad()
def _cloud_depth_map(xyz, c2w, K, height, width, *, device):


    xyz = xyz if torch.is_tensor(xyz) else torch.as_tensor(xyz, dtype=torch.float32, device=device)
    out = torch.full((height * width,), float("inf"), device=device, dtype=torch.float32)
    if xyz.shape[0] == 0:
        return out.view(height, width)
    R = c2w[:3, :3]; t = c2w[:3, 3]
    cam = (xyz - t[None]) @ R
    z = cam[:, 2]
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    px = torch.round(cam[:, 0] / z.clamp(min=1e-6) * fx + cx).long()
    py = torch.round(cam[:, 1] / z.clamp(min=1e-6) * fy + cy).long()
    ok = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if ok.any():
        out.scatter_reduce_(0, py[ok] * width + px[ok], z[ok], reduce="amin", include_self=True)
    return out.view(height, width)


class DA3FrameBank:


    def __init__(self, device="cuda", conf_percentile=30.0,
                 cloud_hygiene=False, hygiene_sat_max=1.0,
                 hygiene_flat_std=0.0, hygiene_flat_win=7,
                 hygiene_conf_pct=0.0, hygiene_conf_abs=0.0,
                 consist_gate=False, consist_tau=0.15,
                 consist_ref_frames=24, consist_min_ref=2000,
                 consist_adaptive=False, consist_tau_lo=0.20, consist_tau_hi=0.30, consist_conf_ref=12.0,
                 consist_probation=0, consist_probation_frac=0.0, consist_probation_win=25,
                 consist_scale_align=False,
                 color_anchor=False, color_anchor_alpha=0.5, color_anchor_ref_windows=4,
                 self_reanchor=False, self_reanchor_scale_thr=1.5, self_reanchor_rej_thr=0.8,
                 self_reanchor_min_gap=8, self_reanchor_keep_windows=3, self_reanchor_lookback=20,
                 self_reanchor_anchor_min_conf=0.0,
                 self_reanchor_pin_prime=False, self_reanchor_pin_windows=4,
                 hist_max_frames=0, reanchor_every=0, reanchor_keep_frames=0,
                 carve=False, carve_margin=0.10, carve_ref_frames=24, carve_min_views=1,
                 carve_strike_windows=1):
        self.device = torch.device(device)
        self.conf_pct = float(conf_percentile)

        self.hygiene = bool(cloud_hygiene)


        self.hyg_conf_abs = float(hygiene_conf_abs)
        self.hyg_sat_max = float(hygiene_sat_max)
        self.hyg_flat_std = float(hygiene_flat_std)
        self.hyg_flat_win = int(hygiene_flat_win)
        self.hyg_conf_pct = float(hygiene_conf_pct)


        self.consist_gate = bool(consist_gate)
        self.consist_tau = float(consist_tau)
        self.consist_ref_frames = int(consist_ref_frames)
        self.consist_min_ref = int(consist_min_ref)


        self.consist_adaptive = bool(consist_adaptive)
        self.consist_tau_lo = float(consist_tau_lo)
        self.consist_tau_hi = float(consist_tau_hi)
        self.consist_conf_ref = float(consist_conf_ref)


        self.consist_probation = int(consist_probation)


        self.consist_probation_frac = float(consist_probation_frac)
        self.consist_probation_win = max(3, int(consist_probation_win) | 1)
        self._probation = {}


        self.hist_max_frames = int(hist_max_frames)
        self.reanchor_every = int(reanchor_every)
        self.reanchor_keep_frames = int(reanchor_keep_frames)
        self._ingest_calls = 0


        self.carve = bool(carve)
        self.carve_margin = float(carve_margin)
        self.carve_ref_frames = int(carve_ref_frames)


        self.carve_min_views = max(1, int(carve_min_views))


        self.carve_strike_windows = max(1, int(carve_strike_windows))
        self._carve_strike = {}


        self.consist_scale_align = bool(consist_scale_align)


        self.color_anchor = bool(color_anchor)
        self.color_anchor_alpha = float(color_anchor_alpha)
        self.color_anchor_ref_windows = max(0, int(color_anchor_ref_windows))
        self._ca_ref = None
        self._ca_seen = 0
        self._ca_ref_acc = []


        self.self_reanchor = bool(self_reanchor)
        self.sr_scale_thr = float(self_reanchor_scale_thr)
        self.sr_rej_thr = float(self_reanchor_rej_thr)
        self.sr_min_gap = int(self_reanchor_min_gap)
        self.sr_keep_windows = max(1, int(self_reanchor_keep_windows))
        self.sr_lookback = max(2, int(self_reanchor_lookback))


        self.sr_anchor_min_conf = float(self_reanchor_anchor_min_conf)

        self.sr_pin_prime = bool(self_reanchor_pin_prime)
        self.sr_pin_windows = max(1, int(self_reanchor_pin_windows))
        self._pinned_wins = []
        self._win_hist = []
        self._sr_ingests = 0
        self._sr_last = -10**9
        self.pts = {}
        self.c2ws = {}
        self.frames = {}


        self._pt_mask = {}

    @property
    def num_frames(self):
        return len(self.pts)

    @torch.no_grad()
    def ingest(self, estimator, frames_rgb, c2w, K_pix, frame_ids):


        frames_rgb = np.asarray(frames_rgb, np.float32); c2w = np.asarray(c2w, np.float32)
        N = frames_rgb.shape[0]
        if N < 3:
            return


        if self.color_anchor:
            _ca_flat = frames_rgb.reshape(-1, 3)
            _ca_mu = _ca_flat.mean(0).astype(np.float32)
            _ca_sig = _ca_flat.std(0).astype(np.float32)
            _ca_dbg = bool(os.environ.get("EVOKE_CONSIST_DEBUG"))
            if self._ca_seen < self.color_anchor_ref_windows:
                self._ca_ref_acc.append((_ca_mu.copy(), _ca_sig.copy()))
                _mus = np.stack([a for a, _ in self._ca_ref_acc], 0)
                _sigs = np.stack([b for _, b in self._ca_ref_acc], 0)
                self._ca_ref = (_mus.mean(0).astype(np.float32), _sigs.mean(0).astype(np.float32))
                if _ca_dbg:
                    print(f"[color-anchor] win_mu={np.round(_ca_mu, 3).tolist()} "
                          f"ref_mu={np.round(self._ca_ref[0], 3).tolist()} alpha=0.000(reference period, no correction "
                          f"{self._ca_seen + 1}/{self.color_anchor_ref_windows})", flush=True)
            elif self._ca_ref is not None:
                _r_mu, _r_sig = self._ca_ref
                _eps = 1e-6
                _xp = (_ca_flat - _ca_mu[None]) / np.maximum(_ca_sig[None], _eps) * _r_sig[None] + _r_mu[None]
                _a = self.color_anchor_alpha
                _xo = _a * _xp + (1.0 - _a) * _ca_flat
                frames_rgb = np.clip(_xo, 0.0, 1.0).reshape(frames_rgb.shape).astype(np.float32)
                if _ca_dbg:
                    print(f"[color-anchor] win_mu={np.round(_ca_mu, 3).tolist()} "
                          f"ref_mu={np.round(_r_mu, 3).tolist()} alpha={_a:.3f}", flush=True)
            self._ca_seen += 1
        Ks = np.stack([np.asarray(K_pix, np.float32)] * N)
        try:
            depth, intr, conf, rgb = estimator.depth_window(frames_rgb, c2w, Ks)
        except Exception as _e:
            print(f"[da3-framebank] WARN ingest failed ({type(_e).__name__}: {_e}); skipping this chunk", flush=True)
            return


        if self.consist_probation > 0 and self._probation:
            _pb_held = 0; _pb_adm = 0; _pb_drop = 0; _pb_unver = 0
            _pb_new = [(torch.as_tensor(depth[i], device=self.device),
                        torch.as_tensor(intr[i], device=self.device),
                        torch.as_tensor(c2w[i], device=self.device)) for i in range(N)]
            for _g, _rec in list(self._probation.items()):
                if _g not in self.frames:
                    continue
                _hm = _rec["mask"]
                _hd = _rec["depth"]
                if not bool(_hm.any()):
                    continue
                _Xh = unproject_depth_torch(_hd, _rec["it"], _rec["cw"])[_hm]
                _ok = torch.zeros(_Xh.shape[0], dtype=torch.int16, device=self.device)
                _bad = torch.zeros_like(_ok)
                for (_dn, _itn, _cwn) in _pb_new:
                    _Rn = _cwn[:3, :3]; _tn = _cwn[:3, 3]
                    _cam = (_Xh - _tn[None]) @ _Rn
                    _z = _cam[:, 2]
                    _fx = _itn[0, 0]; _fy = _itn[1, 1]; _cx = _itn[0, 2]; _cy = _itn[1, 2]
                    _hn, _wn = int(_dn.shape[0]), int(_dn.shape[1])
                    _px = torch.round(_cam[:, 0] / _z.clamp(min=1e-6) * _fx + _cx).long()
                    _py = torch.round(_cam[:, 1] / _z.clamp(min=1e-6) * _fy + _cy).long()
                    _inb = (_z > 1e-4) & (_px >= 0) & (_px < _wn) & (_py >= 0) & (_py < _hn)
                    _dat = _dn.reshape(-1)[(_py.clamp(0, _hn - 1) * _wn + _px.clamp(0, _wn - 1))]
                    _seen = _inb & (_dat > 1e-4)
                    _cons = _seen & ((_z - _dat).abs() <= self.consist_tau * _dat)
                    _ok += _cons.to(torch.int16)
                    _bad += (_seen & ~_cons).to(torch.int16)
                _unver = (_ok == 0) & (_bad == 0)
                _admv = (_ok >= 1) & (_ok >= _bad)
                _adm = _admv | _unver
                _pb_held += int(_Xh.shape[0]); _pb_adm += int(_admv.sum())
                _pb_unver += int(_unver.sum()); _pb_drop += int((~_adm).sum())
                if bool(_adm.any()):
                    _adm_mask = torch.zeros_like(_hm)
                    _adm_mask[_hm] = _adm
                    _dg, _itg, _cwg, _rgbg = self.frames[_g]
                    _dg2 = torch.where(_adm_mask, _hd, _dg)
                    _pm = self._pt_mask.get(_g)
                    _pm2 = (_pm | _adm_mask) if _pm is not None else _adm_mask

                    _keep = torch.isfinite(_dg2) & (_dg2 > 1e-4) & _pm2
                    _Xg2 = unproject_depth_torch(_dg2, _itg, _cwg)
                    self.frames[_g] = (_dg2, _itg, _cwg, _rgbg)
                    self.pts[_g] = (_Xg2[_keep], _rgbg.permute(1, 2, 0)[_keep])
                    self._pt_mask[_g] = _keep
            self._probation = {}
            if bool(os.environ.get("EVOKE_CONSIST_DEBUG")):
                print(f"[probation-dbg] held={_pb_held} admitted={_pb_adm} dropped={_pb_drop} "
                      f"unverifiable={_pb_unver}", flush=True)


        _ref_xyz = None
        if self.consist_gate and self.pts:
            _gs = sorted(self.pts.keys())
            _ref_gids = _gs[-self.consist_ref_frames:] if self.consist_ref_frames > 0 else _gs
            _xs = [self.pts[g][0] for g in _ref_gids if self.pts[g][0].shape[0] > 0]
            if _xs:
                _ref = torch.cat(_xs, 0)
                if _ref.shape[0] >= self.consist_min_ref:
                    if _ref.shape[0] > 300000:
                        _ref = _ref[torch.randint(0, _ref.shape[0], (300000,), device=self.device)]
                    _ref_xyz = _ref


        _cg_dbg = bool(os.environ.get("EVOKE_CONSIST_DEBUG"))
        _cg_track = _cg_dbg or self.self_reanchor
        _cg_have = 0; _cg_rej = 0; _cg_tot = 0; _cg_confs = []; _cg_scales = []


        _cd_cache = {}
        if self.consist_scale_align and _ref_xyz is not None:
            _sa_meds = []; _sa_nov = 0
            for i in range(N):
                _d_i = torch.as_tensor(depth[i], device=self.device)
                _it_i = torch.as_tensor(intr[i], device=self.device)
                _cw_i = torch.as_tensor(c2w[i], device=self.device)
                _cd_i = _cloud_depth_map(_ref_xyz, _cw_i, _it_i,
                                         int(_d_i.shape[0]), int(_d_i.shape[1]), device=self.device)
                _cd_cache[i] = _cd_i
                _ov_i = torch.isfinite(_cd_i) & (_cd_i > 1e-4) & (_d_i > 1e-4)
                _n_i = int(_ov_i.sum()); _sa_nov += _n_i
                if _n_i > 200:
                    _sa_meds.append(float((_d_i[_ov_i] / _cd_i[_ov_i]).median()))
            if _sa_meds and _sa_nov >= 2000:
                _s_raw = float(np.median(_sa_meds))
                if 0.5 < _s_raw < 2.0:
                    _s = float(np.clip(_s_raw, 0.75, 1.33))
                    if abs(_s - 1.0) > 1e-3:
                        depth = depth / _s
                    if _cg_dbg:
                        print(f"[scale-align] window_scale={_s_raw:.3f}"
                              f"{'(clip->%.3f)' % _s if abs(_s - _s_raw) > 1e-3 else ''} "
                              f"applied n_overlap={_sa_nov}", flush=True)
                elif _cg_dbg:
                    print(f"[scale-align] window_scale={_s_raw:.3f} absurd, not in (0.5,2) -> skip", flush=True)
            elif _cg_dbg:
                print(f"[scale-align] overlap too small ({_sa_nov}<2000) -> skip", flush=True)
        for i, gid in enumerate(frame_ids):
            d = torch.as_tensor(depth[i], device=self.device)
            it = torch.as_tensor(intr[i], device=self.device)
            cw = torch.as_tensor(c2w[i], device=self.device)
            wp = unproject_depth_torch(d, it, cw)
            m = torch.isfinite(wp).all(-1) & (d > 1e-4)
            if conf is not None and self.conf_pct > 0:
                thr = float(np.percentile(conf[i].reshape(-1), self.conf_pct))
                m &= torch.as_tensor(conf[i], device=self.device) >= thr
            if rgb is not None:
                r = torch.as_tensor(np.asarray(rgb[i], np.float32), device=self.device)
            else:
                r = torch.zeros_like(wp)

            if self.hygiene and rgb is not None:

                if 0.0 < self.hyg_sat_max < 1.0:


                    _mx = r.max(-1).values
                    _chroma = (_mx - r.min(-1).values).clamp(min=1e-6)
                    _s = _chroma / _mx.clamp(min=1e-6)
                    _f = torch.where(_s > self.hyg_sat_max,
                                     (self.hyg_sat_max * _mx) / _chroma,
                                     torch.ones_like(_mx)).clamp(max=1.0)
                    r = (_mx.unsqueeze(-1) - (_mx.unsqueeze(-1) - r) * _f.unsqueeze(-1)).clamp(0.0, 1.0)

                _drop = torch.zeros_like(m)
                if self.hyg_flat_std > 0.0:
                    _g = (0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]).unsqueeze(0).unsqueeze(0)
                    _w = max(3, int(self.hyg_flat_win) | 1); _pad = _w // 2
                    _mean = torch.nn.functional.avg_pool2d(_g, _w, 1, _pad)
                    _msq = torch.nn.functional.avg_pool2d(_g * _g, _w, 1, _pad)
                    _std = (_msq - _mean * _mean).clamp(min=0.0).sqrt()[0, 0]
                    _drop = _drop | (_std < self.hyg_flat_std)


                if conf is not None and self.hyg_conf_abs > 0.0:
                    _drop = _drop | (torch.as_tensor(conf[i], device=self.device) < self.hyg_conf_abs)

                if conf is not None and self.hyg_conf_pct > 0.0:
                    _thr2 = float(np.percentile(conf[i].reshape(-1), self.hyg_conf_pct))
                    _drop = _drop | (torch.as_tensor(conf[i], device=self.device) < _thr2)
                _keep = ~_drop
                d = torch.where(_keep, d, torch.zeros_like(d))
                m = m & _keep


            if _ref_xyz is not None:


                _cd = _cd_cache.get(i)
                if _cd is None:
                    _cd = _cloud_depth_map(_ref_xyz, cw, it, int(d.shape[0]), int(d.shape[1]), device=self.device)
                _have = torch.isfinite(_cd) & (_cd > 1e-4)

                if self.consist_adaptive and conf is not None:


                    _cf = torch.as_tensor(conf[i], device=self.device).to(torch.float32)
                    _u = (_cf / max(self.consist_conf_ref, 1e-6)).clamp_(0.0, 1.0)
                    _tau = self.consist_tau_lo + (self.consist_tau_hi - self.consist_tau_lo) * _u
                else:
                    _tau = self.consist_tau

                _conflict = _have & (d > 1e-4) & ((d - _cd).abs() > _tau * _cd)
                _keep2 = ~_conflict
                if _cg_track:

                    _ov = _have & (d > 1e-4)
                    _cg_have += int(_ov.sum())
                    _cg_rej += int(_conflict.sum()); _cg_tot += int(d.numel())
                    if conf is not None:
                        _cg_confs.append(float(np.median(conf[i])))


                    if int(_ov.sum()) > 500:
                        _cg_scales.append(float((d[_ov] / _cd[_ov]).median()))
                d = torch.where(_keep2, d, torch.zeros_like(d))
                m = m & _keep2


                if self.consist_probation > 0:
                    _hole = (~_have) & (d > 1e-4) & m


                    if self.consist_probation_frac > 0.0 and bool(_hole.any()):
                        _w = self.consist_probation_win
                        _hf = torch.nn.functional.avg_pool2d(
                            _hole.float()[None, None], _w, stride=1, padding=_w // 2)[0, 0]
                        _hole = _hole & (_hf > self.consist_probation_frac)
                    if bool(_hole.any()):
                        self._probation[int(gid)] = {
                            "mask": _hole.clone(),
                            "depth": torch.where(_hole, d, torch.zeros_like(d)),
                            "rgb": r.clone(), "it": it.clone(), "cw": cw.clone()}
                        d = torch.where(_hole, torch.zeros_like(d), d)
                        m = m & ~_hole

            self.pts[int(gid)] = (wp[m], r[m])
            self._pt_mask[int(gid)] = m
            self.c2ws[int(gid)] = cw

            self.frames[int(gid)] = (d, it, cw, r.permute(2, 0, 1).contiguous())
        if _cg_dbg and self.consist_gate:
            _newest = max(int(g) for g in frame_ids)
            _cstr = f" conf_med={np.median(_cg_confs):.2f}" if _cg_confs else ""
            _sstr = f" scale_ratio_med={np.median(_cg_scales):.3f}" if _cg_scales else ""
            print(f"[consist-dbg] newest_gid={_newest}(~{_newest/24.0:.1f}s) "
                  f"checkable={_cg_have}/{_cg_tot} ({100.0*_cg_have/max(_cg_tot,1):.1f}%) "
                  f"rejected={_cg_rej} ({100.0*_cg_rej/max(_cg_have,1):.1f}% of checkable){_cstr}{_sstr}", flush=True)

        if self.carve:
            self._carve_recent([int(g) for g in frame_ids])

        if self.pts and (self.hist_max_frames > 0 or self.reanchor_every > 0):
            _newest = max(self.pts.keys())
            if self.hist_max_frames > 0:
                self.evict_before(_newest - self.hist_max_frames + 1)
            if self.reanchor_every > 0:
                self._ingest_calls += 1
                if self._ingest_calls % self.reanchor_every == 0 and self.reanchor_keep_frames > 0:
                    self.evict_before(_newest - self.reanchor_keep_frames + 1)

        if self.self_reanchor:
            self._sr_ingests += 1
            _scale = float(np.median(_cg_scales)) if _cg_scales else None
            _rej = _cg_rej / max(_cg_have, 1)
            _conf = float(np.median(_cg_confs)) if _cg_confs else 0.0
            self._win_hist.append({"gids": [int(g) for g in frame_ids],
                                   "conf": _conf, "scale": _scale, "rej": _rej})


            if self.sr_pin_prime and self._sr_ingests <= self.sr_pin_windows:
                self._pinned_wins.append({"gids": [int(g) for g in frame_ids],
                                          "conf": _conf, "scale": _scale, "rej": _rej})
            if len(self._win_hist) > 64:
                self._win_hist = self._win_hist[-64:]
            _diverged = ((_scale is not None and (_scale > self.sr_scale_thr or _scale < 1.0 / self.sr_scale_thr))
                         or (_cg_have > 0 and _rej > self.sr_rej_thr))
            if (_diverged and len(self._win_hist) >= 6
                    and self._sr_ingests - self._sr_last >= self.sr_min_gap):

                _cands = []
                for _wi in range(max(0, len(self._win_hist) - self.sr_lookback), len(self._win_hist) - 1):
                    _w = self._win_hist[_wi]
                    if _w["scale"] is None or not (0.8 <= _w["scale"] <= 1.25) or _w["rej"] >= 0.3:
                        continue
                    if _w["conf"] < self.sr_anchor_min_conf:
                        continue
                    if not all(g in self.pts for g in _w["gids"]):
                        continue
                    _cands.append((_w["conf"], _wi))
                _pin_keep = set()
                if self.sr_pin_prime:
                    for _w in self._pinned_wins:
                        if all(g in self.pts for g in _w["gids"]):
                            _pin_keep.update(_w["gids"])
                if _cands:
                    _best_wi = max(_cands)[1]
                    _keep = set(_pin_keep)
                    for _wi in range(max(0, _best_wi - self.sr_keep_windows + 1), _best_wi + 1):
                        _keep.update(self._win_hist[_wi]["gids"])
                    _n_before = len(self.pts)
                    self._evict_keep(_keep)
                    self._sr_last = self._sr_ingests
                    print(f"[self-reanchor] divergence triggered (scale={_scale if _scale is not None else float('nan'):.2f} "
                          f"rej={_rej:.2f}) -> re-anchoring to the best window in history gid[{min(_keep)}..{max(_keep)}] "
                          f"(conf={self._win_hist[_best_wi]['conf']:.2f}) cloud {_n_before}->{len(self.pts)} frames", flush=True)
                elif self.sr_pin_prime and _pin_keep:

                    _n_before = len(self.pts)
                    self._evict_keep(_pin_keep)
                    self._sr_last = self._sr_ingests
                    print(f"[self-reanchor] divergence triggered (scale={_scale if _scale is not None else float('nan'):.2f} "
                          f"rej={_rej:.2f}) no healthy recent anchor -> [v4] falling back to the prime anchor gid[{min(_pin_keep)}..{max(_pin_keep)}] "
                          f"cloud {_n_before}->{len(self.pts)} frames", flush=True)
                else:
                    print(f"[self-reanchor] divergence triggered (scale={_scale} rej={_rej:.2f}) but no healthy anchor in the lookback -> skipping", flush=True)

    @torch.no_grad()
    def _carve_recent(self, new_gids):


        new_set = set(int(g) for g in new_gids)

        newobs = [self.frames[g][:3] for g in new_gids if g in self.frames]
        if not newobs:
            return
        gmin = min(new_set)
        targets = sorted([g for g in self.frames.keys() if g < gmin])
        if self.carve_ref_frames > 0:
            targets = targets[-self.carve_ref_frames:]
        _dbg = bool(os.environ.get("EVOKE_CARVE_DEBUG"))
        _del_px = 0; _val_px = 0; _ftouch = 0; _strike_pend = 0
        for g in targets:
            d_g, it_g, cw_g, rgb_g = self.frames[g]
            valid = torch.isfinite(d_g) & (d_g > 1e-4)
            if not bool(valid.any()):
                continue
            if _dbg:
                _val_px += int(valid.sum())
            Xg = unproject_depth_torch(d_g, it_g, cw_g)
            votes = torch.zeros_like(d_g, dtype=torch.int16)
            for (d_n, it_n, cw_n) in newobs:
                Rn = cw_n[:3, :3]; tn = cw_n[:3, 3]
                cam = (Xg - tn) @ Rn
                z = cam[..., 2]
                fx = it_n[0, 0]; fy = it_n[1, 1]; cx = it_n[0, 2]; cy = it_n[1, 2]
                hn, wn = int(d_n.shape[0]), int(d_n.shape[1])
                px = torch.round(cam[..., 0] / z.clamp(min=1e-6) * fx + cx).long()
                py = torch.round(cam[..., 1] / z.clamp(min=1e-6) * fy + cy).long()
                inb = (z > 1e-4) & (px >= 0) & (px < wn) & (py >= 0) & (py < hn)
                idx = (py.clamp(0, hn - 1) * wn + px.clamp(0, wn - 1)).reshape(-1)
                d_at = d_n.reshape(-1)[idx].reshape(z.shape)

                freespace = inb & (d_at > 1e-4) & (z < d_at * (1.0 - self.carve_margin))
                votes += (valid & freespace).to(torch.int16)


            if self.carve_strike_windows > 1:
                _confirmed = valid & (votes >= self.carve_min_views)
                _st = self._carve_strike.get(g)
                if _st is None:
                    _st = torch.zeros_like(votes)
                _st = torch.where(_confirmed, _st + 1, torch.zeros_like(_st))
                carve = valid & (_st >= self.carve_strike_windows)
                if _dbg:
                    _strike_pend += int((_confirmed & ~carve).sum())
                _st = torch.where(carve, torch.zeros_like(_st), _st)
                self._carve_strike[g] = _st
            else:

                carve = valid & (votes >= self.carve_min_views)
            if bool(carve.any()):
                d_g2 = torch.where(carve, torch.zeros_like(d_g), d_g)
                keep = torch.isfinite(d_g2) & (d_g2 > 1e-4)

                if g in self._pt_mask:
                    keep = keep & self._pt_mask[g]
                Xg2 = unproject_depth_torch(d_g2, it_g, cw_g)
                self.frames[g] = (d_g2, it_g, cw_g, rgb_g)
                self.pts[g] = (Xg2[keep], rgb_g.permute(1, 2, 0)[keep])
                self._pt_mask[g] = keep
                if _dbg:
                    _del_px += int(carve.sum()); _ftouch += 1
        if _dbg:
            _newest = max(new_set)
            _spstr = f" strike_pending={_strike_pend}" if self.carve_strike_windows > 1 else ""
            print(f"[carve-dbg] newest_gid={_newest}(~{_newest/24.0:.1f}s) targets={len(targets)} "
                  f"frames_touched={_ftouch} deleted_px={_del_px}/{_val_px} "
                  f"({100.0*_del_px/max(_val_px,1):.1f}%){_spstr}", flush=True)

    def evict_before(self, min_gid):

        for gid in [g for g in self.pts if g < int(min_gid)]:
            self.pts.pop(gid, None); self.c2ws.pop(gid, None); self.frames.pop(gid, None)
            self._pt_mask.pop(gid, None)
        for gid in [g for g in self._probation if g < int(min_gid)]:
            self._probation.pop(gid, None)
        for gid in [g for g in self._carve_strike if g < int(min_gid)]:
            self._carve_strike.pop(gid, None)

    def _evict_keep(self, keep_gids):

        keep = set(int(g) for g in keep_gids)
        for gid in [g for g in self.pts if g not in keep]:
            self.pts.pop(gid, None); self.c2ws.pop(gid, None); self.frames.pop(gid, None)
            self._pt_mask.pop(gid, None); self._probation.pop(gid, None)
            self._carve_strike.pop(gid, None)


@torch.no_grad()
def recall_frames(bank, pool_ids, target_c2ws, K_pix, *, recall_k=12, n_nearby=4,
                  n_tframe=6, grid_div=8, mask_pts=8000, height, width, device="cuda"):


    device = torch.device(device)
    pool = [int(g) for g in pool_ids if int(g) in bank.pts]
    if len(pool) < 3:
        return sorted(pool)
    Np = len(pool)
    K_t = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    if torch.is_tensor(target_c2ws):
        tc = target_c2ws.detach().to(device=device, dtype=torch.float32)
    else:
        tc = torch.as_tensor(np.asarray(target_c2ws, np.float32), device=device)
    S = int(n_tframe)
    tframes = tc[torch.linspace(0, tc.shape[0] - 1, S).round().long()]
    gh, gw = max(1, height // int(grid_div)), max(1, width // int(grid_div))
    Gc = gh * gw; G = S * Gc; mp = int(mask_pts)
    K = min(int(recall_k), Np)


    P = torch.empty((Np, mp, 3), device=device, dtype=torch.float32)
    for i, g in enumerate(pool):
        xyz = bank.pts[g][0]; n = int(xyz.shape[0])
        if n == 0:
            P[i].zero_()
        else:
            P[i] = xyz[torch.randint(0, n, (mp,), device=device)]


    sx = width / gw; sy = height / gh
    n_ar = torch.arange(Np, device=device)[:, None].expand(Np, mp)
    M = torch.zeros((Np, G), dtype=torch.bool, device=device)
    for s in range(S):
        cam = torch.einsum('nmj,jk->nmk', P - tframes[s, :3, 3][None, None], tframes[s, :3, :3])
        z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * K_t[0, 0] + K_t[0, 2]
        py = cam[..., 1] / z.clamp(min=1e-6) * K_t[1, 1] + K_t[1, 2]
        ok = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        cell = s * Gc + (py / sy).long().clamp(0, gh - 1) * gw + (px / sx).long().clamp(0, gw - 1)
        M[n_ar[ok], cell[ok]] = True


    order_recent = torch.argsort(torch.tensor(pool, device=device), descending=True)[:int(n_nearby)].tolist()
    cov_frac = M.float().mean(1)
    covered = torch.zeros(G, dtype=torch.bool, device=device)
    chosen = []
    for i in order_recent:
        if float(cov_frac[i]) > 0.005:
            chosen.append(i); covered |= M[i]
    chosen_set = set(chosen)
    while len(chosen) < K:
        gains = (M & ~covered[None]).sum(1)
        if chosen_set:
            gains[torch.tensor(list(chosen_set), device=device)] = -1
        gmax, best = torch.max(gains, 0)
        if int(gmax) <= 0:
            break
        bi = int(best); chosen.append(bi); chosen_set.add(bi); covered |= M[bi]
    return sorted(int(pool[i]) for i in chosen)


@torch.no_grad()
def render_recalled(bank, sel_ids, target_c2ws, K_pix, height, width, *, splat_radius=2, device="cuda"):

    device = torch.device(device)
    xs = [bank.pts[int(g)][0] for g in sel_ids if int(g) in bank.pts]
    rs = [bank.pts[int(g)][1] for g in sel_ids if int(g) in bank.pts]
    xyz = torch.cat(xs, 0) if xs else torch.zeros((0, 3), device=device)
    rgb = torch.cat(rs, 0) if rs else torch.zeros((0, 3), device=device)

    tc_np = (target_c2ws.detach().cpu().numpy() if torch.is_tensor(target_c2ws)
             else np.asarray(target_c2ws, np.float32))
    K_render = np.stack([np.asarray(K_pix, np.float32)] * int(tc_np.shape[0]))
    return render_cloud_batched(xyz, rgb, tc_np, K_render, int(height), int(width),
                                device=device, splat_radius=int(splat_radius))


@torch.no_grad()
def build_recall_cloud_warp(estimator, raw_video_b, lingbot_c2ws_b, K_pix, target_c2ws,
                            *, pix_start, pix_stride, window_pix, height, width,
                            ingest_n=12, recall_k=12, n_nearby=4, lag=1, history=16,
                            n_tframe=6, grid_div=8, mask_pts=8000, conf_pct=30.0,
                            splat_radius=2, device="cuda"):


    _reset_depth_stream(estimator)
    import os as _os, time as _time
    _timing = _os.environ.get("EVOKE_DA3_TIMING", "") == "1"

    def _sync():
        if _timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    T = int(raw_video_b.shape[1])
    kc = int(pix_start) // int(pix_stride)
    newest = kc - 1 - int(lag)
    oldest = max(0, newest - int(history) + 1)
    bank = DA3FrameBank(device=device, conf_percentile=float(conf_pct))
    pool_ids = []
    _n_ingest = 0
    _sync(); _t0 = _time.perf_counter()
    for j in range(oldest, newest + 1):
        s = j * int(pix_stride)
        if s < 0 or s >= T:
            continue
        e = min(s + int(window_pix), T)
        if e - s < 3:
            continue
        idxs = torch.unique(torch.linspace(s, e - 1, int(ingest_n)).round().long().clamp(0, T - 1))
        if idxs.numel() < 3:
            continue
        frames = (raw_video_b[:, idxs].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
        c2w = lingbot_c2ws_b[idxs].float().cpu().numpy()
        ids = idxs.tolist()
        bank.ingest(estimator, frames, c2w, K_pix, ids)
        pool_ids.extend(ids); _n_ingest += 1
    pool_ids = [g for g in pool_ids if g in bank.pts]
    _sync(); _t_ingest = _time.perf_counter() - _t0

    _sync(); _t0 = _time.perf_counter()
    sel = recall_frames(bank, pool_ids, target_c2ws, K_pix, recall_k=recall_k, n_nearby=n_nearby,
                        n_tframe=n_tframe, grid_div=grid_div, mask_pts=mask_pts,
                        height=height, width=width, device=device)
    _sync(); _t_recall = _time.perf_counter() - _t0

    _sync(); _t0 = _time.perf_counter()
    warp, vis = render_recalled(bank, sel, target_c2ws, K_pix, height, width,
                                splat_radius=splat_radius, device=device)
    _sync(); _t_render = _time.perf_counter() - _t0
    if _timing:
        print(f"[da3-warp-timing] kc={kc} lag={lag} hist={history} pool={len(pool_ids)} sel={len(sel)} | "
              f"ingest({_n_ingest}ch x {ingest_n}f)={_t_ingest*1e3:.0f}ms recall={_t_recall*1e3:.1f}ms "
              f"render={_t_render*1e3:.0f}ms (recall+render={(_t_recall+_t_render)*1e3:.0f}ms)", flush=True)
    return warp, vis


def frame_signature(frame):


    return tuple((id(x), x._version) for x in frame)


@torch.no_grad()
def prepare_multisrc(store, ids_all, device, cache=None, count=2000, generator=None, fused_sampling=False, return_aux=True):

    cache = {} if cache is None else cache
    for g in set(cache) - set(ids_all):
        del cache[g]
    P_all = None if fused_sampling else torch.full((len(ids_all), count, 3), 1e6, device=device)
    subpts = {}; dense = {}; valid_sources = []
    for row, g in enumerate(ids_all):
        frame = store[g]; signature = frame_signature(frame)
        entry = cache.get(g)
        if entry is None or entry[0] != signature:
            d, it, cwi, _ = frame
            wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
            valid = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]
            entry = (signature, frame, wp, valid)
            cache[g] = entry
        wp, valid = entry[2:]
        if return_aux:dense[g] = wp
        if fused_sampling:
            valid_sources.append(valid)
            continue
        if valid.shape[0] > 0:
            sp = valid[torch.randint(0, valid.shape[0], (count,), device=device, generator=generator)]
            subpts[g] = sp; P_all[row] = sp
        else:
            subpts[g] = valid
    if fused_sampling:
        from ui.warp_sampling import sample_points
        P_all = sample_points(valid_sources, count=count, device=device, generator=generator)
        if return_aux:
            for row,g in enumerate(ids_all):
                subpts[g] = P_all[row] if valid_sources[row].shape[0] else valid_sources[row]
    return P_all, subpts, dense


@torch.no_grad()
def _render_multisrc(store, ids_all, target_c2ws, K_pix, height, width, *,
                     nsrc=8, nearby=16, splat_radius=1, dens_thresh=0.45, dens_win=7,
                     recall_min_cov=0.5, recall_margin=0.15, device="cuda"):


    import torch.nn.functional as _F
    H, W = int(height), int(width); HW = H * W; F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)


    M = 2000; FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}
    P_all = torch.full((len(ids_all), M, 3), FAR, device=device); subpts = {}
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]
        if wp.shape[0] > 0:
            sp = wp[torch.randint(0, wp.shape[0], (M,), device=device)]
            subpts[g] = sp; P_all[id2row[g]] = sp
        else:
            subpts[g] = wp

    def covis_vec(rows, tpose):
        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def est_scale(g, ref_pts, tposepts_n=200):
        d, it, cwi, _ = store[g]; h, w = d.shape
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0
        D = torch.full((h * w,), float("inf"), device=device); D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        if int(v.sum()) < tposepts_n:
            return 1.0
        s = float((D[v] / df[v]).median())
        return s if 0.2 < s < 5.0 else 1.0

    def render_one(g, R, t, scale=1.0):
        d, it, cwi, rr = store[g]
        if scale != 1.0:
            d = d * scale
        wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3); rgb = rr.permute(1, 2, 0).reshape(-1, 3)
        npt = wp.shape[0]; cam = (R @ wp.T).T + t; zc = cam[:, 2]
        px = torch.round(cam[:, 0] / zc.clamp(min=1e-6) * fx + cx).long(); py = torch.round(cam[:, 1] / zc.clamp(min=1e-6) * fy + cy).long()
        zq = (zc.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1); idx = torch.arange(npt, device=device)
        INF = torch.full((HW,), (1 << 62), dtype=torch.long, device=device)
        front = (zc > 1e-4) & (d.reshape(-1) > 1e-4)

        hit0 = torch.zeros((HW,), dtype=torch.bool, device=device)
        ok0 = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        hit0[(py * W + px)[ok0]] = True
        for dy in range(-splat_radius, splat_radius + 1):
            for dx in range(-splat_radius, splat_radius + 1):
                qx = px + dx; qy = py + dy; ok = front & (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
                INF.scatter_reduce_(0, (qy * W + qx)[ok], (zq[ok] << 24) | idx[ok], reduce="amin", include_self=True)
        valid = INF < (1 << 62); win = (INF & ((1 << 24) - 1)).clamp(max=npt - 1)
        col = torch.zeros((HW, 3), device=device); col[valid] = rgb[win[valid]]
        return col, valid, hit0

    tc = target_c2ws.to(device).float()
    nearby_ids = ids_all[-int(nearby):]; old_ids = [g for g in ids_all if g not in nearby_ids]
    nearby_rows = [id2row[g] for g in nearby_ids]; old_rows = [id2row[g] for g in old_ids]
    nref = torch.cat([subpts[g] for g in nearby_ids if subpts[g].shape[0] > 0]) if nearby_ids else None

    nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]) if nearby_ids else None
    old_cen = torch.stack([store[g][2][:3, 3] for g in old_ids]) if old_ids else None
    old_fwd = torch.stack([store[g][2][:3, 2] for g in old_ids]) if old_ids else None
    warps = []; viss = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(nearby_rows, tpose)
        order = torch.argsort(cv, descending=True).tolist()
        srcs = [nearby_ids[i] for i in order[:int(nsrc)]]
        scales = {}
        if old_ids:
            cvo = covis_vec(old_rows, tpose); oi = int(cvo.argmax())


            tcen = tpose[:3, 3]; tfwd = tpose[:3, 2]
            d_old = float((old_cen[oi] - tcen).norm())
            d_nb_min = float((nb_cen - tcen).norm(dim=1).min()) if nb_cen is not None else 1e9
            cos_fwd = float((old_fwd[oi] @ tfwd) / (old_fwd[oi].norm() * tfwd.norm() + 1e-9))
            pose_ok = (d_old <= max(d_nb_min * 1.5, 1e-6)) and (cos_fwd > 0.5)
            if float(cvo[oi]) >= recall_min_cov and float(cvo[oi]) > float(cv.max()) + recall_margin and pose_ok:
                bo = old_ids[oi]; srcs = srcs[:int(nsrc) - 1] + [bo]
                if nref is not None:
                    scales[bo] = est_scale(bo, nref)
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = w2c[:3, 3]
        fused = torch.zeros((HW, 3), device=device); filled = torch.zeros((HW,), dtype=torch.bool, device=device)
        filled_true = torch.zeros((HW,), dtype=torch.bool, device=device)
        large_holes = torch.zeros((HW,), dtype=torch.bool, device=device)
        _fg = 2
        _kl = 6

        for si, g in enumerate(srcs):
            col, valid, hit0 = render_one(g, R, t, scale=scales.get(g, 1.0))
            if si == 0:
                wmask = valid
                prim_guard = _F.max_pool2d(valid.reshape(1, 1, H, W).float(), 2 * _fg + 1, 1, _fg)[0, 0].reshape(-1) > 0

                _holes = (~prim_guard).reshape(1, 1, H, W).float()
                _er = (_F.avg_pool2d(_holes, 2 * _kl + 1, 1, _kl) >= 0.999).float()
                large_holes = (_F.max_pool2d(_er, 2 * _kl + 1, 1, _kl)[0, 0].reshape(-1) > 0)
            else:
                wmask = valid & (~filled) & large_holes
            fused[wmask] = col[wmask]; filled |= wmask
            filled_true |= (hit0 & wmask)
        vis = filled.reshape(H, W).float()


        dens = _F.avg_pool2d(filled_true.reshape(1, 1, H, W).float(), int(dens_win), 1, int(dens_win) // 2)[0, 0]
        keep = (vis > 0) & (dens >= dens_thresh)
        fused = fused * keep.reshape(-1, 1)
        warps.append(fused.reshape(H, W, 3).clamp(0, 1).permute(2, 0, 1) * 2 - 1)
        viss.append(keep.float())
    return torch.stack(warps, 1)[None].contiguous(), torch.stack(viss)[None, None].contiguous()


def _render_backward(store, ids_all, target_c2ws, K_pix, height, width, *,
                     nearby=16, fill_iters=12, recall_min_cov=0.5, recall_margin=0.15, device="cuda"):


    import torch.nn.functional as _F
    H, W = int(height), int(width); F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)
    M = 2000; FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}
    P_all = torch.full((len(ids_all), M, 3), FAR, device=device); subpts = {}
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]
        if wp.shape[0] > 0:
            sp = wp[torch.randint(0, wp.shape[0], (M,), device=device)]
            subpts[g] = sp; P_all[id2row[g]] = sp
        else:
            subpts[g] = wp

    def covis_vec(rows, tpose):
        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def est_scale(g, ref_pts, tposepts_n=200):


        d, it, cwi, _ = store[g]; h, w = d.shape
        if ref_pts is None or ref_pts.shape[0] == 0:
            return 1.0
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0
        D = torch.full((h * w,), float("inf"), device=device)
        D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        if int(v.sum()) < tposepts_n:
            return 1.0
        s = float((D[v] / df[v]).median())
        return s if 0.2 < s < 5.0 else 1.0

    def bwarp_one(g, tpose, scale=1.0):
        d, it, cwi, rr = store[g]; h, w = d.shape
        if scale != 1.0:
            d = d * scale
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        z = d
        Xc = (xs - it[0, 2]) / it[0, 0] * z; Yc = (ys - it[1, 2]) / it[1, 1] * z
        cam = torch.stack([Xc, Yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (cwi @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(tpose); ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]; zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        src_flat = torch.arange(h * w, device=device)
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        key = (yt * W + xt)[ok]; zq = (zt.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)[ok]
        packed = (zq << 24) | src_flat[ok]
        INF = torch.full((H * W,), (1 << 62), dtype=torch.long, device=device)
        INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = INF < (1 << 62); owner = (INF & ((1 << 24) - 1)).clamp(max=h * w - 1)
        us = (owner % w).float(); vs = (owner // w).float()
        uv = torch.stack([us, vs], 0).reshape(2, H, W); vmask = valid.reshape(H, W)
        kern = torch.ones(1, 1, 3, 3, device=device); cur_uv = uv * vmask[None]; cur_v = vmask.float()[None, None]
        for _ in range(int(fill_iters)):
            if cur_v.min() > 0:
                break
            num = _F.conv2d((cur_uv * vmask[None])[None].reshape(2, 1, H, W), kern, padding=1).reshape(2, H, W)
            cnt = _F.conv2d(cur_v, kern, padding=1)[0, 0]
            newly = (cnt > 0) & (~vmask); filled_uv = num / cnt.clamp(min=1)[None]
            cur_uv = torch.where(vmask[None], cur_uv, filled_uv); vmask = vmask | newly; cur_v = vmask.float()[None, None]
        gx = cur_uv[0] / max(w - 1, 1) * 2 - 1; gy = cur_uv[1] / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        samp = _F.grid_sample(rr[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        col = torch.where(vmask[None].expand(3, H, W), samp, torch.zeros_like(samp))
        return col, vmask

    tc = target_c2ws.to(device).float()
    nearby_ids = ids_all[-int(nearby):]; old_ids = [g for g in ids_all if g not in nearby_ids]
    nearby_rows = [id2row[g] for g in nearby_ids]; old_rows = [id2row[g] for g in old_ids]
    nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]) if nearby_ids else None
    nref = torch.cat([subpts[g] for g in nearby_ids if subpts[g].shape[0] > 0]) if nearby_ids else None
    warps = []; viss = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(nearby_rows, tpose)
        primary = nearby_ids[int(cv.argmax())]
        fused, filled = bwarp_one(primary, tpose)
        if old_ids:
            cvo = covis_vec(old_rows, tpose); oi = int(cvo.argmax()); go = old_ids[oi]
            tcen = tpose[:3, 3]; tfwd = tpose[:3, 2]
            d_old = float((store[go][2][:3, 3] - tcen).norm())
            d_nb_min = float((nb_cen - tcen).norm(dim=1).min()) if nb_cen is not None else 1e9
            of = store[go][2][:3, 2]; cos_fwd = float((of @ tfwd) / (of.norm() * tfwd.norm() + 1e-9))
            pose_ok = (d_old <= max(d_nb_min * 1.5, 1e-6)) and (cos_fwd > 0.5)
            if pose_ok and float(cvo[oi]) >= recall_min_cov and float(cvo[oi]) > float(cv.max()) + recall_margin:
                cr, vr = bwarp_one(go, tpose, scale=est_scale(go, nref)); m = vr & (~filled)
                fused = torch.where(m[None].expand(3, H, W), cr, fused); filled = filled | vr
        warps.append((fused.clamp(0, 1) * 2 - 1)); viss.append(filled.float())
    return torch.stack(warps, 1)[None].contiguous(), torch.stack(viss)[None, None].contiguous()


@torch.no_grad()
def estimate_zbuf_scales(store, ids, nearby_set, ref_pts, device, batch_size=32):


    result = {g:(1.0,0) for g in ids if g in nearby_set}
    old = [g for g in ids if g not in nearby_set]
    if ref_pts is None or ref_pts.shape[0] == 0:
        return {g:(1.0,0) for g in ids}
    for start in range(0,len(old),batch_size):
        projected=[]; counts=[]
        for g in old[start:start+batch_size]:
            d,it,cwi,_=store[g];h,w=d.shape
            w2c=torch.linalg.inv(cwi)
            cam=(w2c[:3,:3] @ ref_pts.T).T + w2c[:3,3];z=cam[:,2]
            px=(cam[:,0]/z.clamp(min=1e-6)*it[0,0]+it[0,2]).round().long()
            py=(cam[:,1]/z.clamp(min=1e-6)*it[1,1]+it[1,2]).round().long()
            ok=(z>1e-4)&(px>=0)&(px<w)&(py>=0)&(py<h)
            projected.append((g,z,px,py,ok));counts.append(ok.sum())
        overlap=[];valid_counts=[]
        for (g,z,px,py,ok),count in zip(projected,torch.stack(counts).tolist()):
            if count<200:
                result[g]=(1.0,count);continue
            d=store[g][0];h,w=d.shape
            D=torch.full((h*w,),float('inf'),device=device)
            D.scatter_reduce_(0,py[ok]*w+px[ok],z[ok],reduce='amin',include_self=True)
            df=d.reshape(-1);v=(D<float('inf'))&(df>1e-4)
            overlap.append((g,D,df,v));valid_counts.append(v.sum())
        medians=[];median_ids=[]
        numbers=torch.stack(valid_counts).tolist() if valid_counts else []
        for (g,D,df,v),count in zip(overlap,numbers):
            if count<200:
                result[g]=(1.0,count);continue
            medians.append((D[v]/df[v]).median());median_ids.append((g,count))
        values=torch.stack(medians).tolist() if medians else []
        for (g,count),scale in zip(median_ids,values):
            result[g]=(scale if 0.2<scale<5.0 else 1.0,count)
    return result


def _render_backward_multisrc_zbuf(store, ids_all, target_c2ws, K_pix, height, width, *,
                                   nearby=16, fill_iters=12, recall_min_cov=0.5, recall_margin=0.15,
                                   depth_thresh=0.02, topk=8,
                                   fg_covis=0.3, fg_factor=1.5,
                                   fg_scale_exempt=1.0,
                                   zbuf_despeckle=False, zbuf_despeckle_ksize=3, zbuf_despeckle_fill_iters=4,
                                   device="cuda", _prepared=None, _prepare_only=False,
                                   _geometry_cache=None, _world_cache=None, _scale_batch=False,
                                   _reuse_target_inverse=False, _device_fusion_gate=False,
                                   _static_splat=False, _reuse_source_index=False, _fused_source_sampling=False,
                                   _reuse_full_source_rows=False, _fused_covis=False):


    import torch.nn.functional as _F
    H, W = int(height), int(width); F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)

    depth_thresh = float(os.environ.get("WARP_ZBUF_DEPTH_THRESH", str(depth_thresh)))
    topk = int(os.environ.get("WARP_ZBUF_TOPK", str(topk)))
    M = int(os.environ.get("WARP_ZBUF_COVIS_M", "2000"))
    _covis_min = float(os.environ.get("WARP_ZBUF_COVIS_MIN", "0"))
    FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}

    _wseed = os.environ.get("EVOKE_WARP_SEED")
    _cgen = torch.Generator(device=device).manual_seed(int(_wseed)) if _wseed is not None else None


    if _prepared is None:
        P_all, _, _ = prepare_multisrc(store, ids_all, device, _geometry_cache, M, _cgen,
                                     fused_sampling=_fused_source_sampling, return_aux=not _fused_source_sampling)
    else:
        P_all = _prepared['points']
    sample_done = torch.cuda.Event(enable_timing=True) if _prepare_only and torch.device(device).type=='cuda' else None
    if sample_done is not None:sample_done.record()
    world_cache = {} if _world_cache is None else _world_cache
    for g in set(world_cache)-set(ids_all): del world_cache[g]


    source_indices = {} if _reuse_source_index else None
    full_source_rows = list(range(len(ids_all))) if _reuse_full_source_rows else None

    def covis_vec(rows, tpose):

        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        if (_reuse_full_source_rows and isinstance(rows, list) and rows == full_source_rows
                and P_all.shape[0] == len(full_source_rows) and P_all.is_contiguous()
                and P_all.stride() == (P_all.shape[1] * 3, 3, 1)):


            P = P_all
        else:
            P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]
        cam = torch.einsum('cmj,kj->cmk', P - t, R)
        if _fused_covis:
            from ui.warp_covis import fused_covis
            return fused_covis(cam,P,fx,fy,cx,cy,W,H,FAR)
        z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def splat_one(g, tpose, scale=1.0, target_inverse=None):


        d, it, cwi, rr = store[g]; h, w = d.shape
        signature = (frame_signature(store[g]), scale)
        entry = world_cache.get(g)
        if entry is None or entry[0] != signature:
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                    torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
            z = d.float()
            if scale != 1.0:
                z = z * scale
            Xc = (xs - it[0, 2]) / it[0, 0] * z; Yc = (ys - it[1, 2]) / it[1, 1] * z
            cam = torch.stack([Xc, Yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
            world = (cwi @ cam.T).T[:, :3]
            entry = (signature, store[g], world, z)
            world_cache[g] = entry
        world, z = entry[2:]
        w2c = torch.linalg.inv(tpose) if target_inverse is None else target_inverse
        ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]; zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        if source_indices is None:
            src_flat = torch.arange(h * w, device=device)
        else:
            index_key = (h, w, world.device)
            src_flat = source_indices.get(index_key)
            if src_flat is None:
                src_flat = torch.arange(h * w, device=device)
                source_indices[index_key] = src_flat
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        col = torch.zeros(3, H, W, device=device)
        zbuf = torch.full((H * W,), float("inf"), device=device)
        if not _static_splat and not bool(ok.any()):
            return col, zbuf, torch.zeros(H, W, dtype=torch.bool, device=device)
        if _static_splat:


            key = torch.where(ok, yt * W + xt, src_flat % (H * W))
            zt_ok, src_ok = zt, src_flat
        else:
            key = (yt * W + xt)[ok]; zt_ok = zt[ok]; src_ok = src_flat[ok]

        zq = (zt_ok.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)
        packed = (zq << 24) | src_ok
        if _static_splat:


            packed = torch.where(ok, packed, 1 << 62)
        INF = torch.full((H * W,), (1 << 62), dtype=torch.long, device=device)
        INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = INF < (1 << 62); owner = (INF & ((1 << 24) - 1)).clamp(max=h * w - 1)
        if _static_splat:
            zbuf = torch.where(valid, ((INF >> 24).float()) / 1000.0, zbuf)
        else:
            zbuf[valid] = ((INF[valid] >> 24).float()) / 1000.0
        us = (owner % w).float(); vs = (owner // w).float()
        gx = us / max(w - 1, 1) * 2 - 1; gy = vs / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx.reshape(H, W), gy.reshape(H, W)], -1)[None]
        samp = _F.grid_sample(rr[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        vmask = valid.reshape(H, W)
        col = torch.where(vmask[None].expand(3, H, W), samp, torch.zeros_like(samp))
        return col, zbuf, vmask

    def est_scale(g, ref_pts, tposepts_n=200):


        d, it, cwi, _ = store[g]; h, w = d.shape
        if ref_pts is None or ref_pts.shape[0] == 0:
            return 1.0, 0
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0, int(ok.sum())
        D = torch.full((h * w,), float("inf"), device=device)
        D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        nov = int(v.sum())
        if nov < tposepts_n:
            return 1.0, nov
        s = float((D[v] / df[v]).median())
        return (s if 0.2 < s < 5.0 else 1.0), nov


    nearby_ids = ids_all[-int(nearby):]; nearby_set = set(nearby_ids)


    _fg_covis = float(os.environ.get("WARP_ZBUF_FG_COVIS", fg_covis))
    _fg_factor = float(os.environ.get("WARP_ZBUF_FG_FACTOR", fg_factor))
    _nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]).to(device).float() if (nearby_ids and _fg_covis > 0) else None
    _all_cen = {g: store[g][2][:3, 3].to(device).float() for g in ids_all} if _fg_covis > 0 else None
    if _prepared is None:
        _nref_l = [P_all[id2row[g]][P_all[id2row[g]][:, 0] < FAR * 0.5] for g in nearby_ids]
        _nref_l = [p for p in _nref_l if p.shape[0] > 0]
        nref = torch.cat(_nref_l) if _nref_l else None
        selection_done = None
        if _prepare_only:


            _tc = target_c2ws.to(device).float()
            _rows = [id2row[g] for g in ids_all]
            covis = torch.stack([covis_vec(_rows, _tc[f]) for f in range(F_t)])
            orders = torch.stack([torch.topk(cv, k=min(int(topk),len(ids_all))).indices for cv in covis]).tolist()
            scale_ids = sorted({ids_all[i] for order in orders for i in order})
            if sample_done is not None:
                selection_done=torch.cuda.Event(enable_timing=True);selection_done.record()
        else:
            scale_ids=ids_all
        _es = (estimate_zbuf_scales(store,scale_ids,nearby_set,nref,device) if _scale_batch else
               {g: ((1.0, 0) if g in nearby_set else est_scale(g, nref)) for g in scale_ids})


        _es = {g:_es.get(g,(1.0,0)) for g in ids_all}
    else:
        _es = _prepared['scales']; nref = None
    if _prepare_only:
        return {'points':P_all, 'scales':_es, 'sampleDone':sample_done,
                'selectionDone':selection_done,'covis':covis,'orders':orders,'scaleSourceCount':len(scale_ids)}
    src_scale = {g: _es[g][0] for g in ids_all}
    _fg_scale_exempt = float(os.environ.get("WARP_ZBUF_FG_SCALE_EXEMPT", str(fg_scale_exempt)))

    src_aligned = ({g: (g not in nearby_set and _es[g][1] >= 200 and 0.8 <= _es[g][0] <= 1.25) for g in ids_all}
                   if _fg_scale_exempt > 0 else {})

    tc = target_c2ws.to(device).float()
    all_rows = [id2row[g] for g in ids_all]
    warps = []; viss = []
    _age_dbg = bool(os.environ.get("EVOKE_WARP_AGE_DEBUG"))
    _age_rows = []
    for f in range(F_t):
        tpose = tc[f]
        if _prepared is not None and 'covis' in _prepared:
            cv = _prepared['covis'][f]; order = _prepared['orders'][f]
        else:
            cv = covis_vec(all_rows, tpose)
            k = min(int(topk), len(ids_all)); order = torch.topk(cv, k=k).indices.tolist()
        if _covis_min > 0:
            order = [i for i in order if float(cv[i]) >= _covis_min]
        if _fg_covis > 0 and _nb_cen is not None:
            _tcen = tpose[:3, 3]; _dnb = float((_nb_cen - _tcen).norm(dim=1).min())
            order = [i for i in order if float(cv[i]) >= _fg_covis
                     or float((_all_cen[ids_all[i]] - _tcen).norm()) <= _fg_factor * _dnb
                     or src_aligned.get(ids_all[i], False)]
        cand = [ids_all[i] for i in order]


        target_inverse = torch.linalg.inv(tpose) if _reuse_target_inverse and cand else None
        fused = torch.zeros(3, H, W, device=device)
        fused_depth = torch.full((H * W,), float("inf"), device=device)
        covered = torch.zeros(H, W, dtype=torch.bool, device=device)
        winner_row = torch.full((H * W,), -1, dtype=torch.long, device=device) if _age_dbg else None
        for g in cand:
            col, zbuf, _vm = splat_one(g, tpose, scale=src_scale[g], target_inverse=target_inverse)
            update = torch.isfinite(zbuf) & (zbuf < fused_depth - depth_thresh)
            if _device_fusion_gate:


                fallback = torch.isfinite(zbuf) & (zbuf < fused_depth)
                update = torch.where(update.any(), update, fallback)
            elif not bool(update.any()):
                update = torch.isfinite(zbuf) & (zbuf < fused_depth)
                if not bool(update.any()):
                    continue
            um = update.reshape(H, W)
            fused = torch.where(um[None].expand(3, H, W), col, fused)
            fused_depth = torch.where(update, zbuf, fused_depth); covered = covered | um
            if _age_dbg:
                winner_row[update] = int(id2row[g])
        if _age_dbg:
            _age_rows.append(winner_row[covered.reshape(-1)])


        if zbuf_despeckle:
            k = int(zbuf_despeckle_ksize)
            erosion = lambda x: -_F.max_pool2d(-x, k, 1, k // 2)
            dilation = lambda x: _F.max_pool2d(x, k, 1, k // 2)
            V = covered.float()[None, None]
            opened = dilation(erosion(V))
            hv = erosion(dilation(opened))
            hv_b = hv[0, 0] > 0.5
            fill_mask = hv_b & (~covered)
            removed = covered & (~hv_b)
            col = fused.clone()
            m = covered.float()[None, None]
            for _ in range(int(zbuf_despeckle_fill_iters)):
                num = _F.avg_pool2d(col[None] * m, 3, 1, 1); den = _F.avg_pool2d(m, 3, 1, 1)
                upd = fill_mask & (den[0, 0] > 1e-6) & (m[0, 0] < 0.5)
                col = torch.where(upd[None].expand(3, -1, -1), (num / den.clamp(min=1e-6))[0], col)
                m[0, 0] = torch.where(upd, torch.ones_like(m[0, 0]), m[0, 0])
            col[:, removed] = 0.0
            warps.append((col.clamp(0, 1) * 2 - 1)); viss.append(hv_b.float())
            continue
        warps.append((fused.clamp(0, 1) * 2 - 1)); viss.append(covered.float())
    vis_out = torch.stack(viss)[None, None].contiguous()

    _mean_cov = float(vis_out.mean())
    if _mean_cov < 0.25:
        _n_old = int(sum(1 for g in ids_all if g not in nearby_set))
        _n_aligned = int(sum(1 for g in ids_all if abs(float(src_scale[g]) - 1.0) > 1e-6))
        print(f"[zbuf-render] LOW-COV mean_cov={_mean_cov*100:.1f}% src={len(ids_all)} "
              f"nearby={len(nearby_ids)} old(recall)={_n_old} aligned={_n_aligned} "
              f"nref={'0' if nref is None else nref.shape[0]} F={F_t}", flush=True)
    if _age_dbg and _age_rows:
        _wr = torch.cat(_age_rows)
        if _wr.numel() > 0:
            _ids_t = torch.as_tensor(ids_all, device=_wr.device)
            _gid = _ids_t[_wr]; _age = (int(ids_all[-1]) - _gid).float() / 24.0
            _n = float(_wr.numel())
            _b = lambda lo, hi: 100.0 * float(((_age >= lo) & (_age < hi)).sum()) / _n
            _oldest = int(_gid.min().item()); _newest_win = int(_gid.max().item())
            _n_old_pool = int(sum(1 for g in ids_all if (int(ids_all[-1]) - g) > 120))
            print(f"[warp-age] winners={int(_n)} bank_gid=[{ids_all[0]}..{ids_all[-1]}] pool={len(ids_all)} old_in_pool={_n_old_pool} | "
                  f"age<2s={_b(0,2):.1f}% 2-5s={_b(2,5):.1f}% 5-10s={_b(5,10):.1f}% "
                  f"10-20s={_b(10,20):.1f}% 20s+={_b(20,1e9):.1f}% | "
                  f"winner_gid=[{_oldest}..{_newest_win}] frac_from_ge5s={_b(5,1e9):.1f}%", flush=True)
    return torch.stack(warps, 1)[None].contiguous(), vis_out


@torch.no_grad()
def build_multisrc_warp(estimator, raw_video_b, lingbot_c2ws_b, K_pix, target_c2ws,
                        *, pix_start, pix_stride, window_pix, height, width,
                        ingest_n=12, lag=1, history=16, nsrc=8, nearby=16,
                        splat_radius=1, dens_thresh=0.45, dens_win=7,
                        recall_min_cov=0.5, recall_margin=0.15, device="cuda"):


    _reset_depth_stream(estimator)
    import torch.nn.functional as _F
    H, W = int(height), int(width); HW = H * W
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    T = int(raw_video_b.shape[1]); F_t = int(target_c2ws.shape[0])
    kc = int(pix_start) // int(pix_stride); newest = kc - 1 - int(lag); oldest = max(0, newest - int(history) + 1)
    K_np = np.asarray(K_pix, np.float32)


    store = {}; ids_all = []
    for j in range(oldest, newest + 1):
        s = j * int(pix_stride)
        if s < 0 or s >= T:
            continue
        e = min(s + int(window_pix), T)
        if e - s < 3:
            continue
        idxs = torch.unique(torch.linspace(s, e - 1, int(ingest_n)).round().long().clamp(0, T - 1))
        if idxs.numel() < 3:
            continue
        frames = (raw_video_b[:, idxs].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
        c2w = lingbot_c2ws_b[idxs].float().cpu().numpy()
        try:
            dep, intr, _conf, _rgb = estimator.depth_window(frames, c2w, np.stack([K_np] * int(idxs.numel())))
        except Exception as _e:
            print(f"[da3-multisrc] WARN chunk@{s} ingest failed ({type(_e).__name__}); skipping", flush=True); continue
        for i, g in enumerate(idxs.tolist()):
            d = torch.as_tensor(dep[i], device=device); it = torch.as_tensor(intr[i], device=device)
            cwi = torch.as_tensor(c2w[i], device=device)
            if _rgb is not None:
                r = torch.as_tensor(np.asarray(_rgb[i], np.float32), device=device).permute(2, 0, 1).contiguous()
            else:
                r = _F.interpolate(((raw_video_b[:, int(g)] * 0.5 + 0.5).clamp(0, 1))[None].to(device),
                                   tuple(d.shape), mode="bilinear", align_corners=False)[0]
            store[g] = (d, it, cwi, r); ids_all.append(g)

    return _render_multisrc(store, ids_all, target_c2ws, K_pix, H, W,
                            nsrc=nsrc, nearby=nearby, splat_radius=splat_radius,
                            dens_thresh=dens_thresh, dens_win=dens_win,
                            recall_min_cov=recall_min_cov, recall_margin=recall_margin, device=device)


CHUNK0_TARGET_DISPARITY_PX_DEFAULT = 135.0


@torch.no_grad()
def solve_chunk0_disparity_scale(depth, intr_depth, ref_c2w, target_c2ws, K_out, target_px,
                                 *, quantile=0.9, k_limits=(1.0 / 64.0, 64.0)):


    d = depth.float()
    h, w = int(d.shape[0]), int(d.shape[1])
    tgt = float(target_px)
    if not (tgt > 0):
        return 1.0, {"reason": "target_px<=0"}

    fx_d, fy_d = float(intr_depth[0, 0]), float(intr_depth[1, 1])
    cx_d, cy_d = float(intr_depth[0, 2]), float(intr_depth[1, 2])
    ys, xs = torch.meshgrid(torch.arange(h, device=d.device, dtype=torch.float32),
                            torch.arange(w, device=d.device, dtype=torch.float32), indexing="ij")
    m_x = (xs - cx_d) / fx_d
    m_y = (ys - cy_d) / fy_d
    fx_o, fy_o = float(K_out[0, 0]), float(K_out[1, 1])

    R0 = ref_c2w[:3, :3].float()
    t0 = ref_c2w[:3, 3].float()
    valid = torch.isfinite(d) & (d > 0)
    if not bool(valid.any()):
        return 1.0, {"reason": "no valid depth"}
    d_safe = torch.where(valid, d, torch.ones_like(d))

    best = (0.0, -1, 0.0)
    for i in range(int(target_c2ws.shape[0])):
        delta = R0.transpose(0, 1) @ (t0 - target_c2ws[i, :3, 3].float())
        dx_, dy_, dz = float(delta[0]), float(delta[1]), float(delta[2])
        C = torch.sqrt((fx_o * (dx_ - m_x * dz)) ** 2 + (fy_o * (dy_ - m_y * dz)) ** 2)
        k_p = ((C / tgt) - dz) / d_safe


        k_p = torch.where(valid, k_p, torch.zeros_like(k_p)).clamp_min(0.0)
        k_i = float(torch.quantile(k_p.reshape(-1), float(quantile)))
        if k_i > best[0]:
            best = (k_i, i, float(np.linalg.norm([dx_, dy_, dz])))

    k, frame, tnorm = best
    info = {"frame": frame, "trans_norm": tnorm, "k_raw": k, "clamped": False}
    if not np.isfinite(k) or k <= 0:
        return 1.0, {**info, "reason": "degenerate solve (no commanded translation?)"}
    lo, hi = float(k_limits[0]), float(k_limits[1])
    if k < lo or k > hi:
        info["clamped"] = True
        k = min(max(k, lo), hi)


    i = frame
    delta = R0.transpose(0, 1) @ (t0 - target_c2ws[i, :3, 3].float())
    dx_, dy_, dz = float(delta[0]), float(delta[1]), float(delta[2])
    C = torch.sqrt((fx_o * (dx_ - m_x * dz)) ** 2 + (fy_o * (dy_ - m_y * dz)) ** 2)
    z_after = d_safe + dz
    front = valid & (z_after > 1e-6)
    n_valid = int(valid.sum())
    behind_frac = 1.0 - (float(front.sum()) / max(n_valid, 1))
    info["behind_frac_before"] = behind_frac
    if behind_frac > (1.0 - float(quantile)) or not bool(front.any()):
        info["disp_p90_before"] = float("inf")
    else:
        disp0 = torch.where(front, C / z_after.clamp_min(1e-6), torch.zeros_like(C))


        q_adj = (float(quantile) - behind_frac) / max(1.0 - behind_frac, 1e-6)
        vals = disp0[front].reshape(-1)
        info["disp_p90_before"] = float(torch.quantile(vals, min(max(q_adj, 0.0), 1.0)))
    return float(k), info


@torch.no_grad()
def build_single_source_warp_mono(estimator, ref_frame_pix, ref_pose_c2w, K_pix, target_c2ws,
                                  *, height, width, splat_radius=2, device="cuda",
                                  chunk0_target_disparity_px=CHUNK0_TARGET_DISPARITY_PX_DEFAULT):


    H, W = int(height), int(width)
    target_c2ws = target_c2ws.to(device=device, dtype=torch.float32)
    F_t = int(target_c2ws.shape[0])

    def _blank():
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))

    depth_single = getattr(estimator, "depth_single", None)
    if not callable(depth_single):
        return _blank()
    ref = ref_frame_pix
    if ref.ndim == 4:
        ref = ref[0]
    frame = (ref.detach().permute(1, 2, 0).float() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    K_np = np.asarray(K_pix, np.float32)
    try:
        dep, intr, rgb = depth_single(frame, K_np)
    except NotImplementedError as _e:


        print(f"[da3-single-src-mono] WARN chunk-0 reference warp DISABLED by the depth recipe: {_e}. "
              f"Chunk 0 gets a BLANK warp, so it has no camera signal and will sit static -- the same as "
              f"chunk0_ref_warp=off. Use scale_mode=depth_median to enable it.", flush=True)
        return _blank()
    except Exception as _e:
        print(f"[da3-single-src-mono] WARN reference-frame depth failed "
              f"({type(_e).__name__}: {_e}); blank chunk-0 warp", flush=True)
        return _blank()
    d0 = torch.as_tensor(dep, device=device)
    it0 = torch.as_tensor(intr, device=device)


    if isinstance(ref_pose_c2w, torch.Tensor):
        cw0 = ref_pose_c2w.detach().to(device=device, dtype=torch.float32)
    else:
        cw0 = torch.as_tensor(np.asarray(ref_pose_c2w, np.float32), device=device)
    if float(chunk0_target_disparity_px) > 0:


        _k, _info = solve_chunk0_disparity_scale(
            d0, it0, cw0, target_c2ws, torch.as_tensor(K_np, device=device),
            float(chunk0_target_disparity_px))
        if _info.get("reason"):
            print(f"[da3-single-src-mono] chunk0 disparity rescale SKIPPED ({_info['reason']}); "
                  f"depth left at depth_median scale", flush=True)
        else:
            d0 = d0 * float(_k)
            print(f"[da3-single-src-mono] chunk0 disparity rescale: k={_k:.4g}"
                  f"{' (CLAMPED)' if _info.get('clamped') else ''} -> p90 parallax "
                  f"{_info['disp_p90_before']:.1f} px => {float(chunk0_target_disparity_px):.1f} px "
                  f"(driving frame {_info['frame']}, |t|={_info['trans_norm']:.3f} pose units, "
                  f"{_info['behind_frac_before'] * 100:.0f}% of the cloud was behind the camera before)",
                  flush=True)
    xyz = unproject_depth_torch(d0, it0, cw0).reshape(-1, 3)
    rgb0 = torch.as_tensor(np.asarray(rgb, np.float32), device=device).reshape(-1, 3)
    K_render = torch.as_tensor(K_np, device=device)[None].expand(F_t, 3, 3)
    return render_cloud_batched(xyz, rgb0, target_c2ws, K_render, H, W,
                                device=device, splat_radius=int(splat_radius), invisible_fill="black")


@torch.no_grad()
def build_single_source_warp(estimator, raw_video_b, lingbot_c2ws_b, K_pix, target_c2ws,
                             *, pix_start, pix_stride, window_pix, height, width,
                             ingest_n=6, splat_radius=2, device="cuda"):


    _reset_depth_stream(estimator)
    import torch.nn.functional as _F
    H, W = int(height), int(width)
    T = int(raw_video_b.shape[1]); F_t = int(target_c2ws.shape[0])
    K_np = np.asarray(K_pix, np.float32)
    s = int(pix_start); e = min(s + int(window_pix), T)

    def _blank():
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))

    if e - s < 3:
        return _blank()

    idxs = torch.unique(torch.linspace(s, e - 1, int(ingest_n)).round().long().clamp(0, T - 1))
    if idxs.numel() < 3:
        return _blank()
    frames = (raw_video_b[:, idxs].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
    c2w = lingbot_c2ws_b[idxs].float().cpu().numpy()
    try:
        dep, intr, _conf, _rgb = estimator.depth_window(frames, c2w, np.stack([K_np] * int(idxs.numel())))
    except Exception as _e:
        print(f"[da3-single-src] WARN source@{s} gt-metric depth failed ({type(_e).__name__}: {_e}); blank warp", flush=True)
        return _blank()

    d0 = torch.as_tensor(dep[0], device=device)
    it0 = torch.as_tensor(intr[0], device=device)
    cw0 = torch.as_tensor(c2w[0], device=device)
    xyz = unproject_depth_torch(d0, it0, cw0).reshape(-1, 3)
    if _rgb is not None:
        rgb = torch.as_tensor(np.asarray(_rgb[0], np.float32), device=device).reshape(-1, 3)
    else:
        r = _F.interpolate(((raw_video_b[:, s] * 0.5 + 0.5).clamp(0, 1))[None].to(device),
                           tuple(d0.shape), mode="bilinear", align_corners=False)[0]
        rgb = r.permute(1, 2, 0).reshape(-1, 3)

    Kt = torch.as_tensor(K_np, device=device)
    K_render = Kt[None].expand(F_t, 3, 3)
    return render_cloud_batched(xyz, rgb, target_c2ws, K_render, H, W,
                                device=device, splat_radius=int(splat_radius), invisible_fill="black")


@torch.no_grad()
def build_backward_warp(estimator, raw_video_b, lingbot_c2ws_b, K_pix, target_c2ws,
                        *, pix_start, pix_stride, window_pix, height, width,
                        ingest_n=12, lag=1, history=16, nearby=16, fill_iters=12,
                        recall_min_cov=0.5, recall_margin=0.15, render_mode="backward",
                        zbuf_despeckle=False, zbuf_despeckle_ksize=3, zbuf_despeckle_fill_iters=4,
                        device="cuda"):


    _reset_depth_stream(estimator)
    import torch.nn.functional as _F
    H, W = int(height), int(width); HW = H * W
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    T = int(raw_video_b.shape[1]); F_t = int(target_c2ws.shape[0])
    kc = int(pix_start) // int(pix_stride); newest = kc - 1 - int(lag); oldest = max(0, newest - int(history) + 1)
    K_np = np.asarray(K_pix, np.float32)


    store = {}; ids_all = []
    for j in range(oldest, newest + 1):
        s = j * int(pix_stride)
        if s < 0 or s >= T:
            continue
        e = min(s + int(window_pix), T)
        if e - s < 3:
            continue
        idxs = torch.unique(torch.linspace(s, e - 1, int(ingest_n)).round().long().clamp(0, T - 1))
        if idxs.numel() < 3:
            continue
        frames = (raw_video_b[:, idxs].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
        c2w = lingbot_c2ws_b[idxs].float().cpu().numpy()
        try:
            dep, intr, _conf, _rgb = estimator.depth_window(frames, c2w, np.stack([K_np] * int(idxs.numel())))
        except Exception as _e:
            print(f"[da3-backward] WARN chunk@{s} ingest failed ({type(_e).__name__}); skipping", flush=True); continue
        for i, g in enumerate(idxs.tolist()):
            d = torch.as_tensor(dep[i], device=device); it = torch.as_tensor(intr[i], device=device)
            cwi = torch.as_tensor(c2w[i], device=device)
            if _rgb is not None:
                r = torch.as_tensor(np.asarray(_rgb[i], np.float32), device=device).permute(2, 0, 1).contiguous()
            else:
                r = _F.interpolate(((raw_video_b[:, int(g)] * 0.5 + 0.5).clamp(0, 1))[None].to(device),
                                   tuple(d.shape), mode="bilinear", align_corners=False)[0]
            store[g] = (d, it, cwi, r); ids_all.append(g)


    _render = _render_backward_multisrc_zbuf if str(render_mode) == "backward_zbuf" else _render_backward
    _extra = dict(zbuf_despeckle=zbuf_despeckle, zbuf_despeckle_ksize=zbuf_despeckle_ksize,
                  zbuf_despeckle_fill_iters=zbuf_despeckle_fill_iters) if str(render_mode) == "backward_zbuf" else {}
    return _render(store, ids_all, target_c2ws, K_pix, height, width,
                   nearby=nearby, fill_iters=fill_iters,
                   recall_min_cov=recall_min_cov, recall_margin=recall_margin, device=device, **_extra)

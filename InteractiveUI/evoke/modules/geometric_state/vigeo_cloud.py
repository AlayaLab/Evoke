

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .da3_cloud import _umeyama_scale_collinear_safe

_REPO_ROOT = Path(__file__).resolve().parents[3]


_VIGEO_SRC = Path(os.environ.get("EVOKE_VIGEO_SRC",
                                 str(_REPO_ROOT / "evoke" / "third_party" / "vigeo")))
_VIGEO_WEIGHTS = Path(os.environ.get("EVOKE_VIGEO_WEIGHTS", str(_REPO_ROOT / "models" / "ViGeo1.1")))

_PATCH = 14


_EXTRA_TOKENS = 6


_BUDGET_BLOCK_SKEW = 0.35


_MIN_KEEP_FRAMES = 4


_FATAL_EXC = (MemoryError, KeyError, AttributeError, ImportError, TypeError, ValueError)


def check_assets(src=None, weights=None) -> None:


    src = Path(src) if src else _VIGEO_SRC
    weights = Path(weights) if weights else _VIGEO_WEIGHTS
    if not (src / "vigeo").is_dir():
        raise FileNotFoundError(
            f"ViGeo source not found: {src / 'vigeo'} does not exist. The vendored in-repo copy should be "
            f"at evoke/third_party/vigeo (see its PROVENANCE.md), or set EVOKE_VIGEO_SRC to point at an "
            f"external checkout.")
    if not (weights / "vigeo.pt").is_file():
        raise FileNotFoundError(
            f"ViGeo weights not found: {weights / 'vigeo.pt'} does not exist. Fetch the "
            f"pkqbajng/ViGeo1.1 snapshot or point EVOKE_VIGEO_WEIGHTS at it.")


def _fit_patch_res(h: int, w: int, long_side: int, patch: int = _PATCH):


    s = float(long_side) / float(max(h, w))
    nh = max(patch, int(round(h * s / patch)) * patch)
    nw = max(patch, int(round(w * s / patch)) * patch)
    return nh, nw


class ViGeoDepthEstimator:


    def __init__(self, device="cuda", process_res: int = 644,
                 src: Path = _VIGEO_SRC, weights: Path = _VIGEO_WEIGHTS,
                 num_tokens: Optional[int] = None, mode: str = "chunk",
                 chunk_size: int = 16, intr_source: str = "gt",
                 conf_transform: str = "exp", scale_mode: str = "anchor",
                 anchor_windows: int = 4, total_budget: int = 0,
                 cache_keep_frames: int = 6, scale_value: float = 0.0,
                 depth_median_target: float = 1.0):


        if str(mode) not in ("offline", "chunk", "online"):
            raise ValueError(f"ViGeo mode must be offline/chunk/online, got {mode!r}")
        if str(scale_mode) not in ("per_window", "anchor", "depth_median", "fixed"):
            raise ValueError("ViGeo scale_mode must be per_window/anchor/depth_median/fixed, "
                             f"got {scale_mode!r}")


        if str(scale_mode) == "fixed" and not (float(scale_value) > 0):
            raise ValueError(f"ViGeo scale_mode=fixed needs scale_value > 0, got {scale_value!r}")
        if not (float(depth_median_target) > 0):
            raise ValueError(f"ViGeo depth_median_target must be > 0, got {depth_median_target!r}")
        if str(intr_source) not in ("gt", "vigeo"):
            raise ValueError(f"ViGeo intr_source must be gt/vigeo, got {intr_source!r}")
        if str(conf_transform) not in ("exp", "none"):
            raise ValueError(f"ViGeo conf_transform must be exp/none, got {conf_transform!r}")
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.src = Path(src)
        self.weights = Path(weights)


        self.num_tokens = int(num_tokens) if num_tokens else None
        self.mode = str(mode)
        self.chunk_size = int(chunk_size)
        self.intr_source = str(intr_source)
        self.conf_transform = str(conf_transform)
        self.scale_mode = str(scale_mode)
        self.scale_value = float(scale_value)
        self.depth_median_target = float(depth_median_target)
        self._skip_reason = None
        self.anchor_windows = int(anchor_windows)
        self.total_budget = int(total_budget)
        self.cache_keep_frames = int(cache_keep_frames)
        self._model = None
        self._dbg = bool(os.environ.get("EVOKE_VIGEO_DEBUG"))
        self._focal_ratios: list[float] = []
        self._kv = None
        self._win_seen = 0
        self._anchor_scales: list[float] = []
        self._scale_locked: Optional[float] = None
        self._parallax_warned = False

    @property
    def streaming(self) -> bool:
        return self.mode in ("chunk", "online")

    def _resolve_budget(self, h: int, w: int) -> int:


        if self.total_budget > 0:
            return self.total_budget
        tpf = (h // _PATCH) * (w // _PATCH) + _EXTRA_TOKENS
        nblk = int(getattr(self._model.pretrained, "num_global_blocks", 0))
        if nblk <= 0:

            raise RuntimeError(
                f"ViGeo reports num_global_blocks={nblk}; cannot size the streaming kv-cache budget. "
                f"Use mode='offline' or set total_budget explicitly.")
        keep = max(_MIN_KEEP_FRAMES, self.cache_keep_frames)
        budget = nblk * keep * tpf
        if self._dbg:
            worst = int(budget * _BUDGET_BLOCK_SKEW / nblk)
            print(f"[vigeo] cache budget: {nblk} global blocks x {keep} frames x {tpf} tokens/frame "
                  f"= {budget}; weakest block ~{worst} tokens = {worst / tpf:.2f} frames "
                  f"(must exceed 1.0 or eviction silently stops)", flush=True)
        return budget

    def _invalidate_geometry(self, why: str):


        had = self._kv is not None or self._scale_locked is not None or bool(self._anchor_scales)
        self._kv = None
        self._scale_locked = None
        self._anchor_scales = []
        if had:
            if self.scale_mode == "fixed":
                how = "the fixed scale is unchanged (it does not depend on the cache)"
            elif self.scale_mode == "depth_median":
                how = "the depth-median scale will be re-derived on the next window"
            else:
                how = f"the depth scale will re-anchor over the next {max(1, self.anchor_windows)} window(s)"
            print(f"[vigeo] stream geometry invalidated ({why}); {how}", flush=True)

    def reset_stream(self):

        if self._dbg and self._focal_ratios:
            r = np.asarray(self._focal_ratios, dtype=np.float64)
            print(f"[vigeo] stream focal ratio (GT/ViGeo) over {r.size} windows: "
                  f"median={np.median(r):.3f} min={r.min():.3f} max={r.max():.3f}", flush=True)
        self._kv = None
        self._win_seen = 0
        self._anchor_scales = []
        self._scale_locked = None
        self._focal_ratios = []
        self._parallax_warned = False
        if self._model is not None:
            self._model.reset_cache_state()

    def _lazy(self):
        if self._model is not None:
            return
        check_assets(self.src, self.weights)
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
        from vigeo import ViGeo
        self._model = ViGeo.from_pretrained(str(self.weights)).to(self.device).eval()
        print(f"[vigeo] loaded {self.weights} (mask_head={self._model.mask_head is not None}) "
              f"process_res={self.process_res} num_tokens={self.num_tokens} mode={self.mode} "
              f"scale_mode={self.scale_mode} intr_source={self.intr_source}", flush=True)

    def _infer_window(self, frames_rgb: np.ndarray):


        x = torch.as_tensor(np.ascontiguousarray(frames_rgb), dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).clamp(0, 1)
        N, _, H, W = x.shape
        if self.num_tokens is None:
            h, w = _fit_patch_res(H, W, self.process_res)
            if (h, w) != (H, W):
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.to(self.device)

        if self.streaming:


            out = self._model.infer(
                x, mode=self.mode, chunk_size=self.chunk_size,
                num_tokens=self.num_tokens, resize_output=True,
                kv_caches=self._kv, reset_cache=(self._kv is None),
                total_budget=self._resolve_budget(int(x.shape[-2]), int(x.shape[-1])))


            self._kv = out["kv_caches"]
        else:

            out = self._model.infer(
                x, mode=self.mode, num_tokens=self.num_tokens, resize_output=True)

        depth = out["depth_pred"][:, 0].float()
        conf = out["conf_pred"]
        conf = None if conf is None else conf[:, 0].float()
        pose = out["pose_pred"].float().cpu().numpy()
        c2w_pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
        c2w_pred[:, :3, :4] = pose
        h, w = int(depth.shape[-2]), int(depth.shape[-1])
        rgb = x.permute(0, 2, 3, 1).float().cpu().numpy()
        f_pix = self._vigeo_focal_pix(out["points_pred"], depth)
        return depth, conf, c2w_pred, rgb, (h, w), f_pix

    @staticmethod
    def _scale_K(K_gt: np.ndarray, src_hw, dst_hw) -> np.ndarray:

        (H, W), (h, w) = src_hw, dst_hw
        sx, sy = float(w) / float(W), float(h) / float(H)
        K = np.asarray(K_gt, np.float32).copy()
        if K.ndim == 2:
            K = K[None]
        K[:, 0, 0] *= sx; K[:, 0, 2] *= sx
        K[:, 1, 1] *= sy; K[:, 1, 2] *= sy
        return K.astype(np.float32)

    @staticmethod
    def _vigeo_focal_pix(points: torch.Tensor, depth: torch.Tensor) -> np.ndarray:


        N, h, w = int(depth.shape[0]), int(depth.shape[-2]), int(depth.shape[-1])
        aspect = w / h
        span_x = aspect / (1 + aspect ** 2) ** 0.5
        span_y = 1.0 / (1 + aspect ** 2) ** 0.5
        u = torch.linspace(-span_x * (w - 1) / w, span_x * (w - 1) / w, w, dtype=torch.float32)
        v = torch.linspace(-span_y * (h - 1) / h, span_y * (h - 1) / h, h, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        xy = points[..., :2].float()
        z = depth.float().unsqueeze(-1)
        uz = torch.stack([uu, vv], -1).unsqueeze(0) * z
        num = (uz * xy).flatten(1).sum(1)
        den = (xy * xy).flatten(1).sum(1).clamp_min(1e-8)
        focal_norm = (num / den)
        diag = float((h ** 2 + w ** 2) ** 0.5)
        return (focal_norm * (diag / 2.0)).cpu().numpy().astype(np.float32)

    def _resolve_scale(self, c_raw: Optional[float]) -> Optional[float]:


        if c_raw is None:
            return None
        if self.scale_mode != "anchor":
            return c_raw
        if self._scale_locked is not None:
            return self._scale_locked
        self._anchor_scales.append(float(c_raw))
        if len(self._anchor_scales) >= max(1, self.anchor_windows):
            self._scale_locked = float(np.median(self._anchor_scales))
            print(f"[vigeo] scale anchored: scale={self._scale_locked:.4g} (median of the first "
                  f"{len(self._anchor_scales)} windows {np.round(self._anchor_scales, 4).tolist()}); "
                  f"reused for all later windows", flush=True)
            return self._scale_locked
        return c_raw

    def _warn_if_no_parallax(self, gt_centres, depth_scaled) -> None:


        if self._parallax_warned:
            return
        gt = np.asarray(gt_centres, np.float64)
        if gt.ndim != 2 or gt.shape[0] < 2:
            return
        baseline = float(np.linalg.norm(gt.max(0) - gt.min(0)))
        med = float(np.median(depth_scaled))
        if not np.isfinite(med) or med <= 0 or baseline <= 0:
            return
        ratio = baseline / med
        if ratio >= 1e-3:
            return
        self._parallax_warned = True
        print(f"[vigeo] WARN zero-parallax geometry: the window's commanded camera travel is {baseline:.4g} "
              f"but the scaled cloud's median depth is {med:.4g} (ratio {ratio:.2e} < 1e-3), so the warp "
              f"will render as a STILL IMAGE (coverage ~1.000) and the model will have no camera signal. "
              f"scale_mode={self.scale_mode}"
              + (" -- anchor/per_window derive the scale as a ratio against the camera motion ViGeo reads "
                 "out of the frames, which collapses when those frames are near-static (i2v chunk 0 has no "
                 "real frames at all); use scale_mode=depth_median, which cannot collapse."
                 if self.scale_mode in ("anchor", "per_window")
                 else " -- check depth_median_target / scale_value against the pose track's units."),
              flush=True)

    def _resolve_scale_baseline_free(self, depth_t) -> float:


        if self._scale_locked is not None:
            return self._scale_locked
        if self.scale_mode == "fixed":
            self._scale_locked = float(self.scale_value)
            print(f"[vigeo] scale fixed: scale={self._scale_locked:.4g} (baseline-free; "
                  f"no Umeyama solve)", flush=True)
            return self._scale_locked
        med = float(torch.median(depth_t.detach().float()).item())
        if not np.isfinite(med) or med <= 0:
            raise ValueError(f"ViGeo depth median is {med!r}; cannot set a depth_median scale")
        self._scale_locked = med / float(self.depth_median_target)
        print(f"[vigeo] scale from depth median: scale={self._scale_locked:.4g} "
              f"(raw median {med:.4g} -> target {self.depth_median_target:g}; baseline-free, "
              f"locked for the rollout)", flush=True)
        return self._scale_locked

    @torch.no_grad()
    def depth_single(self, frame_rgb, K_gt):


        if self.scale_mode not in ("depth_median", "fixed"):
            raise NotImplementedError(
                f"depth_single needs scale_mode depth_median/fixed (baseline-free), got {self.scale_mode!r}")
        fr = np.asarray(frame_rgb, np.float32)
        if fr.ndim != 3 or fr.shape[2] != 3:
            raise ValueError(f"depth_single expects [H,W,3], got {fr.shape}")
        H, W = int(fr.shape[0]), int(fr.shape[1])
        self._lazy()
        self.reset_stream()
        try:
            depth_t, _conf_t, _c2w_pred, rgb, (h, w), f_pix = self._infer_window(fr[None])
            c = self._resolve_scale_baseline_free(depth_t)
            depth = (depth_t / float(c))[0].cpu().numpy().astype(np.float32)
            intr = self._scale_K(K_gt, (H, W), (h, w))[0]
            if self.intr_source == "vigeo":
                intr = np.eye(3, dtype=np.float32)
                intr[0, 0] = float(f_pix[0]); intr[1, 1] = float(f_pix[0])
                intr[0, 2] = (w - 1) / 2.0;   intr[1, 2] = (h - 1) / 2.0
            return depth, intr.astype(np.float32), np.asarray(rgb[0], np.float32)
        finally:
            self.reset_stream()

    @torch.no_grad()
    def depth_window(self, frames_rgb, c2w_gt, K_gt):


        frames_rgb = np.asarray(frames_rgb)
        K = int(frames_rgb.shape[0])
        if K < 3:
            raise ValueError(f"ViGeo GT-pose needs >=3 frames per call (Umeyama scale), got {K}")
        self._skip_reason = None
        deps, intrs, confs, rgbs = self.depth_windows_batched(
            [frames_rgb], [np.asarray(c2w_gt)], [np.asarray(K_gt)])
        if deps[0] is None:


            raise RuntimeError(f"ViGeo depth_window produced no usable window "
                               f"({self._skip_reason or 'reason not recorded'})")
        return (deps[0], intrs[0], confs[0], rgbs[0])

    @torch.no_grad()
    def depth_windows_batched(self, windows_rgb, windows_c2w, windows_K):


        B = len(windows_rgb)
        if B == 0:
            return [], [], [], []


        self._lazy()
        depths, intrs, confs, rgbs = [], [], [], []
        for b in range(B):
            fr = np.asarray(windows_rgb[b])
            N, H, W = int(fr.shape[0]), int(fr.shape[1]), int(fr.shape[2])
            if N < 3:
                raise ValueError(f"ViGeo GT-pose needs >=3 frames per window, window {b} got {N}")
            try:
                depth_t, conf_t, c2w_pred, rgb, (h, w), f_pix = self._infer_window(fr)


                gt_c = np.asarray(windows_c2w[b], np.float64)[:, :3, 3]
                pr_c = c2w_pred[:, :3, 3].astype(np.float64)


                c_raw = _umeyama_scale_collinear_safe(gt_c, pr_c)
                if self.scale_mode in ("depth_median", "fixed"):
                    c = self._resolve_scale_baseline_free(depth_t)
                else:
                    c = self._resolve_scale(c_raw)
                if c is None:


                    self._skip_reason = "zero baseline (static or degenerate camera motion)"
                    print(f"[vigeo] window {self._win_seen + 1} skipped: {self._skip_reason}; "
                          f"no points ingested", flush=True)
                    depths.append(None); intrs.append(None); confs.append(None); rgbs.append(None)
                    continue
                depth = (depth_t / float(c)).cpu().numpy().astype(np.float32)
                self._warn_if_no_parallax(gt_c, depth)
                intr = self._scale_K(windows_K[b], (H, W), (h, w))
                if intr.shape[0] == 1 and N > 1:
                    intr = np.repeat(intr, N, axis=0)
                f_gt = float(np.mean(intr[:, 0, 0]))
                f_vg = float(np.mean(f_pix))
                self._focal_ratios.append(f_gt / max(f_vg, 1e-6))
                if self.intr_source == "vigeo":
                    intr = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
                    intr[:, 0, 0] = f_pix; intr[:, 1, 1] = f_pix
                    intr[:, 0, 2] = (w - 1) / 2.0; intr[:, 1, 2] = (h - 1) / 2.0
                if conf_t is None:
                    conf = None
                elif self.conf_transform == "exp":

                    conf = conf_t.clamp(-30, 30).exp().cpu().numpy().astype(np.float32)
                else:
                    conf = conf_t.cpu().numpy().astype(np.float32)
                self._win_seen += 1
                if self._dbg:
                    print(f"[vigeo] win#{self._win_seen} N={N} {H}x{W}->{h}x{w} "
                          f"scale_raw={('%.4g' % c_raw) if c_raw is not None else 'None'} scale_used={c:.4g}"
                          f"{'(locked)' if self._scale_locked is not None else ''} "
                          f"stream={'on' if self.streaming else 'off'} "
                          f"depth_med={float(np.median(depth)):.3f} f_gt={f_gt:.1f} "
                          f"f_vigeo={f_vg:.1f} ratio={f_gt / max(f_vg, 1e-6):.3f}", flush=True)
                depths.append(depth); intrs.append(intr); confs.append(conf); rgbs.append(rgb)
            except _FATAL_EXC:
                raise
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as _e:


                self._skip_reason = f"{type(_e).__name__}: {_e}"
                self._invalidate_geometry(f"window {self._win_seen + 1} failed")
                print(f"[vigeo] WARN stream window {self._win_seen + 1} (batch item {b}) failed "
                      f"({self._skip_reason}); skipping", flush=True)
                depths.append(None); intrs.append(None); confs.append(None); rgbs.append(None)
        return depths, intrs, confs, rgbs

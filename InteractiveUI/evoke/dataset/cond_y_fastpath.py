

from __future__ import annotations

import os

import torch


M_INDEP = 29

HEAD_PX = 1 + 4 * (M_INDEP - 1)

_K_CACHE: dict = {}


_VERIFY_DONE: dict = {}
_VERIFY_N = int(os.environ.get("SF_CONDY_VERIFY_N", "3") or 3)


_FAST = {"off": False, "why": ""}


def _disable(why: str):
    if not _FAST["off"]:
        _FAST["off"] = True
        _FAST["why"] = why
        print(f"[COND-Y-FAST] x disabling the fast path for this run, falling back to the full encode (training continues, ~13s/step slower): {why}", flush=True)


def _full_encode(vae, raw_video: torch.Tensor, T_px: int) -> torch.Tensor:


    B, C_pix, _, H, W = raw_video.shape
    full_px = torch.zeros(1, C_pix, int(T_px), H, W, device=raw_video.device, dtype=raw_video.dtype)
    full_px[:, :, :1] = raw_video[:1, :, :1]
    with torch.no_grad():
        return vae.encode(full_px.to(vae.dtype)).latent_dist.mode()


def _tiling_key(vae):

    return (
        bool(getattr(vae, "use_tiling", False)),
        bool(getattr(vae, "use_slicing", False)),
        getattr(vae, "tile_sample_min_height", None), getattr(vae, "tile_sample_min_width", None),
        getattr(vae, "tile_sample_stride_height", None), getattr(vae, "tile_sample_stride_width", None),
    )


def _zero_tail_const(vae, H: int, W: int, C_pix: int, dtype, device) -> torch.Tensor:


    key = (H, W, C_pix, str(dtype), str(device)) + _tiling_key(vae)
    if key in _K_CACHE:
        return _K_CACHE[key]

    n_px = 1 + 4 * M_INDEP
    with torch.no_grad():
        z = torch.zeros(1, C_pix, n_px, H, W, device=device, dtype=dtype)
        lat = vae.encode(z).latent_dist.mode()
    assert lat.shape[2] == M_INDEP + 1, (
        f"[COND-Y-FAST] expected {n_px} frames to give {M_INDEP + 1} latents, got {lat.shape[2]} "
        f"-- the VAE temporal compression changed, so the fast path premise no longer holds")
    K = lat[:, :, M_INDEP:M_INDEP + 1].contiguous().clone()
    _K_CACHE[key] = K
    print(f"[COND-Y-FAST] constant tail precomputed and cached: M={M_INDEP} HEAD_PX={HEAD_PX} "
          f"K.shape={tuple(K.shape)} key=({H}x{W}, {dtype}, tiling={_tiling_key(vae)[0]}) "
          f"-- each step now encodes {HEAD_PX} frames instead of the full length", flush=True)
    return K


def cond_y_latent(vae, raw_video: torch.Tensor, T_px: int, verify: bool | None = None) -> torch.Tensor:


    assert raw_video.dim() == 5, f"[COND-Y-FAST] expected [B,C,T,H,W], got {tuple(raw_video.shape)}"
    B, C_pix, _, H, W = raw_video.shape
    T_lat = 1 + (T_px - 1) // 4
    if T_lat <= M_INDEP:
        return _full_encode(vae, raw_video, T_px)
    if _FAST["off"]:
        return _full_encode(vae, raw_video, T_px)
    if verify is None:

        _n = _VERIFY_DONE.get(T_px, 0)
        verify = os.environ.get("SF_CONDY_VERIFY") == "1" and _n < _VERIFY_N
        if verify:
            _VERIFY_DONE[T_px] = _n + 1

    try:
        head_px = torch.zeros(1, C_pix, HEAD_PX, H, W, device=raw_video.device, dtype=raw_video.dtype)
        head_px[:, :, :1] = raw_video[:1, :, :1]
        with torch.no_grad():
            head = vae.encode(head_px.to(vae.dtype)).latent_dist.mode()
        if head.shape[2] != M_INDEP:
            _disable(f"encoding {HEAD_PX} frames should give {M_INDEP} latents, got {head.shape[2]} -- VAE temporal compression changed")
            return _full_encode(vae, raw_video, T_px)
        K = _zero_tail_const(vae, H, W, C_pix, raw_video.dtype, raw_video.device)
        tail = K.expand(head.shape[0], -1, T_lat - M_INDEP, -1, -1)
        out = torch.cat([head, tail], dim=2)
    except Exception as _e:
        _disable(f"the fast path itself raised: {type(_e).__name__}: {_e}")
        return _full_encode(vae, raw_video, T_px)

    if verify:


        try:
            del head_px
            ref = _full_encode(vae, raw_video, T_px)
            if ref.shape != out.shape:
                _disable(f"shape mismatch: ref={tuple(ref.shape)} fast={tuple(out.shape)}")
                return ref
            d_head = (ref[:, :, :M_INDEP] - out[:, :, :M_INDEP]).abs().max().item()
            d_tail = (ref[:, :, M_INDEP:] - out[:, :, M_INDEP:]).abs().max().item()
            if d_head != 0.0 or d_tail != 0.0:
                _disable(
                    f"differs from the full encode: first {M_INDEP} frames max|delta|={d_head:.3e}, "
                    f"last {T_lat - M_INDEP} frames max|delta|={d_tail:.3e} -> "
                    f"{'receptive field R>112, or the chunking changed' if d_head else ''}"
                    f"{' and ' if (d_head and d_tail) else ''}"
                    f"{'the M=' + str(M_INDEP) + ' independence boundary does not hold (tiling params changed?)' if d_tail else ''}")
                return ref
            print(f"[COND-Y-VERIFY] ok bit-identical T_px={T_px} T_lat={T_lat}: first {M_INDEP} frames delta=0, "
                  f"last {T_lat - M_INDEP} frames delta=0  ({_VERIFY_DONE.get(T_px, 0)}/{_VERIFY_N} checks; "
                  f"unchecked from here on)", flush=True)
            del ref
        except Exception as _e:


            print(f"[COND-Y-VERIFY] the check itself failed and was skipped; training is unaffected and the fast-path result is still used: "
                  f"{type(_e).__name__}: {_e}", flush=True)
    return out

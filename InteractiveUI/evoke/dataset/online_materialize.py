
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from evoke.utils.utils_base import encode_prompt


_DEPTH_ESTIMATORS: dict = {}


_GT_OD = {"off": False, "n_warn": 0}


def _get_da3_estimator(process_res: int, device, weights=None, backend="da3", src=None, vigeo_opts=None):


    from evoke.modules.geometric_state.depth_backend import build_estimator
    key = (str(backend or "da3").lower(), int(process_res), str(weights or ""), str(src or ""),
           tuple(sorted((k, str(v)) for k, v in (vigeo_opts or {}).items())))
    if key not in _DEPTH_ESTIMATORS:
        _DEPTH_ESTIMATORS[key] = build_estimator(
            backend, device, int(process_res), weights=weights, src=src, vigeo_opts=vigeo_opts)
    return _DEPTH_ESTIMATORS[key]


def _geo_inject_warp_error(args, recycle_vars, w_lat):


    from evoke.utils.utils_recycle_batch import sample_y_error_from_latent_buffer

    ybuf = getattr(recycle_vars, "y_error_buffer", None)
    _, _, _, h, w = w_lat.shape
    if ybuf is None or (h, w) not in ybuf:
        return
    err, _depths = sample_y_error_from_latent_buffer(
        args, recycle_vars, w_lat, dtype=w_lat.dtype, device=w_lat.device
    )
    w_lat.add_(err.to(w_lat.device, dtype=w_lat.dtype))


def apply_warp_token_drop(vis, cfg, generator=None):


    if vis is None or not getattr(cfg, "enabled", False):
        return vis
    probs = [max(0.0, float(p)) for p in cfg.mode_probs]
    s = sum(probs)
    if s <= 0:
        return vis
    probs = [p / s for p in probs]
    c_full = probs[0] + probs[1]
    c_frame = c_full + probs[2]
    B, _, T, H, W = vis.shape
    dev = vis.device
    fr, pr = float(cfg.frame_drop_ratio), float(cfg.patch_drop_ratio)
    out = vis.clone()
    for b in range(B):
        r = torch.rand((), generator=generator, device=dev).item()
        if r < probs[0]:
            continue
        elif r < c_full:
            out[b] = 0.0
        elif r < c_frame:
            drop = torch.rand(T, generator=generator, device=dev) < fr
            out[b, :, drop] = 0.0
        else:
            ph = pw = 2
            gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw
            pd = torch.rand(T, gh, gw, generator=generator, device=dev) < pr
            pd = pd.repeat_interleave(ph, -2).repeat_interleave(pw, -1)[:, :H, :W]
            out[b, 0] = out[b, 0] * (~pd).to(out.dtype)
    return out


def _pose_jitter_rot_xyz(deg_x, deg_y, deg_z):


    rx, ry, rz = np.radians([deg_x, deg_y, deg_z])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _sample_warp_pose_jitter_DT(cfg, mean_interframe_trans=0.0):


    def _signed(rng):
        lo, hi = float(rng[0]), float(rng[1])
        mag = random.uniform(lo, hi)
        return mag if random.random() < 0.5 else -mag
    pitch = _signed(getattr(cfg, "pitch_deg_range", [0.5, 2.0]))
    yaw = _signed(getattr(cfg, "yaw_deg_range", [0.5, 2.0]))
    roll = _signed(getattr(cfg, "roll_deg_range", [0.0, 0.5]))
    DT = np.eye(4, dtype=np.float32)
    DT[:3, :3] = _pose_jitter_rot_xyz(pitch, yaw, roll)
    tr = getattr(cfg, "trans_frac_range", [0.0, 0.0])
    if not (float(tr[0]) == 0.0 and float(tr[1]) == 0.0):
        tf = random.uniform(float(tr[0]), float(tr[1]))
        tmag = tf * float(mean_interframe_trans)

        DT[:3, 3] = np.array([tmag * 0.5, tmag * 0.2, tmag], np.float32)
    return torch.from_numpy(DT)


def _geo_add_noise_to_warp_latents(
    warp_latents: torch.Tensor,
    device: torch.device,
    generator=None,
    sigma_min: float = 0.111,
    sigma_max: float = 0.135,
    visibility_aware_noise: bool = False,
    sigma_invisible: float = 0.8,
    visibility_mask_lat: torch.Tensor | None = None,
) -> torch.Tensor:


    chunk_frames = int(warp_latents.shape[2])
    rand_generator = generator[0] if isinstance(generator, list) else generator
    frame_sigmas = (
        torch.rand(chunk_frames, device=device, generator=rand_generator) * (sigma_max - sigma_min) + sigma_min
    ).to(dtype=warp_latents.dtype)

    if visibility_aware_noise and visibility_mask_lat is not None:

        assert 0.0 < float(sigma_invisible) <= 1.0, (
            f"sigma_invisible must be in (0, 1.0], got {sigma_invisible}"
        )
        mask = visibility_mask_lat.to(device=device, dtype=warp_latents.dtype)
        if mask.shape[2] != chunk_frames:
            raise ValueError(
                f"visibility_mask_lat temporal dim {mask.shape[2]} != warp_latents {chunk_frames}"
            )

        sigma_visible_5d = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
        sigmas = mask * sigma_visible_5d + (1.0 - mask) * float(sigma_invisible)
        noise = randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
        return sigmas * noise + (1.0 - sigmas) * warp_latents
    else:

        frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
        return (
            frame_sigmas
            * randn_tensor(warp_latents.shape, generator=generator, device=device, dtype=warp_latents.dtype)
            + (1 - frame_sigmas) * warp_latents
        )


def _geo_resize_visibility_to_latent(
    visibility_mask_pix: torch.Tensor,
    num_lat_per_chunk: int,
    H_lat: int,
    W_lat: int,
    vae_t_stride: int = 4,
    patch_hw: tuple = (2, 2),
    cov_thresh: float = 0.5,
) -> torch.Tensor:


    device = visibility_mask_pix.device
    sample_ids = torch.arange(num_lat_per_chunk, device=device) * int(vae_t_stride)
    sample_ids = sample_ids.clamp(max=visibility_mask_pix.shape[2] - 1)
    sampled = visibility_mask_pix.index_select(2, sample_ids)
    ph, pw = max(1, int(patch_hw[0])), max(1, int(patch_hw[1]))
    gh, gw = max(1, int(H_lat) // ph), max(1, int(W_lat) // pw)
    patch_cov = F.adaptive_avg_pool3d(sampled, (num_lat_per_chunk, gh, gw))
    patch_bin = (patch_cov > float(cov_thresh)).to(sampled.dtype)
    return F.interpolate(patch_bin, size=(num_lat_per_chunk, H_lat, W_lat), mode="nearest")


def prepare_stage1_latent_v2(
    vae_latent: torch.Tensor,
    history_sizes,
    choice_idx: int,
    is_keep_x0: bool = True,
    base_vae_latent: torch.Tensor | None = None,
):


    source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

    x0_latent = source_latent[0, :, :1, :, :].clone() if is_keep_x0 else None

    total_sections = source_latent.shape[0]
    latent_window_size = source_latent.shape[2]
    history_window_size = sum(history_sizes)
    section_size = history_window_size + latent_window_size


    temp_source = rearrange(source_latent, "b c t h w -> c (b t) h w")
    pad_src = torch.zeros(
        temp_source.shape[0], history_window_size, temp_source.shape[2], temp_source.shape[3],
        device=temp_source.device, dtype=temp_source.dtype,
    )
    continue_source = torch.cat([pad_src, temp_source], dim=1)

    temp_vae = rearrange(vae_latent, "b c t h w -> c (b t) h w")
    pad_vae = torch.zeros_like(pad_src)
    continue_vae = torch.cat([pad_vae, temp_vae], dim=1)


    if choice_idx == 0 and x0_latent is not None:
        x0_latent = torch.zeros_like(x0_latent)

    assert 0 <= choice_idx < total_sections
    start = choice_idx * latent_window_size
    end = start + section_size
    history_latent = continue_source[:, start : start + history_window_size, :, :]
    target_latent = continue_vae[:, start + history_window_size : end, :, :]
    return x0_latent, history_latent, target_latent


def _find_segment_aligned_choice_idx(
    segment_prompts,
    initial_choice_idx: int,
    n_section: int,
    latent_window_size: int,
    vae_temporal_ratio: int = 4,
):


    if not segment_prompts:
        return initial_choice_idx, None


    chunk_pix_frames = (latent_window_size - 1) * vae_temporal_ratio + 1

    def _try(ci):
        if ci < 0 or ci >= n_section:
            return None
        target_start = ci * chunk_pix_frames
        target_end = target_start + chunk_pix_frames
        for seg in segment_prompts:
            if seg["start_frame"] <= target_start and target_end <= seg["end_frame"]:
                return seg["prompt"]
        return None


    candidates = [initial_choice_idx]
    for d in range(1, n_section):
        candidates.extend([initial_choice_idx + d, initial_choice_idx - d])

    for ci in candidates:
        cap = _try(ci)
        if cap is not None:
            return ci, cap


    import warnings as _warnings
    _warnings.warn(
        f"[interleave_caption] no segment fits {chunk_pix_frames}-frame target window; "
        f"falling back to overall caption. segments={[(s['start_frame'], s['end_frame']) for s in segment_prompts]}",
        RuntimeWarning,
        stacklevel=2,
    )
    return initial_choice_idx, None


def materialize_full_rollout_interleave(
    batch: dict,
    latent: "torch.Tensor",
    raw_video: "torch.Tensor",
    vae,
    tokenizer,
    text_encoder,
    latents_mean: "torch.Tensor",
    latents_std: "torch.Tensor",
    latent_window_size: int,
    prefix_sections: int,
    history_sizes,
    device,
    weight_dtype,
    is_keep_x0: bool = True,
    sf_build_teacher_y: bool = True,


    sf_skip_full_encode: bool = False,

    sf_gt_partial: bool = False,
    full_clip_num_frames: Optional[int] = None,


    sf_i2v_prefix_latent_frames: int = 0,
    sf_i2v_hist_latent_mode: str = "static_repeat",


    sf_i2v_ratio: float = 0.0,
):


    B, C_lat, T_lat, H_lat, W_lat = latent.shape
    assert B == 1, f"[SF10S] train_batch_size=1 (matches the GEO constraint and simplifies segment mapping), got B={B}"
    P = int(prefix_sections)
    if sf_skip_full_encode:


        assert full_clip_num_frames is not None, (
            "sf_skip_full_encode=True requires full_clip_num_frames to derive N/T_px"
        )
        full_T_lat = (full_clip_num_frames - 1) // 4 + 1
        full_n_section = full_T_lat // latent_window_size
        n_section = P
        N = full_n_section - P
    else:
        n_section = T_lat // latent_window_size
        full_T_lat = T_lat
        N = n_section - P
    assert P >= 1 and N >= 1, f"[SF10S] invalid section count: n_section={n_section}, prefix={P}"

    sf_prefix_latents = latent[:, :, : P * latent_window_size].contiguous()


    segs_raw = batch.get("segment_prompts", [None])[0] if batch.get("segment_prompts") else None
    overall_caption = list(batch["prompt"])[0]
    T_px = (full_T_lat - 1) * 4 + 1

    def _seg_of_px(px_frame):
        if segs_raw:
            for si, seg in enumerate(segs_raw):
                if int(seg["start_frame"]) <= px_frame < int(seg["end_frame"]):
                    return si
        return None


    def _captions_for_prefix_latents(p_lat: int, tag: str):
        out = []
        for k in range(N):
            mid_lat = int(p_lat) + k * latent_window_size + latent_window_size // 2
            mid_px = min(mid_lat * 4, T_px - 1)
            si = _seg_of_px(mid_px)
            if segs_raw and si is None:
                print(f"[SF10S][warn]{tag} section {k} midpoint frame {mid_px} falls inside no segment (captions do not cover the clip tail?), "
                      f"student uses the overall caption while teacher chunk_context_map records segment0 -- the two sides may diverge semantically")
            out.append(segs_raw[si]["prompt"] if si is not None else overall_caption)
        return out

    per_section_caption = _captions_for_prefix_latents(P * latent_window_size, "")


    per_section_caption_i2v = (
        _captions_for_prefix_latents(int(sf_i2v_prefix_latent_frames), "[i2v]")
        if int(sf_i2v_prefix_latent_frames) > 0 else None
    )


    if segs_raw:
        score_captions = [seg["prompt"] for seg in segs_raw]
        sf_segment_frame_ranges = [
            (max(0, int(seg["start_frame"])), min(T_px, int(seg["end_frame"]))) for seg in segs_raw
        ]
    else:
        score_captions = [overall_caption]
        sf_segment_frame_ranges = [(0, T_px)]


    uniq = list(dict.fromkeys(
        per_section_caption + score_captions + [overall_caption] + (per_section_caption_i2v or [])))
    from evoke.utils import sf_prep_profile as _pp
    _t = _pp.mark()
    uniq_embeds, _ = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder, prompt=uniq, device=device, dtype=weight_dtype,
    )
    _pp.accum("t5", _t, f"{len(uniq)} captions")
    emb_of = {c: uniq_embeds[i : i + 1] for i, c in enumerate(uniq)}
    sf_prompt_embeds_list = [emb_of[c] for c in per_section_caption]
    sf_prompt_embeds_list_i2v = (
        [emb_of[c] for c in per_section_caption_i2v] if per_section_caption_i2v is not None else None
    )
    sf_score_prompt_embeds = torch.stack([emb_of[c] for c in score_captions], dim=1)
    prompt_embeds = emb_of[overall_caption]


    sf_teacher_y = None
    if sf_build_teacher_y:
        from evoke.dataset.cond_y_fastpath import cond_y_latent as _cond_y_fast
        from evoke.modules.evoke_teacher.wrapper import build_i2v_y
        _T_px_y = int(raw_video.shape[2])
        _t = _pp.mark()


        with torch.no_grad():
            cond_lat = _cond_y_fast(vae, raw_video, _T_px_y)
        _pp.accum("vae_cond_y", _t, f"fast path {_cond_y_fast.__module__.split('.')[-1]}: "
                                    f"encoded 113 frames + constant tail (originally {_T_px_y} frames)")
        cond_px = raw_video
        if os.environ.get("SF_VAE_RF_PROBE") == "1" and not getattr(_pp, "_rf_done", False):
            _pp._rf_done = True


            try:
                print(_pp.vae_rf_probe(vae, raw_video, int(cond_px.shape[2]),
                                       latents_mean, latents_std), flush=True)
            except Exception as _e:
                print(f"[VAE-RF-PROBE] the probe itself errored, skipped (training unaffected): {type(_e).__name__}: {_e}", flush=True)
        cond_lat = ((cond_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        sf_teacher_y = build_i2v_y(cond_lat, num_cond_px_frames=1)


    sf_i2v_hist_latent = None
    if (int(sf_i2v_prefix_latent_frames) > 0 and str(sf_i2v_hist_latent_mode) == "static_repeat"
            and float(sf_i2v_ratio) > 0.0):
        _min_f = (int(latent_window_size) - 1) * 4 + 1
        _static_px = raw_video[:, :, :1].repeat(1, 1, _min_f, 1, 1)
        _t = _pp.mark()
        with torch.no_grad():

            _static_lat = vae.encode(_static_px.to(vae.dtype)).latent_dist.mode()
        _pp.accum("vae_hist1x", _t, f"{_static_px.shape[2]} frames")
        _static_lat = ((_static_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        sf_i2v_hist_latent = _static_lat[:, :, -1:].contiguous()


    _lat_ph = latent[:, :, : n_section * latent_window_size] if sf_gt_partial else latent
    latent_sec = rearrange(_lat_ph, "b c (n w) h s -> b n c w h s", n=n_section, w=latent_window_size)
    x0, hist, tgt = prepare_stage1_latent_v2(
        latent_sec[0], history_sizes=history_sizes, choice_idx=n_section - 1, is_keep_x0=is_keep_x0,
    )

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": None,
        "x0_latents": x0.unsqueeze(0) if is_keep_x0 else None,
        "history_latents": hist.unsqueeze(0),
        "target_latents": tgt.unsqueeze(0),
        "clean_all_latents": None,
        "prompt": [overall_caption],
        "uttid": batch.get("uttid"),
        "dataset_name": batch.get("dataset_name"),
        "bucket_key": batch.get("bucket_key"),

        "sf_prefix_latents": sf_prefix_latents,


        "sf_gt_latents": None if (sf_skip_full_encode and not sf_gt_partial) else latent,
        "sf_prompt_embeds_list": sf_prompt_embeds_list,
        "sf_score_prompt_embeds": sf_score_prompt_embeds,
        "sf_teacher_y": sf_teacher_y,
        "sf_segment_frame_ranges": sf_segment_frame_ranges,
        "sf_num_generated_sections": N,


        "sf_prompt_embeds_list_i2v": sf_prompt_embeds_list_i2v,
        "sf_i2v_hist_latent": sf_i2v_hist_latent,

        "sf_sample_is_i2v": False,


        "sf_pose_Ks": batch.get("lingbot_Ks"),
        "sf_pose_c2ws": batch.get("lingbot_c2ws"),
    }


def materialize_i2v_image_only(
    batch: dict,
    raw_video: "torch.Tensor",
    vae,
    tokenizer,
    text_encoder,
    latents_mean: "torch.Tensor",
    latents_std: "torch.Tensor",
    latent_window_size: int,
    history_sizes,
    device,
    weight_dtype,
    is_keep_x0: bool = True,
    num_generated_sections: int = 20,
    prefix_latent_frames: int = 1,
    hist_latent_mode: str = "static_repeat",
    build_teacher_y: bool = True,
):


    B, C_pix, T_pix, H_pix, W_pix = raw_video.shape
    assert T_pix == 1, f"[LW-I2V-SAMPLE] single-frame input required, got T_pix={T_pix}"
    assert B == 1, f"[LW-I2V-SAMPLE] B=1 required, got {B}"
    win = int(latent_window_size)
    N = int(num_generated_sections)
    P_lat = int(prefix_latent_frames)
    assert N >= 1 and P_lat >= 1, f"[LW-I2V-SAMPLE] invalid N={N} P_lat={P_lat}"
    T_lat_i2v = P_lat + N * win
    T_px_i2v = (T_lat_i2v - 1) * 4 + 1
    Hl, Wl = H_pix // 8, W_pix // 8
    img = raw_video[:, :, :1]

    from evoke.utils import sf_prep_profile as _ppi

    def _enc(px):
        _t = _ppi.mark()
        with torch.no_grad():

            z = vae.encode(px.to(vae.dtype)).latent_dist.mode()

        _ppi.accum(f"vae_i2v_{px.shape[2]}f", _t, f"{px.shape[2]} frames")
        return ((z - latents_mean) * latents_std).to(dtype=weight_dtype)


    sf_prefix_latents = _enc(img)[:, :, :P_lat].contiguous()
    assert sf_prefix_latents.shape[2] == P_lat, (
        f"[LW-I2V-SAMPLE] the single-frame encode yields only {sf_prefix_latents.shape[2]} latents, need {P_lat}")


    sf_i2v_hist_latent = None
    if str(hist_latent_mode) == "static_repeat":
        _min_f = (win - 1) * 4 + 1
        sf_i2v_hist_latent = _enc(img.repeat(1, 1, _min_f, 1, 1))[:, :, -1:].contiguous()


    sf_teacher_y = None
    if build_teacher_y:
        from evoke.dataset.cond_y_fastpath import cond_y_latent as _cond_y_fast
        from evoke.modules.evoke_teacher.wrapper import build_i2v_y


        _t = _ppi.mark()
        with torch.no_grad():
            _cond_lat = _cond_y_fast(vae, img, int(T_px_i2v))
        _cond_lat = ((_cond_lat - latents_mean) * latents_std).to(dtype=weight_dtype)
        _ppi.accum("vae_i2v_condy_fast", _t, f"encoded 113 frames + constant tail (originally {T_px_i2v} frames)")
        sf_teacher_y = build_i2v_y(_cond_lat, num_cond_px_frames=1)
        assert sf_teacher_y.shape[2] == T_lat_i2v, (
            f"[LW-I2V-SAMPLE] y frame count {sf_teacher_y.shape[2]} != scoring sequence {T_lat_i2v}")


    overall_caption = list(batch["prompt"])[0]
    _emb, _ = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder, prompt=[overall_caption],
        device=device, dtype=weight_dtype,
    )
    prompt_embeds = _emb[0:1]
    sf_prompt_embeds_list = [prompt_embeds for _ in range(N)]


    _n_ph = 1 + N
    _ph = torch.zeros(1, sf_prefix_latents.shape[1], _n_ph * win, Hl, Wl,
                      device=sf_prefix_latents.device, dtype=sf_prefix_latents.dtype)
    _ph_sec = rearrange(_ph, "b c (n w) h s -> b n c w h s", n=_n_ph, w=win)
    x0, hist, tgt = prepare_stage1_latent_v2(
        _ph_sec[0], history_sizes=history_sizes, choice_idx=_n_ph - 1, is_keep_x0=is_keep_x0,
    )

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": None,
        "x0_latents": x0.unsqueeze(0) if is_keep_x0 else None,
        "history_latents": hist.unsqueeze(0),
        "target_latents": tgt.unsqueeze(0),
        "clean_all_latents": None,
        "prompt": [overall_caption],
        "uttid": batch.get("uttid"),
        "dataset_name": batch.get("dataset_name"),
        "bucket_key": batch.get("bucket_key"),

        "sf_prefix_latents": sf_prefix_latents,
        "sf_gt_latents": None,
        "sf_prompt_embeds_list": sf_prompt_embeds_list,
        "sf_prompt_embeds_list_i2v": sf_prompt_embeds_list,
        "sf_score_prompt_embeds": prompt_embeds.unsqueeze(1),
        "sf_teacher_y": sf_teacher_y,
        "sf_segment_frame_ranges": [(0, T_px_i2v)],
        "sf_num_generated_sections": N,
        "sf_i2v_hist_latent": sf_i2v_hist_latent,

        "sf_sample_is_i2v": True,
        "sf_pose_Ks": None,
        "sf_pose_c2ws": None,
    }


def materialize_online_batch(
    batch: dict,
    vae,
    tokenizer,
    text_encoder,
    history_sizes,
    latent_window_size: int,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    is_keep_x0: bool = True,
    seed: int = 42,
    epoch: int = 0,

    use_geometric_state: bool = False,

    geo_keep_clean_anchor: bool = False,

    geo_retrieve_cfg=None,

    geo_condition_t2v_ratio: float = 0.0,
    geo_condition_i2v_ratio: float = 0.0,


    geo_cloud_warp_cfg=None,

    geo_visibility_aware_noise: bool = False,
    geo_sigma_invisible: float = 0.8,
    geo_sigma_visible_min: float = 0.111,
    geo_sigma_visible_max: float = 0.135,


    recycle_vars=None,
    args=None,
    geo_warp_error_inject_enabled: bool = False,
    geo_warp_error_prob: float = 0.0,


    geo_pose_jitter_cfg=None,

    sf_full_rollout_interleave: bool = False,
    sf_prefix_sections: int = 1,
    sf_build_teacher_y: bool = True,


    sf_skip_full_encode: bool = False,
    full_clip_num_frames: Optional[int] = None,


    sf_i2v_prefix_latent_frames: int = 0,
    sf_i2v_hist_latent_mode: str = "static_repeat",


    sf_i2v_ratio: float = 0.0,


    sf_gt_encode_px: Optional[int] = None,


    sf_num_generated_sections: int = 0,
):


    raw_video = batch["raw_video"].to(device=device, dtype=vae.dtype, non_blocking=True)
    B, C_pix, T_pix, H_pix, W_pix = raw_video.shape


    _px_per_chunk = (int(latent_window_size) - 1) * 4 + 1
    if sf_full_rollout_interleave and T_pix < _px_per_chunk:


        def _who():
            def _one(k):
                v = batch.get(k)
                return (v[0] if isinstance(v, (list, tuple)) and v else v)
            return f"uttid={_one('uttid')} dataset={_one('dataset_name')} bucket={_one('bucket_key')}"

        assert T_pix == 1, (
            f"[LW-I2V-SAMPLE] only 1-frame image samples are supported, got T_pix={T_pix} (<{_px_per_chunk} but >1): "
            f"filter short video samples out with require_full_length, or extend this branch explicitly. sample: {_who()}")
        assert B == 1, f"[LW-I2V-SAMPLE] train_batch_size=1 required (same constraint as SF10S), got B={B}. sample: {_who()}"
        assert sf_i2v_prefix_latent_frames > 0, (
            "[LW-I2V-SAMPLE] the data contains image-only samples but sf_i2v_ratio=0 (i2v path not enabled) => "
            "sf_i2v_prefix_latent_frames=0, so no i2v conditioning can be built for them. either enable the i2v path or drop the image source from select. "
            f"sample: {_who()}")
        return materialize_i2v_image_only(
            batch, raw_video=raw_video, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, history_sizes=history_sizes,
            device=device, weight_dtype=weight_dtype, is_keep_x0=is_keep_x0,
            num_generated_sections=int(sf_num_generated_sections),
            prefix_latent_frames=int(sf_i2v_prefix_latent_frames),
            hist_latent_mode=str(sf_i2v_hist_latent_mode),
            build_teacher_y=bool(sf_build_teacher_y),
        )


    if sf_full_rollout_interleave and sf_skip_full_encode:
        assert full_clip_num_frames is not None, (
            "sf_skip_full_encode=True requires full_clip_num_frames"
        )
        P = int(sf_prefix_sections)
        P_px = (P * latent_window_size - 1) * 4 + 1
        with torch.no_grad():
            prefix_latent = vae.encode(raw_video[:, :, :P_px]).latent_dist.sample()
        prefix_latent = (prefix_latent - latents_mean) * latents_std
        prefix_latent = prefix_latent.to(dtype=weight_dtype)
        return materialize_full_rollout_interleave(
            batch, latent=prefix_latent, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0,
            sf_build_teacher_y=sf_build_teacher_y,
            sf_skip_full_encode=True,
            full_clip_num_frames=full_clip_num_frames,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )


    from evoke.utils import sf_prep_profile as _ppc
    if sf_gt_encode_px is not None and int(sf_gt_encode_px) < T_pix and not _GT_OD["off"]:
        _gt_px = max(1, int(sf_gt_encode_px))
        _T_lat_full = 1 + (int(T_pix) - 1) // 4


        _do_verify = os.environ.get("SF_GT_VERIFY") == "1" and _ppc.gt_verify_take(int(_gt_px))
        _dev = raw_video.device


        _st_cpu = torch.get_rng_state()
        _st_cu = torch.cuda.get_rng_state(_dev) if (_dev.type == "cuda") else None
        _tc = _ppc.mark()
        with torch.no_grad():
            _dist = vae.encode(raw_video[:, :, :_gt_px]).latent_dist
            _mu, _sd = _dist.mean, _dist.std
            _n = int(_mu.shape[2])
            _eps = randn_tensor(
                (_mu.shape[0], _mu.shape[1], _T_lat_full, _mu.shape[3], _mu.shape[4]),
                generator=None, device=_dist.parameters.device, dtype=_dist.parameters.dtype)
            _part_raw = _mu + _sd * _eps[:, :, :_n]
            del _eps
        _part = ((_part_raw - latents_mean) * latents_std).to(dtype=weight_dtype)
        _ppc.accum("vae_gt_ondemand", _tc, f"{_gt_px} frames -> {_n} lat (full clip {T_pix} -> {_T_lat_full} lat)")

        _st_cpu_after = torch.get_rng_state()
        _st_cu_after = torch.cuda.get_rng_state(_dev) if (_dev.type == "cuda") else None

        if _do_verify:


            _fail = None
            try:
                torch.set_rng_state(_st_cpu)
                if _st_cu is not None:
                    torch.cuda.set_rng_state(_st_cu, _dev)
                with torch.no_grad():
                    _fd = vae.encode(raw_video).latent_dist
                    _full_raw = _fd.sample()
                _dmu = (_fd.mean[:, :, :_n] - _mu).abs().amax(dim=(0, 1, 3, 4))
                _dsd = (_fd.std[:, :, :_n] - _sd).abs().amax(dim=(0, 1, 3, 4))
                _dz = (_full_raw[:, :, :_n] - _part_raw).abs().amax(dim=(0, 1, 3, 4))


                del _fd, _full_raw
                _bad = {k: [i for i in range(_n) if float(v[i]) != 0.0]
                        for k, v in (("μ", _dmu), ("σ", _dsd), ("z", _dz))}
                _fmt = lambda v: "[" + " ".join(f"{float(v[i]):.3e}" for i in range(_n)) + "]"
                _detail = (f"encoded {_gt_px} frames -> {_n} latent (full clip {T_pix} -> {_T_lat_full} latent) | "
                           f"per-frame max|Δ|: μ={_fmt(_dmu)} σ={_fmt(_dsd)} z={_fmt(_dz)}")
                if any(_bad.values()):
                    _fail = (f"frames where μ differs={_bad['μ']} where σ differs={_bad['σ']} where z differs={_bad['z']}\n"
                             f"    {_detail}\n"
                             f"    => 'the VAE is strictly causal in time, truncating the input does not change already produced chunks' is refuted"
                             f" (differences concentrated in the last few frames => the receptive field is longer than R=112; uniform differences over all frames => there is another non-causal operator)")
                else:
                    print(f"[GT-VERIFY] OK bit-identical to the baseline (μ/σ/z all 0): {_detail}", flush=True)
            except Exception as _e:
                print(f"[GT-VERIFY] the check itself errored, skipped (training unaffected): {type(_e).__name__}: {_e}", flush=True)
            finally:


                torch.set_rng_state(_st_cpu_after)
                if _st_cu_after is not None:
                    torch.cuda.set_rng_state(_st_cu_after, _dev)
            if _fail is not None:
                _GT_OD["off"] = True
                print(f"[GT-VERIFY] FAIL premise refuted -- **on-demand GT encoding is disabled for the rest of this run, falling back to full-length encoding**"
                      f" (training is not killed, it just goes back to 16s/step)\n    {_fail}", flush=True)
                torch.set_rng_state(_st_cpu)
                if _st_cu is not None:
                    torch.cuda.set_rng_state(_st_cu, _dev)

    if sf_gt_encode_px is not None and int(sf_gt_encode_px) < T_pix and not _GT_OD["off"]:
        return materialize_full_rollout_interleave(
            batch, latent=_part, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0, sf_build_teacher_y=sf_build_teacher_y,
            sf_skip_full_encode=True, sf_gt_partial=True, full_clip_num_frames=T_pix,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )


    from evoke.utils import sf_prep_profile as _pp0
    _t0 = _pp0.mark()
    with torch.no_grad():
        latent = vae.encode(raw_video).latent_dist.sample()
    _pp0.accum("vae_gt_full", _t0, f"{raw_video.shape[2]} real video frames")
    latent = (latent - latents_mean) * latents_std
    latent = latent.to(dtype=weight_dtype)

    _, C_lat, T_lat, H_lat, W_lat = latent.shape
    assert T_lat % latent_window_size == 0, (
        f"VAE output T_lat={T_lat} must be divisible by latent_window_size={latent_window_size}. "
        f"check num_frames in the training yaml (= (latent_window * n_section - 1) * 4 + 1)."
    )
    n_section = T_lat // latent_window_size


    if sf_full_rollout_interleave:
        return materialize_full_rollout_interleave(
            batch, latent=latent, raw_video=raw_video, vae=vae,
            tokenizer=tokenizer, text_encoder=text_encoder,
            latents_mean=latents_mean, latents_std=latents_std,
            latent_window_size=latent_window_size, prefix_sections=sf_prefix_sections,
            history_sizes=history_sizes, device=device, weight_dtype=weight_dtype,
            is_keep_x0=is_keep_x0,
            sf_build_teacher_y=sf_build_teacher_y,
            sf_i2v_prefix_latent_frames=sf_i2v_prefix_latent_frames,
            sf_i2v_hist_latent_mode=sf_i2v_hist_latent_mode,
            sf_i2v_ratio=sf_i2v_ratio,
        )


    latent_sec = rearrange(latent, "b c (n w) h s -> b n c w h s", n=n_section, w=latent_window_size)


    segs_per_sample = batch.get("segment_prompts", [None] * B)
    if segs_per_sample is None:
        segs_per_sample = [None] * B


    _is_skill_raw = batch.get("is_skill", False)
    _is_skill = bool(_is_skill_raw[0]) if isinstance(_is_skill_raw, (list, tuple)) and _is_skill_raw else bool(_is_skill_raw)
    _event_win_raw = batch.get("event_window", None)
    _event_win = _event_win_raw[0] if isinstance(_event_win_raw, (list, tuple)) and _event_win_raw else _event_win_raw

    x0_list, hist_list, tgt_list = [], [], []
    sample_prompts = list(batch["prompt"])
    choice_idx_per_sample = [None] * B
    for b in range(B):
        g = torch.Generator().manual_seed(seed + epoch * 1_000_003 + b)
        initial_choice_idx = int(torch.randint(0, n_section, (1,), generator=g).item())
        segs = segs_per_sample[b] if b < len(segs_per_sample) else None
        if segs:
            choice_idx, matched_caption = _find_segment_aligned_choice_idx(
                segs, initial_choice_idx, n_section, latent_window_size,
            )

            if matched_caption is not None:
                sample_prompts[b] = matched_caption
        else:
            choice_idx = initial_choice_idx


        if _is_skill and _event_win is not None:
            ev_lo, ev_hi = int(_event_win[0]), int(_event_win[1])
            valid = [s for s in range(n_section) if ev_lo <= s * latent_window_size * 4 <= ev_hi]
            if valid:
                choice_idx = valid[int(torch.randint(0, len(valid), (1,), generator=g).item())]
        choice_idx_per_sample[b] = choice_idx
        x0, hist, tgt = prepare_stage1_latent_v2(
            latent_sec[b], history_sizes=history_sizes,
            choice_idx=choice_idx, is_keep_x0=is_keep_x0,
        )
        x0_list.append(x0)
        hist_list.append(hist)
        tgt_list.append(tgt)
    x0_latents = torch.stack(x0_list, dim=0) if is_keep_x0 else None
    history_latents = torch.stack(hist_list, dim=0)
    target_latents = torch.stack(tgt_list, dim=0)


    target_pose_Ks = None
    target_pose_c2ws = None

    lingbot_c2ws_full = None
    lingbot_Ks_full = None
    if "lingbot_Ks" in batch and "lingbot_c2ws" in batch:
        lingbot_Ks_full = batch["lingbot_Ks"]
        lingbot_c2ws_full = batch["lingbot_c2ws"]
        N_pix = lingbot_c2ws_full.shape[1]
        pix_window_len = (latent_window_size - 1) * 4 + 1
        pose_per_sample = []
        for b in range(B):
            latent_start = choice_idx_per_sample[b] * latent_window_size
            pix_start = latent_start * 4
            pix_end = pix_start + pix_window_len

            if pix_end > N_pix:
                need = pix_end - N_pix
                seg = torch.cat([
                    lingbot_c2ws_full[b, pix_start:N_pix],
                    lingbot_c2ws_full[b, N_pix - 1:N_pix].expand(need, -1, -1),
                ], dim=0)
            else:
                seg = lingbot_c2ws_full[b, pix_start:pix_end]
            pose_per_sample.append(seg)
        target_pose_c2ws = torch.stack(pose_per_sample, dim=0).to(device=device, dtype=weight_dtype)
        target_pose_Ks = lingbot_Ks_full.to(device=device, dtype=weight_dtype)


    if _is_skill and target_pose_c2ws is None:
        _pix_window_len = (latent_window_size - 1) * 4 + 1
        _eye = torch.eye(4, device=device, dtype=weight_dtype)
        target_pose_c2ws = _eye.view(1, 1, 4, 4).expand(B, _pix_window_len, 4, 4).contiguous()
        _bk = batch["bucket_key"]
        _h = int(_bk[1]) if isinstance(_bk, (tuple, list)) else int(_bk[1].item())
        _w = int(_bk[2]) if isinstance(_bk, (tuple, list)) else int(_bk[2].item())
        target_pose_Ks = torch.tensor(
            [float(_w), float(_h), _w / 2.0, _h / 2.0], device=device, dtype=weight_dtype
        ).view(1, 4).expand(B, 4).contiguous()


    prompt_embeds, prompt_attention_mask = encode_prompt(
        tokenizer=tokenizer, text_encoder=text_encoder,
        prompt=sample_prompts, device=device, dtype=weight_dtype,
    )

    bkey = batch["bucket_key"]
    num_frame = int(bkey[0]) if isinstance(bkey, (tuple, list)) else int(bkey[0].item())
    height = int(bkey[1]) if isinstance(bkey, (tuple, list)) else int(bkey[1].item())
    width = int(bkey[2]) if isinstance(bkey, (tuple, list)) else int(bkey[2].item())


    import random as _random_mode
    _geo_mode = "full_geo"
    if use_geometric_state and not _is_skill and (geo_condition_t2v_ratio + geo_condition_i2v_ratio) > 0:
        _r = _random_mode.random()
        if _r < geo_condition_t2v_ratio:
            _geo_mode = "t2v"
        elif _r < geo_condition_t2v_ratio + geo_condition_i2v_ratio:
            _geo_mode = "i2v"
        else:
            _geo_mode = "full_geo"


    warp_video_latents = None
    warp_visibility_mask = None
    geo_source_image_latent = None


    if use_geometric_state and _geo_mode != "t2v" and not _is_skill:
        assert B == 1, (
            f"GEO training requires train_batch_size=1 (Bug 3 cross-batch mask consistency constraint), got B={B}. "
            f"set in the yaml: train_batch_size: 1, gradient_accumulation_steps: 4."
        )
        assert target_pose_c2ws is not None, (
            "GEO training requires target_pose_c2ws (i.e. the batch contains lingbot_c2ws). check pose_dir in the yaml."
        )

        pix_window_len = (latent_window_size - 1) * 4 + 1

        warp_lat_list = []
        warp_mask_list = []
        source_latent_list = []

        _geo_pix_stride = latent_window_size * 4

        import math as _math
        _r_init_k = int(getattr(geo_retrieve_cfg, "init_k", 10)) if geo_retrieve_cfg is not None else 10
        _r_bank_max = int(getattr(geo_retrieve_cfg, "bank_max", 0)) if geo_retrieve_cfg is not None else 0
        _r_score = str(getattr(geo_retrieve_cfg, "score", "v1")) if geo_retrieve_cfg is not None else "v1"
        _r_nearby_k = int(getattr(geo_retrieve_cfg, "nearby_k", 0)) if geo_retrieve_cfg is not None else 0
        _r_select_k = int(getattr(geo_retrieve_cfg, "select_k", 5)) if geo_retrieve_cfg is not None else 5
        _r_score_kwargs = {}
        if _r_score == "v3" and geo_retrieve_cfg is not None:
            _r_score_kwargs = {
                "depth": float(getattr(geo_retrieve_cfg, "v3_depth", 5.0)),
                "fov_rad": _math.radians(float(getattr(geo_retrieve_cfg, "v3_fov_deg", 60.0))),
            }
        _geo_init_max = _r_init_k


        _cloud_enabled = bool(getattr(geo_cloud_warp_cfg, "enabled", False)) if geo_cloud_warp_cfg is not None else False
        if _cloud_enabled:
            import random as _random_mode


            from evoke.modules.geometric_state.da3_cloud import build_recall_cloud_warp, build_multisrc_warp, build_backward_warp
            _cw = geo_cloud_warp_cfg
            _cw_render_mode = str(getattr(_cw, "render_mode", "multisrc"))
            _cw_nsrc = int(getattr(_cw, "nsrc", 8))
            _cw_nearby_win = int(getattr(_cw, "nearby_window", 16))
            _cw_ms_splat = int(getattr(_cw, "multisrc_splat", 1))
            _cw_dens_thresh = float(getattr(_cw, "dens_thresh", 0.45))
            _cw_dens_win = int(getattr(_cw, "dens_win", 7))
            _cw_recall_min_cov = float(getattr(_cw, "recall_min_cov", 0.5))
            _cw_recall_margin = float(getattr(_cw, "recall_margin", 0.15))
            _cw_bw_fill_iters = int(getattr(_cw, "bw_fill_iters", 12))

            _cw_zbuf_despeckle = bool(getattr(_cw, "zbuf_despeckle", False))
            _cw_zbuf_despeckle_ksize = int(getattr(_cw, "zbuf_despeckle_ksize", 3))
            _cw_zbuf_despeckle_fill_iters = int(getattr(_cw, "zbuf_despeckle_fill_iters", 4))

            _cw_render_mode_mix_prob_zbuf = float(getattr(_cw, "render_mode_mix_prob_zbuf", 0.0))


            from evoke.utils.train_config import resolve_cloud_warp, vigeo_opts_from_cfg
            _est_cfg = resolve_cloud_warp(_cw)
            _da3_est = _get_da3_estimator(
                int(_est_cfg.get("da3_process_res", 644)), device,
                weights=_est_cfg.get("da3_weights"), backend=_est_cfg.get("depth_backend", "da3"),
                src=_est_cfg.get("da3_src"), vigeo_opts=vigeo_opts_from_cfg(_est_cfg))
            _cw_ingest_n = int(getattr(_cw, "update_frames_per_chunk", 12))
            _cw_splat = int(getattr(_cw, "splat_radius", 2))
            _cw_n_tframe = int(getattr(_cw, "n_tframe", 6))
            _cw_grid_div = int(getattr(_cw, "recall_grid_div", 8))
            _cw_mask_pts = int(getattr(_cw, "recall_mask_pts", 8000))
            _cw_conf_pct = float(getattr(_cw, "conf_percentile", 30.0))
            _cw_recall_k_default = int(getattr(_cw, "recall_k", 12))
            _cw_n_nearby_default = int(getattr(_cw, "n_nearby", 4))

            def _choices_probs(samp, default):
                ch = list(getattr(samp, "choices", []) or []) if samp is not None else []
                if not ch:
                    return [int(default)], "uniform"
                return ch, getattr(samp, "probs", "uniform")
            _cw_lag_choices, _cw_lag_probs = _choices_probs(getattr(_cw, "lag_sampling", None), 1)
            _cw_hist_choices, _cw_hist_probs = _choices_probs(getattr(_cw, "history_chunks_sampling", None), 16)
            _cw_rk_choices, _cw_rk_probs = _choices_probs(getattr(_cw, "recall_k_sampling", None), _cw_recall_k_default)
            _cw_nn_choices, _cw_nn_probs = _choices_probs(getattr(_cw, "n_nearby_sampling", None), _cw_n_nearby_default)

            def _sample_choice(choices, probs):
                if isinstance(probs, str) or probs is None:
                    return int(_random_mode.choice(choices))
                return int(_random_mode.choices(choices, weights=[float(p) for p in probs], k=1)[0])
        for b in range(B):

            latent_start = choice_idx_per_sample[b] * latent_window_size
            pix_start = latent_start * 4
            target_poses_b = target_pose_c2ws[b].to(dtype=torch.float32)


            if (
                geo_pose_jitter_cfg is not None
                and bool(getattr(geo_pose_jitter_cfg, "enabled", False))
                and random.random() < float(getattr(geo_pose_jitter_cfg, "prob", 0.0))
            ):
                _tp = target_poses_b

                _t = _tp[:, :3, 3].detach().to(torch.float64)
                _mean_mot = (
                    float(torch.median(torch.linalg.norm(_t[1:] - _t[:-1], dim=1)).item())
                    if _tp.shape[0] >= 2 else 0.0
                )
                _DT = _sample_warp_pose_jitter_DT(geo_pose_jitter_cfg, mean_interframe_trans=_mean_mot).to(
                    device=_tp.device, dtype=_tp.dtype)

                target_poses_b = _tp.clone() @ _DT

            if _cloud_enabled and _geo_mode != "i2v":

                assert lingbot_c2ws_full is not None and lingbot_Ks_full is not None, (
                    "cloud_warp requires lingbot_c2ws + lingbot_Ks (GT trajectory + pixel intrinsics)"
                )
                first_frame_pix = raw_video[b:b+1, :, pix_start].to(dtype=torch.float32)
                _lag = _sample_choice(_cw_lag_choices, _cw_lag_probs)
                _hist = _sample_choice(_cw_hist_choices, _cw_hist_probs)
                _recall_k = _sample_choice(_cw_rk_choices, _cw_rk_probs)
                _n_nearby = _sample_choice(_cw_nn_choices, _cw_nn_probs)

                _kn = lingbot_Ks_full[b].cpu().numpy()
                _Kpix = np.array([[_kn[0], 0.0, _kn[2]], [0.0, _kn[1], _kn[3]], [0.0, 0.0, 1.0]], dtype=np.float32)


                _rm_sample = _cw_render_mode
                if _cw_render_mode_mix_prob_zbuf > 0:
                    _rm_sample = "backward_zbuf" if torch.rand(1).item() < _cw_render_mode_mix_prob_zbuf else "backward"
                if _rm_sample in ("backward", "backward_zbuf"):


                    warp_video_b, visibility_mask_b = build_backward_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        lag=_lag, history=_hist, nearby=_cw_nearby_win, fill_iters=_cw_bw_fill_iters,
                        recall_min_cov=_cw_recall_min_cov, recall_margin=_cw_recall_margin,
                        render_mode=_rm_sample,
                        zbuf_despeckle=_cw_zbuf_despeckle, zbuf_despeckle_ksize=_cw_zbuf_despeckle_ksize,
                        zbuf_despeckle_fill_iters=_cw_zbuf_despeckle_fill_iters, device=device)
                elif _cw_render_mode == "multisrc":
                    warp_video_b, visibility_mask_b = build_multisrc_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        lag=_lag, history=_hist, nsrc=_cw_nsrc, nearby=_cw_nearby_win,
                        splat_radius=_cw_ms_splat, dens_thresh=_cw_dens_thresh, dens_win=_cw_dens_win,
                        recall_min_cov=_cw_recall_min_cov, recall_margin=_cw_recall_margin, device=device)
                else:
                    warp_video_b, visibility_mask_b = build_recall_cloud_warp(
                        _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                        _Kpix, target_poses_b,
                        pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                        height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n,
                        recall_k=_recall_k, n_nearby=_n_nearby, lag=_lag, history=_hist,
                        n_tframe=_cw_n_tframe, grid_div=_cw_grid_div, mask_pts=_cw_mask_pts,
                        conf_pct=_cw_conf_pct, splat_radius=_cw_splat, device=device)
                warp_video_b = warp_video_b.to(device=device, dtype=torch.float32)
                visibility_mask_b = visibility_mask_b.to(device=device, dtype=torch.float32)
            elif _geo_mode == "i2v":


                assert _cloud_enabled and lingbot_c2ws_full is not None and lingbot_Ks_full is not None, (
                    "i2v DA3 single-source warp requires cloud_warp.enabled=true + lingbot_c2ws_full + lingbot_Ks_full"
                )
                from evoke.modules.geometric_state.da3_cloud import build_single_source_warp
                first_frame_pix = raw_video[b:b+1, :, pix_start].to(dtype=torch.float32)
                _kn = lingbot_Ks_full[b].cpu().numpy()
                _Kpix = np.array([[_kn[0], 0.0, _kn[2]], [0.0, _kn[1], _kn[3]], [0.0, 0.0, 1.0]], dtype=np.float32)
                warp_video_b, visibility_mask_b = build_single_source_warp(
                    _da3_est, raw_video[b].to(torch.float32), lingbot_c2ws_full[b].to(torch.float32),
                    _Kpix, target_poses_b,
                    pix_start=int(pix_start), pix_stride=int(_geo_pix_stride), window_pix=int(pix_window_len),
                    height=int(H_pix), width=int(W_pix), ingest_n=_cw_ingest_n, splat_radius=_cw_ms_splat, device=device)
                warp_video_b = warp_video_b.to(device=device, dtype=torch.float32)
                visibility_mask_b = visibility_mask_b.to(device=device, dtype=torch.float32)
            else:
                raise RuntimeError(
                    "[GEO] warp requires cloud_warp.enabled=true (DA3 backend); the Pi3X mirror path has been removed."
                )


            if bool(geo_keep_clean_anchor):
                _anc = first_frame_pix.to(device=device)
                _anc = _anc[0:1] if _anc.ndim == 4 else _anc.unsqueeze(0)
                if tuple(_anc.shape[-2:]) != tuple(warp_video_b.shape[-2:]):
                    _anc = torch.nn.functional.interpolate(
                        _anc, size=tuple(warp_video_b.shape[-2:]), mode="bilinear", align_corners=False
                    )
                warp_video_b[:, :, 0] = _anc.to(dtype=warp_video_b.dtype).clamp(-1, 1)
                visibility_mask_b[:, :, 0] = 1.0


            source_pix_5d = first_frame_pix.unsqueeze(2) if first_frame_pix.ndim == 4 else first_frame_pix.unsqueeze(0).unsqueeze(2)
            with torch.no_grad():
                source_lat_dist = vae.encode(source_pix_5d.to(dtype=vae.dtype)).latent_dist
                source_lat_b = source_lat_dist.sample()
            source_lat_b = ((source_lat_b - latents_mean) * latents_std).to(dtype=torch.float32)
            source_latent_list.append(source_lat_b)


            _cw_warm_encode = bool(getattr(_cw, "warp_warm_encode", False))
            with torch.no_grad():
                if _cw_warm_encode:
                    _vt = 4
                    _min_f = (int(latent_window_size) - 1) * _vt + 1
                    _vid = warp_video_b[:, :, -_min_f:]
                    _warm = _vid[:, :, :1].repeat(1, 1, _vt, 1, 1)
                    _padded = torch.cat([_warm, _vid], dim=2).to(dtype=vae.dtype)
                    w_lat = vae.encode(_padded).latent_dist.sample()[:, :, 1:]
                else:
                    w_lat = vae.encode(warp_video_b.to(dtype=vae.dtype)).latent_dist.sample()
            w_lat = (w_lat - latents_mean) * latents_std
            w_lat = w_lat.to(dtype=torch.float32)


            if w_lat.shape[2] > latent_window_size:
                w_lat = w_lat[:, :, -latent_window_size:]


            if (
                geo_warp_error_inject_enabled
                and recycle_vars is not None
                and args is not None
                and random.random() < float(geo_warp_error_prob)
            ):
                with torch.no_grad():
                    _geo_inject_warp_error(args, recycle_vars, w_lat)


            vis_lat = _geo_resize_visibility_to_latent(
                visibility_mask_b, num_lat_per_chunk=latent_window_size,
                H_lat=H_lat, W_lat=W_lat, vae_t_stride=4,
            )
            w_lat = _geo_add_noise_to_warp_latents(
                w_lat,
                device=device,
                sigma_min=float(geo_sigma_visible_min),
                sigma_max=float(geo_sigma_visible_max),
                visibility_aware_noise=bool(geo_visibility_aware_noise),
                sigma_invisible=float(geo_sigma_invisible),
                visibility_mask_lat=vis_lat,
            )

            warp_lat_list.append(w_lat)
            warp_mask_list.append(vis_lat)
        warp_video_latents = torch.cat(warp_lat_list, dim=0).to(dtype=weight_dtype)
        warp_visibility_mask = torch.cat(warp_mask_list, dim=0).to(dtype=torch.float32)

        geo_source_image_latent = torch.cat(source_latent_list, dim=0).to(dtype=weight_dtype)


    if use_geometric_state and _is_skill:
        W = int(latent_window_size)
        warp_video_latents = torch.zeros(B, C_lat, W, H_lat, W_lat, device=device, dtype=weight_dtype)
        warp_visibility_mask = torch.zeros(B, 1, W, H_lat, W_lat, device=device, dtype=torch.float32)
        _src_list = []
        for b in range(B):
            _head = raw_video[b:b + 1, :, 0:1]
            with torch.no_grad():
                _sl = vae.encode(_head.to(dtype=vae.dtype)).latent_dist.sample()
            _sl = ((_sl - latents_mean) * latents_std).to(dtype=torch.float32)
            _src_list.append(_sl)
        geo_source_image_latent = torch.cat(_src_list, dim=0).to(dtype=weight_dtype)

    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_masks": prompt_attention_mask,
        "x0_latents": x0_latents,
        "history_latents": history_latents,
        "target_latents": target_latents,
        "clean_all_latents": None,

        "target_pose_Ks": target_pose_Ks,
        "target_pose_c2ws": target_pose_c2ws,

        "warp_video_latents": warp_video_latents,
        "warp_visibility_mask": warp_visibility_mask,

        "geo_source_image_latent": geo_source_image_latent,


        "geo_condition_mode": _geo_mode if use_geometric_state else None,


        "is_skill": _is_skill,
        "uttid": batch.get("uttid", []),
        "dataset_name": batch.get("dataset_name", []),
        "bucket_key": bkey,
        "num_frame": num_frame,
        "height": height,
        "width": width,
    }

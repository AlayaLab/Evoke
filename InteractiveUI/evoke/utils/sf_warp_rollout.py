

import torch


class SFWarpRollout:


    def __init__(
        self,
        geo_cfg,
        vae,
        target_pose_Ks,
        target_pose_c2ws,
        prefix_latents,
        latent_window_size: int,
        num_rollout_sections: int,
        warp_tail_chunks: int,
        num_score_sections: int,
        height_px: int,
        width_px: int,
        device,
    ):
        import numpy as np
        from evoke.modules.geometric_state.da3_cloud import DA3FrameBank
        from evoke.dataset.online_materialize import _get_da3_estimator

        assert target_pose_c2ws is not None and target_pose_Ks is not None, \
            "[warp-rollout] warp inside the rollout needs GT pose (passed through from the data side)"


        assert str(getattr(geo_cfg, "warp_rope_mode", "overlap_noise")) == "overlap_noise", \
            "[warp-rollout] this helper only implements warp_rope_mode=overlap_noise"
        assert bool(getattr(geo_cfg, "rope_alignment", True)), \
            "[warp-rollout] this helper only implements rope_alignment=true (warp idx = noise idx)"
        assert str(getattr(geo_cfg, "prefix_idx_mode", "zero")) == "zero", \
            "[warp-rollout] this helper only implements prefix_idx_mode=zero"
        assert not bool(getattr(geo_cfg, "warp_rope_noise_center_align", False)), \
            "[warp-rollout] center_align is not implemented (and with plucker off it collapses into blocky artefacts)"
        assert not bool(getattr(geo_cfg, "warp_keep_clean_anchor", False)), \
            "[warp-rollout] warp_keep_clean_anchor is not implemented"
        _cw_chk = getattr(geo_cfg, "cloud_warp", None)
        assert not bool(getattr(_cw_chk, "zbuf_despeckle", False) if _cw_chk is not None else False), \
            "[warp-rollout] zbuf_despeckle is not implemented (the mirrored inference recipe has despeckle off anyway)"
        self.cfg = geo_cfg
        self.vae = vae
        self.device = device
        self.win = int(latent_window_size)
        self.N = int(num_rollout_sections)
        self.K = min(int(warp_tail_chunks), self.N)
        self.W = min(int(num_score_sections), self.K)
        self.vae_t = 4
        self.sec_px = (self.win - 1) * self.vae_t + 1


        self.P_lat = int(prefix_latents.shape[2])
        self.P = self.P_lat // self.win

        _cw = getattr(geo_cfg, "cloud_warp", None)
        _get = (lambda k, d: getattr(_cw, k, d) if _cw is not None else d)

        kn = target_pose_Ks.detach().float().cpu().numpy().reshape(-1)[:4]
        fx, fy, cx, cy = float(kn[0]), float(kn[1]), float(kn[2]), float(kn[3])
        src_w = max(2.0 * cx, 1.0); src_h = max(2.0 * cy, 1.0)
        sx = float(width_px) / src_w; sy = float(height_px) / src_h
        self.K_pix = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], np.float32)
        self.c2w_full = target_pose_c2ws.detach().float()
        self.H, self.W_px = int(height_px), int(width_px)


        from evoke.utils.train_config import resolve_cloud_warp, vigeo_opts_from_cfg
        from evoke.modules.geometric_state.depth_backend import reset_stream as _reset_depth_stream
        _est_cfg = resolve_cloud_warp(_cw)
        self.est = _get_da3_estimator(
            int(_est_cfg.get("da3_process_res", 644)), device,
            weights=_est_cfg.get("da3_weights"), backend=_est_cfg.get("depth_backend", "da3"),
            src=_est_cfg.get("da3_src"), vigeo_opts=vigeo_opts_from_cfg(_est_cfg))
        _reset_depth_stream(self.est)
        self.bank = DA3FrameBank(device=device, conf_percentile=30.0)
        self.ingest_n = int(_get("update_frames_per_chunk", 12))
        self.lag = 0
        self.render_mode = str(_get("render_mode", "backward_zbuf"))
        self.bw_fill_iters = int(_get("bw_fill_iters", 12))
        self.nsrc = int(_get("nsrc", 8)); self.nearby = int(_get("nearby_window", 16))
        self.ms_splat = int(_get("multisrc_splat", 1))
        self.dens_thresh = float(_get("dens_thresh", 0.45)); self.dens_win = int(_get("dens_win", 7))
        self.recall_min_cov = float(_get("recall_min_cov", 0.5)); self.recall_margin = float(_get("recall_margin", 0.15))

        self.sigma_min = float(getattr(geo_cfg, "warp_noise_sigma_min", 0.0))
        self.sigma_max = float(getattr(geo_cfg, "warp_noise_sigma_max", 0.135))
        self.sigma_invisible = float(getattr(geo_cfg, "warp_noise_sigma_invisible", 1.0))
        self.visibility_aware = bool(getattr(geo_cfg, "visibility_aware_noise", True))
        self.vis_token_threshold = float(getattr(geo_cfg, "visible_token_threshold", 0.5))
        self.warp_stage0_only = bool(getattr(geo_cfg, "warp_stage0_only", False))

        self._seed_prefix(prefix_latents)


    def is_tail_section(self, k: int) -> bool:
        return k >= self.N - self.K

    def is_warp_section(self, k: int) -> bool:
        return k >= self.N - self.W

    def _sec_pix_start(self, k: int) -> int:

        return (self.P_lat + k * self.win) * self.vae_t

    @torch.no_grad()
    def _decode_latents_to_px(self, latents):

        vae = self.vae
        lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        vae.clear_cache()
        px = vae.decode(latents.to(vae.device, vae.dtype) / ls + lm, return_dict=False)[0]
        vae.clear_cache()
        return px.clamp(-1, 1).float()

    @torch.no_grad()
    def _ingest_px(self, frames_px, global_pix_start: int):


        f = frames_px[0]
        nloc = int(f.shape[1])
        if nloc < 3:
            return
        loc = torch.unique(torch.linspace(0, nloc - 1, self.ingest_n).round().long().clamp(0, nloc - 1))
        if loc.numel() < 3:
            return
        gids = [int(global_pix_start) + int(i) for i in loc.tolist()]
        pose_idx = torch.tensor([min(g, self.c2w_full.shape[1] - 1) for g in gids])
        c2w = self.c2w_full[0, pose_idx].cpu().numpy()
        frames01 = (f[:, loc].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
        self.bank.ingest(self.est, frames01, c2w, self.K_pix, gids)

    @torch.no_grad()
    def _seed_prefix(self, prefix_latents):


        px = self._decode_latents_to_px(prefix_latents)
        self._ingest_px(px, 0)

    @torch.no_grad()
    def ingest_section(self, k: int, pred_x0_latents):

        if not self.is_tail_section(k):
            return
        px = self._decode_latents_to_px(pred_x0_latents.detach())
        self._ingest_px(px, self._sec_pix_start(k))

    @torch.no_grad()
    def render_warp_latents(self, k: int, generator=None):


        from evoke.modules.geometric_state.da3_cloud import (
            _render_backward, _render_backward_multisrc_zbuf, _render_multisrc)
        from evoke.dataset.online_materialize import _geo_resize_visibility_to_latent

        p0 = self._sec_pix_start(k)
        pose_idx = torch.arange(p0, p0 + self.sec_px).clamp(max=self.c2w_full.shape[1] - 1)
        tgt = self.c2w_full[0, pose_idx].to(device=self.device, dtype=torch.float32)

        _keys = list(self.bank.frames.keys())
        stride = self.win * self.vae_t
        if _keys:
            _cutoff = max(_keys) - self.lag * stride
            pool = [g for g in _keys if g <= _cutoff]
        else:
            pool = []


        if not pool:
            print(f"[warp-rollout][warn] section {k} has an empty warp pool (bank={len(_keys)}; ingest skipped for degenerate motion?) "
                  f"-> fully invisible warp, i.e. pure noise tokens", flush=True)
            warp_video = torch.zeros(1, 3, self.sec_px, self.H, self.W_px, device=self.device)
            vis_mask = torch.zeros(1, 1, self.sec_px, self.H, self.W_px, device=self.device)
            return self._encode_warp_with_visibility(warp_video, vis_mask, generator)
        store = {g: self.bank.frames[g] for g in pool}

        if self.render_mode in ("backward", "backward_zbuf"):
            _render_bw = _render_backward_multisrc_zbuf if self.render_mode == "backward_zbuf" else _render_backward
            warp_video, vis_mask = _render_bw(
                store, pool, tgt, self.K_pix, self.H, self.W_px,
                nearby=self.nearby, fill_iters=self.bw_fill_iters,
                recall_min_cov=self.recall_min_cov, recall_margin=self.recall_margin, device=self.device)
        else:
            warp_video, vis_mask = _render_multisrc(
                store, pool, tgt, self.K_pix, self.H, self.W_px,
                nsrc=self.nsrc, nearby=self.nearby, splat_radius=self.ms_splat,
                dens_thresh=self.dens_thresh, dens_win=self.dens_win,
                recall_min_cov=self.recall_min_cov, recall_margin=self.recall_margin, device=self.device)
        warp_video = warp_video.to(device=self.device, dtype=torch.float32)
        vis_mask = vis_mask.to(device=self.device, dtype=torch.float32)
        return self._encode_warp_with_visibility(warp_video, vis_mask, generator)

    @torch.no_grad()
    def _encode_warp_with_visibility(self, warp_video, vis_mask, generator=None):

        from evoke.dataset.online_materialize import _geo_resize_visibility_to_latent
        vae = self.vae
        lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        vae.clear_cache()
        wl = vae.encode(warp_video[:, :, -self.sec_px:].to(vae.device, vae.dtype)).latent_dist.sample(generator=generator)
        vae.clear_cache()
        warp_latents = ((wl - lm) * ls).float()
        assert int(warp_latents.shape[2]) == self.win, \
            f"[warp-rollout] warp latent frame count {int(warp_latents.shape[2])} != win {self.win}"


        h_lat, w_lat = int(warp_latents.shape[3]), int(warp_latents.shape[4])
        mask_lat = _geo_resize_visibility_to_latent(
            vis_mask.to(device=self.device, dtype=torch.float32), self.win, h_lat, w_lat, vae_t_stride=self.vae_t,
        ).to(dtype=warp_latents.dtype, device=warp_latents.device)


        frame_sigmas = (torch.rand(self.win, device=warp_latents.device)
                        * (self.sigma_max - self.sigma_min) + self.sigma_min).to(warp_latents.dtype)
        noise = torch.randn_like(warp_latents)
        if self.visibility_aware:
            sig = mask_lat * frame_sigmas.view(1, 1, self.win, 1, 1) + (1.0 - mask_lat) * self.sigma_invisible
            warp_latents = sig * noise + (1.0 - sig) * warp_latents
        else:
            fs = frame_sigmas.view(1, 1, self.win, 1, 1)
            warp_latents = fs * noise + (1.0 - fs) * warp_latents
        return warp_latents, mask_lat

    @torch.no_grad()
    def build_warp_short_tier(self, k, latents_prefix, latents_history_1x,
                              indices_hidden_states, indices_prefix, indices_1x, generator=None):


        warp_latents, mask_lat = self.render_warp_latents(k, generator=generator)
        short = torch.cat([
            latents_prefix.to(warp_latents.device, warp_latents.dtype),
            warp_latents,
            latents_history_1x.to(warp_latents.device, warp_latents.dtype)], dim=2)
        idx_warp = indices_hidden_states.clone()
        idx_short = torch.cat([indices_prefix, idx_warp, indices_1x], dim=0)
        ones = torch.ones_like(mask_lat[:, :, :1])
        mask_short = torch.cat([ones, mask_lat, ones], dim=2)
        attn_kwargs = {
            "history_visible_token_threshold": self.vis_token_threshold,
            "geo_warp_frames": self.win,
            "geo_prev_short_frames": 1,
            "geo_warp_stage0_only": self.warp_stage0_only,
        }
        return short, idx_short, mask_short, attn_kwargs

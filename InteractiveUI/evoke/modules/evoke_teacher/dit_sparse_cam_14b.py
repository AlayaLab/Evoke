

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List
from einops import rearrange


from .dit_sparse_14b import (
    DiTBlock, WanModel,
    flash_attention, RMSNorm, AttentionModule,
    SelfAttention, CrossAttention, GateModule, LinearAttention,
    Head, MLP,
    modulate, sinusoidal_embedding_1d,
    precompute_freqs_cis_3d, precompute_freqs_cis, rope_apply,
    _scale_to_tokens,
)


@torch.no_grad()
def compute_plucker_embedding(cam_c2w, intrinsic, height, width):


    F_frames = cam_c2w.shape[0]
    device = cam_c2w.device
    dtype = cam_c2w.dtype


    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) + 0.5
    grid_xy = grid_xy.unsqueeze(0).expand(F_frames, -1, -1)


    fx = intrinsic[0, 0] * width
    fy = intrinsic[1, 1] * height
    cx = intrinsic[0, 2] * width
    cy = intrinsic[1, 2] * height


    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    zs = torch.ones_like(xs)

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)


    rays_d = directions @ cam_c2w[:, :3, :3].transpose(-1, -2)
    rays_o = cam_c2w[:, :3, 3].unsqueeze(1).expand_as(rays_d)

    plucker = torch.cat([rays_o, rays_d], dim=-1)
    plucker = plucker.view(F_frames, height, width, 6)
    return plucker


@torch.no_grad()
def compute_plucker_embedding_from_Ks(cam_c2w, Ks, height, width):


    F_frames = cam_c2w.shape[0]
    device = cam_c2w.device
    dtype = cam_c2w.dtype


    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) + 0.5
    grid_xy = grid_xy.unsqueeze(0).expand(F_frames, -1, -1)


    fx, fy, cx, cy = Ks.chunk(4, dim=-1)


    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    zs = torch.ones_like(xs)

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)


    rays_d = directions @ cam_c2w[:, :3, :3].transpose(-1, -2)
    rays_o = cam_c2w[:, :3, 3].unsqueeze(1).expand_as(rays_d)

    plucker = torch.cat([rays_o, rays_d], dim=-1)
    plucker = plucker.view(F_frames, height, width, 6)
    return plucker


def prepare_plucker_for_model(plucker, vae_stride_t=4, vae_stride_h=8, vae_stride_w=8,
                              subsample_temporal=True):


    F_in = plucker.shape[0]


    plucker = rearrange(
        plucker,
        'f (h c1) (w c2) c -> f h w (c c1 c2)',
        c1=vae_stride_h, c2=vae_stride_w
    )

    if subsample_temporal:

        F_lat = (F_in - 1) // vae_stride_t + 1
        temporal_indices = torch.arange(F_lat, device=plucker.device) * vae_stride_t
        temporal_indices = temporal_indices.clamp(max=F_in - 1)
        plucker = plucker[temporal_indices]


    plucker = rearrange(plucker, 'f h w c -> 1 c f h w')
    return plucker


def build_cam_plucker_emb(c2w_abs, Ks_pixel, height, width, vae_stride_t=4, framewise=None):


    from diffsynth.core.data.operators import (
        interpolate_camera_poses, compute_relative_poses_lingbot)
    if not torch.is_tensor(c2w_abs):
        c2w_abs = torch.as_tensor(c2w_abs, dtype=torch.float32)
    c2w_abs = c2w_abs.float()
    N = c2w_abs.shape[0]
    lat_f = (N - 1) // vae_stride_t + 1
    c2w_lat = interpolate_camera_poses(c2w_abs, lat_f)


    if framewise is None:
        import os
        framewise = os.environ.get("CAM_PLUCKER_FRAMEWISE", "0") == "1"
    rel = compute_relative_poses_lingbot(c2w_lat, framewise=framewise, normalize_trans=True)
    Ks = torch.as_tensor(Ks_pixel, dtype=torch.float32).reshape(-1)[:4]
    Ks = Ks.unsqueeze(0).expand(lat_f, -1)
    plucker = compute_plucker_embedding_from_Ks(rel, Ks, height, width)
    return prepare_plucker_for_model(plucker, subsample_temporal=False)


CAM_ACTION_SCALE = (0.389, 0.255, 2.867, 0.040, 0.186, 0.027)


def build_cam_action_6d(c2w_abs, scale=None, vae_stride_t=4):


    from diffsynth.core.data.operators import interpolate_camera_poses
    from scipy.spatial.transform import Rotation as _R
    if not torch.is_tensor(c2w_abs):
        c2w_abs = torch.as_tensor(c2w_abs, dtype=torch.float32)
    c2w_abs = c2w_abs.float()
    N = c2w_abs.shape[0]
    lat_f = (N - 1) // vae_stride_t + 1
    c2w_lat = interpolate_camera_poses(c2w_abs, lat_f).cpu().numpy().astype(np.float64)
    act = np.zeros((lat_f, 6), dtype=np.float32)
    for i in range(1, lat_f):
        rel = np.linalg.inv(c2w_lat[i - 1]) @ c2w_lat[i]
        t_rel = rel[:3, 3]
        r_rel = _R.from_matrix(rel[:3, :3]).as_euler("xyz")
        act[i] = np.concatenate([t_rel, r_rel])
    sc = np.asarray(scale if scale is not None else CAM_ACTION_SCALE, dtype=np.float32)
    act = act / np.maximum(sc, 1e-6)
    return torch.from_numpy(act)


class ActionAdaLNEmbedder(nn.Module):


    def __init__(self, dim, action_dim=6, freq_dim_per_axis=32, freq_scale=1000.0):
        super().__init__()
        self.action_dim = action_dim
        self.freq_dim_per_axis = freq_dim_per_axis
        self.freq_scale = freq_scale
        in_dim = action_dim * freq_dim_per_axis
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def _sinusoidal(self, x):

        half = self.freq_dim_per_axis // 2
        freqs = torch.exp(-math.log(self.freq_scale)
                          * torch.arange(half, device=x.device, dtype=torch.float32) / max(half, 1))
        ang = x.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return emb.flatten(-2)

    def forward(self, action):
        e = self._sinusoidal(action)
        return self.mlp(e.to(self.mlp[0].weight.dtype))


class CamDiTBlock(DiTBlock):


    def __init__(self, *args, cam_ctrl=False, cam_rank=128, cam_full=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam_ctrl = cam_ctrl
        self.cam_rank = cam_rank
        self.cam_full = cam_full
        if cam_ctrl and cam_full:


            self.cam_injector_layer1 = nn.Linear(self.dim, self.dim)
            self.cam_injector_layer2 = nn.Linear(self.dim, self.dim)
            self.cam_scale_layer = nn.Linear(self.dim, self.dim)
            self.cam_shift_layer = nn.Linear(self.dim, self.dim)


            for _m in (self.cam_injector_layer1, self.cam_injector_layer2,
                       self.cam_scale_layer, self.cam_shift_layer):
                nn.init.zeros_(_m.weight)
                nn.init.zeros_(_m.bias)
        elif cam_ctrl:

            self.cam_scale_down_proj = nn.Linear(self.dim, cam_rank, bias=False)
            self.cam_scale_up_proj = nn.Linear(cam_rank, self.dim, bias=True)
            self.cam_shift_down_proj = nn.Linear(self.dim, cam_rank, bias=False)
            self.cam_shift_up_proj = nn.Linear(cam_rank, self.dim, bias=True)


            nn.init.zeros_(self.cam_scale_down_proj.weight)
            nn.init.zeros_(self.cam_scale_up_proj.weight)
            nn.init.zeros_(self.cam_scale_up_proj.bias)
            nn.init.zeros_(self.cam_shift_down_proj.weight)
            nn.init.zeros_(self.cam_shift_up_proj.weight)
            nn.init.zeros_(self.cam_shift_up_proj.bias)

    def forward(
        self, x, context, t_mod, freqs,
        tokens_per_frame: int = None,
        spatial_hw: tuple = None,
        **kwargs,
    ):


        cam_plucker_emb = kwargs.get('cam_plucker_emb', None)


        if self.training and self.inner_checkpoint:
            linear_attn_out, state, z = torch.utils.checkpoint.checkpoint(
                self.linear_attn, self.linear_attn_norm(x), use_reentrant=False)
        else:
            linear_attn_out, state, z = self.linear_attn(self.linear_attn_norm(x))


        _sp_active = 'sp_num_frames_global' in kwargs
        if _sp_active:
            from .sp_runtime import allreduce_sum, allgather_frames_no_grad, get_sp_frame_info
            la = self.linear_attn

            _sp_nfg = kwargs['sp_num_frames_global']
            _sp_offset = kwargs.get('sp_frame_offset', 0)
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(_sp_nfg)
            _at_s = (_orig_f_start - _sp_offset) * self.per_frame_tokens
            _at_e = (_orig_f_end - _sp_offset) * self.per_frame_tokens

            _x_assigned = self.linear_attn_norm(x[:, _at_s:_at_e])
            _B_a, _N_a = _x_assigned.shape[:2]
            _k_a = la.feature_map(
                la.norm_k(la.k(_x_assigned)).view(_B_a, _N_a, la.num_heads, la.head_dim).transpose(1, 2))
            _v_a = la.v(_x_assigned).view(_B_a, _N_a, la.num_heads, la.head_dim).transpose(1, 2)
            state = torch.einsum('bhnd,bhnv->bhdv', _k_a, _v_a)
            z = _k_a.sum(dim=2)
            del _k_a, _v_a, _x_assigned

            state = allreduce_sum(state)
            z = allreduce_sum(z)
            q_la = la._cached_q_mapped
            del la._cached_q_mapped
            qkv = torch.einsum('bhnd,bhdv->bhnv', q_la, state)
            qk_sum = torch.einsum('bhnd,bhd->bhn', q_la, z).unsqueeze(-1) + la.eps
            del q_la
            la_out = (qkv / qk_sum).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], la.inner_dim)
            del qkv, qk_sum
            linear_attn_out = la.o(la_out)


        B_la = x.shape[0]
        pf_la = self.per_frame_tokens
        num_frames_la = x.shape[1] // pf_la
        _fk_pooled = linear_attn_out.view(B_la, num_frames_la, pf_la, -1).mean(dim=2)
        frame_keys_local = self.chunk_to_state_proj(_fk_pooled)


        if _sp_active:
            from .sp_runtime import get_sp_size, get_sp_group, get_sp_frame_info
            import torch.distributed as dist
            sp_size = get_sp_size()
            _num_frames_global = kwargs.get('sp_num_frames_global', num_frames_la)
            _sp_frame_offset = kwargs.get('sp_frame_offset', 0)

            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(_num_frames_global)
            _local_assigned_start = _orig_f_start - _sp_frame_offset
            _local_assigned_end = _orig_f_end - _sp_frame_offset
            frame_keys_assigned = frame_keys_local[:, _local_assigned_start:_local_assigned_end]

            if frame_keys_assigned.shape[1] < _fpr:
                _pad = torch.zeros(B_la, _fpr - frame_keys_assigned.shape[1], frame_keys_assigned.shape[-1],
                                   device=frame_keys_assigned.device, dtype=frame_keys_assigned.dtype)
                frame_keys_padded = torch.cat([frame_keys_assigned, _pad], dim=1)
            else:
                frame_keys_padded = frame_keys_assigned

            gathered = [torch.zeros_like(frame_keys_padded) for _ in range(sp_size)]
            dist.all_gather(gathered, frame_keys_padded.contiguous(), group=get_sp_group())
            frame_keys = torch.cat(gathered, dim=1)
            frame_keys = frame_keys[:, :_num_frames_global]
        else:
            frame_keys = frame_keys_local


        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)


        if self.sparse_attn:
            attn_out = self.sparse_self_attention(
                input_x, freqs, frame_keys, tokens_per_frame,
                spatial_hw, state, z,
                sp_frame_offset=kwargs.get('sp_frame_offset', 0),
                sp_num_frames_global=kwargs.get('sp_num_frames_global', None),
                freqs_full=kwargs.get('freqs_full', None),
                freqs_3d=kwargs.get('freqs_3d', None),
                select_gate_t_frac=kwargs.get('select_gate_t_frac', None))
            x = self.gate(x, gate_msa, attn_out)
        else:
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))


        if self.cam_ctrl and cam_plucker_emb is not None:

            if self.cam_full:

                c2ws_hidden = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(cam_plucker_emb)))
                c2ws_hidden = c2ws_hidden + cam_plucker_emb
                cam_scale = self.cam_scale_layer(c2ws_hidden)
                cam_shift = self.cam_shift_layer(c2ws_hidden)
            else:

                cam_scale = self.cam_scale_up_proj(self.cam_scale_down_proj(cam_plucker_emb))
                cam_shift = self.cam_shift_up_proj(self.cam_shift_down_proj(cam_plucker_emb))
            x = (1.0 + cam_scale) * x + cam_shift


        segment_contexts_encoded = kwargs.get('segment_contexts_encoded', None)
        chunk_context_map = kwargs.get('chunk_context_map', None)
        if segment_contexts_encoded is not None and chunk_context_map is not None:
            pf = self.per_frame_tokens
            num_frames = x.shape[1] // pf
            chunk_f = self.chunk_size
            num_chunks = (num_frames + chunk_f - 1) // chunk_f
            cbs = getattr(self, 'chunk_batch_size', None) or num_chunks
            x_normed = self.norm3(x)
            cross_out = torch.zeros_like(x)
            B, _, D = x.shape
            for batch_start in range(0, num_chunks, cbs):
                batch_end = min(batch_start + cbs, num_chunks)
                nb = batch_end - batch_start
                q_list, ranges, seg_indices = [], [], []
                for ci in range(batch_start, batch_end):
                    t_s = ci * chunk_f * pf
                    t_e = min((ci + 1) * chunk_f, num_frames) * pf
                    q_list.append(x_normed[:, t_s:t_e])
                    ranges.append((t_s, t_e))
                    seg_indices.append(int(chunk_context_map[ci]))

                max_q_len = max(q.shape[1] for q in q_list)
                q_batch = torch.zeros(nb * B, max_q_len, D, device=x.device, dtype=x.dtype)
                q_lens = []
                for i, q in enumerate(q_list):
                    ql = q.shape[1]
                    q_batch[i*B:(i+1)*B, :ql] = q
                    q_lens.append(ql)

                seg_idx_t = torch.tensor(seg_indices, device=x.device, dtype=torch.long)
                ctx_batch = segment_contexts_encoded[:, seg_idx_t]
                L_text = ctx_batch.shape[2]
                kv_batch = ctx_batch.permute(1, 0, 2, 3).reshape(nb * B, L_text, D)

                out_batch = self.cross_attn(q_batch, kv_batch)

                for i, (t_s, t_e) in enumerate(ranges):
                    ql = q_lens[i]
                    cross_out[:, t_s:t_e] = out_batch[i*B:(i+1)*B, :ql]
            x = x + cross_out
        else:
            _CHUNK_THRESHOLD = 100000
            _chunk_size = 20000
            if x.shape[1] > _CHUNK_THRESHOLD:
                x_normed = self.norm3(x)
                cross_out = torch.empty_like(x)
                for _i, _c in enumerate(x_normed.split(_chunk_size, dim=1)):
                    _s = _i * _chunk_size
                    cross_out[:, _s:_s + _c.shape[1]] = self.cross_attn(_c, context)
                x = x + cross_out
            else:
                x = x + self.cross_attn(self.norm3(x), context)


        _FFN_CHUNK_THRESHOLD = 100000
        _ffn_chunk_size = 20000
        if x.shape[1] > _FFN_CHUNK_THRESHOLD:
            for _s in range(0, x.shape[1], _ffn_chunk_size):
                _e = min(_s + _ffn_chunk_size, x.shape[1])
                _chunk_in = modulate(self.norm2(x[:, _s:_e]), shift_mlp, scale_mlp)
                x[:, _s:_e] = x[:, _s:_e] + gate_mlp * self.ffn(_chunk_in)
        else:
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            if self.training and self.inner_checkpoint:
                ffn_out = torch.utils.checkpoint.checkpoint(self.ffn, input_x, use_reentrant=False)
            else:
                ffn_out = self.ffn(input_x)
            x = self.gate(x, gate_mlp, ffn_out)
        return x


class WanModelCam(WanModel):


    def __init__(
        self,

        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        sparse_attn: bool = True,
        chunk_size: int = 8,
        overlap_size: int = 1,
        num_global_tokens: int = 8,
        per_frame_tokens: int = 30 * 52,
        num_retained_tokens: int = 1024,
        num_select_frames: int = 4,
        num_nearby_frames: int = 3,
        teacher_dim: int = 1024,
        teacher_config: dict = None,
        chunk_batch_size: int = None,
        inner_checkpoint: bool = False,
        lazy_qkv: bool = False,
        select_scales: list = None,

        select_gate_mode: str = 'none',
        select_gate_kappa: float = 2.0,
        select_gate_cos_floor: float = 0.0,
        select_gate_min_candidates: int = 8,
        select_gate_mad_floor: float = 1e-3,
        select_gate_min_keep: int = 0,

        select_gate_temp: float = 0.6667,
        select_gate_budget_target: float = 0.5,
        select_gate_budget_weight: float = 0.0,

        sink_decay_mode: str = 'none',
        sink_decay_onset: int = 40,
        sink_decay_factor: int = 2,

        cam_ctrl: bool = False,
        cam_ctrl_layers: list = None,
        cam_ctrl_dim: int = 6,
        cam_rank: int = 128,
        cam_full: bool = False,
        cam_input_inject: bool = False,
        cam_adaln: bool = False,

        memory_ctrl: bool = False,
        memory_encoder_out_dim: int = None,
        memory_encoder_num_heads: int = 8,
    ):

        super().__init__(
            dim=dim, in_dim=in_dim, ffn_dim=ffn_dim, out_dim=out_dim,
            text_dim=text_dim, freq_dim=freq_dim, eps=eps, patch_size=patch_size,
            num_heads=num_heads, num_layers=num_layers,
            has_image_input=has_image_input, has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv, add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            sparse_attn=sparse_attn, chunk_size=chunk_size, overlap_size=overlap_size,
            num_global_tokens=num_global_tokens, per_frame_tokens=per_frame_tokens,
            num_retained_tokens=num_retained_tokens,
            num_select_frames=num_select_frames, num_nearby_frames=num_nearby_frames,
            teacher_dim=teacher_dim, teacher_config=teacher_config,
            chunk_batch_size=chunk_batch_size, inner_checkpoint=inner_checkpoint,
            lazy_qkv=lazy_qkv, select_scales=select_scales,
            select_gate_mode=select_gate_mode,
            select_gate_kappa=select_gate_kappa,
            select_gate_cos_floor=select_gate_cos_floor,
            select_gate_min_candidates=select_gate_min_candidates,
            select_gate_mad_floor=select_gate_mad_floor,
            select_gate_min_keep=select_gate_min_keep,
            select_gate_temp=select_gate_temp,
            select_gate_budget_target=select_gate_budget_target,
            select_gate_budget_weight=select_gate_budget_weight,
            sink_decay_mode=sink_decay_mode,
            sink_decay_onset=sink_decay_onset,
            sink_decay_factor=sink_decay_factor,
        )

        self.cam_ctrl = cam_ctrl
        self.cam_ctrl_dim = cam_ctrl_dim
        self.cam_rank = cam_rank
        self.cam_full = cam_full
        self.cam_input_inject = cam_input_inject
        self.cam_adaln = cam_adaln


        self._init_memory_ctrl(
            memory_ctrl=memory_ctrl,
            memory_encoder_out_dim=memory_encoder_out_dim,
            memory_encoder_num_heads=memory_encoder_num_heads,
            dim=dim,
            in_dim=in_dim,
        )

        if not cam_ctrl:
            self.cam_ctrl_layer_set = set()
            return

        if cam_adaln:


            self.cam_ctrl_layer_set = set()
            self.action_adaln_embedder = ActionAdaLNEmbedder(dim=dim, action_dim=6)
            self.action_adaln_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.normal_(self.action_adaln_projection[-1].weight, std=1e-3)
            nn.init.zeros_(self.action_adaln_projection[-1].bias)
            print(f"[CamCtrl] AdaLN mode: per-latent-frame 6D action -> modulate timestep AdaLN (6x{dim}); near-zero init; no per-block / no plucker encoder")
            return

        if cam_input_inject:


            self.cam_ctrl_layer_set = set()
            print(f"[CamCtrl] input-inject mode (CameraCtrl style): encoder -> add per-token to x at DiT input; no per-block injector; DiT adapts via LoRA")
        else:

            cam_layer_set = set(cam_ctrl_layers) if cam_ctrl_layers is not None else set(range(num_layers))
            self.cam_ctrl_layer_set = cam_layer_set
            _mode = "FULL (original LingBot full-rank: per-layer injector + full-rank scale/shift)" if cam_full else f"lowrank{cam_rank}"
            print(f"[CamCtrl] Enabling camera control [{_mode}] on {len(cam_layer_set)}/{num_layers} layers: "
                  f"{sorted(cam_layer_set) if len(cam_layer_set) <= 10 else f'[{min(cam_layer_set)}..{max(cam_layer_set)}]'}")
            for i in cam_layer_set:
                old_block = self.blocks[i]
                new_block = CamDiTBlock(
                    has_image_input=has_image_input,
                    dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, eps=eps,
                    sparse_attn=sparse_attn, chunk_size=chunk_size,
                    overlap_size=overlap_size, num_global_tokens=num_global_tokens,
                    per_frame_tokens=per_frame_tokens, num_retained_tokens=num_retained_tokens,
                    num_select_frames=num_select_frames, num_nearby_frames=num_nearby_frames,
                    chunk_batch_size=chunk_batch_size, inner_checkpoint=inner_checkpoint,
                    lazy_qkv=lazy_qkv, select_scales=select_scales,
                    select_gate_mode=select_gate_mode,
                    select_gate_kappa=select_gate_kappa,
                    select_gate_cos_floor=select_gate_cos_floor,
                    select_gate_min_candidates=select_gate_min_candidates,
                    select_gate_mad_floor=select_gate_mad_floor,
                    select_gate_min_keep=select_gate_min_keep,
                    select_gate_temp=select_gate_temp,
                    select_gate_budget_target=select_gate_budget_target,
                    select_gate_budget_weight=select_gate_budget_weight,
                    sink_decay_mode=sink_decay_mode,
                    sink_decay_onset=sink_decay_onset,
                    sink_decay_factor=sink_decay_factor,
                    cam_ctrl=True,
                    cam_rank=cam_rank,
                    cam_full=cam_full,
                )

                new_block.load_state_dict(old_block.state_dict(), strict=False)
                self.blocks[i] = new_block


        vae_spatial = 64
        plucker_input_dim = cam_ctrl_dim * vae_spatial * math.prod(patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(plucker_input_dim, dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        if cam_input_inject:


            self.cam_input_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.cam_input_out.weight); nn.init.zeros_(self.cam_input_out.bias)
        print(f"[CamCtrl] Plucker embedding: {plucker_input_dim} → {dim}"
              + (" | input-inject zero-conv gate (cam_input_out) built" if cam_input_inject else ""))

    def _process_cam_plucker(self, cam_plucker_emb, f, h, w):


        if cam_plucker_emb is None:
            return None


        pt, ph, pw = self.patch_size
        processed = rearrange(
            cam_plucker_emb,
            'b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)',
            pt=pt, ph=ph, pw=pw
        )


        processed = self.patch_embedding_wancamctrl(processed)


        hidden = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(processed)))
        processed = processed + hidden

        if getattr(self, 'cam_input_inject', False):

            processed = self.cam_input_out(processed)

        return processed

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):

        cam_plucker_emb = kwargs.pop('cam_plucker_emb', None)


        memory_latents = kwargs.pop('memory_latents', None)
        memory_latents_lr = kwargs.pop('memory_latents_lr', None)
        omega_indices = kwargs.pop('omega_indices', None)
        rollout_anchors = kwargs.pop('rollout_anchors', None)
        target_lat_indices = kwargs.pop('target_lat_indices', None)


        kwargs.pop('input_latents_lr', None)

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)


        if isinstance(timestep, torch.Tensor) and timestep.numel() >= 1:
            _σ_step = float(timestep.flatten()[0].item() / 1000.0)
        else:
            _σ_step = None

        x_latent = x


        if y is not None:
            x = torch.cat([x, y], dim=1)
        else:
            pad_channels = self.in_dim - x.shape[1]
            if pad_channels > 0:
                x = torch.cat([
                    x,
                    torch.zeros(x.shape[0], pad_channels, *x.shape[2:],
                                device=x.device, dtype=x.dtype),
                ], dim=1)

        if clip_feature is not None and self.has_image_input:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)


        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()


        cam_emb_processed = None
        if self.cam_ctrl and cam_plucker_emb is not None:
            cam_emb_processed = self._process_cam_plucker(cam_plucker_emb, f, h, w)


        _n_mem_tokens = 0


        _mem_token_start = 0
        _mem_token_len = 0

        if self.memory_ctrl and memory_latents is not None:

            assert omega_indices is not None, (
                "memory_ctrl=True + memory_latents provided requires omega_indices to be provided as well"
            )
            _mem_tokens, _mem_info = self.history_encoder(
                memory_latents, patch_embedding=self.patch_embedding,
                memory_latents_lr=memory_latents_lr,
                current_sigma=_σ_step,
            )
            _n_mem_tokens = _mem_tokens.shape[1]
            _mem_token_start = 0
            _mem_token_len = _n_mem_tokens

            freqs = self._build_omega_rope_freqs(omega_indices, h, w, x.device)
            mem_freqs = self._build_memory_rope_freqs(_mem_info, x.device)

            x = torch.cat([_mem_tokens.to(x.dtype), x], dim=1)
            freqs = torch.cat([mem_freqs, freqs], dim=0)

        elif self.memory_ctrl and rollout_anchors is not None:

            assert target_lat_indices is not None, (
                "rollout_anchors provided requires target_lat_indices to be provided as well"
            )
            first_ref_lat  = rollout_anchors["first_ref_lat"]
            mem_lat        = rollout_anchors["mem_lat"]
            nearby_ref_lat = rollout_anchors["nearby_ref_lat"]
            mem_start_lat  = int(rollout_anchors["mem_start_lat"])

            mem_lat_lr     = rollout_anchors.get("mem_lat_lr", None)


            _mem_tokens, _mem_info = self.history_encoder(
                mem_lat, patch_embedding=self.patch_embedding,
                memory_latents_lr=mem_lat_lr,
                current_sigma=_σ_step,
            )
            _n_mem_tokens = _mem_tokens.shape[1]


            def _embed_anchor_frame(lat_1f):

                B_, C_, _, H_, W_ = lat_1f.shape
                noise_slot = torch.zeros(B_, 16, 1, H_, W_, device=lat_1f.device, dtype=lat_1f.dtype)
                mask_slot  = torch.ones (B_,  4, 1, H_, W_, device=lat_1f.device, dtype=lat_1f.dtype)

                lat_36 = torch.cat([noise_slot, mask_slot, lat_1f], dim=1)
                patched = self.patchify(lat_36)
                return rearrange(patched, 'b c 1 h w -> b (h w) c').contiguous()

            first_ref_tokens  = _embed_anchor_frame(first_ref_lat)
            nearby_ref_tokens = _embed_anchor_frame(nearby_ref_lat)


            anchor_tokens = torch.cat([
                first_ref_tokens,
                _mem_tokens.to(x.dtype),
                nearby_ref_tokens,
            ], dim=1)
            _n_mem_tokens = anchor_tokens.shape[1]
            _mem_token_start = first_ref_tokens.shape[1]
            _mem_token_len = _mem_tokens.shape[1]


            freqs = self._build_omega_rope_freqs(target_lat_indices, h, w, x.device)


            nearby_ref_t = mem_start_lat + int(_mem_info["T_mem_lat"]) - 1
            anchor_freqs = self._build_rollout_anchor_rope_freqs(
                first_ref_t=0,
                mem_info=_mem_info,
                nearby_ref_t=nearby_ref_t,
                h=h, w=w,
                device=x.device,
                mem_t_offset=mem_start_lat,
            )

            x = torch.cat([anchor_tokens, x], dim=1)
            freqs = torch.cat([anchor_freqs, freqs], dim=0)

        else:

            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)


        _sp_active = getattr(self, 'sp_enabled', False)
        if _sp_active and _n_mem_tokens > 0:
            raise NotImplementedError(
                "Sequence Parallel + memory_ctrl not implemented yet; the first version only supports DP (see plan §3 M2.5)"
            )
        _sp_total_seq_len = f * h * w
        if _sp_active:
            from .sp_runtime import (
                scatter_frames, get_sp_frame_info, get_sp_rank)
            pf_sp = h * w
            _freqs_full = freqs
            frames_per_rank, _sp_f_start, _sp_f_end, _sp_f_local = get_sp_frame_info(f)

            _overlap_f = self.blocks[0].chunk_size if len(self.blocks) > 0 else 8
            _ghost_f_start = max(0, _sp_f_start - _overlap_f)
            _ghost_f_end = min(f, _sp_f_start + frames_per_rank + _overlap_f)
            _ghost_before = _sp_f_start - _ghost_f_start
            _ghost_after = _ghost_f_end - min(f, _sp_f_start + frames_per_rank)

            x = x[:, _ghost_f_start * pf_sp : _ghost_f_end * pf_sp]
            freqs = _freqs_full[_ghost_f_start * pf_sp : _ghost_f_end * pf_sp]


            if cam_emb_processed is not None:
                cam_emb_processed = cam_emb_processed[:, _ghost_f_start * pf_sp : _ghost_f_end * pf_sp]

            _sp_f_start = _ghost_f_start

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        _block_extra_kw = dict(
            tokens_per_frame=h*w,
            spatial_hw=(h, w),
            freqs_3d=self.freqs,


            select_gate_t_frac=(timestep.detach().float() / 1000.0).clamp(0.0, 1.0),
        )
        if _sp_active:
            _block_extra_kw['sp_num_frames_global'] = f
            _block_extra_kw['sp_frame_offset'] = _sp_f_start
            _block_extra_kw['freqs_full'] = _freqs_full


        if cam_emb_processed is not None:
            _block_extra_kw['cam_plucker_emb'] = cam_emb_processed

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                            **_block_extra_kw,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                        **_block_extra_kw,
                    )
            else:
                x = block(x, context, t_mod, freqs, **_block_extra_kw)


        lr_pred = None
        if _n_mem_tokens > 0:

            anchor_segment = x[:, :_n_mem_tokens]
            x = x[:, _n_mem_tokens:].contiguous()


            if self.memory_ctrl and _mem_info is not None and _mem_token_len > 0:
                T_mem_token = _mem_info["T_mem"]
                H_mem_token = _mem_info["H_mem"]
                W_mem_token = _mem_info["W_mem"]


                x_mem = anchor_segment[:, _mem_token_start : _mem_token_start + _mem_token_len]
                x_mem_proj = self.head(x_mem, t)
                lr_pred = self.unpatchify(x_mem_proj, (T_mem_token, H_mem_token, W_mem_token))


        if _sp_active:
            from .sp_runtime import gather_frames, get_sp_frame_info
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(f)
            pf_hw = h * w
            trim_start = _ghost_before * pf_hw
            orig_local_tokens = (_orig_f_end - _orig_f_start) * pf_hw
            x = x[:, trim_start : trim_start + orig_local_tokens].contiguous()
            x = self.head(x, t)
            if x.shape[1] < _fpr * pf_hw:
                x = torch.nn.functional.pad(x, (0, 0, 0, _fpr * pf_hw - x.shape[1]))
            x = gather_frames(x, _sp_total_seq_len)
        else:
            x = self.head(x, t)

        x = self.unpatchify(x, (f, h, w))


        if lr_pred is not None:
            return x, lr_pred, None, None
        return x


    def _init_memory_ctrl(self, memory_ctrl, memory_encoder_out_dim,
                          memory_encoder_num_heads, dim, in_dim):


        self.memory_ctrl = memory_ctrl
        if not memory_ctrl:
            return

        from diffsynth.models.history_encoder import HistoryEncoder
        out_dim = memory_encoder_out_dim if memory_encoder_out_dim is not None else dim

        memory_in_channels = 16
        self.history_encoder = HistoryEncoder(
            in_channels=memory_in_channels,
            out_dim=out_dim,
            num_attn_heads=memory_encoder_num_heads,
        )
        n_params = sum(p.numel() for p in self.history_encoder.parameters())
        print(f"[MemoryCtrl] HistoryEncoder initialized: "
              f"in_channels={memory_in_channels}, out_dim={out_dim}, params={n_params:,}")

    def _build_memory_rope_freqs(self, mem_info, device, t_offset=0):


        T_mem = mem_info.get('T_mem')
        if T_mem is None:

            T_mem = mem_info['F_eff_mem'] * mem_info['bundle_factor']
        H_mem, W_mem = mem_info['H_mem'], mem_info['W_mem']
        comp_t = mem_info['compress_t']
        T_lat, H_lat, W_lat = mem_info['T_mem_lat'], mem_info['H_lat'], mem_info['W_lat']


        patch_t, patch_h, patch_w = self.patch_size
        H_post = H_lat // patch_h
        W_post = W_lat // patch_w


        t_mem = torch.arange(T_mem)
        max_t_idx = self.freqs[0].shape[0] - 1
        t_real = (int(t_offset) + t_mem * comp_t).clamp(max=min(int(t_offset) + T_lat - 1, max_t_idx)).long()

        h_sub = torch.arange(H_mem)
        w_sub = torch.arange(W_mem)
        h_real = (h_sub * H_post // H_mem).clamp(max=H_post - 1).long()
        w_real = (w_sub * W_post // W_mem).clamp(max=W_post - 1).long()

        freqs_t = self.freqs[0][t_real]
        freqs_h = self.freqs[1][h_real]
        freqs_w = self.freqs[2][w_real]

        T_N = T_mem

        freqs_t_e = freqs_t.view(T_N, 1, 1, -1).expand(T_N, H_mem, W_mem, -1)
        freqs_h_e = freqs_h.view(1, H_mem, 1, -1).expand(T_N, H_mem, W_mem, -1)
        freqs_w_e = freqs_w.view(1, 1, W_mem, -1).expand(T_N, H_mem, W_mem, -1)

        mem_freqs = torch.cat([freqs_t_e, freqs_h_e, freqs_w_e], dim=-1)
        mem_freqs = mem_freqs.reshape(T_N * H_mem * W_mem, 1, -1).to(device)
        return mem_freqs

    def _build_omega_rope_freqs(self, omega_indices, h, w, device):


        max_t_idx = self.freqs[0].shape[0] - 1
        omega_t_cpu = omega_indices.detach().to('cpu').long().clamp(max=max_t_idx)
        F_O = omega_t_cpu.shape[0]

        freqs_t = self.freqs[0][omega_t_cpu]
        freqs_h = self.freqs[1][:h]
        freqs_w = self.freqs[2][:w]

        freqs_t_e = freqs_t.view(F_O, 1, 1, -1).expand(F_O, h, w, -1)
        freqs_h_e = freqs_h.view(1, h, 1, -1).expand(F_O, h, w, -1)
        freqs_w_e = freqs_w.view(1, 1, w, -1).expand(F_O, h, w, -1)

        freqs = torch.cat([freqs_t_e, freqs_h_e, freqs_w_e], dim=-1)
        return freqs.reshape(F_O * h * w, 1, -1).to(device)

    def _build_rollout_anchor_rope_freqs(self, first_ref_t, mem_info, nearby_ref_t, h, w, device,
                                         mem_t_offset=0):


        def _single_frame_freqs(t_idx):

            t_cpu = min(t_idx, self.freqs[0].shape[0] - 1)
            ft = self.freqs[0][t_cpu:t_cpu + 1]
            fh = self.freqs[1][:h]
            fw = self.freqs[2][:w]
            ft_e = ft.view(1, 1, 1, -1).expand(1, h, w, -1)
            fh_e = fh.view(1, h, 1, -1).expand(1, h, w, -1)
            fw_e = fw.view(1, 1, w, -1).expand(1, h, w, -1)
            f_cat = torch.cat([ft_e, fh_e, fw_e], dim=-1)
            return f_cat.reshape(h * w, 1, -1).to(device)

        freqs_first = _single_frame_freqs(first_ref_t)
        freqs_mem   = self._build_memory_rope_freqs(mem_info, device, t_offset=mem_t_offset)
        freqs_near  = _single_frame_freqs(nearby_ref_t)

        return torch.cat([freqs_first, freqs_mem, freqs_near], dim=0)


    def _cam_weights_already_loaded(self):


        for block in self.blocks:
            if not getattr(block, 'cam_ctrl', False):
                continue
            if getattr(block, 'cam_full', False):
                w = block.cam_scale_layer.weight
            else:
                w = block.cam_scale_down_proj.weight
            return w.abs().max() > 1e-8
        return False

    def _print_cam_stats(self, tag=""):


        keys_to_probe = [
            ("patch_embedding_wancamctrl.weight", 0.015),
            ("c2ws_hidden_states_layer1.weight", None),
        ]
        if len(self.cam_ctrl_layer_set) > 0:
            i0 = sorted(self.cam_ctrl_layer_set)[0]
            if getattr(self, 'cam_full', False):
                keys_to_probe += [
                    (f"blocks.{i0}.cam_injector_layer1.weight", None),
                    (f"blocks.{i0}.cam_scale_layer.weight",     None),
                    (f"blocks.{i0}.cam_shift_layer.weight",     None),
                ]
            else:
                keys_to_probe += [
                    (f"blocks.{i0}.cam_scale_down_proj.weight", 0.05),
                    (f"blocks.{i0}.cam_scale_up_proj.weight",   0.05),
                    (f"blocks.{i0}.cam_shift_down_proj.weight", 0.05),
                    (f"blocks.{i0}.cam_shift_up_proj.weight",   0.05),
                ]
        print(f"[CamCtrl] stats {tag}:")
        named = dict(self.named_parameters())
        for k, expected in keys_to_probe:
            if k in named:
                p = named[k]
                try:
                    v = p.detach().abs().float().mean().item()
                    hint = f"(expected ~{expected})" if expected is not None else ""
                    print(f"  {k}: abs_mean={v:.6f}  {hint}")
                except Exception as e:
                    print(f"  {k}: <unreadable under Zero? {e}>")

    def reinit_cam_modules(self, cam_weight_path=None):


        if not self.cam_ctrl:
            return


        if getattr(self, 'cam_adaln', False):
            print("[CamCtrl] adaln: action_adaln modules already near-zero initialized in __init__, no plucker encoder/per-block, skipping reinit")
            return


        if getattr(self, 'cam_input_inject', False):
            if cam_weight_path is not None:
                print(f"[CamCtrl] input-inject: ignoring cam_weight_path (no per-block injector, the encoder is loaded with the base)")
            print("[CamCtrl] input-inject: encoder warm/already initialized, cam_input_out zero-initialized (zero-conv gate), skipping per-block reinit")
            return


        if cam_weight_path is not None:
            print(f"[CamCtrl] Loading cam weights from {cam_weight_path} (explicit path)")
            self._load_cam_weights(cam_weight_path)
            self._print_cam_stats(tag="after loading from cam_weight_path")
            return


        if self._cam_weights_already_loaded():
            print("[CamCtrl] Checkpoint already contains cam weights, skipping reinit.")
            self._print_cam_stats(tag="using existing cam weights")
            return

        try:
            import deepspeed
            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        def safe_init(fn, *args, **kwargs):
            if has_deepspeed and args and hasattr(args[0], 'ds_id'):
                with deepspeed.zero.GatheredParameters(args[0], modifier_rank=0):
                    fn(*args, **kwargs)
            else:
                fn(*args, **kwargs)


        print("[CamCtrl] No cam_weight_path & no existing cam weights → Xavier init fallback")


        safe_init(nn.init.xavier_uniform_, self.patch_embedding_wancamctrl.weight)
        safe_init(nn.init.zeros_, self.patch_embedding_wancamctrl.bias)
        safe_init(nn.init.xavier_uniform_, self.c2ws_hidden_states_layer1.weight)
        safe_init(nn.init.zeros_, self.c2ws_hidden_states_layer1.bias)
        safe_init(nn.init.xavier_uniform_, self.c2ws_hidden_states_layer2.weight)
        safe_init(nn.init.zeros_, self.c2ws_hidden_states_layer2.bias)


        for i in self.cam_ctrl_layer_set:
            block = self.blocks[i]
            if hasattr(block, 'cam_ctrl') and block.cam_ctrl:
                if getattr(block, 'cam_full', False):

                    safe_init(nn.init.xavier_uniform_, block.cam_injector_layer1.weight)
                    safe_init(nn.init.zeros_, block.cam_injector_layer1.bias)
                    safe_init(nn.init.xavier_uniform_, block.cam_injector_layer2.weight)
                    safe_init(nn.init.zeros_, block.cam_injector_layer2.bias)
                    safe_init(nn.init.zeros_, block.cam_scale_layer.weight)
                    safe_init(nn.init.zeros_, block.cam_scale_layer.bias)
                    safe_init(nn.init.zeros_, block.cam_shift_layer.weight)
                    safe_init(nn.init.zeros_, block.cam_shift_layer.bias)
                else:
                    safe_init(nn.init.xavier_uniform_, block.cam_scale_down_proj.weight)
                    safe_init(nn.init.zeros_, block.cam_scale_up_proj.weight)
                    safe_init(nn.init.zeros_, block.cam_scale_up_proj.bias)
                    safe_init(nn.init.xavier_uniform_, block.cam_shift_down_proj.weight)
                    safe_init(nn.init.zeros_, block.cam_shift_up_proj.weight)
                    safe_init(nn.init.zeros_, block.cam_shift_up_proj.bias)

        self._print_cam_stats(tag="after Xavier fallback")

    def _load_cam_weights(self, cam_weight_path):


        import os
        import json
        from safetensors.torch import load_file

        cam_state_dict = {}

        def _collect_cam(st):

            for key, tensor in st.items():
                clean_k = key.replace('_orig_mod.', '')
                if 'cam' in clean_k or 'wancamctrl' in clean_k or 'c2ws_hidden' in clean_k or 'action_adaln' in clean_k:
                    cam_state_dict[clean_k] = tensor

        if os.path.isdir(cam_weight_path):

            index_path = os.path.join(cam_weight_path, "diffusion_pytorch_model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index["weight_map"]


                cam_files = set()
                for key, filename in weight_map.items():
                    clean_k = key.replace('_orig_mod.', '')
                    if 'cam' in clean_k or 'wancamctrl' in clean_k or 'c2ws_hidden' in clean_k or 'action_adaln' in clean_k:
                        cam_files.add(filename)


                for filename in cam_files:
                    filepath = os.path.join(cam_weight_path, filename)
                    _collect_cam(load_file(filepath))
            else:

                for fname in sorted(os.listdir(cam_weight_path)):
                    if fname.endswith('.safetensors'):
                        _collect_cam(load_file(os.path.join(cam_weight_path, fname)))
        elif os.path.isfile(cam_weight_path):
            _collect_cam(load_file(cam_weight_path))

        if not cam_state_dict:
            print(f"[CamCtrl] WARNING: No camera control weights found in {cam_weight_path}")
            return


        try:
            import deepspeed
            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        named = dict(self.named_parameters())
        named_buf = dict(self.named_buffers())
        loaded_cam_keys, skipped_keys = [], []
        for k, v in cam_state_dict.items():
            target = named.get(k, named_buf.get(k))
            if target is None:
                skipped_keys.append(k)
                continue
            is_ds = has_deepspeed and hasattr(target, 'ds_id')

            full_shape = tuple(getattr(target, 'ds_shape', target.shape))
            if full_shape != tuple(v.shape):
                skipped_keys.append(k)
                continue
            if is_ds:
                with deepspeed.zero.GatheredParameters(target, modifier_rank=0):
                    with torch.no_grad():
                        target.data.copy_(v.to(device=target.device, dtype=target.dtype))
            else:
                with torch.no_grad():
                    target.data.copy_(v.to(device=target.device, dtype=target.dtype))
            loaded_cam_keys.append(k)
        print(f"[CamCtrl] Loaded {len(loaded_cam_keys)} cam weight keys from {cam_weight_path}")
        if skipped_keys:
            skipped_layers = set()
            for k in skipped_keys:
                parts = k.split('.')
                if parts[0] == 'blocks' and len(parts) > 1:
                    skipped_layers.add(int(parts[1]))
            if skipped_layers:
                print(f"[CamCtrl] Skipped cam weights for layers not in cam_ctrl_layers: {sorted(skipped_layers)}")


        if len(loaded_cam_keys) == 0:
            raise RuntimeError(
                f"[CamCtrl] FATAL: loaded 0 cam weights from {cam_weight_path} "
                f"(collected {len(cam_state_dict)} keys but all of them unmatched/skipped)! "
                f"Check key names / cam_full architecture / DeepSpeed partitioned writes for consistency; refusing to silently train from scratch.")



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange


_gpu_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = _gpu_cap[0] >= 9
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = _gpu_cap[0] >= 8
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_2_AVAILABLE = False

print(f"[Attention Backend] FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}, GPU={_gpu_cap}")

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    SAGE_ATTN_AVAILABLE = False

from .select_gate import (
    compute_select_keep,
    compute_select_gate_features,
    SelectGateHead,
    gate_to_bias,
)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attn_bias=None):


    if attn_bias is not None:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def _scale_to_tokens(scale_factor: int, per_frame_tokens: int, spatial_hw=(30, 52)) -> int:

    H, W = spatial_hw
    return (H // scale_factor) * (W // scale_factor)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024*8, theta: float = 10000.0):

    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class LinearAttention(nn.Module):


    def __init__(self, dim: int, num_heads: int, state_dim: int = None,
                 inner_dim: int = None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim

        self.inner_dim = inner_dim if inner_dim is not None else dim
        self.num_heads = num_heads if inner_dim is None else max(1, self.inner_dim // (dim // num_heads))
        self.head_dim = self.inner_dim // self.num_heads
        self.state_dim = state_dim if state_dim is not None else self.inner_dim
        self.eps = eps


        self.q = nn.Linear(dim, self.inner_dim)
        self.k = nn.Linear(dim, self.inner_dim)
        self.v = nn.Linear(dim, self.inner_dim)

        self.o = nn.Linear(self.inner_dim, dim)


        self.state_proj = nn.Linear(self.head_dim * self.head_dim, self.state_dim) if state_dim is not None else None


        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:

        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        B, N, D = x.shape


        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)


        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)


        q = self.feature_map(q)
        k = self.feature_map(k)


        state = torch.einsum('bhnd,bhnv->bhdv', k, v)


        z = k.sum(dim=2)


        qkv = torch.einsum('bhnd,bhdv->bhnv', q, state)


        qk_sum = torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + self.eps


        out = qkv / qk_sum


        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        out = self.o(out)


        if self.state_proj is not None:
            state = self.state_proj(state.flatten(2))


        self._cached_q_mapped = q

        return out, state, z


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,
                 sparse_attn: bool = True, chunk_size: int = 256, overlap_size: int = 32, num_global_tokens: int = 8,
                 per_frame_tokens: int = 30 * 52,
                 num_retained_tokens: int = 1024,

                 num_select_frames: int = 4,
                 num_nearby_frames: int = 3,

                 chunk_batch_size: int = None,
                 inner_checkpoint: bool = False,
                 lazy_qkv: bool = False,
                 select_scales: list = None,

                 select_gate_mode: str = 'none',
                 select_gate_kappa: float = 2.0,
                 select_gate_cos_floor: float = -1.0,
                 select_gate_min_candidates: int = 8,
                 select_gate_mad_floor: float = 1e-6,
                 select_gate_min_keep: int = 0,

                 select_gate_temp: float = 0.6667,
                 select_gate_budget_target: float = 0.5,
                 select_gate_budget_weight: float = 0.0,

                 sink_decay_mode: str = 'none',
                 sink_decay_onset: int = 40,
                 sink_decay_factor: int = 2,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.head_dim = dim // num_heads


        self.sparse_attn = sparse_attn
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.num_global_tokens = num_global_tokens
        self.num_retained_tokens = num_retained_tokens


        self.num_select_frames = num_select_frames
        self.num_nearby_frames = num_nearby_frames


        self.chunk_batch_size = chunk_batch_size
        self.inner_checkpoint = inner_checkpoint
        self.lazy_qkv = lazy_qkv

        self.select_scales = select_scales or ['1x', '2x', '4x', '8x']


        self.select_gate_mode = select_gate_mode
        self.select_gate_kappa = select_gate_kappa
        self.select_gate_cos_floor = select_gate_cos_floor
        self.select_gate_min_candidates = select_gate_min_candidates
        self.select_gate_mad_floor = select_gate_mad_floor
        self.select_gate_min_keep = select_gate_min_keep

        self.select_gate_temp = select_gate_temp
        self.select_gate_budget_target = select_gate_budget_target
        self.select_gate_budget_weight = select_gate_budget_weight

        self.sink_decay_mode = sink_decay_mode
        self.sink_decay_onset = sink_decay_onset
        self.sink_decay_factor = sink_decay_factor

        self._select_gate_last_stats = []


        self._select_gate_reg_feats = []


        self.linear_attn = LinearAttention(dim, num_heads, inner_dim=1024, eps=eps)
        self.linear_attn_norm = nn.LayerNorm(dim, eps=eps)


        if self.sparse_attn:

            self.importance_head = nn.Linear(dim, 1)


            _la_inner = self.linear_attn.inner_dim
            _la_heads = self.linear_attn.num_heads
            self.chunk_to_state_proj = nn.Linear(dim, _la_inner)
            self.chunk_to_state_norm = RMSNorm(_la_inner, eps=eps)


            self.global_attn_out_proj = nn.Linear(_la_inner, dim)


            self.global_attn_gate = nn.Parameter(torch.zeros(_la_heads))


            self.blend_sharpness = nn.Parameter(torch.zeros(1))


        self.select_gate_head = None
        if self.sparse_attn and select_gate_mode == 'learned':
            self.select_gate_head = SelectGateHead(
                feat_dim=3, hidden_dim=16, temperature=select_gate_temp)

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.per_frame_tokens = per_frame_tokens

    def _query_global_state(self, chunk_x: torch.Tensor, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:


        B, L, D = chunk_x.shape
        la_heads = self.linear_attn.num_heads
        la_head_dim = self.linear_attn.head_dim
        la_inner = self.linear_attn.inner_dim


        q = self.chunk_to_state_norm(self.chunk_to_state_proj(chunk_x))
        q = q.view(B, L, la_heads, la_head_dim).transpose(1, 2)
        q = F.elu(q) + 1


        qkv = torch.einsum('bhnd,bhdv->bhnv', q, state)
        qk_sum = torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + 1e-6

        out = qkv / qk_sum
        out = out.transpose(1, 2).contiguous().view(B, L, la_inner)
        out = self.global_attn_out_proj(out)
        return out

    def sparse_self_attention(self, x, freqs, frame_keys, sink_len, spatial_hw, state, z,
                              sp_frame_offset=0, sp_num_frames_global=None, freqs_full=None,
                              freqs_3d=None, select_gate_t_frac=None):


        assert freqs_3d is not None, (
            "freqs_3d must be provided for scale-back RoPE on compressed K. "
            "Check that WanModel.forward() passes freqs_3d=self.freqs in _block_extra_kw."
        )
        B, S, D = x.shape
        pf = self.per_frame_tokens
        num_frames = S // pf
        num_frames_global = sp_num_frames_global if sp_num_frames_global else num_frames
        lazy_qkv = getattr(self, 'lazy_qkv', False)
        _sp = sp_num_frames_global is not None


        if not lazy_qkv:

            q_full = self.self_attn.norm_q(self.self_attn.q(x))
            k_full = self.self_attn.norm_k(self.self_attn.k(x))
            v_full = self.self_attn.v(x)
            q_full = rope_apply(q_full, freqs, self.num_heads)
            k_full = rope_apply(k_full, freqs, self.num_heads)

        else:

            q_full = k_full = v_full = None


        x_sink = x[:, :pf]
        freqs_sink = freqs[:pf]
        k_sink = rope_apply(self.self_attn.norm_k(self.self_attn.k(x_sink)), freqs_sink, self.num_heads)
        v_sink = self.self_attn.v(x_sink)
        if _sp:
            from .sp_runtime import broadcast_with_grad, halo_exchange
            k_sink = broadcast_with_grad(k_sink.contiguous(), src_rank=0)
            v_sink = broadcast_with_grad(v_sink.contiguous(), src_rank=0)


        if self.sink_decay_mode == 'downsample' or _sp:
            k_sink = k_sink.detach()
            v_sink = v_sink.detach()


        halo_prev = None
        if _sp and self.num_nearby_frames > 0:
            halo_prev = halo_exchange(x, self.num_nearby_frames, pf)


        def _lazy_kv(x_slice, freqs_slice):
            k_s = rope_apply(self.self_attn.norm_k(self.self_attn.k(x_slice)), freqs_slice, self.num_heads)
            v_s = self.self_attn.v(x_slice)
            return k_s, v_s


        _scale_map = {'1x': 1, '2x': 2, '4x': 4, '8x': 8}
        H_sp, W_sp = spatial_hw if spatial_hw else (30, 52)

        def _downsample_kv(frame_tokens, scale_factor, global_frame_idx=None):


            ft = frame_tokens.view(B, H_sp, W_sp, D).permute(0, 3, 1, 2)
            H_out, W_out = H_sp // scale_factor, W_sp // scale_factor
            ft = F.interpolate(ft, size=(H_out, W_out), mode='bilinear', align_corners=False)
            ft = ft.permute(0, 2, 3, 1).reshape(B, H_out * W_out, D)
            k_s = self.self_attn.norm_k(self.self_attn.k(ft))
            v_s = self.self_attn.v(ft)


            if global_frame_idx is not None and freqs_3d is not None:

                if os.environ.get('DISABLE_SCALE_BACK_ROPE', '0') == '1':
                    return k_s, v_s
                f_cis, h_cis, w_cis = freqs_3d
                h_idx = torch.arange(H_out, device=h_cis.device) * scale_factor
                w_idx = torch.arange(W_out, device=h_cis.device) * scale_factor
                ds_freqs = torch.cat([
                    f_cis[global_frame_idx].view(1, 1, -1).expand(H_out, W_out, -1),
                    h_cis[h_idx].view(H_out, 1, -1).expand(H_out, W_out, -1),
                    w_cis[w_idx].view(1, W_out, -1).expand(H_out, W_out, -1),
                ], dim=-1).reshape(H_out * W_out, 1, -1).to(k_s.device)
                k_s = rope_apply(k_s, ds_freqs, self.num_heads)
            return k_s, v_s


        k_sink_ds = v_sink_ds = None
        if self.sink_decay_mode == 'downsample':


            with torch.no_grad():
                k_sink_ds, v_sink_ds = _downsample_kv(x_sink, self.sink_decay_factor, global_frame_idx=0)
                if _sp:
                    from .sp_runtime import broadcast_with_grad
                    k_sink_ds = broadcast_with_grad(k_sink_ds.contiguous(), src_rank=0)
                    v_sink_ds = broadcast_with_grad(v_sink_ds.contiguous(), src_rank=0)


        if self.select_gate_mode != 'none':
            self._select_gate_last_stats = []


        _sg_t_frac_row = None
        if self.select_gate_mode == 'learned':
            if getattr(self, 'select_gate_head', None) is None:
                raise RuntimeError(
                    "select_gate_mode='learned' but select_gate_head was not built "
                    "(sparse_attn=False, or mode was not 'learned' when the DiTBlock was constructed).")
            if select_gate_t_frac is None:
                raise RuntimeError(
                    "select_gate_mode='learned' requires select_gate_t_frac (normalized timestep, t/1000). "
                    "Check that the caller forwards it: WanModel.forward / model_fn_wan_video (_pass_spatial_hw branch) "
                    "/ WanModelCam.forward via _block_extra_kw['select_gate_t_frac'].")
            _sg_t = select_gate_t_frac.float().reshape(-1)
            if _sg_t.numel() == 1:
                _sg_t = _sg_t.expand(B)
            elif _sg_t.numel() != B:
                raise ValueError(
                    f"select_gate_t_frac numel={_sg_t.numel()} does not match batch B={B} (and is not a scalar).")
            _sg_t_frac_row = _sg_t.to(x.device)

            self._select_gate_reg_feats = []


        chunk_f = self.chunk_size
        overlap_f = self.overlap_size
        overlap_tokens = overlap_f * pf
        num_chunks = (num_frames + chunk_f - 1) // chunk_f


        blend_len = 2 * overlap_tokens
        if overlap_tokens > 0 and blend_len > 0:
            positions = torch.arange(blend_len, device=x.device, dtype=x.dtype)
            t_blend = 2.0 * (positions + 1.0) / (blend_len + 1.0) - 1.0
            sharpness = F.softplus(self.blend_sharpness) + 1.0
            alpha = torch.sigmoid(sharpness * t_blend)
            alpha_next = alpha.unsqueeze(0).unsqueeze(-1)
            alpha_prev = 1.0 - alpha_next


        gate_scalar = torch.sigmoid(self.global_attn_gate).mean() if state is not None else None


        cbs = self.chunk_batch_size if hasattr(self, 'chunk_batch_size') and self.chunk_batch_size else num_chunks


        _sp = sp_num_frames_global is not None


        _scale_map_inv = {'1x': 1, '2x': 2, '4x': 4, '8x': 8}
        precomputed_select = {}
        remote_token_cache = {}
        _exchange_received = None
        _dummy_score_anchors = []

        if _sp and self.num_select_frames > 0:
            from .sp_runtime import get_sp_rank, get_sp_size, exchange_frame_tokens
            _sp_rank = get_sp_rank()
            _sp_size = get_sp_size()
            _frames_per_rank = (sp_num_frames_global + _sp_size - 1) // _sp_size

            remote_frame_requests = {}
            remote_frame_set = set()


            _ns_target = max(1, self.num_select_frames)

            for ci in range(num_chunks):
                f_start = ci * chunk_f
                ext_f_start = max(0, f_start - overlap_f)
                ext_f_end = min(num_frames, min(f_start + chunk_f, num_frames) + overlap_f)
                ext_t_start = ext_f_start * pf
                ext_t_end = ext_f_end * pf
                ext_f_start_g = ext_f_start + sp_frame_offset

                nearby_boundary_g = max(1, ext_f_start_g - self.num_nearby_frames)
                num_available = nearby_boundary_g - 1
                _is_dummy = (num_available <= 0)


                x_ext_ci = x[:, ext_t_start:ext_t_end]
                chunk_score_q = self.chunk_to_state_proj(x_ext_ci.mean(dim=1))
                if _is_dummy:
                    available_keys = frame_keys[:, 0:1]
                else:
                    available_keys = frame_keys[:, 1:nearby_boundary_g]
                scores = torch.einsum('bd,bnd->bn', chunk_score_q, available_keys)


                if scores.shape[1] < _ns_target:
                    _pad_n = _ns_target - scores.shape[1]
                    scores = torch.cat(
                        [scores, scores.new_full((scores.shape[0], _pad_n), float('-inf'))],
                        dim=1)
                _, top_indices_full = scores.topk(_ns_target, dim=1)
                top_indices_full = top_indices_full + 1

                if _is_dummy:


                    _dummy_score_anchors.append(chunk_score_q)


                    if self.select_gate_mode == 'learned':
                        _sg_dummy_feats = torch.zeros(B, 1, 3, device=x.device, dtype=torch.float32)
                        _, _sg_dummy_alpha = self.select_gate_head(_sg_dummy_feats, training=False)
                        _dummy_score_anchors.append(_sg_dummy_alpha.to(x.dtype))
                    continue


                num_select = min(self.num_select_frames, num_available)
                top_indices = top_indices_full[:, :num_select]


                keep_mask = None
                gate_vals_pre = None
                if self.select_gate_mode == 'zscore':
                    keep_mask = compute_select_keep(
                        chunk_score_q, available_keys, top_indices - 1,
                        kappa=self.select_gate_kappa,
                        cos_floor=self.select_gate_cos_floor,
                        min_candidates=self.select_gate_min_candidates,
                        mad_floor=self.select_gate_mad_floor,
                        min_keep=self.select_gate_min_keep,
                    )
                elif self.select_gate_mode == 'learned':
                    _sg_feats = compute_select_gate_features(
                        chunk_score_q, available_keys, top_indices - 1, _sg_t_frac_row)
                    gate_vals_pre, _sg_alpha = self.select_gate_head(
                        _sg_feats, training=self.training)

                    self._select_gate_reg_feats.append(_sg_feats.detach())


                scales = []
                for si in range(num_select):
                    _ss = self.select_scales[min(si, len(self.select_scales) - 1)]
                    sf = _scale_map_inv.get(_ss, 2)
                    scales.append(sf)


                if gate_vals_pre is not None:
                    precomputed_select[ci] = (top_indices, scales, keep_mask, gate_vals_pre)
                else:
                    precomputed_select[ci] = (top_indices, scales, keep_mask)


                for si in range(num_select):
                    gfi = top_indices[0, si].item()
                    _local_start = sp_frame_offset
                    _local_end = _local_start + num_frames
                    if gfi >= _local_start and gfi < _local_end:
                        continue

                    src_rank = min(gfi // _frames_per_rank, _sp_size - 1)
                    if gfi not in remote_frame_set:
                        remote_frame_set.add(gfi)
                        remote_frame_requests.setdefault(src_rank, []).append(gfi)


            if True:
                remote_token_cache, _exchange_received = exchange_frame_tokens(
                    requests=remote_frame_requests,
                    x=x,
                    per_frame_tokens=pf,
                    sp_frame_offset=sp_frame_offset,
                    num_local_frames=num_frames,
                    frames_per_rank=_frames_per_rank,
                )

        def _build_chunk(ci):

            f_start = ci * chunk_f
            f_end = min(f_start + chunk_f, num_frames)
            t_start = f_start * pf
            t_end = f_end * pf
            ext_f_start = max(0, f_start - overlap_f)
            ext_f_end = min(num_frames, f_end + overlap_f)
            ext_t_start = ext_f_start * pf
            ext_t_end = ext_f_end * pf


            f_start_g = f_start + sp_frame_offset
            ext_f_start_g = ext_f_start + sp_frame_offset

            x_ext = x[:, ext_t_start:ext_t_end]

            if not lazy_qkv:
                q_ext = q_full[:, ext_t_start:ext_t_end]
            else:
                q_ext = self.self_attn.norm_q(self.self_attn.q(x_ext))
                q_ext = rope_apply(q_ext, freqs[ext_t_start:ext_t_end], self.num_heads)

            k_parts, v_parts = [], []
            select_bias = None


            if ext_f_start_g > 0:
                if (self.sink_decay_mode == 'downsample' and k_sink_ds is not None
                        and ext_f_start_g >= self.sink_decay_onset):
                    k_parts.append(k_sink_ds); v_parts.append(v_sink_ds)
                else:
                    k_parts.append(k_sink); v_parts.append(v_sink)


            if not lazy_qkv:
                k_parts.append(k_full[:, ext_t_start:ext_t_end]); v_parts.append(v_full[:, ext_t_start:ext_t_end])
            else:
                k_local, v_local = _lazy_kv(x_ext, freqs[ext_t_start:ext_t_end])
                k_parts.append(k_local); v_parts.append(v_local)


            nearby_scale_factors = [2, 4, 8]
            for ni in range(self.num_nearby_frames):
                nearby_f_g = ext_f_start_g - 1 - ni
                if nearby_f_g <= 0: break
                sf = nearby_scale_factors[min(ni, len(nearby_scale_factors) - 1)]
                local_fidx = nearby_f_g - sp_frame_offset
                if 0 <= local_fidx < num_frames:

                    frame_tokens = x[:, local_fidx * pf : (local_fidx + 1) * pf]
                elif _sp and halo_prev is not None and halo_prev.shape[1] > 0:


                    halo_fidx = self.num_nearby_frames + local_fidx
                    if 0 <= halo_fidx < self.num_nearby_frames:
                        frame_tokens = halo_prev[:, halo_fidx * pf : (halo_fidx + 1) * pf]
                    else:
                        continue
                else:
                    continue
                k_near, v_near = _downsample_kv(frame_tokens, sf, global_frame_idx=nearby_f_g)
                k_parts.append(k_near); v_parts.append(v_near)


            nearby_boundary_g = max(1, ext_f_start_g - self.num_nearby_frames)
            num_available = nearby_boundary_g - 1
            if num_available > 0 and self.num_select_frames > 0:

                keep_mask = None
                gate_vals = None
                if _sp and ci in precomputed_select:
                    _sel_entry = precomputed_select[ci]
                    if len(_sel_entry) == 4:
                        top_indices, scales, keep_mask, gate_vals = _sel_entry
                    elif len(_sel_entry) == 3:
                        top_indices, scales, keep_mask = _sel_entry
                    else:
                        top_indices, scales = _sel_entry
                    num_select = top_indices.shape[1]
                else:

                    chunk_score_q = self.chunk_to_state_proj(x_ext.mean(dim=1))
                    available_keys = frame_keys[:, 1:nearby_boundary_g]
                    scores = torch.einsum('bd,bnd->bn', chunk_score_q, available_keys)
                    num_select = min(self.num_select_frames, num_available)
                    _, top_indices = scores.topk(num_select, dim=1)


                    if self.select_gate_mode == 'zscore':
                        keep_mask = compute_select_keep(
                            chunk_score_q, available_keys, top_indices,
                            kappa=self.select_gate_kappa,
                            cos_floor=self.select_gate_cos_floor,
                            min_candidates=self.select_gate_min_candidates,
                            mad_floor=self.select_gate_mad_floor,
                            min_keep=self.select_gate_min_keep,
                        )
                    elif self.select_gate_mode == 'learned':
                        _sg_feats = compute_select_gate_features(
                            chunk_score_q, available_keys, top_indices, _sg_t_frac_row)
                        gate_vals, _sg_alpha = self.select_gate_head(
                            _sg_feats, training=self.training)
                        self._select_gate_reg_feats.append(_sg_feats.detach())
                    top_indices = top_indices + 1
                    scales = [_scale_map.get(self.select_scales[min(si, len(self.select_scales) - 1)], 2)
                              for si in range(num_select)]


                _num_parts_before_select = len(k_parts)
                _prefix_len = sum(p.shape[1] for p in k_parts)
                _sel_si_order = []
                if keep_mask is not None:
                    self._select_gate_last_stats.append(
                        (int(keep_mask.sum().item()), keep_mask.numel()))
                if gate_vals is not None:

                    self._select_gate_last_stats.append(
                        (int((gate_vals > 0).sum().item()), gate_vals.numel()))

                for si in range(num_select):
                    frame_idx = top_indices[:, si]
                    sf = scales[si]
                    gfi = frame_idx[0].item()


                    _local_start = sp_frame_offset
                    _local_end = _local_start + num_frames
                    _is_local = (gfi >= _local_start and gfi < _local_end)

                    if not _is_local and _sp:

                        if gfi in remote_token_cache:
                            _rtk = remote_token_cache[gfi]
                            if sf == 1:

                                _rf = freqs_full[gfi * pf:(gfi + 1) * pf] if freqs_full is not None else None
                                if _rf is not None:
                                    k_sel = rope_apply(self.self_attn.norm_k(self.self_attn.k(_rtk)),
                                                       _rf, self.num_heads)
                                else:
                                    k_sel = self.self_attn.norm_k(self.self_attn.k(_rtk))
                                v_sel = self.self_attn.v(_rtk)
                            else:
                                k_sel, v_sel = _downsample_kv(_rtk, sf, global_frame_idx=gfi)
                            k_parts.append(k_sel); v_parts.append(v_sel)
                            _sel_si_order.append(si)
                        continue


                    local_frame_idx = frame_idx - sp_frame_offset if _sp else frame_idx

                    if sf == 1:

                        if not lazy_qkv:
                            token_offsets = torch.arange(pf, device=x.device)
                            gi = local_frame_idx.unsqueeze(-1) * pf + token_offsets.unsqueeze(0)
                            gi = gi.unsqueeze(-1).expand(-1, -1, D)
                            k_parts.append(torch.gather(k_full, 1, gi))
                            v_parts.append(torch.gather(v_full, 1, gi))
                            _sel_si_order.append(si)
                        else:
                            token_offsets = torch.arange(pf, device=x.device)
                            gi = local_frame_idx.unsqueeze(-1) * pf + token_offsets.unsqueeze(0)
                            gi_x = gi.unsqueeze(-1).expand(-1, -1, D)
                            x_sel = torch.gather(x, 1, gi_x)
                            freqs_sel = freqs[gi[0]]
                            sel_k, sel_v = _lazy_kv(x_sel, freqs_sel)
                            k_parts.append(sel_k); v_parts.append(sel_v)
                            _sel_si_order.append(si)
                    else:

                        t_off = local_frame_idx[0].item() * pf
                        frame_tokens = x[:, t_off : t_off + pf]
                        k_sel, v_sel = _downsample_kv(frame_tokens, sf, global_frame_idx=gfi)
                        k_parts.append(k_sel); v_parts.append(v_sel)
                        _sel_si_order.append(si)


                if gate_vals is not None and _sel_si_order:
                    _g_dtype = gate_vals.to(x.dtype)
                    for _j in range(len(v_parts) - _num_parts_before_select):
                        _pidx = _num_parts_before_select + _j
                        _g_col = _g_dtype[:, _sel_si_order[_j]].view(B, 1, 1)
                        v_parts[_pidx] = v_parts[_pidx] * _g_col


                elif keep_mask is not None and _sel_si_order:
                    _col = _prefix_len
                    _off_spans = []
                    for _j, _part in enumerate(k_parts[_num_parts_before_select:]):
                        _keep_col = keep_mask[:, _sel_si_order[_j]]
                        if not bool(_keep_col.all()):
                            _off_spans.append((_col, _col + _part.shape[1], _keep_col))
                        _col += _part.shape[1]
                    if _off_spans:
                        _total_len = sum(p.shape[1] for p in k_parts)
                        select_bias = torch.zeros(B, _total_len, device=x.device, dtype=x.dtype)
                        for _cs, _ce, _keep_col in _off_spans:
                            select_bias[~_keep_col, _cs:_ce] = -1e9

            k_ctx = torch.cat(k_parts, dim=1)
            v_ctx = torch.cat(v_parts, dim=1)
            return ext_t_start, ext_t_end, t_start, t_end, q_ext, k_ctx, v_ctx, x_ext, select_bias


        chunk_outputs = []

        for batch_start in range(0, num_chunks, cbs):
            batch_end = min(batch_start + cbs, num_chunks)

            batch_data = [_build_chunk(ci) for ci in range(batch_start, batch_end)]
            nb = len(batch_data)

            max_q_len = max(c[4].shape[1] for c in batch_data)
            max_kv_len = max(c[5].shape[1] for c in batch_data)

            q_batch = torch.zeros(nb * B, max_q_len, D, device=x.device, dtype=x.dtype)
            k_batch = torch.zeros(nb * B, max_kv_len, D, device=x.device, dtype=x.dtype)
            v_batch = torch.zeros(nb * B, max_kv_len, D, device=x.device, dtype=x.dtype)


            bias_batch = None
            if any(c[8] is not None for c in batch_data):
                bias_batch = torch.zeros(nb * B, max_kv_len, device=x.device, dtype=x.dtype)

            for ci, (_, _, _, _, q_ci, k_ci, v_ci, _, bias_ci) in enumerate(batch_data):
                ql, kvl = q_ci.shape[1], k_ci.shape[1]
                q_batch[ci*B:(ci+1)*B, :ql] = q_ci
                k_batch[ci*B:(ci+1)*B, :kvl] = k_ci
                v_batch[ci*B:(ci+1)*B, :kvl] = v_ci
                if bias_batch is not None and bias_ci is not None:
                    bias_batch[ci*B:(ci+1)*B, :kvl] = bias_ci


            batch_meta = [(d[0], d[1], d[2], d[3], d[4].shape[1], d[7]) for d in batch_data]
            del batch_data


            _attn_bias = bias_batch.view(nb * B, 1, 1, max_kv_len) if bias_batch is not None else None
            out_batch = flash_attention(q_batch, k_batch, v_batch, num_heads=self.num_heads, attn_bias=_attn_bias)
            del q_batch, k_batch, v_batch, bias_batch, _attn_bias

            for ci, (et_s, et_e, t_s, t_e, ql, x_ext) in enumerate(batch_meta):
                out_local = out_batch[ci*B:(ci+1)*B, :ql]
                if state is not None:
                    out_global = self._query_global_state(x_ext, state, z)
                    out_ext = out_local + gate_scalar * out_global
                else:
                    out_ext = out_local
                chunk_outputs.append((et_s, et_e, t_s, t_e, out_ext))

            del out_batch


        attn_output = torch.zeros(B, S, D, device=x.device, dtype=x.dtype)

        for i, (ext_start_i, ext_end_i, cs_i, ce_i, out_i) in enumerate(chunk_outputs):
            excl_start = ext_start_i if i == 0 else cs_i + overlap_tokens
            excl_end = ext_end_i if i == num_chunks - 1 else ce_i - overlap_tokens

            if excl_end > excl_start:
                offset = excl_start - ext_start_i
                attn_output[:, excl_start:excl_end] = out_i[:, offset:offset + (excl_end - excl_start)]

            if i < num_chunks - 1 and overlap_tokens > 0:
                ext_start_next = chunk_outputs[i + 1][0]
                _, _, _, _, out_next = chunk_outputs[i + 1]
                bz_start = ce_i - overlap_tokens
                bz_end = min(ce_i + overlap_tokens, S)
                bz_len = bz_end - bz_start
                out_curr = out_i[:, (bz_start - ext_start_i):(bz_start - ext_start_i + bz_len)]
                out_next_bz = out_next[:, (bz_start - ext_start_next):(bz_start - ext_start_next + bz_len)]
                attn_output[:, bz_start:bz_end] = alpha_prev[:, :bz_len, :] * out_curr + alpha_next[:, :bz_len, :] * out_next_bz


        if _sp:
            _nccl_anchor = torch.zeros(1, device=attn_output.device, dtype=attn_output.dtype)
            if halo_prev is not None and isinstance(halo_prev, torch.Tensor) and halo_prev.requires_grad:
                _nccl_anchor = _nccl_anchor + halo_prev.sum() * 0
            if _exchange_received is not None and isinstance(_exchange_received, torch.Tensor) and _exchange_received.requires_grad:
                _nccl_anchor = _nccl_anchor + _exchange_received.sum() * 0
            for _dsa in _dummy_score_anchors:
                if isinstance(_dsa, torch.Tensor) and _dsa.requires_grad:
                    _nccl_anchor = _nccl_anchor + _dsa.sum() * 0
            if _nccl_anchor.requires_grad:
                attn_output = attn_output + _nccl_anchor


        if self.select_gate_mode == 'learned' and getattr(self, 'select_gate_head', None) is not None:
            _sg_anchor_feats = torch.zeros(1, 1, 3, device=x.device, dtype=torch.float32)
            _, _sg_anchor_alpha = self.select_gate_head(_sg_anchor_feats, training=False)
            if _sg_anchor_alpha.requires_grad:
                attn_output = attn_output + (_sg_anchor_alpha.sum() * 0).to(attn_output.dtype)

        return self.self_attn.o(attn_output)

    def forward(
        self, x, context, t_mod, freqs,
        tokens_per_frame: int = None,
        spatial_hw: tuple = None,
        **kwargs,
    ):


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


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
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
        select_gate_cos_floor: float = -1.0,
        select_gate_min_candidates: int = 8,
        select_gate_mad_floor: float = 1e-6,
        select_gate_min_keep: int = 0,

        select_gate_temp: float = 0.6667,
        select_gate_budget_target: float = 0.5,
        select_gate_budget_weight: float = 0.0,

        sink_decay_mode: str = 'none',
        sink_decay_onset: int = 40,
        sink_decay_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.num_heads = num_heads


        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        if chunk_batch_size or inner_checkpoint or lazy_qkv or select_scales:
            print(f"[MemOpt] chunk_batch_size={chunk_batch_size}, inner_checkpoint={inner_checkpoint}, lazy_qkv={lazy_qkv}, select_scales={select_scales or ['1x','2x','4x','8x']}")

        if sink_decay_mode != 'none':
            print(f"[SinkDecay] mode={sink_decay_mode}, onset={sink_decay_onset} latent-frames, factor={sink_decay_factor}x (far chunks use a downsampled sink to break first-frame lock-in)")


        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
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
                     sink_decay_factor=sink_decay_factor)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        self.has_image_pos_emb = has_image_pos_emb


        if teacher_config is None:

            teacher_config = {"dino": {"dim": teacher_dim, "enabled": True}}
        self.teacher_config = teacher_config
        self.repr_projs = nn.ModuleDict()
        for name, cfg in teacher_config.items():
            if cfg.get("enabled", False):
                t_dim = cfg["dim"]
                self.repr_projs[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, t_dim),
                )

    def _sparse_weights_already_loaded(self):

        for block in self.blocks:
            if hasattr(block, 'linear_attn') and hasattr(block.linear_attn, 'q'):
                if block.linear_attn.q.weight.abs().max() > 1e-8:
                    return True
                return False
        return False

    def load_sparse_checkpoint(self, ckpt_path):


        from safetensors.torch import load_file
        st = load_file(ckpt_path)
        sparse_keys = [
            'linear_attn', 'importance_head', 'blend_sharpness',
            'global_attn_gate', 'global_attn_out_proj',
            'chunk_to_state_proj', 'chunk_to_state_norm', 'linear_attn_norm',
            'select_gate_head',
        ]
        sparse_sd = {}
        for k, v in st.items():

            clean_k = k.replace('_orig_mod.', '')
            if any(sk in clean_k for sk in sparse_keys):
                sparse_sd[clean_k] = v
        if not sparse_sd:
            print(f"[Sparse] WARNING: No sparse keys found in {ckpt_path}")
            return
        missing, unexpected = self.load_state_dict(sparse_sd, strict=False)
        loaded = len(sparse_sd) - len(unexpected)
        print(f"[Sparse] Loaded {loaded} sparse keys from {ckpt_path}")
        if unexpected:
            print(f"[Sparse] Skipped {len(unexpected)} unexpected keys")

    def reinit_sparse_modules(self):


        if self._sparse_weights_already_loaded():
            print("[reinit_sparse_modules] Checkpoint already contains sparse weights, skipping reinit.")
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

        print("[reinit_sparse_modules] Reinitializing sparse attention modules...")
        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, 'sparse_attn') and block.sparse_attn:

                if hasattr(block, 'importance_head'):
                    safe_init(nn.init.xavier_uniform_, block.importance_head.weight)
                    safe_init(nn.init.zeros_, block.importance_head.bias)


                if hasattr(block, 'chunk_to_state_proj'):
                    safe_init(nn.init.xavier_uniform_, block.chunk_to_state_proj.weight)
                    safe_init(nn.init.zeros_, block.chunk_to_state_proj.bias)

                if hasattr(block, 'chunk_to_state_norm'):
                    safe_init(nn.init.ones_, block.chunk_to_state_norm.weight)

                if hasattr(block, 'global_attn_out_proj'):
                    safe_init(nn.init.zeros_, block.global_attn_out_proj.weight)
                    safe_init(nn.init.zeros_, block.global_attn_out_proj.bias)


                if hasattr(block, 'global_attn_gate'):
                    safe_init(nn.init.zeros_, block.global_attn_gate)


                if hasattr(block, 'blend_sharpness'):
                    safe_init(nn.init.zeros_, block.blend_sharpness)


                if getattr(block, 'select_gate_head', None) is not None:
                    _sgh = block.select_gate_head
                    safe_init(nn.init.xavier_uniform_, _sgh.net[0].weight)
                    safe_init(nn.init.zeros_, _sgh.net[0].bias)
                    safe_init(nn.init.zeros_, _sgh.net[-1].weight)
                    safe_init(nn.init.constant_, _sgh.net[-1].bias, _sgh.logit_bias_init)


                if hasattr(block, 'linear_attn'):
                    la = block.linear_attn
                    for proj_name in ['q', 'k', 'v', 'o']:
                        if hasattr(la, proj_name):
                            proj = getattr(la, proj_name)
                            if hasattr(proj, 'weight') and proj.weight is not None:
                                safe_init(nn.init.xavier_uniform_, proj.weight)
                            if hasattr(proj, 'bias') and proj.bias is not None:
                                safe_init(nn.init.zeros_, proj.bias)


                    for norm_name in ['norm_q', 'norm_k']:
                        if hasattr(la, norm_name):
                            norm = getattr(la, norm_name)
                            if hasattr(norm, 'weight') and norm.weight is not None:
                                safe_init(nn.init.ones_, norm.weight)


                if hasattr(block, 'linear_attn_norm'):
                    norm = block.linear_attn_norm
                    if hasattr(norm, 'weight') and norm.weight is not None:
                        safe_init(nn.init.ones_, norm.weight)
                    if hasattr(norm, 'bias') and norm.bias is not None:
                        safe_init(nn.init.zeros_, norm.bias)


        if hasattr(self, 'repr_projs'):
            for proj_name, proj in self.repr_projs.items():
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        safe_init(nn.init.xavier_uniform_, m.weight)
                        safe_init(nn.init.zeros_, m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                print(f"  [repr_proj] Reinitialized '{proj_name}' projection head")

        print(f"[reinit_sparse_modules] Reinitialized sparse modules in {len(self.blocks)} blocks.")

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

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
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)


        x_latent = x

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)


        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()


        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)


        _sp_active = getattr(self, 'sp_enabled', False)
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
        return x

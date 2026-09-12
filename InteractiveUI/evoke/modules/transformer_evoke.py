

import glob
import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import apply_lora_scale, deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from .evoke_kernels import attn_varlen_func, create_navit_attention_masks


logger = logging.get_logger(__name__)


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def pool_history_visible_mask(mask, patch_size):

    if mask is None:
        return None
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"history visible mask must be 4D/5D, got {tuple(mask.shape)}")
    mask = pad_for_3d_conv(mask.float(), patch_size)
    return torch.nn.functional.avg_pool3d(mask, kernel_size=patch_size, stride=patch_size)


def resolve_history_keep_mask(keep_mask, threshold: float = 0.5):

    if keep_mask is None:
        return None
    if keep_mask.ndim == 5:
        if keep_mask.shape[1] != 1:
            raise ValueError(f"history visible mask channel dimension must be 1, got {tuple(keep_mask.shape)}")
        keep_mask = keep_mask[:, 0]
    if keep_mask.ndim != 4:
        raise ValueError(f"history visible mask must reduce to [B,T,H,W], got {tuple(keep_mask.shape)}")
    keep_flat = keep_mask.flatten(1)
    if keep_flat.dtype != torch.bool:
        keep_flat = keep_flat >= float(threshold)
    if keep_flat.shape[0] == 1:
        return keep_flat[0]
    if not torch.equal(keep_flat, keep_flat[0:1].expand_as(keep_flat)):
        raise ValueError("history visible masking currently requires identical masks across the batch.")
    return keep_flat[0]


def filter_history_tokens_by_mask(hidden_states, rope_freqs, keep_mask, threshold: float = 0.5):

    keep = resolve_history_keep_mask(keep_mask, threshold=threshold)
    if keep is None:
        return hidden_states, rope_freqs
    if bool(keep.all()):
        return hidden_states, rope_freqs
    if not bool(keep.any()):
        return hidden_states[:, :0], rope_freqs[:, :0]
    return hidden_states[:, keep, :], rope_freqs[:, keep, :]


def replace_history_tokens_by_mask(hidden_states, keep_mask, invisible_token, threshold: float = 0.5):

    keep = resolve_history_keep_mask(keep_mask, threshold=threshold)
    if keep is None or bool(keep.all()):
        return hidden_states
    if invisible_token is None:
        raise ValueError(
            "history invisible token mode requires transformer.history_invisible_token to be initialized. "
            "The released checkpoints do not carry this Parameter; use mode='filter' (the default) instead."
        )
    replace = (~keep).view(1, -1, 1)
    token = invisible_token.to(device=hidden_states.device, dtype=hidden_states.dtype)
    token = token.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
    return torch.where(replace, token, hidden_states)


def apply_rotary_emb_transposed(
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


def _compute_noise_slots(
    history_context_length: int,
    original_context_length: int,
    enable_navit: bool,
    original_context_length_list: list,
    warp_len_list: Optional[list] = None,
) -> List[Tuple[int, int]]:


    if enable_navit:
        slots: List[Tuple[int, int]] = []
        _off = 0
        _rev = original_context_length_list[::-1]
        _wrev = (warp_len_list[::-1] if warp_len_list is not None else [0] * len(_rev))
        for cur_len, w_len in zip(_rev, _wrev):
            _off += history_context_length + w_len
            slots.append((_off, _off + cur_len))
            _off += cur_len
        return slots
    return [(history_context_length, history_context_length + original_context_length)]


def _get_qkv_projections(attn: "EvokeAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:

            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:

            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


class Discriminator3DHead(nn.Module):
    def __init__(self, input_channel, cond_map_dim=768):
        super().__init__()

        self.head3d = nn.Sequential(
            nn.Conv3d(input_channel, cond_map_dim, 3, stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 4, stride=[2, 2, 2], padding=(1, 1, 1)),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 4, stride=[2, 2, 2], padding=(1, 1, 1)),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 3, stride=[2, 1, 1], padding=(1, 1, 1)),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(cond_map_dim, cond_map_dim, 3, stride=[2, 1, 1], padding=(1, 1, 1)),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.Conv3d(
                cond_map_dim, cond_map_dim, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=(0, 1, 1)
            ),
            nn.GroupNorm(32, cond_map_dim),
            nn.SiLU(False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(cond_map_dim, 1),
        )

    def forward(self, x):
        return self.head3d(x)


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)


class EvokeOutputNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, original_context_length: int):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2).to(hidden_states.device), scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = (self.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        return hidden_states


class EvokeAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "EvokeAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

        self.kv_cache = None
        self.cache_enabled = False

    def enable_cache(self):
        self.cache_enabled = True
        self.kv_cache = None

    def disable_cache(self):
        self.cache_enabled = False
        self.kv_cache = None

    def clear_cache(self):
        self.kv_cache = None

    def __call__(
        self,
        attn: "EvokeAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        enable_navit: bool = False,
        is_first_denoising_step: bool = False,
        history_seq_len_override: Optional[int] = None,


        sp_plan=None,
    ) -> torch.Tensor:
        use_cache = False
        history_seq_len = None
        enable_cross = attn.is_cross_attention

        if not enable_cross:
            if sp_plan is not None:


                history_seq_len = sp_plan.hist_local
            elif history_seq_len_override is not None:

                history_seq_len = int(history_seq_len_override) // max(1, len(original_context_length_list))
            else:
                history_seq_len = (hidden_states.shape[1] - original_context_length) // len(original_context_length_list)

        if attn.restrict_self_attn:
            use_cache = self.cache_enabled and not is_first_denoising_step and self.kv_cache is not None
            assert not (use_cache and enable_navit), "Cache and NAViT are incompatible"

            if use_cache:
                key_history = self.kv_cache["key_history"]
                value_history = self.kv_cache["value_history"]
                history_hidden_states = self.kv_cache["history_hidden_states"]

                hidden_states = hidden_states[:, history_seq_len:]
                rotary_emb = rotary_emb[:, history_seq_len:] if rotary_emb is not None else None

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.restrict_self_attn and not use_cache:
            if enable_navit:
                seq_start = 0
                num_seqs = len(original_context_length_list)
                query_list = [None] * num_seqs
                key_list = [None] * num_seqs
                value_list = [None] * num_seqs
                query_history_list = [None] * num_seqs
                key_history_list = [None] * num_seqs
                value_history_list = [None] * num_seqs

                if attn.restrict_lora:
                    history_hidden_states_list = [None] * num_seqs

                if rotary_emb is not None:
                    rotary_emb_list = [None] * num_seqs
                    history_rotary_emb_list = [None] * num_seqs

                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    seq_end = seq_start + cur_seq_len + history_seq_len

                    slice_qkv = slice(seq_start, seq_end)
                    cur_query = query[:, slice_qkv, :]
                    cur_key = key[:, slice_qkv, :]
                    cur_value = value[:, slice_qkv, :]

                    query_history_list[idx] = cur_query[:, :history_seq_len]
                    query_list[idx] = cur_query[:, history_seq_len:]

                    key_history_list[idx] = cur_key[:, :history_seq_len]
                    key_list[idx] = cur_key[:, history_seq_len:]

                    value_history_list[idx] = cur_value[:, :history_seq_len]
                    value_list[idx] = cur_value[:, history_seq_len:]

                    if attn.restrict_lora:
                        cur_hidden = hidden_states[:, slice_qkv, :]
                        history_hidden_states_list[idx] = cur_hidden[:, :history_seq_len]

                    if rotary_emb is not None:
                        cur_rotary_emb = rotary_emb[:, slice_qkv, :]
                        history_rotary_emb_list[idx] = cur_rotary_emb[:, :history_seq_len]
                        rotary_emb_list[idx] = cur_rotary_emb[:, history_seq_len:]

                    seq_start = seq_end

                query = torch.cat(query_list, dim=1)
                key = torch.cat(key_list, dim=1)
                value = torch.cat(value_list, dim=1)
                query_history = torch.cat(query_history_list, dim=1)
                key_history = torch.cat(key_history_list, dim=1)
                value_history = torch.cat(value_history_list, dim=1)

                if attn.restrict_lora:
                    history_hidden_states = torch.cat(history_hidden_states_list, dim=1)
                    query_history = query_history + attn.q_loras(history_hidden_states)
                    key_history = key_history + attn.k_loras(history_hidden_states)
                    value_history = value_history + attn.v_loras(history_hidden_states)

                query_history = query_history.unflatten(2, (attn.heads, -1))
                key_history = key_history.unflatten(2, (attn.heads, -1))
                value_history = value_history.unflatten(2, (attn.heads, -1))

                if rotary_emb is not None:
                    rotary_emb = torch.cat(rotary_emb_list, dim=1)
                    history_rotary_emb = torch.cat(history_rotary_emb_list, dim=1)
                    query_history = apply_rotary_emb_transposed(query_history, history_rotary_emb)
                    key_history = apply_rotary_emb_transposed(key_history, history_rotary_emb)
            else:
                history_hidden_states = hidden_states[:, :history_seq_len]
                query_history, query = query[:, :history_seq_len], query[:, history_seq_len:]
                key_history, key = key[:, :history_seq_len], key[:, history_seq_len:]
                value_history, value = value[:, :history_seq_len], value[:, history_seq_len:]

                if attn.restrict_lora:
                    query_history = query_history + attn.q_loras(history_hidden_states)
                    key_history = key_history + attn.k_loras(history_hidden_states)
                    value_history = value_history + attn.v_loras(history_hidden_states)

                query_history = query_history.unflatten(2, (attn.heads, -1))
                key_history = key_history.unflatten(2, (attn.heads, -1))
                value_history = value_history.unflatten(2, (attn.heads, -1))

                if rotary_emb is not None:
                    history_rotary_emb, rotary_emb = (rotary_emb[:, :history_seq_len], rotary_emb[:, history_seq_len:])
                    query_history = apply_rotary_emb_transposed(query_history, history_rotary_emb)
                    key_history = apply_rotary_emb_transposed(key_history, history_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

        if attn.restrict_self_attn:
            if use_cache:
                key = torch.cat([key_history, key], dim=1)
                value = torch.cat([value_history, value], dim=1)
            else:
                if enable_navit:
                    num_seqs = len(original_context_length_list)

                    key_list = [None] * num_seqs
                    value_list = [None] * num_seqs

                    seq_start = 0
                    seq_start_history = 0

                    for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                        key_list[idx] = torch.cat(
                            [
                                key_history[:, seq_start_history : seq_start_history + history_seq_len, :],
                                key[:, seq_start : seq_start + cur_seq_len, :],
                            ],
                            dim=1,
                        )

                        value_list[idx] = torch.cat(
                            [
                                value_history[:, seq_start_history : seq_start_history + history_seq_len, :],
                                value[:, seq_start : seq_start + cur_seq_len, :],
                            ],
                            dim=1,
                        )

                        seq_start += cur_seq_len
                        seq_start_history += history_seq_len

                    key = torch.cat(key_list, dim=1)
                    value = torch.cat(value_list, dim=1)

                    history_hidden_states = attn_varlen_func(
                        query_history,
                        key_history,
                        value_history,
                        attention_mask=attention_mask[1],
                    )
                else:
                    key = torch.cat([key_history, key], dim=1)
                    value = torch.cat([value_history, value], dim=1)

                    history_hidden_states = attn_varlen_func(
                        query_history,
                        key_history,
                        value_history,
                    )
                history_hidden_states = history_hidden_states.flatten(2, 3)
                history_hidden_states = history_hidden_states.type_as(query)

                if self.cache_enabled and is_first_denoising_step and not enable_navit:
                    self.kv_cache = {
                        "key_history": key_history,
                        "value_history": value_history,
                        "history_hidden_states": history_hidden_states,
                    }

        if enable_cross and enable_navit:
            key = key.repeat(1, len(original_context_length_list), 1, 1)
            value = value.repeat(1, len(original_context_length_list), 1, 1)

        if not enable_cross and history_seq_len > 0 and attn.is_amplify_history:
            scale_key = attn.get_scale_key()
            if attn.history_scale_mode == "per_head":
                scale_key = scale_key.view(1, 1, -1, 1)

            if enable_navit:
                key_new = key.clone()
                seq_start = 0
                for cur_seq_len in original_context_length_list[::-1]:
                    hist_slice = slice(seq_start, seq_start + history_seq_len)
                    key_new[:, hist_slice] = key[:, hist_slice] * scale_key
                    seq_start += history_seq_len + cur_seq_len
                key = key_new
            else:
                key = torch.cat([key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]], dim=1)


        _sp_a2a = (sp_plan is not None) and (not enable_cross)
        if _sp_a2a:
            from evoke.modules import student_sp as _stu_sp
            query, key, value = _stu_sp.a2a_qkv_to_head(query, key, value, sp_plan.ctx)
            if sp_plan.has_pad:


                query = _stu_sp.drop_pad_tokens(query, sp_plan)
                key = _stu_sp.drop_pad_tokens(key, sp_plan)
                value = _stu_sp.drop_pad_tokens(value, sp_plan)

        hidden_states = attn_varlen_func(
            query,
            key,
            value,
            attention_mask=attention_mask[0] if isinstance(attention_mask, list) else attention_mask,
        )
        if _sp_a2a:
            if sp_plan.has_pad:
                hidden_states = _stu_sp.restore_pad_tokens(hidden_states, sp_plan)
            hidden_states = _stu_sp.a2a_head_to_seq(hidden_states.contiguous(), sp_plan.ctx)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if attn.restrict_self_attn:
            if enable_navit:
                num_seqs = len(original_context_length_list)
                hidden_states_list = [None] * num_seqs

                seq_start = 0
                seq_start_history = 0

                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    hidden_states_list[idx] = torch.cat(
                        [
                            history_hidden_states[:, seq_start_history : seq_start_history + history_seq_len, :],
                            hidden_states[:, seq_start : seq_start + cur_seq_len, :],
                        ],
                        dim=1,
                    )

                    seq_start += cur_seq_len
                    seq_start_history += history_seq_len

                hidden_states = torch.cat(hidden_states_list, dim=1)
            else:
                hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class EvokeAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The EvokeAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use EvokeAttnProcessor instead. "
        )
        deprecate("EvokeAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return EvokeAttnProcessor(*args, **kwargs)


class EvokeAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = EvokeAttnProcessor
    _available_processors = [EvokeAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
        restrict_self_attn=False,
        is_train_restrict_lora=False,
        restrict_lora=False,
        restrict_lora_rank=128,
        is_amplify_history=False,
        history_scale_mode="per_head",
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

        self.restrict_self_attn = restrict_self_attn
        self.restrict_lora = restrict_lora
        if restrict_lora:
            self.init_lora(is_train=is_train_restrict_lora, lora_rank=restrict_lora_rank)

        self.is_amplify_history = is_amplify_history
        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0
            self.register_buffer("_scale_cache", None)

    def get_scale_key(self):
        if self.history_key_scale.requires_grad:
            scale = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
        else:
            if self._scale_cache is None:
                self._scale_cache = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
            scale = self._scale_cache
        return scale

    def init_lora(self, is_train=False, lora_rank=128):
        dim = self.inner_dim
        self.q_loras = LoRALinearLayer(dim, dim, rank=lora_rank)
        self.k_loras = LoRALinearLayer(dim, dim, rank=lora_rank)
        self.v_loras = LoRALinearLayer(dim, dim, rank=lora_rank)

        requires_grad = is_train
        for lora in [self.q_loras, self.k_loras, self.v_loras]:
            for param in lora.parameters():
                param.requires_grad = requires_grad

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        enable_navit: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            original_context_length,
            original_context_length_list,
            enable_navit,
            **kwargs,
        )


class EvokeTimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        is_return_encoder_hidden_states: bool = True,
    ):
        B = None
        F = None
        if timestep.ndim == 2:
            B, F = timestep.shape
            timestep = timestep.flatten()

        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if B is not None and F is not None:
            temb = temb.reshape(B, F, -1)
            timestep_proj = timestep_proj.reshape(B, F, -1)

        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class EvokeRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.DT), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.DY), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.DX), persistent=False)

    def _get_freqs_base(self, dim):
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    @torch.no_grad()
    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        gy = torch.arange(height, device=device, dtype=torch.float32)
        gx = torch.arange(width, device=device, dtype=torch.float32)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")
        return GY, GX

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device, spatial_scale=None, spatial_offset=None):
        B = frame_indices.shape[0]
        T = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        GY, GX = self._get_spatial_meshgrid(height, width, str(device))


        if spatial_scale is not None:
            _sy, _sx = spatial_scale
            _oy, _ox = (spatial_offset if spatial_offset is not None else (0.0, 0.0))
            GY = GY * float(_sy) + float(_oy)
            GX = GX * float(_sx) + float(_ox)

        GT = frame_indices[:, :, None, None].expand(B, T, height, width)
        GY_batch = GY[None, None, :, :].expand(B, T, -1, -1)
        GX_batch = GX[None, None, :, :].expand(B, T, -1, -1)

        FCT, FST = self.get_frequency_batched(self.freqs_base_t, GT)
        FCY, FSY = self.get_frequency_batched(self.freqs_base_y, GY_batch)
        FCX, FSX = self.get_frequency_batched(self.freqs_base_x, GX_batch)

        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)

        return result.permute(1, 0, 2, 3, 4)


@maybe_allow_in_graph
class EvokeTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        restrict_self_attn: bool = False,
        guidance_cross_attn: bool = False,
        is_train_restrict_lora: bool = False,
        restrict_lora: bool = False,
        restrict_lora_rank: int = 128,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",

        cam_ctrl: bool = False,
        cam_rank: int = 128,
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = EvokeAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=EvokeAttnProcessor(),
            restrict_self_attn=restrict_self_attn,
            is_train_restrict_lora=is_train_restrict_lora,
            restrict_lora=restrict_lora,
            restrict_lora_rank=restrict_lora_rank,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )

        self.attn2 = EvokeAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=EvokeAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.guidance_cross_attn = guidance_cross_attn

        self.cam_ctrl = cam_ctrl
        if cam_ctrl:
            from evoke.modules.camera_control import build_camera_modulation_lowrank_submodules
            build_camera_modulation_lowrank_submodules(self, dim=dim, cam_rank=cam_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        navit_hidden_attention_mask: Optional[torch.Tensor] = None,
        navit_encoder_attention_mask: Optional[torch.Tensor] = None,
        original_context_length: int = None,
        original_context_length_list: list = None,
        is_first_denoising_step: bool = False,
        cam_token_seq: Optional[torch.Tensor] = None,
        cam_noise_slots: Optional[list] = None,
        warp_len_list: Optional[list] = None,


        sp_plan=None,
    ) -> torch.Tensor:
        enable_navit = False
        if len(original_context_length_list) > 1:
            enable_navit = True

        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)

            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            None,
            navit_hidden_attention_mask,
            rotary_emb,
            original_context_length,
            original_context_length_list,
            enable_navit,
            is_first_denoising_step=is_first_denoising_step,


            **({"sp_plan": sp_plan} if sp_plan is not None else {}),
        )

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)


        if self.cam_ctrl and cam_token_seq is not None and cam_noise_slots is not None:
            from evoke.modules.camera_control import apply_cam_modulation
            hidden_states = apply_cam_modulation(
                hidden_states, cam_token_seq.to(hidden_states.dtype),
                self.cam_inj1_down_proj, self.cam_inj1_up_proj,
                self.cam_inj2_down_proj, self.cam_inj2_up_proj,
                self.cam_scale_down_proj, self.cam_scale_up_proj,
                self.cam_shift_down_proj, self.cam_shift_up_proj,
                cam_noise_slots,
            )

        if self.guidance_cross_attn:


            _warp_total = sum(warp_len_list) if warp_len_list is not None else 0
            history_seq_len = (
                hidden_states.shape[1] - original_context_length - _warp_total
            ) // len(original_context_length_list)


            if sp_plan is not None:
                history_seq_len = sp_plan.hist_local
            _wl_rev = (warp_len_list[::-1] if warp_len_list is not None
                       else [0] * len(original_context_length_list))

            if enable_navit:
                num_seqs = len(original_context_length_list)

                hidden_states_list = [None] * num_seqs
                history_hidden_states_list = [None] * num_seqs

                seq_start = 0
                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    _front = history_seq_len + _wl_rev[idx]
                    seq_end = seq_start + cur_seq_len + _front
                    cur_hidden_states = hidden_states[:, seq_start:seq_end, :]

                    history_hidden_states_list[idx] = cur_hidden_states[:, :_front]
                    hidden_states_list[idx] = cur_hidden_states[:, _front:]

                    seq_start += cur_seq_len + _front

                hidden_states = torch.cat(hidden_states_list, dim=1)

                norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states,
                    navit_encoder_attention_mask,
                    None,
                    original_context_length,
                    original_context_length_list,
                    enable_navit,
                )
                hidden_states = hidden_states + attn_output

                seq_start = 0
                for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                    cur_hidden_states = hidden_states[:, seq_start : seq_start + cur_seq_len, :]

                    hidden_states_list[idx] = torch.cat([history_hidden_states_list[idx], cur_hidden_states], dim=1)

                    seq_start += cur_seq_len

                hidden_states = torch.cat(hidden_states_list, dim=1)
            else:
                history_hidden_states, hidden_states = (
                    hidden_states[:, :history_seq_len],
                    hidden_states[:, history_seq_len:],
                )


                assert not (sp_plan is not None and hidden_states.shape[1] == 0), (
                    "[STU-SP] attn2 input is empty: the two-region split should leave every rank noise tokens. "
                    f"hist_local={getattr(sp_plan, 'hist_local', None)} lh={getattr(sp_plan, 'lh', None)} "
                    f"ln={getattr(sp_plan, 'ln', None)} -> the split plan is wrong; continuing would make the group's graphs differ and deadlock")
                norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states,
                    navit_encoder_attention_mask,
                    None,
                    original_context_length,
                    original_context_length_list,
                    enable_navit,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                navit_encoder_attention_mask,
                None,
                original_context_length,
                original_context_length_list,
                enable_navit,
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class EvokeTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):


    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "patch_short",
        "patch_mid",
        "patch_long",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["EvokeTransformerBlock", "EvokeOutputNorm"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "history_key_scale",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["EvokeTransformerBlock"]
    _cp_plan = {

        "blocks.*.attn1": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "rotary_emb": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.attn2": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.ffn": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        **{f"blocks.{i}.attn1": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.attn2": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.ffn": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        restrict_self_attn: bool = False,
        guidance_cross_attn: bool = False,
        is_train_restrict_lora: bool = False,
        restrict_lora: bool = False,
        restrict_lora_rank: int = 128,
        zero_history_timestep: bool = False,
        has_multi_term_memory_patch: bool = False,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
        is_use_gan: bool = False,
        is_use_gan_hooks: bool = False,
        is_use_gan_final: bool = False,
        gan_cond_map_dim: int = 768,
        gan_hooks: List[int] = [5, 15, 25, 35],

        use_raw_sink_frames: bool = False,

        enable_cam_control: bool = False,
        cam_rank: int = 128,
        cam_ctrl_layers: Optional[List[int]] = None,


        geo_warp_plucker_enabled: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = EvokeRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = EvokeTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )


        if enable_cam_control:
            cam_layer_set = set(range(num_layers)) if cam_ctrl_layers is None else set(cam_ctrl_layers)
        else:
            cam_layer_set = set()
        self.cam_ctrl_layer_set = cam_layer_set
        self.enable_cam_control = enable_cam_control
        self.cam_rank = cam_rank
        self.geo_warp_plucker_enabled = bool(geo_warp_plucker_enabled)

        self.blocks = nn.ModuleList(
            [
                EvokeTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    restrict_self_attn=restrict_self_attn,
                    guidance_cross_attn=guidance_cross_attn,
                    is_train_restrict_lora=is_train_restrict_lora,
                    restrict_lora=restrict_lora,
                    restrict_lora_rank=restrict_lora_rank,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                    cam_ctrl=(i in cam_layer_set),
                    cam_rank=cam_rank,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = EvokeOutputNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        self.init_weights()


        if (enable_cam_control or self.geo_warp_plucker_enabled) and not hasattr(self, "patch_embedding_wancamctrl"):
            from evoke.modules.camera_control import (
                build_camera_plucker_encoder_submodules,
                get_plucker_input_dim,
            )
            plucker_input_dim = get_plucker_input_dim(patch_size=tuple(patch_size))
            build_camera_plucker_encoder_submodules(
                self, dim=inner_dim, plucker_input_dim=plucker_input_dim,
            )

        self.zero_history_timestep = zero_history_timestep
        self.inner_dim = inner_dim
        if has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(in_channels, self.inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.patch_mid = nn.Conv3d(in_channels, self.inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
            self.patch_long = nn.Conv3d(in_channels, self.inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            self.initialize_weight_from_another_conv3d(self.patch_embedding)

        self.use_raw_sink_frames = bool(use_raw_sink_frames)

        self.is_use_gan = is_use_gan
        if is_use_gan:
            self.is_use_gan_hooks = is_use_gan_hooks
            self.is_use_gan_final = is_use_gan_final
            if is_use_gan_hooks:
                gan_heads = []
                self.gan_hooks = gan_hooks
                for hook in self.gan_hooks:
                    gan_heads.append((str(hook), Discriminator3DHead(inner_dim, gan_cond_map_dim)))
                self.gan_heads = nn.ModuleDict(gan_heads)
            if is_use_gan_final:
                self.gan_final_head = Discriminator3DHead(out_channels, gan_cond_map_dim)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        weight = weight[:, :16, :, :, :]

        sd = {
            "patch_short.weight": weight.clone(),
            "patch_short.bias": bias.clone(),
            "patch_mid.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2) / 8.0,
            "patch_mid.bias": bias.clone(),
            "patch_long.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4)
            / 64.0,
            "patch_long.bias": bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd, strict=False)

    def gradient_checkpointing_method(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            result = self._gradient_checkpointing_func(block, *args)
        else:
            result = block(*args)
        return result

    def enable_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "enable_cache"):
                block.attn1.processor.enable_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "disable_cache"):
                block.attn1.processor.disable_cache()

    def clear_kv_cache(self):
        for block in self.blocks:
            if hasattr(block.attn1, "processor") and hasattr(block.attn1.processor, "clear_cache"):
                block.attn1.processor.clear_cache()

    def _build_sync_warp_tokens(
        self,
        latents_history_short,
        indices_latents_history_short,
        target_latent_hw,
        mask_pool_factor,
        history_visible_mask_short,
        history_threshold,
        history_invisible_token_mode,
    ):


        lhs = latents_history_short.to(self.device, dtype=self.dtype)
        Hs, Ws = int(target_latent_hw[0]), int(target_latent_hw[1])
        if lhs.shape[-2] != Hs or lhs.shape[-1] != Ws:
            _t = lhs.shape[2]
            lhs = rearrange(lhs, "b c t h w -> (b t) c h w")
            lhs = F.interpolate(lhs, size=(Hs, Ws), mode="bilinear")
            lhs = rearrange(lhs, "(b t) c h w -> b c t h w", t=_t)
        lhs = self.gradient_checkpointing_method(self.patch_short, lhs)
        _, _, T_short, H1, W1 = lhs.shape
        lhs = lhs.flatten(2).transpose(1, 2)
        _warp_mlp = getattr(self, "warp_residual_mlp", None)
        if _warp_mlp is not None:
            lhs = lhs + self.gradient_checkpointing_method(_warp_mlp, lhs)
        rope_s = self.rope(
            frame_indices=indices_latents_history_short, height=H1, width=W1, device=lhs.device,
        )
        rope_s = rope_s.flatten(2).transpose(1, 2)
        if history_visible_mask_short is not None:
            keep_mask = pool_history_visible_mask(history_visible_mask_short, mask_pool_factor)
            if history_invisible_token_mode == "global":
                lhs = replace_history_tokens_by_mask(
                    lhs, keep_mask, getattr(self, "history_invisible_token", None), threshold=history_threshold,
                )
            else:
                lhs, rope_s = filter_history_tokens_by_mask(
                    lhs, rope_s, keep_mask, threshold=history_threshold,
                )
        return lhs, rope_s, int(lhs.shape[1])

    def process_input_hidden_states(
        self,
        latents,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        sink_latents=None,
        nearby_sink_latents=None,
        nearby_sink_indices=None,

        history_visible_mask_short=None,
        history_visible_mask_mid=None,
        history_visible_mask_long=None,
        attention_kwargs=None,
        cam_plucker_emb=None,
    ):

        history_threshold = 0.5
        history_invisible_token_mode = "none"


        _geo_warp_frames = 0
        _geo_prev_short_frames = 0
        _noise_center_on = False
        if attention_kwargs:
            history_threshold = float(attention_kwargs.get("history_visible_token_threshold", 0.5) or 0.5)
            history_invisible_token_mode = str(attention_kwargs.get("history_invisible_token_mode", "none") or "none")
            _geo_warp_frames = int(attention_kwargs.get("geo_warp_frames", 0) or 0)
            _geo_prev_short_frames = int(attention_kwargs.get("geo_prev_short_frames", 0) or 0)
            _noise_center_on = bool(attention_kwargs.get("warp_rope_noise_center_align", False))


        _nc_full_hpp = _nc_full_wpp = None
        if _noise_center_on and latents_history_short is not None:
            _nc_ph, _nc_pw = int(self.config.patch_size[1]), int(self.config.patch_size[2])
            _nc_full_hpp = int(latents_history_short.shape[-2]) // _nc_ph
            _nc_full_wpp = int(latents_history_short.shape[-1]) // _nc_pw

        def _noise_rope(_fi, _H, _W, _dev):

            if _noise_center_on and _nc_full_hpp is not None and (_H < _nc_full_hpp or _W < _nc_full_wpp):
                _sch = _nc_full_hpp / max(1, _H)
                _scw = _nc_full_wpp / max(1, _W)
                return self.rope(frame_indices=_fi, height=_H, width=_W, device=_dev,
                                 spatial_scale=(_sch, _scw), spatial_offset=((_sch - 1.0) / 2.0, (_scw - 1.0) / 2.0))
            return self.rope(frame_indices=_fi, height=_H, width=_W, device=_dev)


        _sync_warp = bool(
            attention_kwargs
            and str(attention_kwargs.get("stage2_warp_compression_mode", "fixed_mem")) == "synchronized"
            and isinstance(latents, list)
            and latents_history_short is not None
            and indices_latents_history_short is not None
        )
        warp_tokens_list = None
        warp_rope_list = None
        warp_len_list = None
        height_list = []
        width_list = []
        temporal_list = []
        seq_list = []
        if isinstance(latents, list):
            hidden_states = None
            rope_freqs = None
            for idx, cur_hidden_states in enumerate(latents):
                cur_hidden_states = self.gradient_checkpointing_method(
                    self.patch_embedding, cur_hidden_states.to(self.device, dtype=self.dtype)
                )
                B, C, T, H, W = cur_hidden_states.shape

                cur_hidden_states = cur_hidden_states.flatten(2).transpose(1, 2)

                if indices_hidden_states is None:
                    indices_hidden_states = torch.arange(0, T).unsqueeze(0).expand(B, -1)

                cur_indices_latents = indices_hidden_states
                cur_rope_freqs = _noise_rope(cur_indices_latents, H, W, cur_hidden_states.device)
                cur_rope_freqs = cur_rope_freqs.flatten(2).transpose(1, 2)

                height_list.append(H)
                width_list.append(W)
                temporal_list.append(T)
                seq_list.append(cur_hidden_states.shape[1])

                if hidden_states is None:
                    hidden_states = cur_hidden_states
                    rope_freqs = cur_rope_freqs
                else:
                    hidden_states = torch.cat([cur_hidden_states, hidden_states], dim=1)
                    rope_freqs = torch.cat([cur_rope_freqs, rope_freqs], dim=1)
        else:
            hidden_states = self.gradient_checkpointing_method(self.patch_embedding, latents)
            B, C, T, H, W = hidden_states.shape

            if indices_hidden_states is None:
                indices_hidden_states = torch.arange(0, T).unsqueeze(0).expand(B, -1)

            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            rope_freqs = _noise_rope(indices_hidden_states, H, W, hidden_states.device)
            rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

            height_list.append(H)
            width_list.append(W)
            temporal_list.append(T)
            seq_list.append(hidden_states.shape[1])


        _geo_plk_token_seq = None
        if getattr(self, "geo_warp_plucker_enabled", False) and cam_plucker_emb is not None:
            if isinstance(latents, list) or isinstance(cam_plucker_emb, list):

                if not getattr(self, "_geo_plk_pyramid_warned", False):
                    print(
                        "[GEO-plucker] WARNING: additive plucker is not wired for the pyramid/list path yet; "
                        "skipping injection (non-pyramid path is unaffected).", flush=True,
                    )
                    self._geo_plk_pyramid_warned = True
            else:
                from evoke.modules.camera_control import process_cam_plucker_to_tokens
                _geo_plk_token_seq = process_cam_plucker_to_tokens(
                    cam_plucker_emb.to(hidden_states.device, dtype=hidden_states.dtype),
                    self.patch_embedding_wancamctrl,
                    self.c2ws_hidden_states_layer1,
                    self.c2ws_hidden_states_layer2,
                    patch_size=(1, 2, 2),
                )

                _n_noise = int(seq_list[-1])
                if _geo_plk_token_seq.shape[1] == _n_noise:
                    hidden_states = hidden_states + _geo_plk_token_seq
                else:
                    print(
                        f"[GEO-plucker] WARNING: cam_token_seq len {_geo_plk_token_seq.shape[1]} != noise "
                        f"token count {_n_noise}; skipping noise add.", flush=True,
                    )
                    _geo_plk_token_seq = None


        if nearby_sink_latents is not None:
            assert indices_latents_history_short is not None or nearby_sink_indices is not None, (
                "nearby_sink_latents needs nearby_sink_indices or indices_latents_history_short for the frame index"
            )
            nearby_sink_lat = nearby_sink_latents.to(hidden_states)
            nearby_emb = self.gradient_checkpointing_method(self.patch_embedding, nearby_sink_lat)
            B_n, _, T_ns, H_ns, W_ns = nearby_emb.shape
            nearby_tokens = nearby_emb.flatten(2).transpose(1, 2)
            if nearby_sink_indices is not None:

                nearby_indices = nearby_sink_indices.to(nearby_sink_lat.device)
                if nearby_indices.ndim == 1:
                    nearby_indices = nearby_indices.unsqueeze(0)
            else:

                nearby_indices = indices_latents_history_short[:, -T_ns:].to(nearby_sink_lat.device)
            nearby_rope = self.rope(
                frame_indices=nearby_indices, height=H_ns, width=W_ns, device=nearby_sink_lat.device,
            )
            nearby_rope = nearby_rope.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([nearby_tokens, hidden_states], dim=1)
            rope_freqs = torch.cat([nearby_rope, rope_freqs], dim=1)


        if _sync_warp:
            _T_full = int(latents_history_short.shape[2])
            if _geo_warp_frames > 0:

                _Pf = _T_full - _geo_warp_frames - _geo_prev_short_frames
                _warp_lat = latents_history_short[:, :, _Pf:_Pf + _geo_warp_frames]
                _warp_idx = indices_latents_history_short[:, _Pf:_Pf + _geo_warp_frames]
                _anchor_lat = torch.cat(
                    [latents_history_short[:, :, :_Pf], latents_history_short[:, :, _Pf + _geo_warp_frames:]], dim=2
                )
                _anchor_idx = torch.cat(
                    [indices_latents_history_short[:, :_Pf], indices_latents_history_short[:, _Pf + _geo_warp_frames:]], dim=1
                )
                if history_visible_mask_short is not None:
                    _warp_vis = history_visible_mask_short[:, :, _Pf:_Pf + _geo_warp_frames]
                else:
                    _warp_vis = None
            else:

                _anchor_n = 1
                _anchor_lat = latents_history_short[:, :, :_anchor_n]
                _anchor_idx = indices_latents_history_short[:, :_anchor_n]
                _warp_lat = latents_history_short[:, :, _anchor_n:]
                _warp_idx = indices_latents_history_short[:, _anchor_n:]
                _warp_vis = history_visible_mask_short[:, :, _anchor_n:] if history_visible_mask_short is not None else None


            _anchor_emb = self.gradient_checkpointing_method(
                self.patch_short, _anchor_lat.to(self.device, dtype=self.dtype)
            )
            _, _, _, H1, W1 = _anchor_emb.shape
            _anchor_tok = _anchor_emb.flatten(2).transpose(1, 2)
            _anchor_rope = self.rope(
                frame_indices=_anchor_idx, height=H1, width=W1, device=_anchor_tok.device,
            ).flatten(2).transpose(1, 2)
            hidden_states = torch.cat([_anchor_tok.to(hidden_states), hidden_states], dim=1)
            rope_freqs = torch.cat([_anchor_rope, rope_freqs], dim=1)


            warp_tokens_list, warp_rope_list, warp_len_list = [], [], []
            _full_h = max(int(lt.shape[-2]) for lt in latents)
            _full_w = max(int(lt.shape[-1]) for lt in latents)
            for idx in range(len(latents)):
                _hs, _ws = int(latents[idx].shape[-2]), int(latents[idx].shape[-1])
                _fh = max(1, _full_h // _hs)
                _fw = max(1, _full_w // _ws)
                _wt, _wr, _wl = self._build_sync_warp_tokens(
                    latents_history_short=_warp_lat,
                    indices_latents_history_short=_warp_idx,
                    target_latent_hw=(_hs, _ws),
                    mask_pool_factor=(1, 2 * _fh, 2 * _fw),
                    history_visible_mask_short=_warp_vis,
                    history_threshold=history_threshold,
                    history_invisible_token_mode=history_invisible_token_mode,
                )
                warp_tokens_list.append(_wt)
                warp_rope_list.append(_wr)
                warp_len_list.append(_wl)
            print(f"[GEO-sync] synchronized: anchor={int(_anchor_lat.shape[2])}f (prefix+prev_short, full-res, shared) "
                  f"+ per-stage warp_len={warp_len_list} (noise seq={seq_list})", flush=True)

        if latents_history_short is not None and indices_latents_history_short is not None and not _sync_warp:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.gradient_checkpointing_method(self.patch_short, latents_history_short)
            _, _, T_short, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            _warp_mlp = getattr(self, "warp_residual_mlp", None)
            if _warp_mlp is not None and T_short > 1:
                _tokens_per_frame = int(H1 * W1)
                if _geo_warp_frames > 0:

                    _Pf = T_short - _geo_warp_frames - _geo_prev_short_frames
                    _w0 = int(_Pf * _tokens_per_frame)
                    _w1 = int((_Pf + _geo_warp_frames) * _tokens_per_frame)
                    _pre_part = latents_history_short[:, :_w0, :]
                    _warp_part = latents_history_short[:, _w0:_w1, :]
                    _post_part = latents_history_short[:, _w1:, :]
                    _warp_part = _warp_part + self.gradient_checkpointing_method(_warp_mlp, _warp_part)
                    latents_history_short = torch.cat([_pre_part, _warp_part, _post_part], dim=1)
                else:

                    _anchor_tokens = int(1 * _tokens_per_frame)
                    _anchor_part = latents_history_short[:, :_anchor_tokens, :]
                    _warp_part = latents_history_short[:, _anchor_tokens:, :]
                    _warp_part = _warp_part + self.gradient_checkpointing_method(_warp_mlp, _warp_part)
                    latents_history_short = torch.cat([_anchor_part, _warp_part], dim=1)


            if _geo_plk_token_seq is not None and _geo_warp_frames > 0:
                _tpf = int(H1 * W1)
                _Pf = T_short - _geo_warp_frames - _geo_prev_short_frames
                _w0 = int(_Pf * _tpf)
                _w1 = int((_Pf + _geo_warp_frames) * _tpf)
                _warp_len = _w1 - _w0
                if _warp_len == _geo_plk_token_seq.shape[1]:
                    _plk = _geo_plk_token_seq.to(latents_history_short.dtype)
                    latents_history_short = torch.cat(
                        [
                            latents_history_short[:, :_w0, :],
                            latents_history_short[:, _w0:_w1, :] + _plk,
                            latents_history_short[:, _w1:, :],
                        ],
                        dim=1,
                    )
                else:
                    print(
                        f"[GEO-plucker] WARNING: warp slice len {_warp_len} != cam_token_seq len "
                        f"{_geo_plk_token_seq.shape[1]}; skipping warp add.", flush=True,
                    )

            rope_freqs_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rope_freqs_history_short = rope_freqs_history_short.flatten(2).transpose(1, 2)

            if history_visible_mask_short is not None:
                keep_mask_short = pool_history_visible_mask(history_visible_mask_short, (1, 2, 2))
                if history_invisible_token_mode == "global":
                    latents_history_short = replace_history_tokens_by_mask(
                        latents_history_short,
                        keep_mask_short,
                        getattr(self, "history_invisible_token", None),
                        threshold=history_threshold,
                    )
                else:
                    latents_history_short, rope_freqs_history_short = filter_history_tokens_by_mask(
                        latents_history_short,
                        rope_freqs_history_short,
                        keep_mask_short,
                        threshold=history_threshold,
                    )

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_short, rope_freqs], dim=1)

        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.gradient_checkpointing_method(self.patch_mid, latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rope_freqs_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rope_freqs_history_mid = pad_for_3d_conv(rope_freqs_history_mid, (2, 2, 2))
            rope_freqs_history_mid = center_down_sample_3d(rope_freqs_history_mid, (2, 2, 2))
            rope_freqs_history_mid = rope_freqs_history_mid.flatten(2).transpose(1, 2)

            if history_visible_mask_mid is not None:
                keep_mask_mid = pool_history_visible_mask(history_visible_mask_mid, (2, 4, 4))
                if history_invisible_token_mode == "global":
                    latents_history_mid = replace_history_tokens_by_mask(
                        latents_history_mid,
                        keep_mask_mid,
                        getattr(self, "history_invisible_token", None),
                        threshold=history_threshold,
                    )
                else:
                    latents_history_mid, rope_freqs_history_mid = filter_history_tokens_by_mask(
                        latents_history_mid,
                        rope_freqs_history_mid,
                        keep_mask_mid,
                        threshold=history_threshold,
                    )

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_mid, rope_freqs], dim=1)

        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.gradient_checkpointing_method(self.patch_long, latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rope_freqs_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rope_freqs_history_long = pad_for_3d_conv(rope_freqs_history_long, (4, 4, 4))
            rope_freqs_history_long = center_down_sample_3d(rope_freqs_history_long, (4, 4, 4))
            rope_freqs_history_long = rope_freqs_history_long.flatten(2).transpose(1, 2)

            if history_visible_mask_long is not None:
                keep_mask_long = pool_history_visible_mask(history_visible_mask_long, (4, 8, 8))
                if history_invisible_token_mode == "global":
                    latents_history_long = replace_history_tokens_by_mask(
                        latents_history_long,
                        keep_mask_long,
                        getattr(self, "history_invisible_token", None),
                        threshold=history_threshold,
                    )
                else:
                    latents_history_long, rope_freqs_history_long = filter_history_tokens_by_mask(
                        latents_history_long,
                        rope_freqs_history_long,
                        keep_mask_long,
                        threshold=history_threshold,
                    )

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rope_freqs = torch.cat([rope_freqs_history_long, rope_freqs], dim=1)


        if sink_latents is not None:
            sink_lat = sink_latents.to(hidden_states)
            sink_emb = self.gradient_checkpointing_method(self.patch_embedding, sink_lat)
            B_s, _, T_sink, H_sink, W_sink = sink_emb.shape
            sink_tokens = sink_emb.flatten(2).transpose(1, 2)
            sink_indices = torch.zeros(B_s, T_sink, device=sink_lat.device, dtype=torch.long)
            sink_rope = self.rope(
                frame_indices=sink_indices, height=H_sink, width=W_sink, device=sink_lat.device,
            )
            sink_rope = sink_rope.flatten(2).transpose(1, 2)
            hidden_states = torch.cat([sink_tokens, hidden_states], dim=1)
            rope_freqs = torch.cat([sink_rope, rope_freqs], dim=1)

        return (
            hidden_states,
            rope_freqs,
            height_list,
            width_list,
            temporal_list,
            seq_list,
            warp_tokens_list,
            warp_rope_list,
            warp_len_list,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        sink_latents=None,
        nearby_sink_latents=None,
        nearby_sink_indices=None,

        history_visible_mask_short=None,
        history_visible_mask_mid=None,
        history_visible_mask_long=None,
        is_first_denoising_step: bool = False,
        cam_plucker_emb=None,
        gan_mode: bool = False,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,


        sf_student_sp_ctx=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        _multi_term_present = (
            latents_history_short is not None or latents_history_mid is not None or latents_history_long is not None
        )
        if _multi_term_present:

            assert (
                len(
                    {
                        x is None
                        for x in [
                            indices_hidden_states,
                            indices_latents_history_short,
                            indices_latents_history_mid,
                            indices_latents_history_long,
                            latents_history_short,
                            latents_history_mid,
                            latents_history_long,
                        ]
                    }
                )
                == 1
            ), "All history latents and indices must either all exist or all be None"

        if indices_hidden_states is not None and indices_hidden_states.ndim == 1:
            indices_hidden_states = indices_hidden_states.unsqueeze(0)
        if indices_latents_history_short is not None and indices_latents_history_short.ndim == 1:
            indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        if indices_latents_history_mid is not None and indices_latents_history_mid.ndim == 1:
            indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        if indices_latents_history_long is not None and indices_latents_history_long.ndim == 1:
            indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        if gan_mode:
            assert self.is_use_gan

        if isinstance(hidden_states, list):
            assert gan_mode is False and self.is_use_gan is False
            enable_navit = True
            navit_len = len(hidden_states)
            batch_size = hidden_states[0].shape[0]
        else:
            enable_navit = False
            batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        (
            hidden_states,
            rotary_emb,
            post_patch_height_list,
            post_patch_width_list,
            post_patch_num_frames_list,
            original_context_length_list,
            warp_tokens_list,
            warp_rope_list,
            warp_len_list,
        ) = self.process_input_hidden_states(
            latents=hidden_states,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            sink_latents=sink_latents,
            nearby_sink_latents=nearby_sink_latents,
            nearby_sink_indices=nearby_sink_indices,
            history_visible_mask_short=history_visible_mask_short,
            history_visible_mask_mid=history_visible_mask_mid,
            history_visible_mask_long=history_visible_mask_long,
            attention_kwargs=attention_kwargs,
            cam_plucker_emb=cam_plucker_emb,
        )
        post_patch_num_frames = sum(post_patch_num_frames_list)
        post_patch_height = sum(post_patch_height_list)
        post_patch_width = sum(post_patch_width_list)
        original_context_length = sum(original_context_length_list)
        history_context_length = hidden_states.shape[1] - original_context_length


        _has_sync_warp = warp_len_list is not None and any(int(w) > 0 for w in warp_len_list)
        if _has_sync_warp:
            assert not (gan_mode and self.is_use_gan), (
                "stage2_warp_compression_mode=synchronized is incompatible with GAN "
                "(noise tokens are not trailing-contiguous)."
            )
            assert not self.config.restrict_self_attn, (
                "stage2_warp_compression_mode=synchronized is incompatible with restrict_self_attn "
                "(uniform per-stage history slicing)."
            )

        if indices_hidden_states is not None and self.zero_history_timestep:
            if isinstance(timestep, list):
                timestep_t0 = torch.zeros((1), dtype=timestep[0].dtype, device=timestep[0].device)
            else:
                timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        navit_hidden_attention_mask = None
        navit_encoder_attention_mask = None
        if enable_navit:
            assert navit_len == len(original_context_length_list)
            navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask = (
                create_navit_attention_masks(
                    batch_size=batch_size,
                    original_context_length_list=original_context_length_list[::-1],
                    history_context_length=history_context_length,
                    encoder_hidden_states_seq_len=encoder_hidden_states.shape[1],
                    device=hidden_states.device,
                    restrict_self_attn=self.config.restrict_self_attn,
                    guidance_cross_attn=self.config.guidance_cross_attn,
                    warp_len_list=(warp_len_list[::-1] if warp_len_list is not None else None),
                )
            )
            navit_hidden_attention_mask = [navit_hidden_attention_mask, navit_history_hidden_attention_mask]

            history_hidden_states, hidden_states = (
                hidden_states[:, :history_context_length],
                hidden_states[:, history_context_length:],
            )
            history_rotary_emb, rotary_emb = (
                rotary_emb[:, :history_context_length],
                rotary_emb[:, history_context_length:],
            )
            timestep = timestep[::-1]

            hidden_states_list = [None] * navit_len
            rotary_emb_list = [None] * navit_len
            temb_list = [None] * navit_len
            timestep_proj_list = [None] * navit_len


            _warp_tok_rev = warp_tokens_list[::-1] if warp_tokens_list is not None else None
            _warp_rope_rev = warp_rope_list[::-1] if warp_rope_list is not None else None

            seq_start = 0
            for idx, cur_seq_len in zip(range(navit_len), original_context_length_list[::-1]):
                cur_hidden_states = hidden_states[:, seq_start : seq_start + cur_seq_len, :]
                cur_rotary_emb = rotary_emb[:, seq_start : seq_start + cur_seq_len, :]

                if _warp_tok_rev is not None:

                    _warp_s = _warp_tok_rev[idx]
                    _warp_rope_s = _warp_rope_rev[idx]
                    _warp_len_s = _warp_s.shape[1]
                    hidden_states_list[idx] = torch.cat([history_hidden_states, _warp_s, cur_hidden_states], dim=1)
                    rotary_emb_list[idx] = torch.cat([history_rotary_emb, _warp_rope_s, cur_rotary_emb], dim=1)
                else:
                    _warp_len_s = 0
                    hidden_states_list[idx] = torch.cat([history_hidden_states, cur_hidden_states], dim=1)
                    rotary_emb_list[idx] = torch.cat([history_rotary_emb, cur_rotary_emb], dim=1)

                seq_start += cur_seq_len

                if idx == 0:
                    cur_temb, cur_timestep_proj, encoder_hidden_states = self.condition_embedder(
                        timestep[idx], encoder_hidden_states
                    )
                else:
                    cur_temb, cur_timestep_proj, _ = self.condition_embedder(
                        timestep[idx], encoder_hidden_states, is_return_encoder_hidden_states=False
                    )

                cur_temb = cur_temb.view(batch_size, 1, -1).expand(-1, cur_seq_len, -1)
                cur_timestep_proj = cur_timestep_proj.view(batch_size, 6, 1, -1).expand(-1, -1, cur_seq_len, -1)

                if self.zero_history_timestep:

                    _t0_w = history_context_length + _warp_len_s
                    _temb_t0_s = temb_t0[:, :1, :].expand(batch_size, _t0_w, -1)
                    _tsp_t0_s = timestep_proj_t0[:, :, :1, :].expand(batch_size, 6, _t0_w, -1)
                    temb_list[idx] = torch.cat([_temb_t0_s, cur_temb], dim=1)
                    timestep_proj_list[idx] = torch.cat([_tsp_t0_s, cur_timestep_proj], dim=2)
                else:
                    temb_list[idx] = cur_temb
                    timestep_proj_list[idx] = cur_timestep_proj

            hidden_states = torch.cat(hidden_states_list, dim=1)
            rotary_emb = torch.cat(rotary_emb_list, dim=1)
            temb = torch.cat(temb_list, dim=1)
            timestep_proj = torch.cat(timestep_proj_list, dim=2)
        else:
            temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
            timestep_proj = timestep_proj.unflatten(-1, (6, -1))

            if indices_hidden_states is not None and not self.zero_history_timestep:
                main_repeat_size = hidden_states.shape[1]
            else:
                main_repeat_size = original_context_length
            temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
            timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

            if indices_hidden_states is not None and self.zero_history_timestep:
                temb = torch.cat([temb_t0, temb], dim=1)
                timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)


        _noise_slots = _compute_noise_slots(
            history_context_length=history_context_length,
            original_context_length=original_context_length,
            enable_navit=enable_navit,
            original_context_length_list=original_context_length_list,
            warp_len_list=warp_len_list,
        )


        cam_token_seq = None
        cam_noise_slots = None
        if getattr(self, "enable_cam_control", False) and cam_plucker_emb is not None:
            from evoke.modules.camera_control import process_cam_plucker_to_tokens
            ps = tuple(self.config.patch_size)
            if isinstance(cam_plucker_emb, list):
                assert len(cam_plucker_emb) == len(original_context_length_list), (
                    f"[CamCtrl] cam_plucker_emb list len {len(cam_plucker_emb)} != "
                    f"hidden_states list len {len(original_context_length_list)}"
                )
                _cam_acc = None
                for _emb in cam_plucker_emb:
                    _tok = process_cam_plucker_to_tokens(
                        _emb.to(hidden_states.device, dtype=hidden_states.dtype),
                        self.patch_embedding_wancamctrl,
                        self.c2ws_hidden_states_layer1,
                        self.c2ws_hidden_states_layer2,
                        patch_size=ps,
                    )
                    _cam_acc = _tok if _cam_acc is None else torch.cat([_tok, _cam_acc], dim=1)
                cam_token_seq = _cam_acc
            else:
                cam_token_seq = process_cam_plucker_to_tokens(
                    cam_plucker_emb.to(hidden_states.device, dtype=hidden_states.dtype),
                    self.patch_embedding_wancamctrl,
                    self.c2ws_hidden_states_layer1,
                    self.c2ws_hidden_states_layer2,
                    patch_size=ps,
                )
            cam_noise_slots = _noise_slots
            _total_cur = sum(e - s for s, e in cam_noise_slots)
            assert cam_token_seq.shape[1] == _total_cur, (
                f"[CamCtrl] cam_token_seq len {cam_token_seq.shape[1]} != total cur len {_total_cur}, "
                f"slots={cam_noise_slots} navit={enable_navit}"
            )

        logits_hidden = []
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()


        _sp_plan = None
        if sf_student_sp_ctx is not None:
            from evoke.modules import student_sp as _stu_sp


            _S_real = int(hidden_states.shape[1])
            assert not enable_navit, "[STU-SP] mechanism B does not support the NAViT path (absolute-position splits throughout the processor)"
            assert len(original_context_length_list) == 1, \
                f"[STU-SP] mechanism B requires a single-stage sequence, got {len(original_context_length_list)}"
            assert self.config.num_attention_heads % sf_student_sp_ctx.size == 0, \
                (f"[STU-SP] num_attention_heads={self.config.num_attention_heads} "
                 f"is not divisible by G_u={sf_student_sp_ctx.size}")
            _b_global = _S_real - original_context_length - (
                sum(warp_len_list) if warp_len_list is not None else 0)
            _sp_plan = _stu_sp.ShardPlan(sf_student_sp_ctx, _S_real, _b_global)


            _stu_sp.assert_same_in_group(_S_real, "S_real (token count)", sf_student_sp_ctx.group)
            _stu_sp.assert_same_in_group(_b_global, "history boundary b", sf_student_sp_ctx.group)

            assert timestep_proj.ndim == 4, (
                f"[STU-SP] mechanism B needs a per-token timestep_proj (ndim=4 after permute), got ndim={timestep_proj.ndim}")
            assert int(timestep_proj.shape[1]) == _S_real, (
                f"[STU-SP] timestep_proj token count {timestep_proj.shape[1]} != hidden_states {_S_real}"
                " -> AdaLN would be misaligned once sharded")


            _sc_h, _sc_tsp, _sc_rot = (
                (hidden_states, timestep_proj, rotary_emb)
                if _stu_sp.selfcheck_pending() else (None, None, None))
            hidden_states = _stu_sp.scatter_tokens(hidden_states, _sp_plan)
            rotary_emb = _stu_sp.scatter_tokens(rotary_emb, _sp_plan)
            timestep_proj = _stu_sp.scatter_tokens(timestep_proj, _sp_plan)


            timestep_proj = _stu_sp.prepare_timestep_projection(timestep_proj)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    cam_token_seq,
                    cam_noise_slots,
                    warp_len_list,
                    _sp_plan,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                    cam_token_seq,
                    cam_noise_slots,
                    warp_len_list,
                    _sp_plan,
                )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])


        if _sp_plan is not None:
            from evoke.modules import student_sp as _stu_sp
            hidden_states = _stu_sp.gather_tokens(hidden_states, _sp_plan)


            hidden_states = _stu_sp.scale_grad(hidden_states, float(_sp_plan.ctx.size))


            if _stu_sp.selfcheck_pending():
                def _sc_run(_h, _t, _r, _plan, _upto=None):


                    _hh = _stu_sp.scatter_tokens(_h, _plan) if _plan is not None else _h
                    _tt = _stu_sp.scatter_tokens(_t, _plan) if _plan is not None else _t
                    _rr = _stu_sp.scatter_tokens(_r, _plan) if _plan is not None else _r
                    for _blk in (self.blocks if _upto is None else self.blocks[:_upto]):
                        _hh = torch.utils.checkpoint.checkpoint(
                            _blk, _hh, encoder_hidden_states, _tt, _rr,
                            navit_hidden_attention_mask, navit_encoder_attention_mask,
                            original_context_length, original_context_length_list,
                            is_first_denoising_step, cam_token_seq, cam_noise_slots,
                            warp_len_list, _plan, use_reentrant=False)
                    return _stu_sp.gather_tokens(_hh, _plan) if _plan is not None else _hh

                _sc_full = (_sc_h, _sc_tsp, _sc_rot)
                with torch.no_grad():
                    _stu_sp.selfcheck_report(hidden_states, _sc_run(*_sc_full, None),
                                             _sp_plan, tag=f"S={_S_real}")
                _sc_ps = [p for p in self.blocks.parameters() if p.requires_grad]
                _stu_sp.set_param_names([(n, p) for n, p in self.blocks.named_parameters() if p.requires_grad])
                _stu_sp.selfcheck_grad(_sc_run, _sc_full, _sc_full, _sp_plan, _sc_ps,
                                       tag=f"S={_S_real}", depths=(1, 5, 10, 20),
                                       n_layers=len(self.blocks))

        if temb.ndim == 3:
            if not enable_navit:
                temb = temb[:, -original_context_length:, :]
            shift, scale = (self.norm_out.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.norm_out.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)


        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if enable_navit:
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

            output = []
            seq_start = 0
            _warp_len_rev = (warp_len_list[::-1] if warp_len_list is not None
                             else [0] * len(original_context_length_list))
            for (
                cur_original_context_length,
                cur_warp_len,
                cur_post_patch_num_frames,
                cur_post_patch_height,
                cur_post_patch_width,
            ) in zip(
                reversed(original_context_length_list),
                _warp_len_rev,
                reversed(post_patch_num_frames_list),
                reversed(post_patch_height_list),
                reversed(post_patch_width_list),
            ):

                _seg = cur_original_context_length + history_context_length + cur_warp_len
                cur_hidden_states = hidden_states[:, seq_start : seq_start + _seg, :]
                cur_hidden_states = cur_hidden_states[:, history_context_length + cur_warp_len:, :]
                cur_hidden_states = self.proj_out(cur_hidden_states)
                seq_start += _seg

                cur_hidden_states = cur_hidden_states.reshape(
                    batch_size,
                    cur_post_patch_num_frames,
                    cur_post_patch_height,
                    cur_post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                cur_hidden_states = cur_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                cur_hidden_states = cur_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                output.append(cur_hidden_states)

            output = output[::-1]
        else:
            hidden_states = hidden_states[:, -original_context_length:, :]
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


            if _sp_plan is not None:
                output = _stu_sp.scale_grad(output, 1.0 / float(_sp_plan.ctx.size))

        logits = []
        if gan_mode and self.is_use_gan:
            if self.is_use_gan_final:
                logits.append(self.gradient_checkpointing_method(self.gan_final_head, output))
            if self.is_use_gan_hooks:
                for idx, (_, gan_head) in enumerate(self.gan_heads.items()):
                    activation = rearrange(
                        logits_hidden[idx],
                        "b (f h w) c -> b c f h w",
                        f=post_patch_num_frames,
                        h=post_patch_height,
                        w=post_patch_width,
                    )
                    logits.append(self.gradient_checkpointing_method(gan_head, activation.contiguous()))
            logits = torch.cat(logits, dim=1) if len(logits) > 1 else logits[0]
            logits_hidden = None
            del logits_hidden

        if not return_dict:
            return (output, logits)

        return Transformer2DModelOutput(sample=output, logits=logits)

    def init_weights(self):


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.condition_embedder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.proj_out.weight)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        subfolder=None,
        transformer_additional_kwargs={},
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
        device_map="cpu",
        max_workers=8,
        use_default_loader=False,
    ):
        if use_default_loader:
            return super().from_pretrained(
                pretrained_model_path, subfolder=subfolder, device_map=device_map, torch_dtype=torch_dtype
            )

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from huggingface_hub import snapshot_download

        from diffusers.utils import WEIGHTS_NAME

        if os.path.exists(pretrained_model_path):
            if subfolder is not None:
                pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            print(f"Downloading from Hugging Face Hub: {pretrained_model_path}")
            cache_dir = snapshot_download(
                repo_id=pretrained_model_path,

            )
            pretrained_model_path = cache_dir
            if subfolder is not None:
                pretrained_model_path = os.path.join(cache_dir, subfolder)

        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        def remap_state_dict_keys(state_dict):

            remapped = {}
            for key, value in state_dict.items():
                new_key = key

                if key == "scale_shift_table":
                    new_key = "norm_out.scale_shift_table"
                    print(f"Remapping key: {key} -> {new_key}")
                remapped[new_key] = value
            return remapped

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.model_loading_utils import load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available

                if is_accelerate_available():
                    import accelerate

                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file

                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file

                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(f"Loading {len(model_files_safetensors)} safetensors files with {max_workers} workers...")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_file = {executor.submit(load_file, f): f for f in model_files_safetensors}
                        for future in as_completed(future_to_file):
                            _state_dict = future.result()
                            state_dict.update(_state_dict)

                state_dict = remap_state_dict_keys(state_dict)

                if diffusers_version >= "0.33.0":

                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                        keep_in_fp32_modules=cls._keep_in_fp32_modules,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )

                return model
            except Exception as e:
                print(f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead.")

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file

            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            print(f"Loading {len(model_files_safetensors)} safetensors files with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(load_file, f): f for f in model_files_safetensors}
                for future in as_completed(future_to_file):
                    _state_dict = future.result()
                    state_dict.update(_state_dict)

        state_dict = remap_state_dict_keys(state_dict)

        tmp_state_dict = {}
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        for name, param in model.named_parameters():
            should_keep_fp32 = any(pattern in name for pattern in cls._keep_in_fp32_modules)
            if should_keep_fp32:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(torch_dtype)
        model = model.to(device_map)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn2." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn2 Parameters: {sum(params) / 1e6} M")

        return model


if __name__ == "__main__":
    import os

    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
    os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"


    gan_mode = False
    is_use_gan_hooks = False
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "restrict_lora": False,
        "is_train_restrict_lora": False,
        "is_amplify_history": False,
        "history_scale_mode": "per_head",
        "is_use_gan": gan_mode,
        "is_use_gan_hooks": is_use_gan_hooks,
        "gan_hooks": [13, 21, 29],
        "gan_cond_map_dim": 768,


    }
    device = "cuda"
    weight_dtype = torch.bfloat16
    transformer = EvokeTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    transformer = transformer.to(device, dtype=weight_dtype)

    is_navit = False
    batch_size = 4
    max_length = 512
    if is_navit:
        noisy_model_input = [
            torch.randn(batch_size, 16, 9, 12, 20),
            torch.randn(batch_size, 16, 9, 24, 40),
            torch.randn(batch_size, 16, 9, 48, 80),
        ]
        timesteps = [
            torch.randint(0, 1000, (batch_size,)).to(device),
            torch.randint(0, 1000, (batch_size,)).to(device),
            torch.randint(0, 1000, (batch_size,)).to(device),
        ]
    else:
        noisy_model_input = torch.randn(batch_size, 16, 9, 48, 80).to(device, dtype=weight_dtype)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

    prompt_embeds = torch.randn(batch_size, max_length, 4096).to(device, dtype=weight_dtype)
    indices_hidden_states = torch.randint(0, 10, (batch_size, 9)).to(device)
    indices_latents_history_short = torch.randint(0, 3, (batch_size, 2)).to(device)
    indices_latents_history_mid = torch.randint(0, 3, (batch_size, 2)).to(device)
    indices_latents_history_long = torch.randint(0, 17, (batch_size, 16)).to(device)
    latents_history_short = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    latents_history_mid = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    latents_history_long = torch.randn(batch_size, 16, 16, 48, 80).to(device, dtype=weight_dtype)

    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short.to(weight_dtype),
        latents_history_mid=latents_history_mid.to(weight_dtype),
        latents_history_long=latents_history_long.to(weight_dtype),
        gan_mode=gan_mode,
        return_dict=False,
    )[0]

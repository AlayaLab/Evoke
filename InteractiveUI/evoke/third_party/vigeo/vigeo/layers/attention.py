

import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0

    def eviction(self, k: Tensor, v: Tensor, cache_budget: int, num_anchor_tokens: int):
        B, H, N, D = k.shape
        if N <= cache_budget:
            return k, v, 0.0

        anchor_k, candidate_k = k.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)
        anchor_v, candidate_v = v.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)

        num_to_keep = cache_budget - num_anchor_tokens
        if num_to_keep <= 0:
            return k, v, 0.0

        candidate_k_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = torch.mean(candidate_k_norm, dim=2, keepdim=True)

        scores = torch.sum(candidate_k_norm * mean_vector, dim=-1)


        avg_scores = scores.mean()
        if not getattr(self, "_evoke_defer_scores", False):
            avg_scores = avg_scores.item()
        trace = getattr(self, "_evoke_score_trace", None)
        if trace is not None:
            trace['evictions'].append({'tokens': N, 'budget': cache_budget,
                                      'anchors': num_anchor_tokens, 'kept': num_to_keep})

        _, top_indices = torch.topk(-scores, k=num_to_keep, dim=-1)
        top_indices = top_indices.sort(dim=-1).values

        expanded_indices = top_indices.unsqueeze(-1).expand(B, H, num_to_keep, D)
        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
        final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)

        return final_k, final_v, avg_scores

    def forward(self, x: Tensor, pos: Tensor = None, segment_lengths: list[int] = None,
            attn_mask=None, past_key_value=None, use_cache=False,
            cache_budget: int = None, tokens_per_frame: int = None):
        B, SN, C = x.shape
        qkv = self.qkv(x).reshape(B, SN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        scores = None

        if self.rope is not None:
            q = self.rope(q, pos).to(v.dtype)
            k = self.rope(k, pos).to(v.dtype)

        if use_cache and self.num_anchor_tokens == 0:
            self.num_anchor_tokens = tokens_per_frame if tokens_per_frame is not None else k.shape[2]

        if use_cache:
            curr_k, curr_v = k, v
            if past_key_value is not None:
                prev_k, prev_v = past_key_value
                attn_k = torch.cat([prev_k, curr_k], dim=2)
                attn_v = torch.cat([prev_v, curr_v], dim=2)
            else:
                attn_k, attn_v = curr_k, curr_v

            x_out = F.scaled_dot_product_attention(q, attn_k, attn_v)

            cache_k, cache_v = attn_k, attn_v
            if cache_budget is not None and cache_k.shape[2] > cache_budget:
                cache_k, cache_v, scores = self.eviction(cache_k, cache_v, cache_budget, self.num_anchor_tokens)

            new_kv = (cache_k, cache_v)
            x_out = x_out.transpose(1, 2).reshape(B, SN, C)
            return self.proj_drop(self.proj(x_out)), new_kv, scores

        if segment_lengths is None or len(segment_lengths) <= 1:
            x_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(attn_mask)[:, None].repeat(1, self.num_heads, 1, 1) if attn_mask is not None else None
            )
        else:
            outputs = []
            curr_idx = 0
            num_frames = sum(segment_lengths)
            tokens_per_frame = SN // num_frames
            for length in segment_lengths:
                segment_tokens = length * tokens_per_frame
                next_idx = curr_idx + segment_tokens
                q_chunk = q[:, :, curr_idx:next_idx]
                k_past  = k[:, :, :next_idx]
                v_past  = v[:, :, :next_idx]
                outputs.append(
                    F.scaled_dot_product_attention(
                        q_chunk, k_past, v_past,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    ))
                curr_idx = next_idx
            x_out = torch.cat(outputs, dim=2)

        x_out = x_out.transpose(1, 2).reshape(B, SN, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class MemEffAttention(Attention):
    def forward(self, x: Tensor, pos=None, attn_mask=None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos, attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

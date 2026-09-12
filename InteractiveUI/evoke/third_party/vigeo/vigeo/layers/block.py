

import os
import torch
from typing import Any
from .mlp import Mlp
from typing import Callable
from torch import nn, Tensor
from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True

    else:

        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, attn_mask=None, segment_lengths=None, use_cache=False, past_key_value=None,
                cache_budget=None, tokens_per_frame=None):
        current_kv = None
        scores = None
        def attn_residual_func(x: Tensor, pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths) -> Tensor:
            nonlocal current_kv, scores
            if use_cache:
                attn_out, current_kv, scores = self.attn(
                    self.norm1(x), pos=pos, use_cache=True, past_key_value=past_key_value,
                    cache_budget=cache_budget, tokens_per_frame=tokens_per_frame
                )
                return self.ls1(attn_out)
            else:
                return self.ls1(self.attn(
                    self.norm1(x), pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths
                ))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths)
        x = x + ffn_residual_func(x)
        return (x, current_kv, scores) if use_cache else x

class NestedTensorBlock(Block):
    def forward_nested(self, x_list: list[Tensor]) -> list[Tensor]:


        assert isinstance(self.attn, MemEffAttention)
        def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

        def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        attn_bias, x = get_attn_bias_and_cat(x_list)
        x = x + attn_residual_func(x, attn_bias=attn_bias)
        x = x + ffn_residual_func(x)
        return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

attn_bias_cache: dict[tuple, Any] = {}

def get_attn_bias_and_cat(x_list, branges=None):


    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors

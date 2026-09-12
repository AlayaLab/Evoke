

import math
import logging
from functools import partial
from collections.abc import Sequence
from typing import Any, Callable
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
from typing import Literal
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint
from .patch_embed import PatchEmbed
from .block import Block
from .swiglu_ffn import SwiGLUFFNFused
from .mlp import Mlp
from .attention import Attention

logger = logging.getLogger("dinov2")

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def init_weights_vit_timm(module: nn.Module, name: str = ""):

    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        use_checkpoint=False,
        interpolate_offset=0.1,
        alt_start=-1,
        rope=None,
    ):


        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start

        self.patch_start_idx = 1 + num_register_tokens

        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.use_checkpoint = use_checkpoint
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
            self.patch_start_idx += 1

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm= i >= alt_start if alt_start != -1 else False,
                rope=rope if i >= alt_start and alt_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.num_global_blocks = 0
        if self.alt_start != -1:
            self.num_global_blocks = sum(1 for i in range(depth) if i >= self.alt_start and i % 2 == 1)
            if self.num_global_blocks > 0:
                self.register_buffer("last_scores", torch.zeros(self.num_global_blocks), persistent=False)
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:


            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:

            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def reset_cache_state(self):
        for module in self.modules():
            if module is not self and hasattr(module, "_reset_cache_state"):
                module._reset_cache_state()
        if hasattr(self, "last_scores"):
            self.last_scores.zero_()

    def _calculate_dynamic_budgets(self, total_budget):
        if total_budget is None or total_budget <= 0:
            return [None] * self.num_global_blocks

        with torch.no_grad():
            diversity_scores = 1.0 - self.last_scores
            scaled_scores = diversity_scores / 0.5
            proportions = torch.softmax(scaled_scores, dim=0)
            budgets = proportions * total_budget
        return budgets.int().tolist()

    def process_attention(self, x, block, attn_type: Literal["global", "local"]="local", pos=None,
        segment_lengths=None, attn_mask=None, use_cache=False, past_key_value=None,
        cache_budget=None, tokens_per_frame=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            segment_lengths = None
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")

        if use_cache:
            x, current_kv, scores = block(
                x, pos=pos, use_cache=True, past_key_value=past_key_value,
                cache_budget=cache_budget, tokens_per_frame=tokens_per_frame
            )
        else:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, pos, attn_mask, segment_lengths, use_reentrant=False)
            else:
                x = block(x, pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths)
            current_kv = None
            scores = None

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return (x, current_kv, scores) if use_cache else x

    def get_semantic_feature(self, x):
        B, S, _, H, W = x.shape

        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.prepare_tokens_with_masks(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x_norm = self.norm(x)
        x_norm = rearrange(x_norm, "(b s) n c -> b s n c", b=B, s=S)

        return x[:, self.num_register_tokens + 1 :]

    def forward(self, x, segment_lengths=None, use_cache=False, past_key_values=None, total_budget=0, **kwargs):
        B, S, _, H, W = x.shape

        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.prepare_tokens_with_masks(x)
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)

        pos = kwargs.get("pos", None)
        new_kv_caches = [] if use_cache else None
        final_output = []
        current_budgets = []
        new_scores = []
        if use_cache and self.alt_start != -1:
            current_budgets = self._calculate_dynamic_budgets(total_budget)

        score_trace = getattr(self, "_evoke_score_trace", None)
        score_record = None
        if score_trace is not None and use_cache:
            score_record = {'currentBudgets': list(current_budgets), 'totalBudget': total_budget}
            score_trace['forwards'].append(score_record)
        defer_scores = getattr(self, "_evoke_defer_scores", False)
        global_block_idx = 0
        for i, blk in enumerate(self.blocks):
            if self.alt_start != -1 and i == self.alt_start:
                is_cont = (use_cache and past_key_values is not None and past_key_values[0] is not None)
                cam = self.camera_token[:, 1:2] if is_cont else self.camera_token[:, 0:1]
                if not is_cont and S > 1:
                    cam = torch.cat([cam, self.camera_token[:, 1:2].expand(-1, S-1, -1)], dim=1)
                else:
                    cam = cam.expand(-1, S, -1)
                cam = cam.expand(B, -1, -1)
                x = torch.cat((x[:, :, :1], cam.unsqueeze(2), x[:, :, 1:]), dim=2)

            tokens_per_frame = x.shape[2]
            is_global = (self.alt_start != -1 and i >= self.alt_start and i % 2 == 1)

            if use_cache and is_global:
                prev_kv = past_key_values[global_block_idx] if past_key_values else None
                current_budget = current_budgets[global_block_idx] if current_budgets else None
                x, current_kv, layer_scores = self.process_attention(
                    x, blk, "global", pos=pos, use_cache=True, past_key_value=prev_kv,
                    cache_budget=current_budget, tokens_per_frame=tokens_per_frame
                )
                new_kv_caches.append(current_kv)
                if layer_scores is not None:
                    new_scores.append(layer_scores)
                elif hasattr(self, "last_scores"):
                    score = self.last_scores[global_block_idx]
                    new_scores.append(score if defer_scores else score.item())
                global_block_idx += 1
            else:
                attn_type = "global" if (is_global and not use_cache) else "local"
                x = self.process_attention(x, blk, attn_type, pos=pos, segment_lengths=segment_lengths)

            if i+1 in [len(self.blocks)-1, len(self.blocks)]:
                final_output.append(x)

        if use_cache and hasattr(self, "last_scores") and len(new_scores) > 0:
            if defer_scores:


                new_scores = torch.stack([
                    score.to(self.last_scores) if isinstance(score, torch.Tensor)
                    else self.last_scores.new_tensor(score) for score in new_scores])
                self.last_scores.copy_(new_scores)
            else:
                self.last_scores.copy_(torch.tensor(new_scores, device=self.last_scores.device, dtype=self.last_scores.dtype))
            if score_record is not None:
                score_record['scores'] = self.last_scores.clone()

        final_output = torch.cat(final_output, dim=-1)
        camera_tokens = final_output[:, :, 1] if self.alt_start != -1 else None

        x_norm = torch.cat(
            [self.norm(final_output[..., : self.embed_dim]), self.norm(final_output[..., self.embed_dim :])], dim=-1)

        output = {
            "x_norm_clstoken": x_norm[:, :, 0:1],
            "x_norm_patchtokens": x_norm[:, :, self.patch_start_idx:],
            "x_norm_regtokens": x_norm[:, :, self.patch_start_idx - self.num_register_tokens : self.patch_start_idx + 1],
            "x_norm": x_norm,
            "camera_tokens": camera_tokens,
        }
        if use_cache:
            output["new_kv_caches"] = new_kv_caches

        return output

def vit_small(patch_size=14, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=14, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        alt_start=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=14, num_register_tokens=0, alt_start=8, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        alt_start=alt_start,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=14, num_register_tokens=0, alt_start=13, **kwargs):


    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        alt_start=alt_start,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens,
        use_checkpoint=True,
        **kwargs,
    )
    return model

def DINOv2(model_name, patch_size=14, num_register_tokens=4, **kwargs):
    model_zoo = {
        "vits": vit_small,
        "vitb": vit_base,
        "vitl": vit_large,
        "vitg": vit_giant2
    }

    return model_zoo[model_name](
        img_size=518,
        patch_size=patch_size,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.1,
        **kwargs
    )

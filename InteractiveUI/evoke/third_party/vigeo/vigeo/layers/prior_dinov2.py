

import math
import torch
from torch import nn
from functools import partial
from torch.nn.init import trunc_normal_
from typing import Literal
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFNFused
from .attention import MemEffAttention
from .block import NestedTensorBlock as Block
from .mask_utils import depth_masking, _compute_depth_invalid_mask
from typing import Callable

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class PriorDinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
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
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        use_additional_prior=False,
    ):


        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.alt_start = alt_start

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.depth_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        if use_additional_prior:
            self.coarse_depth_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))


        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        self.patch_start_idx = 1 + num_register_tokens

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

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
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _interpolate_pos_encoding(self, x, w, h, pos_embed_source):
        npatch, dim = x.shape[1], x.shape[-1]
        N = pos_embed_source.shape[1]
        if npatch == N and w == h: return pos_embed_source

        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M

        patch_pos_embed = nn.functional.interpolate(
            pos_embed_source.float().reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0), mode="bicubic", antialias=self.interpolate_antialias,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2).to(x.dtype)


    def prepare_tokens_with_masks(self, x_img, x_depth, x_depth_coarse: torch.Tensor | None = None):
        B, S, _, h_img, w_img = x_img.shape
        _, _, _, h_depth, w_depth = x_depth.shape
        BS = B * S

        valid_sparse_mask = (x_depth != 0)

        if x_depth_coarse is not None and hasattr(self, 'coarse_depth_patch_embed'):
            x_depth_combined_raw = x_depth_coarse.clone()
            x_depth_combined_raw[valid_sparse_mask] = x_depth[valid_sparse_mask]
            x_depth_coarse = x_depth_coarse.reshape(BS, x_depth_coarse.shape[2], h_depth, w_depth)
            x_coarse_depth_tokens = self.coarse_depth_patch_embed(x_depth_coarse)

        else:
            x_depth_combined_raw = x_depth.clone()
            x_coarse_depth_tokens = 0

        x_depth_combined_raw[x_depth_combined_raw == 0] = -10.0

        x_img = x_img.reshape(BS, -1, h_img, w_img)
        x_depth = x_depth.reshape(BS, -1, h_depth, w_depth)
        x_depth_combined_raw = x_depth_combined_raw.reshape(BS, -1, h_depth, w_depth)

        x_img_tokens = self.patch_embed(x_img)
        x_depth_tokens = self.depth_patch_embed(x_depth)

        patch_h, patch_w = h_depth // self.patch_size, w_depth // self.patch_size
        invalid_depth_patch_mask = _compute_depth_invalid_mask(
            depth_values=x_depth,
            H_patch=patch_h,
            W_patch=patch_w,
            threshold_ratio=None,
            threshold_num=1,
            valid_range=(-9.5, 200.0)
        )
        valid_depth_patch_mask = ~invalid_depth_patch_mask
        valid_mask_expanded = valid_depth_patch_mask.unsqueeze(-1).float()
        fused_depth_tokens = x_coarse_depth_tokens + x_depth_tokens * valid_mask_expanded

        patch_pos_embed = self.pos_embed[:, 1:, :]
        img_pose_enc = 1 + self._interpolate_pos_encoding(x_img_tokens, w_img, h_img, patch_pos_embed).repeat(BS, 1, 1)
        depth_pose_enc = 2 + self._interpolate_pos_encoding(x_depth_tokens, w_depth, h_depth, patch_pos_embed).repeat(BS, 1, 1)

        x_img_tokens = x_img_tokens + img_pose_enc
        fused_depth_tokens_with_pos = fused_depth_tokens + depth_pose_enc

        x_depth_masked, _ = depth_masking(
            fused_depth_tokens_with_pos,
            h_depth // self.patch_size, w_depth // self.patch_size,
            depth_values=x_depth_combined_raw,
            depth_mask_threshold_num=[1] * BS,
            valid_depth_range=(-9.5, 200.0)
        )

        x_cls = self.cls_token.squeeze(0) + self.pos_embed.squeeze(0)[:1]

        x_masked_list = []
        for i in range(BS):
            tokens_to_cat = [x_cls]
            if self.register_tokens is not None:
                tokens_to_cat.append(self.register_tokens.squeeze(0))
            tokens_to_cat.extend([x_img_tokens[i], x_depth_masked[i]])

            x_masked = torch.cat(tokens_to_cat, dim=0).unsqueeze(0)
            x_masked_list.append(x_masked)

        return x_masked_list, (B, S)

    def process_attention(self, x_list: list[torch.Tensor], block: Block, attn_type: Literal["local", "global"], shape_info: tuple) -> list[torch.Tensor]:
        B, S = shape_info
        if attn_type == "local": return block(x_list)
        if attn_type == "global":
            global_context = [
                torch.cat([frame.squeeze(0) for frame in x_list[i*S:(i+1)*S]], dim=0).unsqueeze(0)
                for i in range(B)
            ]
            processed = block(global_context)
            restored = []
            for i in range(B):
                lengths = [frame.shape[1] for frame in x_list[i*S:(i+1)*S]]
                for frame_tensor in torch.split(processed[i].squeeze(0), lengths, dim=0):
                    restored.append(frame_tensor.unsqueeze(0))
            return restored

    def forward(self, x_img: torch.Tensor, x_depth: torch.Tensor, x_depth_coarse: torch.Tensor | None = None):
        x, (B, S) = self.prepare_tokens_with_masks(x_img, x_depth, x_depth_coarse)

        local_x = x
        for i, blk in enumerate(self.blocks):
            is_global_turn = (self.alt_start != -1 and i >= self.alt_start and (i - self.alt_start) % 2 == 1)

            if is_global_turn:
                x = self.process_attention(x, blk, attn_type="global", shape_info=(B, S))
            else:
                x = self.process_attention(x, blk, attn_type="local", shape_info=(B, S))
                local_x = x

        processed_list = [torch.cat([self.norm(l), self.norm(c)], dim=-1) for l, c in zip(local_x, x)]

        h_img, w_img = x_img.shape[-2:]
        ph, pw = h_img // self.patch_size, w_img // self.patch_size
        num_img_patches = ph * pw

        cls_tokens = torch.cat(
            [t[:, 0:1] for t in processed_list],
            dim=0).squeeze(1).view(B, S, self.embed_dim * 2)
        img_patch_tokens = torch.cat(
            [t[:, self.patch_start_idx : self.patch_start_idx + num_img_patches] for t in processed_list],
            dim=0).view(B, S, num_img_patches, self.embed_dim * 2)

        return {
            "x_norm_clstoken": cls_tokens,
            "x_norm_patchtokens": img_patch_tokens,
        }

def init_weights_vit_timm(module: nn.Module, name: str = ""):

    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = PriorDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = PriorDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = PriorDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        alt_start=8,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):


    model = PriorDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def PriorDINOv2(model_name, patch_size=14, num_register_tokens=4, **kwargs):
    assert model_name == 'vitl', "Currently, only the 'vitl' model is supported."
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
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs
    )

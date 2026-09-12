import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from .layers import Mlp, Block, Attention
from torch.utils.checkpoint import checkpoint


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                attn_class=Attention,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, pos=None):
        if pos is not None:
            seq_len = hidden.shape[2]
            pos = pos[:, :, -seq_len:]
            pos = rearrange(pos, "b s n c -> (b s) n c")

        hidden = rearrange(hidden, "b s n c -> (b s) n c")
        hidden = self.projects(hidden)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=pos)
        out = self.linear_out(hidden)
        return out

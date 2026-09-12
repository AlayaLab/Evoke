

from typing import Dict as TyDict
from typing import List, Sequence
import torch
import torch.nn as nn

from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.utils.head_utils import activate_head_gs, custom_interpolate


class GSDPT(DPT):

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "linear",
        conf_activation: str = "sigmoid",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        conf_dim: int = 1,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            head_name="raw_gs",
            use_sky_head=False,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
        )
        self.conf_dim = conf_dim
        if conf_dim and conf_dim > 1:
            assert (
                conf_activation == "linear"
            ), "use linear prediction when using view-dependent opacity"

        merger_out_dim = features if feature_only else features // 2
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, merger_out_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 4, merger_out_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(merger_out_dim // 2, merger_out_dim, 3, 1, 1),
            nn.GELU(),
        )


    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        images: torch.Tensor,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)


        fused = self._fuse(resized_feats)
        fused = self.scratch.output_conv1(fused)


        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)


        fused = fused + self.images_merger(images)

        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)


        feat = fused


        main_logits = self.scratch.output_conv2(feat)
        outs: TyDict[str, torch.Tensor] = {}
        if self.has_conf:
            pred, conf = activate_head_gs(
                main_logits,
                activation=self.activation,
                conf_activation=self.conf_activation,
                conf_dim=self.conf_dim,
            )
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(main_logits).squeeze(1)

        return outs

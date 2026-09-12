from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from .conv_head import ConvHead
from .layers.dinov2 import DINOv2
from .camera_head import CameraHead
from .transformer_decoder import TransformerDecoder
from .layers import RotaryPositionEmbedding2D, PositionGetter
from .utils import depth_to_pointmap, pose_encoding_to_extri, recover_focal_from_xy


class ViGeo(nn.Module):
    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitg",
        with_mask_head: bool = False,
    ):
        super().__init__()
        self.patch_size = 14

        rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()

        self.pretrained = DINOv2(encoder, rope=rope)

        self.decoder = TransformerDecoder(
            in_dim=2 * self.pretrained.embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=rope,
        )

        conv_kwargs = dict(
            projects=nn.Identity(), dim_proj=1024, dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2, num_res_blocks=2,
            last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True
        )

        self.point_head = ConvHead(dim_out=3, **conv_kwargs)

        self.ray_head = ConvHead(dim_out=6, **conv_kwargs)

        self.camera_head = CameraHead(dim_in=2 * self.pretrained.embed_dim)
        self.normal_head = ConvHead(dim_out=3, **conv_kwargs)
        self.conf_head = ConvHead(dim_out=1, **conv_kwargs)
        self.mask_head = ConvHead(dim_out=1, **conv_kwargs) if with_mask_head else None

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        filename: str = "vigeo.pt",
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitg",
        **hf_kwargs,
    ) -> "ViGeo":
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            checkpoint_path = path / filename
        elif path.exists():
            checkpoint_path = path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=filename,
                repo_type="model",
                **hf_kwargs,
            )

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
        state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
        model = cls(
            encoder=encoder,
            with_mask_head=any(key.startswith("mask_head.") for key in state_dict),
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def _prepare_rope(self, B, T, H, W, device):
        pos = self.position_getter(B * T, H // self.patch_size, W // self.patch_size, device=device)
        pos = rearrange(pos, "(b t) n c -> b t n c", b=B)
        if self.pretrained.patch_start_idx > 0:
            pos_special = torch.zeros(B, T, self.pretrained.patch_start_idx, 2, dtype=pos.dtype, device=device)
            pos = torch.cat([pos_special, pos + 1], dim=2)
        return pos

    def _resolve_model_size(self, height: int, width: int, num_tokens: int | None):
        if num_tokens is None:
            patch_h = max(1, height // self.patch_size)
            patch_w = max(1, width // self.patch_size)
        else:
            aspect_ratio = width / height
            patch_h = max(1, round((num_tokens / aspect_ratio) ** 0.5))
            patch_w = max(1, round((num_tokens * aspect_ratio) ** 0.5))
        return patch_h * self.patch_size, patch_w * self.patch_size, patch_h, patch_w

    def _remap(self, points):
        xy, z = points.split([2, 1], dim=-3)
        z = z.clamp(-10, 10).exp()
        return torch.cat([xy * z, z], dim=-3), z

    def reset_cache_state(self):
        if hasattr(self.pretrained, "reset_cache_state"):
            self.pretrained.reset_cache_state()

    def forward(
        self,
        image_chunk: torch.Tensor,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        total_budget: int = 0,
        num_tokens: int | None = None,
        resize_output: bool = False,
        skip_unused_heads: bool = False,
    ):
        B, curr_step, C, orig_H, orig_W = image_chunk.shape
        H, W, patch_h, patch_w = self._resolve_model_size(orig_H, orig_W, num_tokens)
        if H != orig_H or W != orig_W:
            image_chunk = F.interpolate(
                image_chunk.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False
            ).view(B, curr_step, C, H, W)

        x = (image_chunk - self.image_mean) / self.image_std
        device_type = self.device.type

        with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
            pos = self._prepare_rope(B, curr_step, H, W, device=image_chunk.device)
            out = self.pretrained(
                x,
                pos=pos,
                use_cache=use_cache,
                past_key_values=past_key_values,
                total_budget=total_budget,
            )
            new_kv_caches = out.get("new_kv_caches", past_key_values)
            hidden = self.decoder(out["x_norm_patchtokens"], pos=pos)

        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            point_logits = self.point_head(hidden, patch_h, patch_w)
            pose = self.camera_head(out["camera_tokens"])


            normal = None if skip_unused_heads else self.normal_head(hidden, patch_h, patch_w)
            conf = self.conf_head(hidden, patch_h, patch_w)
            mask = self.mask_head(hidden, patch_h, patch_w) if self.mask_head is not None and not skip_unused_heads else None

            if resize_output:
                def resize_back(tensor: torch.Tensor | None):
                    if tensor is None:
                        return None
                    if tensor.shape[-2:] == (orig_H, orig_W):
                        return tensor
                    return F.interpolate(
                        tensor, size=(orig_H, orig_W), mode='bilinear', align_corners=False
                    )

                point_logits = resize_back(point_logits)
                normal = resize_back(normal)
                conf = resize_back(conf)
                mask = resize_back(mask)

            out_H, out_W = (orig_H, orig_W) if resize_output else (H, W)
            points, depth = self._remap(point_logits)

            def format_pred(t):
                if t is None:
                    return None
                return t.reshape(B, curr_step, -1, out_H, out_W)

        return {
            'point_logits': format_pred(point_logits),
            'points': format_pred(points),
            'depth': format_pred(depth),
            'pose': pose,
            'normal': format_pred(normal),
            'conf': format_pred(conf),
            'mask': format_pred(mask),
            'new_kv_caches': new_kv_caches,
        }

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        mode: Literal["offline", "chunk", "online"] = "offline",
        chunk_size: int = 16,
        num_tokens: int | None = 1369,
        total_budget: int = 0,
        resize_output: bool = True,
        kv_caches: tuple | list | None = None,
        reset_cache: bool | None = None,
        skip_unused_heads: bool = False,
    ):
        if reset_cache is None:
            reset_cache = kv_caches is None
        if reset_cache:
            self.reset_cache_state()
            kv_caches = None

        is_unbatched = image.dim() == 4
        if is_unbatched:
            image = image.unsqueeze(0)

        image = image.to(dtype=self.dtype)
        B, T = image.shape[:2]

        if mode == "offline":
            step_size, use_cache = T, False
        elif mode == "chunk":
            step_size, use_cache = max(1, min(chunk_size, T)), True
        elif mode == "online":
            step_size, use_cache = 1, True
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        all_points, all_depths, all_point_logits, all_poses = [], [], [], []
        all_normals, all_confs, all_masks = [], [], []

        for t in range(0, T, step_size):
            end_t = min(t + step_size, T)
            image_chunk = image[:, t:end_t].to(device=self.device)

            output = self.forward(
                image_chunk=image_chunk,
                use_cache=use_cache,
                past_key_values=kv_caches,
                total_budget=total_budget,
                num_tokens=num_tokens,
                resize_output=resize_output,
                skip_unused_heads=skip_unused_heads,
            )
            kv_caches = output['new_kv_caches']

            if not resize_output:
                all_points.append(output['points'].cpu())
            all_depths.append(output['depth'].cpu())
            all_point_logits.append(output['point_logits'].cpu())
            all_poses.append(output['pose'].cpu())
            if output['normal'] is not None:
                all_normals.append(output['normal'].cpu())
            if output['conf'] is not None:
                all_confs.append(output['conf'].cpu())
            if output['mask'] is not None:
                all_masks.append(output['mask'].cpu())

        depth_pred = torch.cat(all_depths, dim=1)
        point_logits_pred = torch.cat(all_point_logits, dim=1)
        pose_pred = torch.cat(all_poses, dim=1)
        normal_pred = torch.cat(all_normals, dim=1) if all_normals else None
        conf_pred = torch.cat(all_confs, dim=1) if all_confs else None
        mask_pred = torch.cat(all_masks, dim=1) > 0.5 if all_masks else None

        if resize_output:
            focal_length = recover_focal_from_xy(point_logits_pred[:, :, :2])
            points_pred = depth_to_pointmap(depth_pred, focal_length)
        else:
            points_chw = torch.cat(all_points, dim=1)
            points_pred = points_chw.permute(0, 1, 3, 4, 2)

        out_dict = {
            'points_pred': points_pred,
            'depth_pred': depth_pred,
            'pose_pred': pose_encoding_to_extri(pose_pred),
            'normal_pred': normal_pred.permute(0, 1, 3, 4, 2) if normal_pred is not None else None,
            'conf_pred': conf_pred,
            'mask_pred': mask_pred,
            'kv_caches': kv_caches,
        }

        if is_unbatched:
            out_dict = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in out_dict.items()
            }
        return out_dict

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


ViGeoModel = ViGeo

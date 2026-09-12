

from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):


    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None):


        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:


        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
        )

        H, W = x.shape[-2], x.shape[-1]


        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)

        output = self._process_mono_sky_estimation(output)


        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        return output

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output

        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)


        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:

        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic)
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:

        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])

            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf


            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."


        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics

        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)


        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf


        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:

        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):

            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):


    def __init__(self, anyview: DictConfig, metric: DictConfig):


        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:


        output = self.da3(
            x, extrinsics, intrinsics, export_feat_layers=export_feat_layers, infer_gs=infer_gs, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy
        )
        metric_output = self.da3_metric(x)


        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:


        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:


        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)


        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"


        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)


        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )


        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)


        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:

        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)


        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)


        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output

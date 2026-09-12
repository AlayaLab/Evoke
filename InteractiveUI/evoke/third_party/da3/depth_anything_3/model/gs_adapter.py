

from typing import Optional
import torch
from einops import einsum, rearrange, repeat
from torch import nn

from depth_anything_3.model.utils.transform import cam_quat_xyzw_to_world_quat_wxyz
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.geometry import affine_inverse, get_world_rays, sample_image_grid
from depth_anything_3.utils.pose_align import batch_align_poses_umeyama
from depth_anything_3.utils.sh_helpers import rotate_sh


class GaussianAdapter(nn.Module):

    def __init__(
        self,
        sh_degree: int = 0,
        pred_color: bool = False,
        pred_offset_depth: bool = False,
        pred_offset_xy: bool = True,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max


        if not pred_color:
            self.register_buffer(
                "sh_mask",
                torch.ones((self.d_sh,), dtype=torch.float32),
                persistent=False,
            )
            for degree in range(1, sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depths: torch.Tensor,
        opacities: torch.Tensor,
        raw_gaussians: torch.Tensor,
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        gt_extrinsics: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        H, W = image_shape
        b, v = raw_gaussians.shape[:2]


        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H


        if self.pred_offset_depth:
            gs_depths = depths + raw_gaussians[..., -1]
            raw_gaussians = raw_gaussians[..., :-1]
        else:
            gs_depths = depths

        if gt_extrinsics is not None and not torch.equal(extrinsics, gt_extrinsics):
            try:
                _, _, pose_scales = batch_align_poses_umeyama(
                    gt_extrinsics.detach().float(),
                    extrinsics.detach().float(),
                )
            except Exception:
                pose_scales = torch.ones_like(extrinsics[:, 0, 0, 0])
            pose_scales = torch.clamp(pose_scales, min=1 / 3.0, max=3.0)
            cam2worlds[:, :, :3, 3] = cam2worlds[:, :, :3, 3] * rearrange(
                pose_scales, "b -> b () ()"
            )
            gs_depths = gs_depths * rearrange(pose_scales, "b -> b () () ()")

        xy_ray, _ = sample_image_grid((H, W), device)
        xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

        if self.pred_offset_xy:
            pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
            offset_xy = raw_gaussians[..., :2]
            xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]

        origins, directions = get_world_rays(
            xy_ray,
            repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
            repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        )
        gs_means_world = origins + directions * gs_depths[..., None]
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")


        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)


        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")


        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        cam_quat_xyzw = rearrange(rotations, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(
            cam2worlds,
            "b v i j -> b (v h w) i j",
            h=H,
            w=W,
        )
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        gs_rotations_world = world_quat_wxyz


        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if not self.pred_color:
            sh = sh * self.sh_mask

        if self.pred_color or self.sh_degree == 0:

            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")


        gs_opacities = rearrange(opacities, "b v h w ... -> b (v h w) ...")

        return Gaussians(
            means=gs_means_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities,
            scales=gs_scales,
            rotations=gs_rotations_world,
        )

    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,
        pixel_size: torch.Tensor,
        multiplier: float = 0.1,
    ) -> torch.Tensor:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return 1 if self.pred_color else (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:

        raw_gs_dim = 0
        if self.pred_offset_xy:
            raw_gs_dim += 2
        raw_gs_dim += 3
        raw_gs_dim += 4
        raw_gs_dim += 3 * self.d_sh
        if self.pred_offset_depth:
            raw_gs_dim += 1

        return raw_gs_dim

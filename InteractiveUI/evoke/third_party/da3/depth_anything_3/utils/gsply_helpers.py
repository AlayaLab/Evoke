

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from einops import rearrange, repeat
from plyfile import PlyData, PlyElement
from torch import Tensor

from depth_anything_3.specs import Gaussians


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Tensor,
    scales: Tensor,
    rotations: Tensor,
    harmonics: Tensor,
    opacities: Tensor,
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    if shift_and_scale:

        means = means - means.median(dim=0).values


        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    rotations = rotations.detach().cpu().numpy()


    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    if match_3dgs_mcmc_dev:
        sh_degree = 3
        n_rest = 3 * (sh_degree + 1) ** 2 - 3
        f_rest = repeat(
            torch.zeros_like(harmonics[..., :1]), "... i -> ... (n i)", n=(n_rest // 3)
        ).flatten(start_dim=1)
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(num_rest=n_rest)
            if attribute not in ("nx", "ny", "nz")
        ]
    else:
        dtype_full = [
            (attribute, "f4")
            for attribute in construct_list_of_attributes(
                0 if save_sh_dc_only else f_rest.shape[1]
            )
        ]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if match_3dgs_mcmc_dev:
        attributes.pop(1)
    elif save_sh_dc_only:
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def save_gaussian_ply(
    gaussians: Gaussians,
    save_path: str,
    ctx_depth: torch.Tensor,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    gs_views_interval: int = 1,
    inv_opacity: Optional[bool] = True,
    prune_by_depth_percent: Optional[float] = 1.0,
    prune_border_gs: Optional[bool] = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
):
    b = gaussians.means.shape[0]
    assert b == 1, "must set batch_size=1 when exporting 3D gaussians"
    src_v, out_h, out_w, _ = ctx_depth.shape


    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = inverse_sigmoid(gaussians.opacities) if inv_opacity else gaussians.opacities


    if prune_border_gs:
        mask = torch.zeros_like(ctx_depth, dtype=torch.bool)
        gstrim_h = int(8 / 256 * out_h)
        gstrim_w = int(8 / 256 * out_w)
        mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w, :] = 1
    else:
        mask = torch.ones_like(ctx_depth, dtype=torch.bool)


    if prune_by_depth_percent is not None and prune_by_depth_percent < 1:
        in_depths = ctx_depth
        d_percentile = torch.quantile(
            in_depths.view(in_depths.shape[0], -1), q=prune_by_depth_percent, dim=1
        ).view(-1, 1, 1)
        d_mask = (in_depths[..., 0] <= d_percentile).unsqueeze(-1)
        mask = mask & d_mask
    mask = mask.squeeze(-1)


    def trim_select_reshape(element):
        selected_element = rearrange(
            element[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w
        )
        selected_element = selected_element[::gs_views_interval][mask[::gs_views_interval]]
        return selected_element

    export_ply(
        means=trim_select_reshape(world_means),
        scales=trim_select_reshape(gs_scales),
        rotations=trim_select_reshape(world_rotations),
        harmonics=trim_select_reshape(world_shs),
        opacities=trim_select_reshape(gs_opacities),
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=match_3dgs_mcmc_dev,
    )

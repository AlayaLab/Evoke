import numpy as np
import torch
import torch.nn.functional as F

def se3_inverse(T):


    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
        ], dim=-2)
    else:
        R = T[..., :3, :3]
        t = T[..., :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
        bottom_row[..., :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=-2)

    return T_inv

def get_pixel(H, W):

    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))


    pixels_a = np.stack([
        u_a.flatten() + 0.5,
        v_a.flatten() + 0.5,
        np.ones_like(u_a.flatten())
    ], axis=0)

    return pixels_a

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):


    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam
    if camera_pose is not None:


        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]


        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):


    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape


    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


    valid_mask = (depthmap > 0.0)

    valid_mask = valid_mask
    return X_cam, valid_mask

def homogenize_points(
    points,
):

    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):

    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):


    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":

        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1,
                  smooth_mask = smooth_mask,
                  return_relative_depth_error = return_relative_depth_error,
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1,
                  smooth_mask = smooth_mask,
                  return_relative_depth_error = return_relative_depth_error,
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest)
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp


    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )


    nonzero_mask = kpts0_depth > 0


    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)
    kpts0_cam = kpts0_n


    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]


    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )


    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )


    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]

    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0


def geotrf(Trf, pts, ncol=None, norm=False):


    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)


    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]


    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:

                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:

                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):


    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def opencv_camera_to_plucker(poses, K, H, W):
    device = poses.device
    B = poses.shape[0]

    pixel = torch.from_numpy(get_pixel(H, W).astype(np.float32)).to(device).T.reshape(H, W, 3)[None].repeat(B, 1, 1, 1)
    pixel = torch.einsum('bij, bhwj -> bhwi', torch.inverse(K), pixel)
    ray_directions = torch.einsum('bij, bhwj -> bhwi', poses[..., :3, :3], pixel)

    ray_origins = poses[..., :3, 3][:, None, None].repeat(1, H, W, 1)

    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_ray = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_ray


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:


    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

def recover_intrinsic_from_rays_d(
    rays_d: torch.Tensor,
    ndc_coords: bool = False,
    force_center_principal_point: bool = False
):
    device = rays_d.device
    dtype = rays_d.dtype

    *batch_dims, H, W, _ = rays_d.shape
    num_points = H * W

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=dtype),
        indexing="ij"
    )

    if ndc_coords:
        grid_x = grid_x * (2.0 / (W - 1)) - 1.0
        grid_y = grid_y * (2.0 / (H - 1)) - 1.0

    u_targets = grid_x.reshape(1, num_points).expand(*batch_dims, num_points)
    v_targets = grid_y.reshape(1, num_points).expand(*batch_dims, num_points)

    z_abs = rays_d[..., 2].abs() + 1e-10
    x_projs = (rays_d[..., 0] / z_abs).reshape(*batch_dims, num_points)
    y_projs = (rays_d[..., 1] / z_abs).reshape(*batch_dims, num_points)

    if force_center_principal_point:
        if ndc_coords:
            cx_val, cy_val = 0.0, 0.0
        else:
            cx_val, cy_val = (W - 1) / 2.0, (H - 1) / 2.0

        u_centered = u_targets - cx_val
        v_centered = v_targets - cy_val

        fx = (x_projs * u_centered).sum(dim=-1) / (x_projs * x_projs).sum(dim=-1)
        fy = (y_projs * v_centered).sum(dim=-1) / (y_projs * y_projs).sum(dim=-1)

        cx = torch.full_like(fx, cx_val)
        cy = torch.full_like(fy, cy_val)

    else:
        def solve_linear_least_squares(X, Y):
            mean_X = X.mean(dim=-1, keepdim=True)
            mean_Y = Y.mean(dim=-1, keepdim=True)
            X_centered = X - mean_X
            Y_centered = Y - mean_Y

            a = (X_centered * Y_centered).sum(dim=-1) / (X_centered * X_centered).sum(dim=-1)
            b = mean_Y.squeeze(-1) - a * mean_X.squeeze(-1)
            return a, b

        fx, cx = solve_linear_least_squares(x_projs, u_targets)
        fy, cy = solve_linear_least_squares(y_projs, v_targets)

    K = torch.zeros((*batch_dims, 3, 3), device=device, dtype=dtype)
    K[..., 0, 0] = fx
    K[..., 0, 2] = cx
    K[..., 1, 1] = fy
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0

    return K

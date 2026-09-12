

import torch
from einops import repeat
from .geometry import unproject_depth


def compute_optimal_rotation_intrinsics_batch(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2, weights=None,
    n_sample = None,
    n_iter=100,
    num_sample_for_ransac=8,
    rand_sample_iters_idx=None,
):


    device = rays_origin.device
    B, N, _ = rays_origin.shape
    z_mask = torch.logical_and(
        torch.abs(rays_target[:, :, 2]) > z_threshold, torch.abs(rays_origin[:, :, 2]) > z_threshold
    )
    rays_origin = rays_origin.clone()
    rays_target = rays_target.clone()
    rays_origin[:, :, 0][z_mask] /= rays_origin[:, :, 2][z_mask]
    rays_origin[:, :, 1][z_mask] /= rays_origin[:, :, 2][z_mask]
    rays_target[:, :, 0][z_mask] /= rays_target[:, :, 2][z_mask]
    rays_target[:, :, 1][z_mask] /= rays_target[:, :, 2][z_mask]

    rays_origin = rays_origin[:, :, :2]
    rays_target = rays_target[:, :, :2]
    assert weights is not None, "weights must be provided"
    weights[~z_mask] = 0

    A_list = []
    max_chunk_size = 2
    for i in range(0, rays_origin.shape[0], max_chunk_size):
        A = ransac_find_homography_weighted_fast_batch(
            rays_origin[i:i+max_chunk_size],
            rays_target[i:i+max_chunk_size],
            weights[i:i+max_chunk_size],
            n_iter=n_iter,
            n_sample = n_sample,
            num_sample_for_ransac=num_sample_for_ransac,
            reproj_threshold=reproj_threshold,
            rand_sample_iters_idx=rand_sample_iters_idx,
            max_inlier_num=8000,
        )
        A = A.to(device)
        A_need_inv_mask = torch.linalg.det(A) < 0
        A[A_need_inv_mask] = -A[A_need_inv_mask]
        A_list.append(A)

    A = torch.cat(A_list, dim=0)

    R_list = []
    f_list = []
    pp_list = []
    for i in range(A.shape[0]):
        R, L = ql_decomposition(A[i])
        L = L / L[2][2]

        f = torch.stack((L[0][0], L[1][1]))
        pp = torch.stack((L[2][0], L[2][1]))
        R_list.append(R)
        f_list.append(f)
        pp_list.append(pp)

    R = torch.stack(R_list)
    f = torch.stack(f_list)
    pp = torch.stack(pp_list)

    return R, f, pp


def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L

def find_homography_least_squares_weighted_torch(src_pts, dst_pts, confident_weight):


    assert src_pts.shape == dst_pts.shape
    N = src_pts.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute homography.")
    assert confident_weight.shape == (N,)

    w = confident_weight.sqrt().unsqueeze(1)

    x = src_pts[:, 0:1]
    y = src_pts[:, 1:2]
    u = dst_pts[:, 0:1]
    v = dst_pts[:, 1:2]

    zeros = torch.zeros_like(x)


    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=1)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=1)
    A = torch.cat([A1, A2], dim=0)


    _, _, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H = H / H[-1, -1]
    return H


def ransac_find_homography_weighted(
    src_pts,
    dst_pts,
    confident_weight,
    n_iter=100,
    sample_ratio=0.2,
    reproj_threshold=3.0,
    num_sample_for_ransac=16,
    random_seed=None,
):


    if random_seed is not None:
        torch.manual_seed(random_seed)
    N = src_pts.shape[0]
    assert N >= 4

    sorted_idx = torch.argsort(confident_weight, descending=True)
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))
    candidate_idx = sorted_idx[:n_sample]
    best_inlier_mask = None
    best_score = 0
    for _ in range(n_iter):

        idx = candidate_idx[torch.randperm(n_sample)[:num_sample_for_ransac]]

        try:
            H = find_homography_least_squares_weighted_torch(
                src_pts[idx], dst_pts[idx], confident_weight[idx]
            )
        except Exception:
            H = torch.eye(3, dtype=src_pts.dtype, device=src_pts.device)

        src_homo = torch.cat(
            [src_pts, torch.ones(N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=1
        )
        proj = (H @ src_homo.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        error = ((proj - dst_pts) ** 2).sum(dim=1).sqrt()
        inlier_mask = error < reproj_threshold
        total_score = (inlier_mask * confident_weight).sum().item()
        n_inlier = inlier_mask.sum().item()
        if n_inlier < 4:
            continue

        if total_score > best_score:
            best_score = total_score
            best_inlier_mask = inlier_mask


    H_inlier = find_homography_least_squares_weighted_torch(
        src_pts[best_inlier_mask], dst_pts[best_inlier_mask], confident_weight[best_inlier_mask]
    )

    return H_inlier


def find_homography_least_squares_weighted_torch_batch(
    src_pts_batch, dst_pts_batch, confident_weight_batch
):


    B, K, _ = src_pts_batch.shape
    w = confident_weight_batch.sqrt().unsqueeze(2)
    x = src_pts_batch[:, :, 0:1]
    y = src_pts_batch[:, :, 1:2]
    u = dst_pts_batch[:, :, 0:1]
    v = dst_pts_batch[:, :, 1:2]
    zeros = torch.zeros_like(x)
    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=2)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=2)
    A = torch.cat([A1, A2], dim=1)

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].reshape(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H


def ransac_find_homography_weighted_fast(
    src_pts,
    dst_pts,
    confident_weight,
    n_sample,
    n_iter=100,
    reproj_threshold=3.0,
    num_sample_for_ransac=8,
    random_seed=None,
    rand_sample_iters_idx=None,
):


    if random_seed is not None:
        torch.manual_seed(random_seed)
    N = src_pts.shape[0]
    device = src_pts.device
    assert N >= 4

    sorted_idx = torch.argsort(confident_weight, descending=True)
    candidate_idx = sorted_idx[:n_sample]
    if rand_sample_iters_idx is None:
        rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )


    rand_idx = candidate_idx[rand_sample_iters_idx]

    src_pts_batch = src_pts[rand_idx]
    dst_pts_batch = dst_pts[rand_idx]
    confident_weight_batch = confident_weight[rand_idx]

    H_batch = find_homography_least_squares_weighted_torch_batch(
        src_pts_batch, dst_pts_batch, confident_weight_batch
    )

    src_homo = torch.cat(
        [src_pts, torch.ones(N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=1
    )
    src_homo_expand = src_homo.unsqueeze(0).expand(n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(0).expand(n_iter, N, 2)
    confident_weight_expand = confident_weight.unsqueeze(0).expand(n_iter, N)

    proj = torch.bmm(src_homo_expand, H_batch.transpose(1, 2))
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3]
    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=2).sqrt()
    inlier_mask = error < reproj_threshold
    total_score = (inlier_mask * confident_weight_expand).sum(dim=1)

    best_idx = torch.argmax(total_score)
    best_inlier_mask = inlier_mask[best_idx]
    inlier_src_pts = src_pts[best_inlier_mask]
    inlier_dst_pts = dst_pts[best_inlier_mask]
    inlier_confident_weight = confident_weight[best_inlier_mask]

    max_inlier_num = 10000
    sorted_idx = torch.argsort(inlier_confident_weight, descending=True)


    sorted_idx = sorted_idx[torch.randperm(len(sorted_idx))[:max_inlier_num]]

    inlier_src_pts = inlier_src_pts[sorted_idx]
    inlier_dst_pts = inlier_dst_pts[sorted_idx]
    inlier_confident_weight = inlier_confident_weight[sorted_idx]

    H_inlier = find_homography_least_squares_weighted_torch(
        inlier_src_pts, inlier_dst_pts, inlier_confident_weight
    )
    return H_inlier


def ransac_find_homography_weighted_fast_batch(
    src_pts,
    dst_pts,
    confident_weight,
    n_sample,
    n_iter=100,
    reproj_threshold=3.0,
    num_sample_for_ransac=8,
    max_inlier_num=10000,
    random_seed=None,
    rand_sample_iters_idx=None,
):


    if random_seed is not None:
        torch.manual_seed(random_seed)
    B, N, _ = src_pts.shape
    assert N >= 4

    device = src_pts.device


    sorted_idx = torch.argsort(confident_weight, descending=True, dim=1)
    candidate_idx = sorted_idx[:, :n_sample]


    if rand_sample_iters_idx is None:
        rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )

    rand_idx = candidate_idx[:, rand_sample_iters_idx]


    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_iter, num_sample_for_ransac)
    src_pts_batch = src_pts[b_idx, rand_idx]
    dst_pts_batch = dst_pts[b_idx, rand_idx]
    confident_weight_batch = confident_weight[b_idx, rand_idx]


    cB, cN = src_pts_batch.shape[:2]
    H_batch = find_homography_least_squares_weighted_torch_batch(
        src_pts_batch.flatten(0, 1), dst_pts_batch.flatten(0, 1), confident_weight_batch.flatten(0, 1)
    )
    H_batch = H_batch.unflatten(0, (cB, cN))


    src_homo = torch.cat(
        [src_pts, torch.ones(B, N, 1, dtype=src_pts.dtype, device=src_pts.device)], dim=2
    )
    src_homo_expand = src_homo.unsqueeze(1).expand(B, n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(1).expand(B, n_iter, N, 2)
    confident_weight_expand = confident_weight.unsqueeze(1).expand(B, n_iter, N)


    H_batch_flat = H_batch.reshape(-1, 3, 3)
    src_homo_expand_flat = src_homo_expand.reshape(-1, N, 3)
    proj = torch.bmm(src_homo_expand_flat, H_batch_flat.transpose(1, 2))
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3]
    proj_xy = proj_xy.reshape(B, n_iter, N, 2)
    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=3).sqrt()
    inlier_mask = error < reproj_threshold
    total_score = (inlier_mask * confident_weight_expand).sum(dim=2)


    best_idx = torch.argmax(total_score, dim=1)
    best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_idx]


    H_inlier_list = []
    for b in range(B):
        mask = best_inlier_mask[b]
        inlier_src_pts = src_pts[b][mask]
        inlier_dst_pts = dst_pts[b][mask]
        inlier_confident_weight = confident_weight[b][mask]

        sorted_idx = torch.argsort(inlier_confident_weight, descending=True)


        if len(sorted_idx) > max_inlier_num:

            keep_len = max(int(len(sorted_idx) * 0.95), max_inlier_num)
            sorted_idx = sorted_idx[:keep_len]
            perm = torch.randperm(len(sorted_idx), device=device)[:max_inlier_num]
            sorted_idx = sorted_idx[perm]
        inlier_src_pts = inlier_src_pts[sorted_idx]
        inlier_dst_pts = inlier_dst_pts[sorted_idx]
        inlier_confident_weight = inlier_confident_weight[sorted_idx]

        H_inlier = find_homography_least_squares_weighted_torch(
            inlier_src_pts, inlier_dst_pts, inlier_confident_weight
        )
        H_inlier_list.append(H_inlier)
    H_inlier = torch.stack(H_inlier_list, dim=0)
    return H_inlier

def get_params_for_ransac(N, device):
    n_iter=100
    sample_ratio=0.3
    num_sample_for_ransac=8
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))
    rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
            dim=0,
        )
    return n_iter, num_sample_for_ransac, n_sample, rand_sample_iters_idx


def camray_to_caminfo(camray, confidence=None, reproj_threshold=0.2, training=False):


    if confidence is None:
        confidence = torch.ones_like(camray[:, :, :, :, 0])
    B, S, num_patches_y, num_patches_x, _ = camray.shape

    I_K = torch.eye(3, dtype=camray.dtype, device=camray.device)
    I_K[0, 2] = 1.0
    I_K[1, 2] = 1.0

    I_K = I_K.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)

    cam_plane_depth = torch.ones(
        B, S, num_patches_y, num_patches_x, 1, dtype=camray.dtype, device=camray.device
    )
    I_cam_plane_unproj = unproject_depth(
        cam_plane_depth,
        I_K,
        c2w=None,
        ixt_normalized=True,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )

    camray = camray.flatten(0, 1).flatten(1, 2)
    I_cam_plane_unproj = I_cam_plane_unproj.flatten(0, 1).flatten(
        1, 2
    )
    confidence = confidence.flatten(0, 1).flatten(1, 2)


    N = camray.shape[-2]
    device = camray.device
    n_iter, num_sample_for_ransac, n_sample, rand_sample_iters_idx = get_params_for_ransac(N, device)


    if training:
        camray = camray.clone().detach()
        I_cam_plane_unproj = I_cam_plane_unproj.clone().detach()
        confidence = confidence.clone().detach()
    R, focal_lengths, principal_points = compute_optimal_rotation_intrinsics_batch(
        I_cam_plane_unproj,
        camray[:, :, :3],
        reproj_threshold=reproj_threshold,
        weights=confidence,
        n_sample = n_sample,
        n_iter=n_iter,
        num_sample_for_ransac=num_sample_for_ransac,
        rand_sample_iters_idx=rand_sample_iters_idx,
    )

    T = torch.sum(camray[:, :, 3:] * confidence.unsqueeze(-1), dim=1) / torch.sum(
        confidence, dim=-1, keepdim=True
    )

    R = R.reshape(B, S, 3, 3)
    T = T.reshape(B, S, 3)
    focal_lengths = focal_lengths.reshape(B, S, 2)
    principal_points = principal_points.reshape(B, S, 2)

    return R, T, 1.0 / focal_lengths, principal_points + 1.0

def get_extrinsic_from_camray(camray, conf, patch_size_y, patch_size_x, training=False):
    pred_R, pred_T, pred_focal_lengths, pred_principal_points = camray_to_caminfo(
        camray, confidence=conf.squeeze(-1), training=training
    )

    pred_extrinsic = torch.cat(
        [
            torch.cat([pred_R, pred_T.unsqueeze(-1)], dim=-1),
            repeat(
                torch.tensor([0, 0, 0, 1], dtype=pred_R.dtype, device=pred_R.device),
                "c -> b s 1 c",
                b=pred_R.shape[0],
                s=pred_R.shape[1],
            ),
        ],
        dim=-2,
    )
    return pred_extrinsic, pred_focal_lengths, pred_principal_points

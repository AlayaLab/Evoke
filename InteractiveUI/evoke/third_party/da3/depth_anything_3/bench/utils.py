

from typing import Dict as TDict, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from addict import Dict
from scipy.spatial import KDTree

from depth_anything_3.utils.geometry import mat_to_quat


def quat2rotmat(qvec: list) -> np.ndarray:


    rotmat = np.array(
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ]
    )
    rotmat = rotmat.reshape(3, 3)
    return rotmat


def nn_correspondance(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:


    if len(verts1) == 0 or len(verts2) == 0:
        return np.array([])

    kdtree = KDTree(verts1)
    distances, _ = kdtree.query(verts2)
    return distances.reshape(-1)


def evaluate_3d_reconstruction(
    pcd_pred: Union[o3d.geometry.PointCloud, np.ndarray],
    pcd_trgt: Union[o3d.geometry.PointCloud, np.ndarray],
    threshold: float = 0.05,
    down_sample: Optional[float] = None,
) -> TDict[str, float]:


    if isinstance(pcd_pred, np.ndarray):
        pcd_pred_o3d = o3d.geometry.PointCloud()
        pcd_pred_o3d.points = o3d.utility.Vector3dVector(pcd_pred)
        pcd_pred = pcd_pred_o3d
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt_o3d = o3d.geometry.PointCloud()
        pcd_trgt_o3d.points = o3d.utility.Vector3dVector(pcd_trgt)
        pcd_trgt = pcd_trgt_o3d


    if down_sample is not None and down_sample > 0:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)


    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {
            "acc": float("inf"),
            "comp": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }


    dist_pred_to_gt = nn_correspondance(verts_trgt, verts_pred)
    dist_gt_to_pred = nn_correspondance(verts_pred, verts_trgt)


    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    overall = (accuracy + completeness) / 2

    precision = float(np.mean((dist_pred_to_gt < threshold).astype(float)))
    recall = float(np.mean((dist_gt_to_pred < threshold).astype(float)))

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "acc": accuracy,
        "comp": completeness,
        "overall": overall,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def create_tsdf_volume(
    voxel_length: float = 4.0 / 512.0,
    sdf_trunc: float = 0.04,
    color_type: str = "RGB8",
) -> o3d.pipelines.integration.ScalableTSDFVolume:


    if color_type == "RGB8":
        color_enum = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    else:
        color_enum = o3d.pipelines.integration.TSDFVolumeColorType.Gray32

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_enum,
    )
    return volume


def fuse_depth_to_tsdf(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    depths: np.ndarray,
    images: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    max_depth: float = 10.0,
) -> o3d.geometry.TriangleMesh:


    for i in range(len(depths)):
        depth = depths[i]
        image = images[i]
        ixt = intrinsics[i]
        ext = extrinsics[i]

        h, w = depth.shape[:2]


        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(image.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )


        ixt_o3d = o3d.camera.PinholeCameraIntrinsic(
            w, h, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        )


        volume.integrate(rgbd, ixt_o3d, ext)


    mesh = volume.extract_triangle_mesh()
    return mesh


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 1000000,
) -> o3d.geometry.PointCloud:


    try:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            colors = np.clip(colors, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:

        rng = np.random.default_rng(seed=42)
        points = rng.uniform(-1, 1, size=(num_points, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def build_pair_index(N: int, B: int = 1):


    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = ((i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_])
    return i1, i2


def compute_pose(pred_se3: torch.Tensor, gt_se3: torch.Tensor) -> Dict:


    pred_se3 = align_to_first_camera(pred_se3)
    gt_se3 = align_to_first_camera(gt_se3)

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, len(pred_se3))
    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()

    output = Dict()
    output.auc30, _ = calculate_auc_np(rError, tError, max_threshold=30)
    output.auc15, _ = calculate_auc_np(rError, tError, max_threshold=15)
    output.auc05, _ = calculate_auc_np(rError, tError, max_threshold=5)
    output.auc03, _ = calculate_auc_np(rError, tError, max_threshold=3)
    return output


def align_to_first_camera(camera_poses: torch.Tensor) -> torch.Tensor:


    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses


def rotation_angle(
    rot_gt: torch.Tensor, rot_pred: torch.Tensor, batch_size: int = None, eps: float = 1e-15
) -> torch.Tensor:


    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(
    tvec_gt: torch.Tensor,
    tvec_pred: torch.Tensor,
    batch_size: int = None,
    ambiguity: bool = True,
) -> torch.Tensor:


    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(
    t_gt: torch.Tensor, t: torch.Tensor, eps: float = 1e-15, default_err: float = 1e6
) -> torch.Tensor:


    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(
    r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30
) -> tuple:


    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(
    pred_se3: torch.Tensor, gt_se3: torch.Tensor, num_frames: int
) -> tuple:


    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)


    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])


    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def closed_form_inverse_se3(
    se3: torch.Tensor, R: torch.Tensor = None, T: torch.Tensor = None
) -> torch.Tensor:


    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

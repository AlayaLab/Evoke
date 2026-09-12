

import os
from typing import Dict as TDict

import cv2
import imageio
import numpy as np
import open3d as o3d
from addict import Dict

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    fuse_depth_to_tsdf,
    nn_correspondance,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    SCANNETPP_DOWN_SAMPLE,
    SCANNETPP_EVAL_DATA_ROOT,
    SCANNETPP_EVAL_THRESHOLD,
    SCANNETPP_INPUT_H,
    SCANNETPP_INPUT_W,
    SCANNETPP_MAX_DEPTH,
    SCANNETPP_SAMPLING_NUMBER,
    SCANNETPP_SCENES,
    SCANNETPP_SDF_TRUNC,
    SCANNETPP_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama
from depth_anything_3.utils.read_write_model import read_model


@MV_REGISTRY.register(name="scannetpp")
@MONO_REGISTRY.register(name="scannetpp")
class ScanNetPP(Dataset):


    data_root = SCANNETPP_EVAL_DATA_ROOT
    SCENES = SCANNETPP_SCENES


    input_h = SCANNETPP_INPUT_H
    input_w = SCANNETPP_INPUT_W


    max_depth = SCANNETPP_MAX_DEPTH
    sampling_number = SCANNETPP_SAMPLING_NUMBER
    voxel_length = SCANNETPP_VOXEL_LENGTH
    sdf_trunc = SCANNETPP_SDF_TRUNC
    eval_threshold = SCANNETPP_EVAL_THRESHOLD
    down_sample = SCANNETPP_DOWN_SAMPLE

    def __init__(self):
        super().__init__()
        self._scene_cache = {}


    def get_data(self, scene: str) -> Dict:


        if scene in self._scene_cache:
            return self._scene_cache[scene]

        input_path = os.path.join(self.data_root, scene, "merge_dslr_iphone")
        colmap_path = os.path.join(input_path, "colmap/sparse_render_rgb")
        image_path = os.path.join(input_path, "images")
        depth_path_dir = os.path.join(input_path, "render_depth")


        cams, images, points3d = read_model(colmap_path)


        name2id = {image.name: k for k, image in images.items()}
        names = sorted([image.name for k, image in images.items()])

        names = [name for name in names if "iphone" in name]

        gt_mesh_path = os.path.join(
            input_path.replace("merge_dslr_iphone", "scans"), "mesh_aligned_0.05.ply"
        )

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_mesh_path": gt_mesh_path,
                "dist_list": [],
                "roi_list": [],
                "cam_hw_list": [],
                "ixt_raw_list": [],
                "gt_depth_files": [],
            }),
        })

        for name in names:
            image = images[name2id[name]]
            img_path = os.path.join(image_path, name)

            if not os.path.exists(img_path):
                continue


            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = image.qvec2rotmat()
            ext[:3, 3] = image.tvec


            cam_id = image.camera_id
            camera = cams[cam_id]
            cam_height, cam_width = camera.height, camera.width


            ixt = np.eye(3, dtype=np.float32)
            ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
            ixt[:2, 2] -= 0.5
            ixt_raw = ixt.copy()


            dist = np.zeros(5, dtype=np.float32)
            roi = (0, 0, cam_width, cam_height)
            if camera.model == "OPENCV":
                dist[:4] = camera.params[4:]
                ixt, roi = cv2.getOptimalNewCameraMatrix(
                    ixt, dist, (cam_width, cam_height), 1, (cam_width, cam_height)
                )


            frame_name = os.path.basename(name)[:-4]
            depth_file = os.path.join(depth_path_dir, f"{frame_name}.png")

            out.image_files.append(img_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt)
            out.aux.dist_list.append(dist)
            out.aux.roi_list.append(roi)
            out.aux.cam_hw_list.append((cam_height, cam_width))
            out.aux.ixt_raw_list.append(ixt_raw)
            out.aux.gt_depth_files.append(depth_file)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[ScanNet++] {scene}: {len(out.image_files)} images")

        self._scene_cache[scene] = out
        return out

    def load_image(self, img_path: str, idx: int, aux: Dict) -> np.ndarray:


        image = imageio.imread(img_path).astype(np.uint8)
        ixt_raw = aux.ixt_raw_list[idx]
        ixt = aux.intrinsics[idx] if hasattr(aux, 'intrinsics') else None
        dist = aux.dist_list[idx]
        roi = aux.roi_list[idx]


        stored_ixt = self._scene_cache.get(aux.scene, {}).get('intrinsics', [None])[idx] if hasattr(aux, 'scene') else None
        if stored_ixt is None:

            cam_h, cam_w = aux.cam_hw_list[idx]
            ixt_for_undistort = ixt_raw.copy()
            ixt_for_undistort, _ = cv2.getOptimalNewCameraMatrix(
                ixt_raw, dist, (cam_w, cam_h), 1, (cam_w, cam_h)
            )
        else:
            ixt_for_undistort = stored_ixt

        image = cv2.undistort(image, ixt_raw, dist, newCameraMatrix=ixt_for_undistort)


        x, y, w, h = roi
        image = image[y:y+h, x:x+w]


        image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)

        return image

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:


        gt_data = self.get_data(scene)
        gt_mesh_path = gt_data.aux.gt_mesh_path


        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, self.sampling_number)


        pred_pcd = o3d.io.read_point_cloud(fuse_path)


        aabb = gt_pcd.get_axis_aligned_bounding_box()
        points = np.asarray(pred_pcd.points)
        inside_mask = (
            (points[:, 0] >= aabb.min_bound[0] - 0.1) &
            (points[:, 0] <= aabb.max_bound[0] + 0.1) &
            (points[:, 1] >= aabb.min_bound[1] - 0.1) &
            (points[:, 1] <= aabb.max_bound[1] + 0.1) &
            (points[:, 2] >= aabb.min_bound[2] - 0.1) &
            (points[:, 2] <= aabb.max_bound[2] + 0.1)
        )
        pred_pcd = pred_pcd.select_by_index(inside_mask.nonzero()[0])


        if self.down_sample > 0:
            pred_pcd = pred_pcd.voxel_down_sample(self.down_sample)
            gt_pcd = gt_pcd.voxel_down_sample(self.down_sample)

        verts_pred = np.asarray(pred_pcd.points)
        verts_gt = np.asarray(gt_pcd.points)

        if len(verts_pred) == 0 or len(verts_gt) == 0:
            return {
                "acc": float("inf"),
                "comp": float("inf"),
                "overall": float("inf"),
                "precision": 0.0,
                "recall": 0.0,
                "fscore": 0.0,
            }


        dist_pred_to_gt = nn_correspondance(verts_gt, verts_pred)
        dist_gt_to_pred = nn_correspondance(verts_pred, verts_gt)


        accuracy = float(np.mean(dist_pred_to_gt))
        completeness = float(np.mean(dist_gt_to_pred))
        overall = (accuracy + completeness) / 2

        precision = float(np.mean((dist_pred_to_gt < self.eval_threshold).astype(float)))
        recall = float(np.mean((dist_gt_to_pred < self.eval_threshold).astype(float)))

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

    def _load_gt_meta(self, result_path: str) -> Dict:

        export_dir = os.path.dirname(result_path)
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")

        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            image_files = list(data["image_files"])


            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": image_files,
            })
        return None

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:


        full_gt_data = self.get_data(scene)


        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta

            image_indices = [
                full_gt_data.image_files.index(f)
                for f in gt_data.image_files
                if f in full_gt_data.image_files
            ]
        else:
            gt_data = full_gt_data
            image_indices = list(range(len(full_gt_data.image_files)))

        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})


        images = []
        for idx, img_idx in enumerate(image_indices):
            img_path = full_gt_data.image_files[img_idx]
            image = imageio.imread(img_path).astype(np.uint8)


            ixt_raw = full_gt_data.aux.ixt_raw_list[img_idx]
            ixt = full_gt_data.intrinsics[img_idx]
            dist = full_gt_data.aux.dist_list[img_idx]
            roi = full_gt_data.aux.roi_list[img_idx]

            image = cv2.undistort(image, ixt_raw, dist, newCameraMatrix=ixt)
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
            image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)

            images.append(image)

        images = np.stack(images, axis=0)


        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, full_gt_data, image_indices, scene=scene
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, full_gt_data, image_indices, scene=scene
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")


        volume = create_tsdf_volume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
        )
        mesh = fuse_depth_to_tsdf(
            volume, depths, images, intrinsics, extrinsics, max_depth=self.max_depth
        )


        pcd = sample_points_from_mesh(mesh, self.sampling_number)


        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        o3d.io.write_point_cloud(fuse_path, pcd)


    def _prep_unposed(
        self, pred_data: Dict, gt_data: Dict, full_gt_data: Dict,
        image_indices: list, scene: str = None
    ) -> tuple:


        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]

        depths_out = []
        intrinsics_out = []
        for i in range(len(pred_data.depth)):
            img_idx = image_indices[i]


            orig_h, orig_w = full_gt_data.aux.cam_hw_list[img_idx]


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            depth = cv2.resize(
                depth,
                (self.input_w, self.input_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)


            gt_zero_mask = self._load_gt_mask(full_gt_data.aux.gt_depth_files[img_idx])


            depth = self._mask_invalid_depth(depth, gt_zero_mask)


            depth = depth * scale


            h_ratio = self.input_h / model_h
            w_ratio = self.input_w / model_w
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= w_ratio
            ixt[1, :] *= h_ratio

            depths_out.append(depth)
            intrinsics_out.append(ixt)

        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(
        self, pred_data: Dict, gt_data: Dict, full_gt_data: Dict,
        image_indices: list, scene: str = None
    ) -> tuple:


        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        depths_out = []
        intrinsics_out = []
        extrinsics_out = []

        for i in range(len(pred_data.depth)):
            img_idx = image_indices[i]


            orig_h, orig_w = full_gt_data.aux.cam_hw_list[img_idx]


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            depth = cv2.resize(
                depth,
                (self.input_w, self.input_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)


            gt_zero_mask = self._load_gt_mask(full_gt_data.aux.gt_depth_files[img_idx])


            depth = self._mask_invalid_depth(depth, gt_zero_mask)


            depth = depth * scale

            depths_out.append(depth)


            ixt = full_gt_data.intrinsics[img_idx].copy()
            cam_h, cam_w = full_gt_data.aux.cam_hw_list[img_idx]
            ixt[:2, 2] += 0.5
            ixt[0, :] *= self.input_w / cam_w
            ixt[1, :] *= self.input_h / cam_h
            intrinsics_out.append(ixt)

            extrinsics_out.append(full_gt_data.extrinsics[img_idx])

        return np.stack(depths_out), np.stack(intrinsics_out), np.stack(extrinsics_out)

    def _load_gt_mask(self, gt_depth_path: str) -> np.ndarray:


        if not os.path.exists(gt_depth_path):
            return None

        gt_depth = imageio.imread(gt_depth_path) / 1000.0


        gt_depth = cv2.resize(
            gt_depth,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)


        valid_mask = np.logical_and(gt_depth > 0, gt_depth != np.inf)
        return valid_mask

    def _mask_invalid_depth(
        self, depth: np.ndarray, gt_zero_mask: np.ndarray = None
    ) -> np.ndarray:

        depth = depth.copy()

        if gt_zero_mask is not None:
            pred_invalid = np.isnan(depth) | np.isinf(depth)
            combined_mask = np.logical_and(gt_zero_mask, np.logical_not(pred_invalid))
            depth = depth * combined_mask.astype(np.float32)
        else:
            invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
            depth[invalid_mask] = 0.0

        return depth

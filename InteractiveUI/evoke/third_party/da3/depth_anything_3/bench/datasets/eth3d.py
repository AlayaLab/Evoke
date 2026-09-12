

import glob
import os
from typing import Dict as TDict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from addict import Dict
from PIL import Image

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    fuse_depth_to_tsdf,
    quat2rotmat,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    ETH3D_DOWN_SAMPLE,
    ETH3D_EVAL_DATA_ROOT,
    ETH3D_EVAL_THRESHOLD,
    ETH3D_FILTER_KEYS,
    ETH3D_MAX_DEPTH,
    ETH3D_SAMPLING_NUMBER,
    ETH3D_SCENES,
    ETH3D_SDF_TRUNC,
    ETH3D_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


@MV_REGISTRY.register(name="eth3d")
@MONO_REGISTRY.register(name="eth3d")
class ETH3D(Dataset):


    data_root = ETH3D_EVAL_DATA_ROOT
    SCENES = ETH3D_SCENES


    max_depth = ETH3D_MAX_DEPTH
    sampling_number = ETH3D_SAMPLING_NUMBER
    voxel_length = ETH3D_VOXEL_LENGTH
    sdf_trunc = ETH3D_SDF_TRUNC
    eval_threshold = ETH3D_EVAL_THRESHOLD
    down_sample = ETH3D_DOWN_SAMPLE

    def __init__(self):
        super().__init__()

        self._scene_cache = {}


    def _parse_cameras_txt(self, filepath: str) -> dict:


        camera_dict = {}
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines[3:]:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                cam_id = parts[0]

                camera_dict[cam_id] = {
                    "width": float(parts[2]),
                    "height": float(parts[3]),
                    "fx": float(parts[4]),
                    "fy": float(parts[5]),
                    "cx": float(parts[6]),
                    "cy": float(parts[7]),
                }
        return camera_dict

    def _parse_images_txt(self, filepath: str) -> dict:


        pose_dict = {}
        with open(filepath) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[4:]):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if idx % 2 == 0:
                    parts = line.split()
                    if len(parts) < 10:
                        continue

                    image_id = parts[0]
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    camera_id = parts[8]
                    name = parts[9]
                    pose_dict[name] = {
                        "image_id": image_id,
                        "quat": [qw, qx, qy, qz],
                        "trans": [tx, ty, tz],
                        "camera_id": camera_id,
                    }
        return pose_dict

    def _should_filter_image(self, scene: str, image_name: str) -> bool:

        filter_keys = ETH3D_FILTER_KEYS.get(scene, [])
        for key in filter_keys:
            if image_name.endswith(key):
                return True
        return False


    def get_data(self, scene: str) -> Dict:


        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = os.path.join(self.data_root, scene)


        cameras_file = os.path.join(scene_dir, "dslr_calibration_jpg", "cameras.txt")
        images_file = os.path.join(scene_dir, "dslr_calibration_jpg", "images.txt")
        camera_dict = self._parse_cameras_txt(cameras_file)
        pose_dict = self._parse_images_txt(images_file)


        gt_mesh_path = os.path.join(scene_dir, "combined_mesh.ply")

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_mesh_path": gt_mesh_path,
                "heights": [],
                "widths": [],
            }),
        })


        filtered_count = 0
        for image_name, pose_info in pose_dict.items():

            if self._should_filter_image(scene, image_name):
                filtered_count += 1
                continue

            image_path = os.path.join(scene_dir, "images", image_name)
            if not os.path.exists(image_path):
                continue

            cam_info = camera_dict.get(pose_info["camera_id"])
            if cam_info is None:
                continue


            ixt = np.array([
                [cam_info["fx"], 0, cam_info["cx"]],
                [0, cam_info["fy"], cam_info["cy"]],
                [0, 0, 1],
            ], dtype=np.float32)


            rot = quat2rotmat(pose_info["quat"])
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = rot
            ext[:3, 3] = pose_info["trans"]

            out.image_files.append(image_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt)
            out.aux.heights.append(cam_info["height"])
            out.aux.widths.append(cam_info["width"])

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)


        total_images = len(pose_dict)
        used_images = len(out.image_files)
        print(f"[ETH3D] {scene}: {used_images}/{total_images} images "
              f"(filtered {filtered_count}, missing {total_images - used_images - filtered_count})")

        if used_images < 3:
            print(f"[ETH3D] ⚠️  WARNING: {scene} has only {used_images} images - evaluation may fail!")


        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:


        gt_data = self.get_data(scene)
        gt_mesh_path = gt_data.aux.gt_mesh_path


        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, self.sampling_number)


        pred_pcd = o3d.io.read_point_cloud(fuse_path)


        metrics = evaluate_3d_reconstruction(
            pred_pcd,
            gt_pcd,
            threshold=self.eval_threshold,
            down_sample=self.down_sample,
        )

        return metrics

    def _load_gt_meta(self, result_path: str) -> Dict:


        export_dir = os.path.dirname(result_path)
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")

        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": data["image_files"] if "image_files" in data else None,
            })
        return None

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:


        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
        else:
            gt_data = self.get_data(scene)
        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})


        images = []
        orig_sizes = []
        for img_path in gt_data.image_files:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))


        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, orig_sizes, scene=scene
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, orig_sizes, scene=scene
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        images = np.stack(images, axis=0)


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
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str = None
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
            orig_h, orig_w = orig_sizes[i]
            image_name = os.path.basename(gt_data.image_files[i])


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            gt_zero_mask = None
            if scene is not None:
                gt_zero_mask = self._load_gt_mask(scene, image_name, (orig_h, orig_w))


            depth = self._mask_invalid_depth(depth, gt_zero_mask)


            depth = depth * scale


            h_ratio = orig_h / model_h
            w_ratio = orig_w / model_w
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= w_ratio
            ixt[1, :] *= h_ratio

            depths_out.append(depth)
            intrinsics_out.append(ixt)

        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str = None
    ) -> tuple:


        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )


        depths_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            image_name = os.path.basename(gt_data.image_files[i])


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            gt_zero_mask = None
            if scene is not None:
                gt_zero_mask = self._load_gt_mask(scene, image_name, (orig_h, orig_w))


            depth = self._mask_invalid_depth(depth, gt_zero_mask)


            depth = depth * scale

            depths_out.append(depth)


        return np.stack(depths_out), gt_data.intrinsics.copy(), gt_data.extrinsics.copy()

    def _load_gt_mask(self, scene: str, image_name: str, shape: tuple) -> np.ndarray:


        h, w = shape


        gt_mask_path = os.path.join(
            self.data_root, scene, "masks_for_images", "dslr_images",
            image_name.replace(".JPG", ".png")
        )


        gt_depth_path = os.path.join(
            self.data_root, scene, "ground_truth_depth", "dslr_images", image_name
        )


        if os.path.exists(gt_depth_path):
            gt_depth = np.fromfile(gt_depth_path, dtype=np.float32).reshape(h, w)
        else:
            gt_depth = np.ones((h, w), dtype=np.float32)


        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = np.asarray(gt_mask)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)


        invalid_mask_from_gt = gt_mask == 1
        gt_depth_copy = gt_depth.copy()
        gt_depth_copy[gt_mask == 1] = 0

        invalid_mask_from_gt_depth = np.logical_or(gt_depth_copy == 0, gt_depth_copy == np.inf)


        zero_mask = np.logical_and(
            np.logical_not(invalid_mask_from_gt),
            np.logical_not(invalid_mask_from_gt_depth)
        )

        return zero_mask

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

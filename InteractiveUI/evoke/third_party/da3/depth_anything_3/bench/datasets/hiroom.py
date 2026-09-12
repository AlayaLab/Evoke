

import os
from typing import Dict as TDict, List

import cv2
import numpy as np
import open3d as o3d
from addict import Dict

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    fuse_depth_to_tsdf,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    HIROOM_DOWN_SAMPLE,
    HIROOM_EVAL_DATA_ROOT,
    HIROOM_EVAL_THRESHOLD,
    HIROOM_GT_ROOT_PATH,
    HIROOM_MAX_DEPTH,
    HIROOM_SAMPLING_NUMBER,
    HIROOM_SCENE_LIST_PATH,
    HIROOM_SDF_TRUNC,
    HIROOM_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


def _load_scene_list() -> List[str]:

    if os.path.exists(HIROOM_SCENE_LIST_PATH):
        with open(HIROOM_SCENE_LIST_PATH, "r") as f:
            return f.read().splitlines()
    return []


@MV_REGISTRY.register(name="hiroom")
@MONO_REGISTRY.register(name="hiroom")
class HiRoomDataset(Dataset):


    data_root = HIROOM_EVAL_DATA_ROOT
    gt_root_path = HIROOM_GT_ROOT_PATH
    SCENES = _load_scene_list()


    max_depth = HIROOM_MAX_DEPTH
    sampling_number = HIROOM_SAMPLING_NUMBER
    voxel_length = HIROOM_VOXEL_LENGTH
    sdf_trunc = HIROOM_SDF_TRUNC
    eval_threshold = HIROOM_EVAL_THRESHOLD
    down_sample = HIROOM_DOWN_SAMPLE

    def __init__(self):
        super().__init__()
        self._scene_cache = {}


    def get_data(self, scene: str) -> Dict:


        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = os.path.join(self.data_root, scene)
        image_dir = os.path.join(scene_dir, "image")


        scene_name = "-".join(scene.split("/")[-3:])
        gt_pcd_path = os.path.join(self.gt_root_path, f"{scene_name}.ply")


        intrin_path = os.path.join(scene_dir, "cam_K.npy")
        ixt_shared = np.load(intrin_path).astype(np.float32)


        image_names = sorted(os.listdir(image_dir))

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_pcd_path": gt_pcd_path,
                "gt_depth_files": [],
                "aliasing_mask_files": [],
            }),
        })

        for img_name in image_names:
            img_path = os.path.join(image_dir, img_name)
            frame_name = img_name.split(".")[0]


            depth_path = os.path.join(scene_dir, "depth", f"{frame_name}.png")
            pose_path = os.path.join(scene_dir, "pose", f"{frame_name}.npy")
            aliasing_mask_path = os.path.join(scene_dir, "aliasing_mask", f"{frame_name}.png")

            if not os.path.exists(pose_path):
                continue


            ext = np.load(pose_path).astype(np.float32)

            out.image_files.append(img_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt_shared.copy())
            out.aux.gt_depth_files.append(depth_path)
            out.aux.aliasing_mask_files.append(aliasing_mask_path)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[HiRoom] {scene}: {len(out.image_files)} images")

        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:


        gt_data = self.get_data(scene)
        gt_pcd_path = gt_data.aux.gt_pcd_path


        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)


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
        orig_sizes = []
        for img_idx in image_indices:
            img_path = full_gt_data.image_files[img_idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))

        images = np.stack(images, axis=0)


        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes, scene=scene
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes, scene=scene
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
        image_indices: list, orig_sizes: list, scene: str = None
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
            img_idx = image_indices[i]


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            gt_zero_mask = self._load_gt_mask(
                full_gt_data.aux.gt_depth_files[img_idx],
                full_gt_data.aux.aliasing_mask_files[img_idx],
            )


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
        self, pred_data: Dict, gt_data: Dict, full_gt_data: Dict,
        image_indices: list, orig_sizes: list, scene: str = None
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
            img_idx = image_indices[i]


            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )


            gt_zero_mask = self._load_gt_mask(
                full_gt_data.aux.gt_depth_files[img_idx],
                full_gt_data.aux.aliasing_mask_files[img_idx],
            )


            depth = self._mask_invalid_depth(depth, gt_zero_mask)


            depth = depth * scale

            depths_out.append(depth)


        gt_intrinsics = np.stack([full_gt_data.intrinsics[idx] for idx in image_indices])
        gt_extrinsics = np.stack([full_gt_data.extrinsics[idx] for idx in image_indices])

        return np.stack(depths_out), gt_intrinsics, gt_extrinsics

    def _load_gt_mask(self, gt_depth_path: str, aliasing_mask_path: str) -> np.ndarray:


        if os.path.exists(gt_depth_path):
            gt_depth = cv2.imread(gt_depth_path, -1) / 65535.0 * 100.0
        else:
            return None


        aliasing_mask = None
        if os.path.exists(aliasing_mask_path):
            aliasing_mask = cv2.imread(aliasing_mask_path, -1) > 0


        valid_mask = gt_depth > 0
        if aliasing_mask is not None:
            valid_mask = np.logical_and(valid_mask, np.logical_not(aliasing_mask))

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

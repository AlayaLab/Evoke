

import glob
import os
from typing import Dict as TDict

import numpy as np
from addict import Dict

from depth_anything_3.bench.dataset import Dataset
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.utils.constants import (
    DTU64_CAMERA_ROOT,
    DTU64_EVAL_DATA_ROOT,
    DTU64_SCENES,
)


@MV_REGISTRY.register(name="dtu64")
@MONO_REGISTRY.register(name="dtu64")
class DTU64(Dataset):


    data_root = DTU64_EVAL_DATA_ROOT
    camera_root = DTU64_CAMERA_ROOT
    SCENES = DTU64_SCENES

    def __init__(self):
        super().__init__()
        self._scene_cache = {}


    def read_cam_file(self, filename: str) -> tuple:


        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]

        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))

        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
        return intrinsics, extrinsics


    def get_data(self, scene: str) -> Dict:


        if scene in self._scene_cache:
            return self._scene_cache[scene]

        rgb_folder = os.path.join(self.data_root, scene, "image")


        files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))


        if len(files) > 33:
            files = [files[33]] + files[:33] + files[34:]

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({}),
        })

        for rgb_file in files:
            basename = os.path.basename(rgb_file)

            file_idx = basename.split(".")[0]
            cam_idx = int(file_idx)


            cam_file = os.path.join(self.camera_root, f"{cam_idx:0>8}_cam.txt")

            if not os.path.exists(cam_file):
                print(f"[DTU-64] Warning: Camera file not found: {cam_file}")
                continue

            intrinsics, extrinsics = self.read_cam_file(cam_file)

            out.image_files.append(rgb_file)
            out.extrinsics.append(extrinsics)
            out.intrinsics.append(intrinsics)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[DTU-64] {scene}: {len(out.image_files)} images (pose evaluation only)")

        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:


        raise NotImplementedError(
            "DTU-64 dataset is for POSE EVALUATION ONLY. "
            "3D reconstruction evaluation is not supported. "
            "Use the standard 'dtu' dataset for 3D reconstruction evaluation."
        )

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:


        raise NotImplementedError(
            "DTU-64 dataset is for POSE EVALUATION ONLY. "
            "3D reconstruction (fuse3d) is not supported. "
            "Use the standard 'dtu' dataset for 3D reconstruction."
        )

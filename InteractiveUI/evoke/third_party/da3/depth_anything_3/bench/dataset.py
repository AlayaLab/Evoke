

import os
import time
from abc import abstractmethod
from typing import Dict as TDict

import numpy as np
import torch
from addict import Dict

from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous


def _wait_for_file_ready(path: str, timeout: float = 3.0, interval: float = 0.2) -> None:

    last_size = -1
    stable_count = 0
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(interval)
        size = os.path.getsize(path)
        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= 2:
                return
        else:
            stable_count = 0
        last_size = size


class Dataset:


    SCENES: list = []
    data_root: str = ""

    def __init__(self):
        pass

    def eval_pose(self, scene: str, result_path: str) -> TDict[str, float]:


        _wait_for_file_ready(result_path)
        pred = np.load(result_path)
        gt = self.get_data(scene)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt["extrinsics"])),
        )

    @abstractmethod
    def get_data(self, scene: str) -> Dict:


        raise NotImplementedError

    @abstractmethod
    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:


        raise NotImplementedError

    @abstractmethod
    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:


        raise NotImplementedError

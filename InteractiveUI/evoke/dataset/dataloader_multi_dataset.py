
from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from evoke.dataset.evoke_data import load_data_config


try:
    if mp.get_sharing_strategy() != "file_system":
        mp.set_sharing_strategy("file_system")
except Exception:
    pass


def _stack_pil_frames(frames) -> torch.Tensor:

    arr_list = []
    for f in frames:
        a = np.asarray(f, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        arr_list.append(a)
    arr = np.stack(arr_list, axis=0)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return t * 2.0 - 1.0


class BucketedFeatureDataset(Dataset):


    def __init__(
        self,
        data_yaml_path: str,
        single_height: int = 384,
        single_width: int = 640,
        num_frames: int = 105,
        target_fps: int = 24,
        history_sizes=(16, 2, 1),
        is_keep_x0: bool = True,
        seed: int = 42,


        resample_ratio_each_epoch: bool = False,

        **_unused,
    ):
        self.inner = load_data_config(
            data_yaml_path,
            height=single_height,
            width=single_width,
            num_frames=num_frames,
            target_fps=target_fps,
            resample_each_epoch=bool(resample_ratio_each_epoch),
            seed=int(seed),
        )
        self.resample_ratio_each_epoch = bool(resample_ratio_each_epoch)
        self.single_height = single_height
        self.single_width = single_width
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.history_sizes = list(history_sizes)
        self.is_keep_x0 = is_keep_x0
        self.seed = seed
        self._epoch = 0


        bkey = (num_frames, single_height, single_width)
        self.buckets = {bkey: list(range(len(self.inner)))}
        self.samples = [{"dataset_name": "online_multi", "bucket_key": bkey} for _ in range(len(self.inner))]

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch: int):
        self._epoch = epoch


        if getattr(self, "resample_ratio_each_epoch", False):
            from evoke.dataset.evoke_data.data_config import set_dataset_epoch
            _n = set_dataset_epoch(self.inner, epoch)
            if __import__("os").environ.get("RANK", "0") == "0":
                print(f"[EPOCH-RESAMPLE] epoch={epoch}: re-drew {_n} subsampled parts"
                      f" (total sample count {len(self.inner)} unchanged)", flush=True)

    def __getitem__(self, idx: int):


        if __import__("os").environ.get("SF_DECOUPLE_EQUIV_MODE"):
            from scripts.training.tmp.test_decouple_equivalence import deterministic_dataset_item
            data = deterministic_dataset_item(self.inner, idx)
        else:
            data = self.inner[idx]
        video = _stack_pil_frames(data["video"])
        out = {
            "raw_video": video,
            "prompt": data["prompt"],

            "segment_prompts": data.get("segment_prompts"),
            "bucket_key": (self.num_frames, self.single_height, self.single_width),
            "uttid": f"online_{idx}",
            "dataset_name": "online_multi",
        }

        if "lingbot_Ks" in data and "lingbot_c2ws" in data:
            out["lingbot_Ks"] = torch.as_tensor(data["lingbot_Ks"], dtype=torch.float32)
            out["lingbot_c2ws"] = torch.as_tensor(data["lingbot_c2ws"], dtype=torch.float32)


        if data.get("__skill__"):
            out["is_skill"] = True
            out["event_window"] = data.get("__event_window__")
        return out


class BucketedSampler:


    def __init__(
        self,
        dataset: BucketedFeatureDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        num_sp_groups: int = 1,
        sp_world_size: int = 1,
        global_rank: int = 0,
        decouple_rollout: bool = False,
        **_unused,
    ):
        self.dataset = dataset


        self.buckets = dataset.buckets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank

        self.ith_sp_group = global_rank // max(sp_world_size, 1)


        self.decouple_rollout = bool(decouple_rollout)
        if self.decouple_rollout:
            self.shard_key = int(global_rank)
            self.num_shards = int(num_sp_groups) * max(int(sp_world_size), 1)
        else:
            self.shard_key = self.ith_sp_group
            self.num_shards = int(num_sp_groups)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * 1000003)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        shard = indices[self.shard_key :: max(self.num_shards, 1)]
        for i in range(0, len(shard), self.batch_size):
            batch = shard[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self):
        per_rank = len(self.dataset) // max(self.num_sp_groups, 1)
        if self.drop_last:
            return per_rank // self.batch_size
        return (per_rank + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    out = {
        "raw_video": torch.stack([b["raw_video"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],

        "segment_prompts": [b.get("segment_prompts") for b in batch],
        "uttid": [b["uttid"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "bucket_key": batch[0]["bucket_key"],
    }

    if batch[0].get("lingbot_Ks") is not None:
        out["lingbot_Ks"] = torch.stack([b["lingbot_Ks"] for b in batch], dim=0)
        out["lingbot_c2ws"] = torch.stack([b["lingbot_c2ws"] for b in batch], dim=0)


    if any(b.get("is_skill") for b in batch):
        out["is_skill"] = [bool(b.get("is_skill", False)) for b in batch]
        out["event_window"] = [b.get("event_window") for b in batch]
    return out

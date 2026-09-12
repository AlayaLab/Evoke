

import gc
import math
import os
import random
import yaml
import torch
from .unified_dataset import UnifiedDataset
from .caption_operator import ResolveCaptionFile
from .operators import ToAbsolutePath, LoadVideo, ImageCropAndResize, RouteByType, RouteByExtensionName, LoadGIF, LoadImage, ToList


class ConfigAwareDataset(torch.utils.data.Dataset):


    _GC_EVERY = 2

    def __init__(self, inner, ds_meta, target_fps=24):
        self.inner = inner
        self.ds_meta = ds_meta
        self.target_fps = target_fps
        self._gc_counter = 0

    def __getitem__(self, idx):

        MAX_RETRIES = 20
        last_err = None
        tried_idx = idx
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    data = self._load_and_process(tried_idx)
                    if attempt > 0:
                        print(f"[BadSample recovered] orig_idx={idx} → final_idx={tried_idx} after {attempt} retries", flush=True)
                    return data
                except (OSError, IOError, RuntimeError, ValueError, AttributeError, KeyError,
                        IndexError, StopIteration) as e:


                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    if attempt == 0:

                        print(f"[BadSample] idx={tried_idx}: {last_err} → retrying with random idx", flush=True)
                    tried_idx = random.randrange(len(self.inner))

            raise RuntimeError(
                f"ConfigAwareDataset.__getitem__: gave up after {MAX_RETRIES} retries starting from idx={idx}. "
                f"Last error: {last_err}"
            )
        finally:

            self._gc_counter += 1
            if self._gc_counter >= self._GC_EVERY:
                self._gc_counter = 0
                gc.collect()

    def _load_and_process(self, idx):
        data = self.inner[idx]
        data["__ds_config__"] = self.ds_meta


        clip_start_time = data.get("__start_time__", 0) or 0
        if isinstance(data.get("video"), tuple):
            if len(data["video"]) == 3:
                data["video"], clip_start_time, data["__frame_indices__"] = data["video"]
            elif len(data["video"]) == 2:
                data["video"], clip_start_time = data["video"]

        data["__start_time__"] = clip_start_time


        if isinstance(data.get("prompt"), dict):
            cap = data["prompt"]
            data["prompt"] = cap["text"]
            if "segment_prompts" in cap:

                data["segment_prompts"] = self._align_segment_prompts(
                    cap["segment_prompts"], clip_start_time, len(data.get("video", []))
                )
        return data

    def _align_segment_prompts(self, segment_prompts, clip_start_time, num_frames):

        if not segment_prompts or clip_start_time == 0:
            return segment_prompts

        clip_start_frame = int(clip_start_time * self.target_fps)
        clip_end_frame = clip_start_frame + num_frames
        aligned = []
        for seg in segment_prompts:
            seg_start = seg["start_frame"]
            seg_end = seg["end_frame"]

            if seg_end <= clip_start_frame or seg_start >= clip_end_frame:
                continue

            aligned.append({
                "start_frame": max(0, seg_start - clip_start_frame),
                "end_frame": min(num_frames, seg_end - clip_start_frame),
                "prompt": seg["prompt"],
            })
        return aligned if aligned else None

    def __len__(self):
        return len(self.inner)


class SubsampledDataset(torch.utils.data.Dataset):


    def __init__(self, inner, subset_size, part_id: int = 0,
                 resample_each_epoch: bool = False, base_seed: int = 0):
        self.inner = inner
        self.subset_size = int(min(subset_size, len(inner)))
        self.part_id = int(part_id)
        self.resample_each_epoch = bool(resample_each_epoch)
        self.base_seed = int(base_seed)
        self._epoch = None
        if self.resample_each_epoch:


            self.set_epoch(0)
        else:
            self.indices = random.sample(range(len(inner)), self.subset_size)

    def set_epoch(self, epoch: int):


        if not self.resample_each_epoch:
            return
        e = int(epoch)
        if e == self._epoch:
            return
        rng = random.Random((self.base_seed & 0x7FFFFFFF) * 1000003
                            + e * 1000033 + self.part_id * 7919 + 20260729)
        self.indices = rng.sample(range(len(self.inner)), self.subset_size)
        self._epoch = e

    def __getitem__(self, idx):
        return self.inner[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def set_dataset_epoch(dataset, epoch: int) -> int:


    n = 0
    for part in (getattr(dataset, "datasets", None) or []):
        if isinstance(part, SubsampledDataset):
            part.set_epoch(epoch)
            n += 1
    return n


def load_data_config(config_path, max_pixels=1024 * 1024, height=None, width=None,
                     num_frames=None, target_fps=None,
                     resample_each_epoch: bool = False, seed: int = 0):


    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_datasets = cfg["datasets"]
    defaults = cfg.get("defaults", {})
    select = cfg["select"]
    ratio = cfg["ratio"]


    global_num_frames = num_frames if num_frames is not None else defaults.get("num_frames", 193)
    global_target_fps = target_fps if target_fps is not None else defaults.get("target_fps", 24)

    assert len(select) == len(ratio), f"select ({len(select)}) and ratio ({len(ratio)}) must have same length"

    sub_datasets = []
    raw_sizes = []

    for name in select:
        assert name in all_datasets, f"Dataset '{name}' not found in datasets config"

        ds_cfg = {**defaults, **all_datasets[name]}

        for nested_key in ("v2v", "task_ratio", "caption_key"):
            if nested_key in defaults and nested_key not in all_datasets[name]:
                ds_cfg[nested_key] = defaults[nested_key]

        source_fps = ds_cfg.get("source_fps", None)
        source_resolution = ds_cfg.get("source_resolution", None)
        if source_resolution is not None:
            source_h, source_w = int(source_resolution[0]), int(source_resolution[1])
        else:
            source_h, source_w = None, None


        task_ratio = ds_cfg.get("task_ratio", {}) or {}
        needs_first_frame = "memory_rollout" in task_ratio
        frame_processor = ImageCropAndResize(height, width, max_pixels, 16, 16)

        video_loader = LoadVideo(
            global_num_frames, 4, 1,
            frame_processor=frame_processor,
            target_fps=global_target_fps,
            source_fps=source_fps,
            random_start=True,
            return_first_frame=needs_first_frame,


            require_full_length=ds_cfg.get("require_full_length", False),
        )
        video_operator = RouteByType(operator_map=[
            (str, ToAbsolutePath(ds_cfg["video_dir"]) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                (("gif",), LoadGIF(global_num_frames, 4, 1, frame_processor=frame_processor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), video_loader),
            ])),
        ])


        skill_source = bool(ds_cfg.get("skill_source", False))
        event_window_keys = ds_cfg.get("event_window_keys", None)

        caption_operator = ResolveCaptionFile(
            caption_dir=ds_cfg.get("caption_dir", ""),
            caption_key=ds_cfg.get("caption_key", "overall_caption"),
            target_fps=global_target_fps,

            inline=ds_cfg.get("inline_caption", False),


            segment_text_key=ds_cfg.get("segment_text_key", "full_prompt"),
            strip_camera_tags=ds_cfg.get("strip_camera_tags", False),
        )


        if os.environ.get("RANK", "0") == "0":
            print(f"[LW-ALIGN][data] {name}: caption_key={ds_cfg.get('caption_key', 'overall_caption')} "
                  f"segment_text_key={ds_cfg.get('segment_text_key', 'full_prompt')} "
                  f"strip_camera_tags={bool(ds_cfg.get('strip_camera_tags', False))} "
                  f"require_full_length={bool(ds_cfg.get('require_full_length', False))}", flush=True)


        has_pose = "pose_dir" in ds_cfg and ds_cfg["pose_dir"]
        data_file_keys = ("video", "prompt", "pose") if has_pose else ("video", "prompt")


        jsonl_key_map = ds_cfg.get("jsonl_key_map", None)
        if jsonl_key_map is None:

            jsonl_key_map = {
                "video_path": "video",
                "pose_path": "pose",
                "prompt_path": "prompt",
            }

        ds = UnifiedDataset(
            base_path=ds_cfg["video_dir"],
            metadata_path=ds_cfg["jsonl"],
            data_file_keys=data_file_keys,
            main_data_operator=video_operator,
            special_operator_map={"prompt": caption_operator},
            jsonl_key_map=jsonl_key_map,
            pose_dir=ds_cfg.get("pose_dir", None),
            target_h=height,
            target_w=width,
            source_h=source_h,
            source_w=source_w,

            fallback_default_intrinsic=ds_cfg.get("fallback_default_intrinsic", False),
            event_window_keys=event_window_keys,
            skill_source=skill_source,
            video_loader=video_loader,
        )


        path_replace = ds_cfg.get("path_replace", None)
        if path_replace:
            for entry in ds.data:
                for key, replace_map in path_replace.items():
                    val = entry.get(key)
                    if not isinstance(val, str):
                        continue
                    for old, new in replace_map.items():
                        val = val.replace(old, new, 1)
                    entry[key] = val

        raw_sizes.append(len(ds))

        ds_meta = {
            "task_ratio": ds_cfg.get("task_ratio", {"t2v": 0.1, "i2v": 0.7, "v2v": 0.2}),
            "v2v": ds_cfg.get("v2v", {}),
            "has_pose": has_pose,
            "skill": skill_source,
        }
        sub_datasets.append(ConfigAwareDataset(ds, ds_meta, target_fps=global_target_fps))


    parts = []
    total_effective = 0


    _sub_part_id = 0
    _n_resampled = 0

    for ds, r, name, raw_sz in zip(sub_datasets, ratio, select, raw_sizes):
        effective = int(raw_sz * r)

        def _mk_sub(_inner, _size):
            nonlocal _sub_part_id, _n_resampled
            _sub = SubsampledDataset(_inner, _size, part_id=_sub_part_id,
                                     resample_each_epoch=resample_each_epoch, base_seed=seed)
            _sub_part_id += 1
            if resample_each_epoch:
                _n_resampled += 1
            return _sub

        if r >= 1:
            full_repeats = int(r)
            frac = r - full_repeats

            for _ in range(full_repeats):
                parts.append(ds)

            if frac > 0:
                frac_size = int(raw_sz * frac)
                if frac_size > 0:
                    parts.append(_mk_sub(ds, frac_size))
        else:

            subset_size = max(1, int(raw_sz * r))
            parts.append(_mk_sub(ds, subset_size))

        total_effective += effective


        print(f"[DataConfig] {name}: {raw_sz} x {r} = {effective} effective samples"
              f"{'  [re-drawn each epoch]' if (resample_each_epoch and r != int(r)) else ''}")

    dataset = torch.utils.data.ConcatDataset(parts)
    print(f"[DataConfig] Total: {len(dataset)} samples/epoch  (select={select}, ratio={ratio})")
    if resample_each_epoch:
        print(f"[EPOCH-RESAMPLE] on: {_n_resampled} randomly subsampled parts are re-drawn every epoch"
              f" (same size, so the epoch length does not change); seed = f(seed={seed}, epoch, part_id), rank-independent."
              f" Requires persistent_workers=false for workers to see the new subset each epoch.", flush=True)
    return dataset

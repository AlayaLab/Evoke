

from __future__ import annotations

import time
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

torch.backends.cudnn.benchmark = False


SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):


    _commit_hash: str | None = None

    def __init__(self, model_name: str = "da3-large", **kwargs):


        super().__init__()
        self.model_name = model_name


        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()


        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()


        self.device = None

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:


        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
                return self.model(
                    image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
                )

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,

        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,

        feat_vis_fps: int = 15,

        export_kwargs: Optional[dict] = {},
    ) -> Prediction:


        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."


        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )


        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)


        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)


        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(
            imgs, ex_t_norm, in_t, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
        )


        prediction = self._convert_to_prediction(raw_output)


        prediction = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale
        )


        prediction = self._add_processed_images(prediction, imgs_cpu)


        if export_dir is not None:

            if "gs" in export_format:
                if infer_gs and "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format:
                    if "gs_video" not in export_kwargs:
                        export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update(
                        {
                            "extrinsics": render_exts,
                            "intrinsics": render_ixts,
                            "out_image_hw": render_hw,
                        }
                    )

            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update(
                    {
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "num_max_points": num_max_points,
                        "show_cameras": show_cameras,
                    }
                )

            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update(
                    {
                        "fps": feat_vis_fps,
                    }
                )

            if "colmap" in export_format:
                if "colmap" not in export_kwargs:
                    export_kwargs["colmap"] = {}
                export_kwargs["colmap"].update(
                    {
                        "image_paths": image,
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "process_res_method": process_res_method,
                    }
                )
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:

        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(
        self,
        imgs_cpu: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:

        device = self._get_model_device()


        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()


        ex_t = (
            extrinsics.to(device, non_blocking=True)[None].float()
            if extrinsics is not None
            else None
        )
        in_t = (
            intrinsics.to(device, non_blocking=True)[None].float()
            if intrinsics is not None
            else None
        )

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:

        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:

        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor | None,
        in_t: torch.Tensor | None,
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:

        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs, use_ray_pose, ref_view_strategy)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:

        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:


        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()


        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(
        self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:

        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_model_device(self) -> torch.device:


        if self.device is not None:
            return self.device


        for param in self.parameters():
            self.device = param.device
            return param.device


        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")

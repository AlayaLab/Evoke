

import glob
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.gs import export_to_gs_video


class ModelInference:


    def __init__(self):

        self.model = None

    def initialize_model(self, device: str = "cuda") -> None:


        if self.model is None:

            model_dir = os.environ.get(
                "DA3_MODEL_DIR", "/dev/shm/da3_models/DA3HF-VITG-METRIC_VITL"
            )
            self.model = DepthAnything3.from_pretrained(model_dir)
            self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)

        self.model.eval()

    def run_inference(
        self,
        target_dir: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        show_camera: bool = True,
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
    ) -> Tuple[Any, Dict[int, Dict[str, Any]]]:


        print(f"Processing images from {target_dir}")


        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)


        self.initialize_model(device)


        print("Loading images...")
        image_folder_path = os.path.join(target_dir, "images")
        all_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*")))


        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        all_image_paths = [
            path
            for path in all_image_paths
            if any(path.lower().endswith(ext) for ext in image_extensions)
        ]

        print(f"Found {len(all_image_paths)} images")
        print(f"All image paths: {all_image_paths}")


        image_paths = all_image_paths
        print(f"Reference view selection strategy: {ref_view_strategy}")

        if len(image_paths) == 0:
            raise ValueError("No images found. Check your upload.")


        method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}
        actual_method = method_mapping.get(process_res_method, "upper_bound_crop")


        print(f"Running inference with method: {actual_method}")
        with torch.no_grad():
            prediction = self.model.inference(
                image_paths,
                export_dir=None,
                process_res_method=actual_method,
                infer_gs=infer_gs,
                ref_view_strategy=ref_view_strategy,
            )

        export_to_glb(
            prediction,
            filter_black_bg=filter_black_bg,
            filter_white_bg=filter_white_bg,
            export_dir=target_dir,
            show_cameras=show_camera,
            conf_thresh_percentile=save_percentage,
            num_max_points=int(num_max_points),
        )


        if infer_gs:
            mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
            print(f"GS mode: {gs_trj_mode}; Backend mode: {mode_mapping[gs_trj_mode]}")
            export_to_gs_video(
                prediction,
                export_dir=target_dir,
                chunk_size=4,
                trj_mode=mode_mapping.get(gs_trj_mode, "extend"),
                enable_tqdm=True,
                vis_depth="hcat",
                video_quality=gs_video_quality,
            )


        self._save_predictions_cache(target_dir, prediction)


        processed_data = self._process_results(target_dir, prediction, image_paths)


        cleanup_cuda_memory()

        return prediction, processed_data

    def _save_predictions_cache(self, target_dir: str, prediction: Any) -> None:


        try:
            output_file = os.path.join(target_dir, "predictions.npz")


            save_dict = {}


            if prediction.processed_images is not None:
                save_dict["images"] = prediction.processed_images


            if prediction.depth is not None:
                save_dict["depths"] = np.round(prediction.depth, 6)


            if prediction.conf is not None:
                save_dict["conf"] = np.round(prediction.conf, 2)


            if prediction.extrinsics is not None:
                save_dict["extrinsics"] = prediction.extrinsics
            if prediction.intrinsics is not None:
                save_dict["intrinsics"] = prediction.intrinsics


            np.savez_compressed(output_file, **save_dict)
            print(f"Saved predictions cache to: {output_file}")

        except Exception as e:
            print(f"Warning: Failed to save predictions cache: {e}")

    def _process_results(
        self, target_dir: str, prediction: Any, image_paths: list
    ) -> Dict[int, Dict[str, Any]]:


        processed_data = {}


        depth_vis_dir = os.path.join(target_dir, "depth_vis")

        if os.path.exists(depth_vis_dir):
            depth_files = sorted(glob.glob(os.path.join(depth_vis_dir, "*.jpg")))
            for i, depth_file in enumerate(depth_files):

                processed_image = None
                if prediction.processed_images is not None and i < len(
                    prediction.processed_images
                ):
                    processed_image = prediction.processed_images[i]

                processed_data[i] = {
                    "depth_image": depth_file,
                    "image": processed_image,
                    "original_image_path": image_paths[i] if i < len(image_paths) else None,
                    "depth": prediction.depth[i] if i < len(prediction.depth) else None,
                    "intrinsics": (
                        prediction.intrinsics[i]
                        if prediction.intrinsics is not None and i < len(prediction.intrinsics)
                        else None
                    ),
                    "mask": None,
                }

        return processed_data

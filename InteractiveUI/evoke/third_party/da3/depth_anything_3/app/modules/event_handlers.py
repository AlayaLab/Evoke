

import os
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import numpy as np
import torch

from depth_anything_3.app.modules.file_handlers import FileHandler
from depth_anything_3.app.modules.model_inference import ModelInference
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.app.modules.visualization import VisualizationHandler


class EventHandlers:


    def __init__(self):

        self.model_inference = ModelInference()
        self.file_handler = FileHandler()
        self.visualization_handler = VisualizationHandler()

    def clear_fields(self) -> None:


        return None

    def update_log(self) -> str:


        return "Loading and Reconstructing..."

    def save_current_visualization(
        self,
        target_dir: str,
        save_percentage: float,
        show_cam: bool,
        filter_black_bg: bool,
        filter_white_bg: bool,
        processed_data: Optional[Dict],
        scene_name: str = "",
    ) -> str:


        if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
            return "No reconstruction available. Please run 'Reconstruct' first."

        if processed_data is None:
            return "No processed data available. Please run 'Reconstruct' first."

        try:

            print("[DEBUG] save_current_visualization called with:")
            print(f"  target_dir: {target_dir}")
            print(f"  save_percentage: {save_percentage}")
            print(f"  show_cam: {show_cam}")
            print(f"  filter_black_bg: {filter_black_bg}")
            print(f"  filter_white_bg: {filter_white_bg}")
            print(f"  processed_data: {processed_data is not None}")


            import datetime

            from .utils import save_to_gallery_func

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if scene_name and scene_name.strip():
                gallery_name = f"{scene_name.strip()}_{timestamp}_pct{save_percentage:.0f}"
            else:
                gallery_name = f"save_{timestamp}_pct{save_percentage:.0f}"

            print(f"[DEBUG] Saving to gallery with name: {gallery_name}")


            success, message = save_to_gallery_func(
                target_dir=target_dir, processed_data=processed_data, gallery_name=gallery_name
            )

            if success:
                print(f"[DEBUG] Gallery save completed successfully: {message}")
                return (
                    "Successfully saved to gallery!\n"
                    f"Gallery name: {gallery_name}\n"
                    f"Save percentage: {save_percentage}%\n"
                    f"Show cameras: {show_cam}\n"
                    f"Filter black bg: {filter_black_bg}\n"
                    f"Filter white bg: {filter_white_bg}\n\n"
                    f"{message}"
                )
            else:
                print(f"[DEBUG] Gallery save failed: {message}")
                return f"Failed to save to gallery: {message}"

        except Exception as e:
            return f"Error saving visualization: {str(e)}"

    def gradio_demo(
        self,
        target_dir: str,
        show_cam: bool = True,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
    ) -> Tuple[
        Optional[str],
        str,
        Optional[Dict],
        Optional[np.ndarray],
        Optional[np.ndarray],
        str,
        gr.Dropdown,
        Optional[str],
        gr.update,
        gr.update,
    ]:


        if not os.path.isdir(target_dir) or target_dir == "None":
            return (
                None,
                "No valid target directory found. Please upload first.",
                None,
                None,
                None,
                "",
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=True),
            )

        start_time = time.time()
        cleanup_cuda_memory()


        target_dir_images = os.path.join(target_dir, "images")
        all_files = (
            sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
        )

        print("Running DepthAnything3 model...")
        print(f"Reference view strategy: {ref_view_strategy}")

        with torch.no_grad():
            prediction, processed_data = self.model_inference.run_inference(
                target_dir,
                process_res_method=process_res_method,
                show_camera=show_cam,
                save_percentage=save_percentage,
                num_max_points=int(num_max_points * 1000),
                infer_gs=infer_gs,
                ref_view_strategy=ref_view_strategy,
                gs_trj_mode=gs_trj_mode,
                gs_video_quality=gs_video_quality,
            )


        glbfile = os.path.join(target_dir, "scene.glb")


        gsvideo_path = None
        gs_video_visible = False
        gs_info_visible = True

        if infer_gs:
            try:
                gsvideo_path = sorted(glob(os.path.join(target_dir, "gs_video", "*.mp4")))[-1]
                gs_video_visible = True
                gs_info_visible = False
            except IndexError:
                gsvideo_path = None
                print("3DGS video not found, but infer_gs was enabled")


        cleanup_cuda_memory()

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."


        depth_vis, measure_img, measure_depth_vis, measure_pts = (
            self.visualization_handler.populate_visualization_tabs(processed_data)
        )


        depth_selector, measure_selector = self.visualization_handler.update_view_selectors(
            processed_data
        )

        return (
            glbfile,
            log_msg,
            processed_data,
            measure_img,
            measure_depth_vis,
            "",
            measure_selector,
            gsvideo_path,
            gr.update(visible=gs_video_visible),
            gr.update(visible=gs_info_visible),
        )

    def update_visualization(
        self,
        target_dir: str,
        show_cam: bool,
        is_example: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
    ) -> Tuple[gr.update, str]:


        if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
            return (
                gr.update(),
                "No reconstruction available. Please click the Reconstruct button first.",
            )


        glbfile = os.path.join(target_dir, "scene.glb")
        if os.path.exists(glbfile):
            return (
                glbfile,
                (
                    "Visualization loaded from cache."
                    if is_example == "True"
                    else "Visualization updated."
                ),
            )


        if is_example == "True":
            return (
                gr.update(),
                "No reconstruction available. Please click the Reconstruct button first.",
            )


        predictions_path = os.path.join(target_dir, "predictions.npz")
        if not os.path.exists(predictions_path):
            error_message = (
                f"No reconstruction available at {predictions_path}. "
                "Please run 'Reconstruct' first."
            )
            return gr.update(), error_message

        loaded = np.load(predictions_path, allow_pickle=True)
        predictions = {key: loaded[key] for key in loaded.keys()}

        return (
            glbfile,
            "Visualization updated.",
        )

    def handle_uploads(
        self,
        input_video: Optional[str],
        input_images: Optional[List],
        s_time_interval: float = 10.0,
    ) -> Tuple[Optional[str], Optional[str], Optional[List], Optional[str]]:


        return self.file_handler.update_gallery_on_upload(
            input_video, input_images, s_time_interval
        )

    def load_example_scene(self, scene_name: str, examples_dir: str = None) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[List],
        str,
        Optional[Dict],
        gr.Dropdown,
        Optional[str],
        gr.update,
        gr.update,
    ]:


        if examples_dir is None:

            workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "gradio_workspace")
            examples_dir = os.path.join(workspace_dir, "examples")

        reconstruction_output, target_dir, image_paths, log_message = (
            self.file_handler.load_example_scene(scene_name, examples_dir)
        )


        processed_data = None
        measure_view_selector = gr.Dropdown(choices=["View 1"], value="View 1")
        gs_video_path = None
        gs_video_visible = False
        gs_info_visible = True

        if target_dir and target_dir != "None":
            predictions_path = os.path.join(target_dir, "predictions.npz")
            if os.path.exists(predictions_path):
                try:

                    loaded = np.load(predictions_path, allow_pickle=True)
                    predictions = {key: loaded[key] for key in loaded.keys()}


                    num_images = len(predictions.get("images", []))
                    processed_data = {}

                    for i in range(num_images):
                        processed_data[i] = {
                            "image": predictions["images"][i] if "images" in predictions else None,
                            "depth": predictions["depths"][i] if "depths" in predictions else None,
                            "depth_image": os.path.join(
                                target_dir, "depth_vis", f"{i:04d}.jpg"
                            ),
                            "intrinsics": (
                                predictions["intrinsics"][i]
                                if "intrinsics" in predictions
                                and i < len(predictions["intrinsics"])
                                else None
                            ),
                            "mask": None,
                        }


                    choices = [f"View {i + 1}" for i in range(num_images)]
                    measure_view_selector = gr.Dropdown(choices=choices, value=choices[0])

                except Exception as e:
                    print(f"Error loading cached data: {e}")


            gs_video_dir = os.path.join(target_dir, "gs_video")
            if os.path.exists(gs_video_dir):
                try:
                    from glob import glob

                    gs_videos = sorted(glob(os.path.join(gs_video_dir, "*.mp4")))
                    if gs_videos:
                        gs_video_path = gs_videos[-1]
                        gs_video_visible = True
                        gs_info_visible = False
                        print(f"Loaded cached 3DGS video: {gs_video_path}")
                except Exception as e:
                    print(f"Error loading cached 3DGS video: {e}")

        return (
            reconstruction_output,
            target_dir,
            image_paths,
            log_message,
            processed_data,
            measure_view_selector,
            gs_video_path,
            gr.update(visible=gs_video_visible),
            gr.update(visible=gs_info_visible),
        )

    def navigate_depth_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector: str,
        direction: int,
    ) -> Tuple[str, Optional[str]]:


        return self.visualization_handler.navigate_depth_view(
            processed_data, current_selector, direction
        )

    def update_depth_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[str]:


        return self.visualization_handler.update_depth_view(processed_data, view_index)

    def navigate_measure_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector: str,
        direction: int,
    ) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], List]:


        return self.visualization_handler.navigate_measure_view(
            processed_data, current_selector, direction
        )

    def update_measure_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:


        return self.visualization_handler.update_measure_view(processed_data, view_index)

    def measure(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        measure_points: List,
        current_view_selector: str,
        event: gr.SelectData,
    ) -> List:


        return self.visualization_handler.measure(
            processed_data, measure_points, current_view_selector, event
        )

    def select_first_frame(
        self, image_gallery: List, selected_index: int = 0
    ) -> Tuple[List, str, str]:


        try:
            if not image_gallery or len(image_gallery) == 0:
                return image_gallery, "No images available to select as first frame.", ""


            if (
                selected_index is None
                or selected_index < 0
                or selected_index >= len(image_gallery)
            ):
                selected_index = 0
                print(f"Invalid selected_index: {selected_index}, using default: 0")


            selected_image = image_gallery[selected_index]
            print(f"Selected image index: {selected_index}")
            print(f"Total images: {len(image_gallery)}")


            selected_frame_path = ""
            print(f"Selected image type: {type(selected_image)}")
            print(f"Selected image: {selected_image}")

            if isinstance(selected_image, tuple):

                selected_frame_path = selected_image[0]
            elif isinstance(selected_image, str):
                selected_frame_path = selected_image
            elif hasattr(selected_image, "name"):
                selected_frame_path = selected_image.name
            elif isinstance(selected_image, dict):
                if "name" in selected_image:
                    selected_frame_path = selected_image["name"]
                elif "path" in selected_image:
                    selected_frame_path = selected_image["path"]
                elif "src" in selected_image:
                    selected_frame_path = selected_image["src"]
            else:

                selected_frame_path = str(selected_image)

            print(f"Extracted path: {selected_frame_path}")


            import os

            selected_filename = os.path.basename(selected_frame_path)
            print(f"Selected filename: {selected_filename}")


            updated_gallery = [selected_image] + [
                img for img in image_gallery if img != selected_image
            ]

            log_message = (
                f"Selected frame: {selected_filename}. "
                f"Moved to first position. Total frames: {len(updated_gallery)}"
            )
            return updated_gallery, log_message, selected_filename

        except Exception as e:
            print(f"Error selecting first frame: {e}")
            return image_gallery, f"Error selecting first frame: {e}", ""

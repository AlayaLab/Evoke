

from typing import Any, Dict, List, Optional, Union
import numpy as np
import requests
import typer

from ..api import DepthAnything3


class InferenceService:


    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):

        if self.model is None:
            typer.echo(f"Loading model from {self.model_dir}...")
            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
        return self.model

    def run_local_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Any:

        if export_feat_layers is None:
            export_feat_layers = []

        model = self.load_model()


        inference_kwargs = {
            "image": image_paths,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
        }


        if extrinsics is not None:
            inference_kwargs["extrinsics"] = extrinsics
        if intrinsics is not None:
            inference_kwargs["intrinsics"] = intrinsics


        typer.echo(f"Running inference on {len(image_paths)} images...")
        prediction = model.inference(**inference_kwargs)

        typer.echo(f"Results saved to {export_dir}")
        typer.echo(f"Export format: {export_format}")

        return prediction

    def run_backend_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        backend_url: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Dict[str, Any]:

        if export_feat_layers is None:
            export_feat_layers = []


        if not self._check_backend_status(backend_url):
            raise typer.BadParameter(f"Backend service is not running at {backend_url}")


        payload = {
            "image_paths": image_paths,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
        }


        if extrinsics is not None:
            payload["extrinsics"] = [ext.astype(np.float64).tolist() for ext in extrinsics]
        if intrinsics is not None:
            payload["intrinsics"] = [intr.astype(np.float64).tolist() for intr in intrinsics]


        typer.echo("Submitting inference task to backend...")
        try:
            response = requests.post(f"{backend_url}/inference", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["success"]:
                task_id = result["task_id"]
                typer.echo("Task submitted successfully!")
                typer.echo(f"Task ID: {task_id}")
                typer.echo(f"Results will be saved to: {export_dir}")
                typer.echo(f"Check backend logs for progress updates with task ID: {task_id}")
                return result
            else:
                raise typer.BadParameter(
                    f"Backend inference submission failed: {result['message']}"
                )
        except requests.exceptions.RequestException as e:
            raise typer.BadParameter(f"Backend inference submission failed: {e}")

    def _check_backend_status(self, backend_url: str) -> bool:

        try:
            response = requests.get(f"{backend_url}/status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def run_inference(
    image_paths: List[str],
    export_dir: str,
    model_dir: str,
    device: str = "cuda",
    backend_url: Optional[str] = None,
    export_format: str = "mini_npz-glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    export_feat_layers: List[int] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    align_to_input_ext_scale: bool = True,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    feat_vis_fps: int = 15,
) -> Union[Any, Dict[str, Any]]:


    service = InferenceService(model_dir, device)

    if backend_url:
        return service.run_backend_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            backend_url=backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
    else:
        return service.run_local_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )



import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def create_depth_visualization(depth: np.ndarray) -> Optional[np.ndarray]:


    if depth is None:
        return None


    depth_min = depth[depth > 0].min() if (depth > 0).any() else 0
    depth_max = depth.max()

    if depth_max <= depth_min:
        return None


    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_norm = np.clip(depth_norm, 0, 1)


    import matplotlib.cm as cm


    depth_colored = cm.viridis(depth_norm)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored


def save_to_gallery_func(
    target_dir: str, processed_data: Dict[int, Dict[str, Any]], gallery_name: Optional[str] = None
) -> Tuple[bool, str]:


    try:

        gallery_dir = os.environ.get(
            "DA3_GALLERY_DIR",
            "workspace/gallery",
        )
        if not os.path.exists(gallery_dir):
            os.makedirs(gallery_dir)


        if gallery_name is None or gallery_name.strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gallery_name = f"reconstruction_{timestamp}"

        gallery_path = os.path.join(gallery_dir, gallery_name)


        if os.path.exists(gallery_path):
            return False, f"Save failed: folder '{gallery_name}' already exists"


        os.makedirs(gallery_path, exist_ok=True)


        glb_source = os.path.join(target_dir, "scene.glb")
        glb_dest = os.path.join(gallery_path, "scene.glb")
        if os.path.exists(glb_source):
            shutil.copy2(glb_source, glb_dest)


        depth_vis_dir = os.path.join(target_dir, "depth_vis")
        if os.path.exists(depth_vis_dir):
            gallery_depth_vis = os.path.join(gallery_path, "depth_vis")
            shutil.copytree(depth_vis_dir, gallery_depth_vis)


        images_source = os.path.join(target_dir, "images")
        if os.path.exists(images_source):
            gallery_images = os.path.join(gallery_path, "images")
            shutil.copytree(images_source, gallery_images)

        scene_preview_source = os.path.join(target_dir, "scene.jpg")
        scene_preview_dest = os.path.join(gallery_path, "scene.jpg")
        shutil.copy2(scene_preview_source, scene_preview_dest)


        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_images": len(processed_data) if processed_data else 0,
            "gallery_name": gallery_name,
        }

        with open(os.path.join(gallery_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved reconstruction to gallery: {gallery_path}")
        return True, f"Save successful: saved to {gallery_path}"

    except Exception as e:
        print(f"Error saving to gallery: {e}")
        return False, f"Save failed: {str(e)}"


def get_scene_info(examples_dir: str) -> List[Dict[str, Any]]:


    import glob

    scenes = []
    if not os.path.exists(examples_dir):
        return scenes

    for scene_folder in sorted(os.listdir(examples_dir)):
        scene_path = os.path.join(examples_dir, scene_folder)
        if os.path.isdir(scene_path):

            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(scene_path, ext)))
                image_files.extend(glob.glob(os.path.join(scene_path, ext.upper())))

            if image_files:

                image_files = sorted(image_files)
                first_image = image_files[0]
                num_images = len(image_files)

                scenes.append(
                    {
                        "name": scene_folder,
                        "path": scene_path,
                        "thumbnail": first_image,
                        "num_images": num_images,
                        "image_files": image_files,
                    }
                )

    return scenes


def get_logo_base64() -> Optional[str]:


    import base64

    logo_path = "examples/WAI-Logo/wai_logo.png"
    try:
        with open(logo_path, "rb") as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{base64_str}"
    except FileNotFoundError:
        return None

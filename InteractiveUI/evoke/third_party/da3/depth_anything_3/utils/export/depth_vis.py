

import os
import imageio
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth


def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
):

    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images

    os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    for idx in range(prediction.depth.shape[0]):
        depth_vis = visualize_depth(prediction.depth[idx])
        image_vis = images_u8[idx]
        depth_vis = depth_vis.astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        save_path = os.path.join(export_dir, f"depth_vis/{idx:04d}.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)

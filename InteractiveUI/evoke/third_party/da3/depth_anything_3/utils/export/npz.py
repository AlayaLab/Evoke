

import os
import numpy as np

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.parallel_utils import async_call


@async_call
def export_to_npz(
    prediction: Prediction,
    export_dir: str,
):
    output_file = os.path.join(export_dir, "exports", "npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    image = prediction.processed_images


    save_dict = {
        "image": image,
        "depth": np.round(prediction.depth, 6),
    }

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics


    np.savez_compressed(output_file, **save_dict)


@async_call
def export_to_mini_npz(
    prediction: Prediction,
    export_dir: str,
):
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    save_dict = {
        "depth": np.round(prediction.depth, 8),
    }

    if prediction.conf is not None:
        save_dict["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        save_dict["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        save_dict["intrinsics"] = prediction.intrinsics

    np.savez_compressed(output_file, **save_dict)

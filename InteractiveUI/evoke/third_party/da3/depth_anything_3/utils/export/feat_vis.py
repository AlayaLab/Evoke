

import os
import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm

from depth_anything_3.utils.parallel_utils import async_call
from depth_anything_3.utils.pca_utils import PCARGBVisualizer


@async_call
def export_to_feat_vis(
    prediction,
    export_dir,
    fps=15,
):


    out_dir = os.path.join(export_dir, "feat_vis")
    os.makedirs(out_dir, exist_ok=True)

    images = prediction.processed_images
    for k, v in prediction.aux.items():
        if not k.startswith("feat_layer_"):
            continue
        os.makedirs(os.path.join(out_dir, k), exist_ok=True)
        viz = PCARGBVisualizer(basis_mode="fixed", percentile_mode="global", clip_percent=10.0)
        viz.fit_reference(v)
        feats_vis = viz.transform_video(v)
        for idx in tqdm(range(len(feats_vis))):
            img = images[idx]
            feat_vis = (feats_vis[idx] * 255).astype(np.uint8)
            feat_vis = cv2.resize(
                feat_vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            save_path = os.path.join(out_dir, f"{k}/{idx:06d}.jpg")
            save = np.concatenate([img, feat_vis], axis=1)
            imageio.imwrite(save_path, save, quality=95)
        cmd = (
            "ffmpeg -loglevel error -hide_banner -y "
            f"-framerate {fps} -start_number 0 "
            f"-i {out_dir}/{k}/%06d.jpg "
            f"-c:v libx264 -pix_fmt yuv420p "
            f"{out_dir}/{k}.mp4"
        )
        os.system(cmd)

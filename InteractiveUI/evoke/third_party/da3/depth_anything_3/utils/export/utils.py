

import numpy as np
import torch


def _denorm_and_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:

    resnet_mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device
    )
    resnet_std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device
    )
    img = image_tensor * resnet_std[None, :, None, None] + resnet_mean[None, :, None, None]
    img = torch.clamp(img, 0.0, 1.0)
    img = (img.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return img

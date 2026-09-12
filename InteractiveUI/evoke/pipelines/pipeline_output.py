from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class EvokePipelineOutput(BaseOutput):


    frames: torch.Tensor

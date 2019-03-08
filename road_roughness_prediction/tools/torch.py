import numpy as np

import torch

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import resize


def to_image(tensor: torch.Tensor):
    '''Return [W, H, C] numpy array'''
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_resized_grid(tensor: torch.Tensor, size, normalize=False) -> torch.Tensor:
    grid = make_grid(tensor)
    return to_tensor(resize(to_pil_image(grid), size))

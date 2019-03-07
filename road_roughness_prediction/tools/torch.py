import numpy as np

import torch


def to_image(tensor: torch.Tensor):
    '''Return [W, H, C] numpy array'''
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

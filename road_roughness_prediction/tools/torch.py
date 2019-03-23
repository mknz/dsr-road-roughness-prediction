import numpy as np

import torch


def to_image(tensor: torch.Tensor):
    '''Return [W, H, C] numpy array'''
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_device(use_cpu=True, device_id=0):
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else "cpu")
    return device


def set_seeds(seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in device.type:
        torch.cuda.manual_seed(seed)

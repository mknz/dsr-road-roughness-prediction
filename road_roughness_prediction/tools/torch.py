import numpy as np

from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import torch

from road_roughness_prediction.tools.image_utils import fig_to_pil


def to_image(tensor: torch.Tensor):
    '''Return [W, H, C] numpy array'''
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_segmentated_images_tensor(tensor: torch.Tensor, dim) -> torch.Tensor:
    tensors = []
    for i in range(tensor.shape[0]):
        n_class = tensor.shape[dim]
        segmented = tensor[i, :, :, :].argmax(dim=dim).float()
        segmented /= n_class - 1  # normalize 0 to 1
        tensors.append(get_segmentated_image_tensor(segmented))
    return torch.stack(tensors)


def get_segmentated_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    fig = plt.figure()
    plt.imshow(tensor.numpy(), figure=fig, cmap='jet', vmin=0., vmax=1.)
    plt.axis('off')
    plt.tight_layout()
    pil_image = fig_to_pil(fig).convert(mode='RGB')
    plt.close()
    return to_tensor(pil_image)


def make_resized_grid(tensor: torch.Tensor, size, normalize=False) -> torch.Tensor:
    grid = make_grid(tensor, normalize=normalize)
    grid_pil = to_pil_image(grid.float())
    grid_pil_resized = resize(grid_pil, size)
    grid_tensor = to_tensor(grid_pil_resized)
    return grid_tensor

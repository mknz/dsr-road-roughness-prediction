from pathlib import Path
from typing import List

import numpy as np

from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

import cv2

import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from road_roughness_prediction.tools.image_utils import fig_to_pil
import road_roughness_prediction.segmentation.datasets.surface_types as surface_types


class Logger:
    def __init__(self, log_dir, n_save, image_size, category_type):
        self.n_save = n_save
        self.image_size = image_size
        self.writer = SummaryWriter(str(log_dir))

        self.category_type = category_type
        self.is_binary = category_type == surface_types.BinaryCategory
        self.n_class = len(category_type)

    def add_output(self, tag, out_tensor, global_steps=None):

        if self.is_binary:
            # out:  [n_batch, height, width]
            out_ = out_tensor[:self.n_save, :, :]
            save_img = make_resized_grid(out_, size=self.image_size, normalize=True)
        else:
            # out:  [n_batch, n_class, height, width]
            out_ = out_tensor[:self.n_save, :, :, :]
            segmented = get_segmentated_images_tensor(out_, dim=0)
            save_img = make_resized_grid(segmented, size=self.image_size, normalize=False)

        self.writer.add_image(tag, save_img, global_steps)

    def add_input(self, tag, input_tensor, global_steps=None):
        # X: [n_batch, 3, height, width]
        X_ = input_tensor[:self.n_save, :, :, :]
        x_save = make_resized_grid(X_, size=self.image_size, normalize=True)
        self.writer.add_image(tag, x_save, global_steps)

    def add_target(self, tag, target_tensor, global_steps=None):
        if self.is_binary:
            # Y: [n_batch, 1, height, width]
            # Y is 0 or 1
            Y_ = target_tensor[:self.n_save, :, :, :]
            y_save = make_resized_grid(Y_, size=self.image_size, normalize=True)
        else:
            # Y: [n_batch, height, width]
            # Y is 0 to n_class - 1
            Y_ = target_tensor[:self.n_save, :, :]
            images = []
            for i in range(Y_.shape[0]):
                Y_norm = Y_[i, :, :].float() / (self.n_class - 1)   # Normalize 0 to 1
                images.append(get_segmentated_image_tensor(Y_norm))

            y_save = make_resized_grid(torch.stack(images), size=self.image_size, normalize=False)

        self.writer.add_image(tag, y_save, global_steps)

    def add_images_from_path(self, tag, paths: List[str], global_steps=None):
        '''Image sizes can be different'''
        for i, path in enumerate(paths[:self.n_save]):
            image = _load_image(path)
            self.writer.add_image(f'{tag}/{i:00d}', image/255, global_steps, dataformats='HWC')


def get_segmentated_images_tensor(tensor: torch.Tensor, dim) -> torch.Tensor:
    tensors = []
    n_class = tensor.shape[1]
    for i in range(tensor.shape[0]):
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


def _load_image(path, target_height=128):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    factor = target_height / h
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    return resized

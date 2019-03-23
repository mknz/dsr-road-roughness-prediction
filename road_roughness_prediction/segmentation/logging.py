'''Logging module'''
from typing import List

import numpy as np

from torchvision.transforms.functional import resize
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from tensorboardX import SummaryWriter

from road_roughness_prediction.segmentation.datasets import surface_types


class Logger:
    def __init__(self, log_dir, n_save, image_size, category_type):
        self.n_save = n_save
        self.image_size = image_size
        self.writer = SummaryWriter(str(log_dir))

        self.category_type = category_type
        self.is_binary = category_type == surface_types.BinaryCategory
        self.n_class = len(category_type)

        self._save_legend()

    def _save_legend(self):
        '''Save legend figure'''
        fig = create_legend_figure(self.category_type)
        self.writer.add_figure('legend', fig)

    def add_output(self, tag, out_tensor, global_steps=None):
        if self.is_binary:
            # out:  [n_batch, height, width]
            out_ = out_tensor[:self.n_save, :, :]
            save_img = make_resized_grid(out_, size=self.image_size, normalize=True)
            self.writer.add_image(tag, save_img, global_steps)
        else:
            # out:  [n_batch, n_class, height, width]
            out_ = out_tensor[:self.n_save, :, :, :]
            segmented = out_.argmax(1)
            out_rgb = expand_to_rgb(segmented)
            self.add_resized_images(tag, out_rgb, global_steps)

    def add_resized_images(self, tag, tensor, global_steps):
        for i in range(tensor.shape[0]):
            img = tensor[i, :, :, :]  # (N, C, H, W)
            resized = resize_image_tensor(img, self.image_size)
            self.writer.add_image(f'{tag}/{i:03d}', resized, global_steps, dataformats='HWC')

    def add_input(self, tag, input_tensor, global_steps=None):
        # X: [n_batch, 3, height, width]
        X_ = input_tensor[:self.n_save, :, :, :]
        self.add_resized_images(tag, normalize(X_), global_steps)

    def add_target(self, tag, target_tensor, global_steps=None):
        if self.is_binary:
            # Y: [n_batch, 1, height, width]
            # Y is 0 or 1
            Y_ = target_tensor[:self.n_save, :, :, :]
            y_save = make_resized_grid(Y_, size=self.image_size, normalize=True)
            self.writer.add_image(tag, y_save, global_steps)
        else:
            # Y: [n_batch, height, width]
            # Y is 0 to n_class - 1
            Y_ = target_tensor[:self.n_save, :, :]
            y_rgb = expand_to_rgb(Y_)
            self.add_resized_images(tag, y_rgb, global_steps)

    def add_images_from_path(self, tag, paths: List[str], global_steps=None):
        '''Image sizes can be different'''
        for i, path in enumerate(paths[:self.n_save]):
            image = np.array(Image.open(path))
            resized = resize_pil_image(image)
            self.writer.add_image(f'{tag}/{i:03d}', resized / 255, global_steps, dataformats='HWC')


def expand_to_rgb(tensor):
    '''(N, H, W) -> (N, 3, H, W)'''
    rgb = surface_types.COLOR_MAP[tensor]
    return torch.Tensor(rgb).permute(0, 3, 1, 2)


def create_legend_figure(category_type, figsize=(2, 4)):
    '''Save legend figure'''
    patches = [
        mpatches.Patch(
            color=surface_types.COLOR_MAP[category.value],
            label=category.name
        )
        for category in category_type
    ]
    fig = plt.figure(figsize=figsize)
    fig.legend(handles=patches, loc='center')
    fig.tight_layout()
    return fig


def normalize(tensor):
    '''Normalize to 0 - 1'''
    tensor_ = tensor.clone()
    tensor_ -= tensor_.min()
    tensor_ /= tensor_.max()
    return tensor_


def make_resized_grid(tensor: torch.Tensor, size, normalize=False) -> torch.Tensor:
    grid = make_grid(tensor, normalize=normalize)
    grid_pil = to_pil_image(grid.float())
    grid_pil_resized = resize(grid_pil, size)
    grid_tensor = to_tensor(grid_pil_resized)
    return grid_tensor


def resize_pil_image(img: np.array, target_height=128):
    h, w, _ = img.shape
    factor = target_height / h
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    return resized


def resize_image_tensor(img: torch.Tensor, target_height=128) -> np.array:
    # (C, H, W) -> (H, W, C)
    img_ = img.permute(1, 2, 0).numpy()
    return resize_pil_image(img_, target_height=target_height)

from itertools import chain
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from PIL import Image
from albumentations import (Blur, IAAAdditiveGaussianNoise, Resize, RandomCrop, Compose)
from albumentations.pytorch import ToTensor
import numpy as np


class SurfaceCategoryDataset(Dataset):
    def __init__(
            self,
            paths_list: List[Path],
            labels_list,
            categories: List[str],
            transform,
    ) -> None:
        self.paths = list(chain(*paths_list))
        self.labels = list(chain(*labels_list))
        self.categories = categories
        assert len(self.paths) == len(self.labels)

        self._transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path)
        
        # Convert PIL image to numpy array
        image_np = np.array(img)
        
        # Apply transformations
        augmented = self._transform(image=image_np)
        
        img_ = augmented['image']
        
        return img_, label

    @property
    def distributions(self):
        '''Returns class distribution'''
        dist = []
        for i, category in enumerate(self.categories):
            n_images = sum([1 if x == i else 0 for x in self.labels])
            dist.append((i, category, n_images, n_images / len(self)))
        return dist

    def show_dist(self):
        '''Show class distribution'''
        for i, category, n_images, dist in self.distributions:
            print(f'{i:02d} {n_images:07d} {dist:.4f} {category}')


class Rescale:
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape
        
        if isinstance(self.output_size, int):  
            if min (h, w) < self.output_size:
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = h, w
        else:
            new_w, new_h = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_ = Resize(new_h, new_w)(image=image)

        return image_

    
class RndCrop:
    """Random crop along smallest axis"""
    
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        image = RandomCrop(self.crop_size, self.crop_size)(image=image)
        return image


IMAGENET_PARAMS = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def create_surface_category_dataset(
        root: Path,
        categories: List[str],
        target_dir_name: str,
        output_size: int = 256,
):
    def _get_paths(category):
        paths = []
        for record_dir in (root / category).glob('*'):
            if not record_dir.is_dir():
                continue

            data_dir = record_dir / target_dir_name
            for path in data_dir.glob('*jpg'):
                paths.append(path)
        return paths

    paths_list, labels_list = [], []
    for i, category in enumerate(categories):
        img_paths = _get_paths(category)
        paths_list.append(img_paths)
        labels_list.append([i for _ in img_paths])
    
    transform = Compose([
        IAAAdditiveGaussianNoise (scale=(0.01*255, 0.15*255.)),
        Blur (blur_limit=4),
        Rescale(output_size),
        RndCrop(output_size),
        ToTensor(normalize = IMAGENET_PARAMS),
        ])


    return SurfaceCategoryDataset(paths_list, labels_list, categories, transform)


def create_surface_category_test_dataset(
        root: Path,
        categories: List[str],
        output_size=256,
):
    def _get_paths(category):
        paths = []
        for path in (root / category).glob('*jpg'):
            paths.append(path)
        return paths

    paths_list, labels_list = [], []
    for i, category in enumerate(categories):
        img_paths = _get_paths(category)
        paths_list.append(img_paths)
        labels_list.append([i for _ in img_paths])

    transform = Compose([
        IAAAdditiveGaussianNoise (scale=(0.01*255, 0.15*255.)),
        Blur (blur_limit=4),
        Rescale(output_size),
        RndCrop(output_size),
        ToTensor(normalize = IMAGENET_PARAMS),
        ])

    return SurfaceCategoryDataset(paths_list, labels_list, categories, transform)

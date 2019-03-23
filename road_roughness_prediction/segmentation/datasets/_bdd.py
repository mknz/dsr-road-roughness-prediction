'''Surface Segmentation Dataset'''
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from albumentations.augmentations.functional import normalize
from PIL import Image

import numpy as np

from .surface_types import BinaryCategory
from ._datasets import SidewalkSegmentationDatasetBase


class BddDataset(SidewalkSegmentationDatasetBase):
    '''BDD 100K dataset'''

    def __init__(
            self,
            image_paths,
            mask_paths,
            category_type,
            transform,
    ) -> None:
        '''
        Args:
            image_paths (List[Path])            : image paths
            mask_paths (List[Path])             : mask paths
            category_type (SurfaceCategoryBase) : Category type
            transform                           : Transformation
        '''
        assert image_paths, 'Image paths are empty'
        assert mask_paths, 'mask paths are empty'
        assert len(image_paths) == len(mask_paths), 'Number of image/mask does not match'

        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._category_type = category_type
        self._transform = transform

    @property
    def category_type(self):
        return self._category_type

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        mask_path = self._mask_paths[idx]

        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        if self.category_type == BinaryCategory:
            mask[mask == 1] = 1
            mask[mask != 1] = 0
        else:
            raise NotImplementedError

        data = {'image': image, 'mask': mask}
        augmented = self._transform(**data)
        image, mask = augmented['image'], augmented['mask']

        # Imagenet params
        image = normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        X = to_tensor(image)
        if self.category_type == BinaryCategory:
            Y = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            #Y = torch.from_numpy(mask).long()
            raise NotImplementedError

        return dict(
            X=X,
            Y=Y,
            image_path=str(image_path),
            mask_path=str(mask_path),
        )

    def get_raw_image(self, idx):
        image_path = self._image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask_path = self._mask_paths[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.uint8)
        return image, mask


class BddDatasetFactory:

    def __new__(
            cls,
            image_dirs: List[Path],
            mask_dirs: List[Path],
            category_type,
            transform,
    ):
        total_image_paths = []
        total_mask_paths = []
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            image_paths = sorted(list(image_dir.glob('*.jpg')))
            for image_path in image_paths:
                mask_path = mask_dir / (image_path.stem + '_train_id.png')
                assert mask_path.exists(), f'Not found {str(mask_path)}'
                total_image_paths.append(image_path)
                total_mask_paths.append(mask_path)

        return BddDataset(
            total_image_paths,
            total_mask_paths,
            category_type,
            transform,
        )

'''Surface Segmentation Dataset'''
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from albumentations.augmentations.functional import normalize
from PIL import Image

import numpy as np

from .surface_types import convert_mask
from .surface_types import BinaryCategory


class SidewalkSegmentationDataset(Dataset):
    '''Surface segmentation dataset'''
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

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.category_type = category_type
        self._transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        mask = convert_mask(mask, self.category_type)

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
            Y = torch.from_numpy(mask).long()

        return dict(
            X=X,
            Y=Y,
            image_path=str(image_path),
            mask_path=str(mask_path),
        )

    def get_raw_image(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.uint8)
        return image, mask


class SidewalkSegmentationDatasetFactory:
    '''Factory class for SidewalkSegmentaionDataset'''

    def __new__(
            cls,
            directories: List[Path],
            category_type,
            transform,
    ):

        image_paths = []
        mask_paths = []
        for directory in directories:
            images = sorted(list(directory.glob('JPEGImages/*.jpg')))
            masks = [p.parent.parent / 'SegmentationClassPNG' / (p.stem + '.png') for p in images]
            if not all([p.exists() for p in masks]):
                raise FileNotFoundError(f'Some mask file does not exists {str(directory)}')

            image_paths.extend(images)
            mask_paths.extend(masks)

        return SidewalkSegmentationDataset(
            image_paths,
            mask_paths,
            category_type,
            transform,
        )

'''Surface Segmentation Dataset'''
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from albumentations.augmentations.functional import normalize
from PIL import Image

import numpy as np

from .surface_types import convert_mask
from .surface_types import BinaryCategory
from road_roughness_prediction.tools.torch import imagenet_normalize


class SidewalkSegmentationDatasetBase(ABC, Dataset):
    '''Surface segmentation dataset base class'''

    @property
    @abstractmethod
    def category_type(self):
        pass

    @abstractmethod
    def get_raw_image(self, idx) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class SidewalkSegmentationDataset(SidewalkSegmentationDatasetBase):
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
        mask = convert_mask(mask, self.category_type)

        data = {'image': image, 'mask': mask}
        augmented = self._transform(**data)
        image, mask = augmented['image'], augmented['mask']

        # Normalize with imagenet params
        image = imagenet_normalize(image)

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
        image_path = self._image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask_path = self._mask_paths[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.uint8)
        return image, mask


class SidewalkSegmentationDatasetFactory:
    '''Factory class for SidewalkSegmentaionDataset'''

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
                mask_path = mask_dir / (image_path.stem + '.png')
                assert mask_path.exists(), f'Not found {str(mask_path)}'
                total_image_paths.append(image_path)
                total_mask_paths.append(mask_path)

        return SidewalkSegmentationDataset(
            total_image_paths,
            total_mask_paths,
            category_type,
            transform,
        )

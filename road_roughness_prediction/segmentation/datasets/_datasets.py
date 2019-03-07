'''Surface Segmentation Dataset'''
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from torchvision.transforms.functional import to_tensor

import numpy as np
from PIL import Image


class SidewalkSegmentationDataset(Dataset):
    '''Surface segmentation dataset'''
    def __init__(
            self,
            image_paths,
            mask_paths,
            category_type,
            transform,
            is_binary,
    ) -> None:
        '''
        Args:
            image_paths (List[Path])            : image paths
            mask_paths (List[Path])             : mask paths
            category_type (SurfaceCategoryBase) : Category type
            transform                           : Transformation
            is_binary                           :
        '''
        assert image_paths, 'Image paths are empty'
        assert mask_paths, 'mask paths are empty'
        assert len(image_paths) == len(mask_paths), 'Number of image/mask does not match'

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.category_type = category_type
        self._transform = transform
        self.is_binary = is_binary

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)

        mask = np.array(Image.open(mask_path)).astype(np.uint8)

        if self.is_binary:
            for category in self.category_type:
                if category.name == 'BACKGROUND':
                    mask[mask == category.value] = 0
                else:
                    mask[mask == category.value] = 255

        data = {'image': image, 'mask': mask}
        augmented = self._transform(**data)
        image, mask = augmented['image'], augmented['mask']

        return to_tensor(image), to_tensor(mask)

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
            is_binary
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
            is_binary,
        )

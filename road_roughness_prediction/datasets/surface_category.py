'''Surface category Dataset'''
from itertools import chain
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from PIL import Image
import numpy as np

from .surface_types import SurfaceBasicCategory


class SurfaceCategoryDataset(Dataset):
    def __init__(
            self,
            paths_list: List[Path],
            labels_list,
            category_type,
            transform,
    ) -> None:
        self.paths = list(chain(*paths_list))
        self.labels = list(chain(*labels_list))
        assert len(self.paths) == len(self.labels)

        self._transform = transform
        self.category_type = category_type

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path)

        # Convert PIL image to numpy array
        image_np = np.array(img).astype(np.uint8)

        # Apply transformations
        augmented = self._transform.full_transform(image=image_np)

        img_ = augmented['image']

        return img_, label

    @property
    def distributions(self):
        '''Returns class distribution'''
        dist = []
        for category in self.category_type:
            n_images = sum([1 if x == category.value else 0 for x in self.labels])
            dist.append((category.value, category.name, n_images, n_images / len(self)))
        return dist

    def show_dist(self):
        '''Show class distribution'''
        for category_value, category_name, n_images, dist in self.distributions:
            print(f'{category_value:02d} {n_images:07d} {dist:.4f} {category_name}')


class SurfaceCategoryDatasetFactory:

    def __new__(
            cls,
            root: Path,
            target_dir_name: str,
            dir_type: str,
            transform,
    ):
        category_type = SurfaceBasicCategory

        if dir_type == 'deep':
            def _get_paths(category_name):
                paths = []
                for record_dir in (root / category_name).glob('*'):
                    if not record_dir.is_dir():
                        continue

                    data_dir = record_dir / target_dir_name
                    for path in data_dir.glob('*jpg'):
                        paths.append(path)
                return paths

        elif dir_type == 'shallow':
            def _get_paths(category_name):
                paths = []
                for path in (root / category_name).glob('*jpg'):
                    paths.append(path)
                return paths
        else:
            raise NotImplementedError(dir_type)

        paths_list, labels_list = [], []
        for category in category_type:
            img_paths = _get_paths(category.name.lower())
            paths_list.append(img_paths)
            labels_list.append([category.value for _ in img_paths])

        assert paths_list, 'Empty paths'
        assert labels_list, 'Empty labels'

        return SurfaceCategoryDataset(paths_list, labels_list, category_type, transform)

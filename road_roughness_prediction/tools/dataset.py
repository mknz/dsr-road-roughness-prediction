from itertools import chain
from pathlib import Path
from typing import List

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


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
        img_ = self._transform(img)
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
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_w, new_h = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_ = transforms.Resize((new_h, new_w))(image)

        return image_


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

    transform = transforms.Compose([
        Rescale(512),
        transforms.RandomCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_PARAMS),
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

    transform = transforms.Compose([
        Rescale(512),
        transforms.CenterCrop(output_size),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_PARAMS),
    ])

    return SurfaceCategoryDataset(paths_list, labels_list, categories, transform)

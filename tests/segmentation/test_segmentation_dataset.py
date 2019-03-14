'''Test for segmentation dataset'''

from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from albumentations import Compose
from albumentations import CenterCrop
from albumentations import RandomCrop

from road_roughness_prediction.segmentation.datasets import SidewalkSegmentationDatasetFactory
from road_roughness_prediction.segmentation.datasets.surface_types import BinaryCategory
from road_roughness_prediction.segmentation.datasets.surface_types import SimpleCategory
import road_roughness_prediction.tools.torch as torch_tools


ROOT = Path('tests/resources/segmentation/labelme')


@pytest.mark.interactive
def test_transform():
    transform = Compose([
        CenterCrop(800, 800),
    ])
    dataset = SidewalkSegmentationDatasetFactory([ROOT], SimpleCategory, transform)

    img, mask = dataset.get_raw_image(0)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.title('Image')
    plt.imshow(img)

    plt.subplot(2, 1, 2)
    plt.title('Mask')
    plt.imshow(mask)
    plt.show()

    data = dataset[0]
    img = torch_tools.to_image(data['X'])
    mask = torch_tools.to_image(data['Y']).squeeze()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Cropped Image')

    plt.subplot(1, 2, 2)
    plt.title('Cropped Mask')
    plt.imshow(mask)
    plt.show()


def test_create_binary_mask_dataset():
    transform = Compose([
    ])
    dataset = SidewalkSegmentationDatasetFactory([ROOT], BinaryCategory, transform)

    # Only one file
    assert len(dataset) == 1

    data = dataset[0]
    mask = data['Y']

    # Binary mask
    assert set(mask.unique().tolist()) == set([0, 1])


def test_create_multiple_dataset():
    transform = Compose([])
    directories = [ROOT, ROOT]
    dataset = SidewalkSegmentationDatasetFactory(directories, SimpleCategory, transform)
    assert len(dataset) == 2

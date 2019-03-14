'''Test for segmentation dataset'''

from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from albumentations import Compose
from albumentations import CenterCrop
from albumentations import RandomCrop
from albumentations import RandomGamma
from albumentations import Rotate
from albumentations import RandomSizedCrop
from albumentations import HueSaturationValue
from albumentations import RandomBrightnessContrast
from albumentations.imgaug.transforms import IAAPerspective


from road_roughness_prediction.segmentation.datasets import SidewalkSegmentationDatasetFactory
from road_roughness_prediction.segmentation.datasets.surface_types import BinaryCategory
from road_roughness_prediction.segmentation.datasets.surface_types import SimpleCategory
import road_roughness_prediction.tools.torch as torch_tools
from road_roughness_prediction.segmentation import logging


ROOT = Path('tests/resources/segmentation/labelme')


@pytest.mark.interactive
def test_transform():
    rate = 0.9
    size = 640
    rot_deg = 3
    transform = Compose([
        #Rotate(rot_deg, p=1.),
        IAAPerspective(scale=(0.05, 0.1), p=1.0),
        RandomGamma(p=1.),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        RandomBrightnessContrast(p=1.0),
        RandomSizedCrop((int(size * rate) , int(size * rate)), size, size, p=1.),
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

    for i in range(10):
        data = dataset[0]
        img = np.array(to_pil_image(logging.normalize(data['X'])))
        mask = data['Y'].numpy()

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title('Cropped Image')

        plt.subplot(2, 1, 2)
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

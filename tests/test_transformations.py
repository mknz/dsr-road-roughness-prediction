'''Test image transformation. This is basically interactive tests'''
from pathlib import Path

import pytest
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from road_roughness_prediction.config import Config
from road_roughness_prediction.datasets.transformations import TransformFactory


def _get_images(paths):
    return list(map(np.array, map(Image.open, paths)))


class TestTransformations:
    config = Config()
    root = Path('tests/resources/surfaces/')
    paths = list(root.glob('**/*.jpg'))
    images = _get_images(paths)
    transforms = [TransformFactory(config)]
    image_transforms = [tr.image_transform for tr in transforms]

    def _plot_transformation(self, transform, images, paths):
        for image, path in zip(images, paths):
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(image)

            plt.subplot(1, 2, 2)
            plt.title('Transformed')
            image_ = transform(image=image)['image']
            plt.imshow(image_)
            plt.show()

    @pytest.mark.interactive
    def test_cli_image(self, image_path):
        '''Test image given from command line'''
        paths = [Path(image_path)]
        images = _get_images(paths)
        for transform in self.image_transforms:
            self._plot_transformation(transform, images, paths)

    @pytest.mark.interactive
    def test_transformations(self):
        for transform in self.image_transforms:
            self._plot_transformation(transform, self.images, self.paths)

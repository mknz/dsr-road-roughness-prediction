'''Transformations'''
from albumentations import Blur
from albumentations import Compose
from albumentations import IAAAdditiveGaussianNoise
from albumentations import RandomCrop
from albumentations.pytorch import ToTensor

from ..config import Config
from .rescale import Rescale


class Transform:
    '''Wrapper class of transform.

    Attributes:
        image_transform: Transformation without normalization. For display purpose.
        full_transform : Transformation with ToTensor. This creates input tensor to NN.
    '''

    def __init__(self, image_transform, config):
        self.image_transform = image_transform
        self.full_transform = Compose([
            self.image_transform,
            ToTensor(normalize=config.NORMALIZE_PARAMS),
        ])


class TransformFactory:

    def __new__(cls, config: Config):
        if config.TRANSFORMATION == 'BASIC_TRANSFORM':

            image_transform = Compose([
                IAAAdditiveGaussianNoise(scale=config.GAUSSIAN_NOISE_SCALE),
                Blur(blur_limit=config.BLUR_LIMIT),
                Rescale(config.OUTPUT_SIZE),
                RandomCrop(config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            ])

            transform = Transform(image_transform, config)
        else:
            NotImplementedError(config.TRANSFORMATION)

        return transform

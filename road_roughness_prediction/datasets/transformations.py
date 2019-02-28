'''Transformations'''
from enum import auto
from enum import Enum

from albumentations import Blur
from albumentations import Compose
from albumentations import IAAAdditiveGaussianNoise
from albumentations import CenterCrop
from albumentations import HorizontalFlip
from albumentations import RandomBrightness
from albumentations import RandomContrast
from albumentations import RandomCrop
from albumentations import RandomScale
from albumentations import Rotate
from albumentations.pytorch import ToTensor

from .rescale import Rescale


class TransformType(Enum):
    '''Transformation types'''

    BASIC_TRANSFORM = auto()
    BASIC_EVAL_TRANSFORM = auto()
    EXTENSIVE_TRANSFORM = auto()

    @classmethod
    def from_string(cls, transform_type_string):
        if transform_type_string == 'basic_transform':
            type_ = cls.BASIC_TRANSFORM
        elif transform_type_string == 'basic_eval_transform':
            type_ = cls.BASIC_EVAL_TRANSFORM
        elif transform_type_string == 'exetensive_transform':
            type_ = cls.EXTENSIVE_TRANSFORM
        else:
            raise ValueError(transform_type_string)

        return type_


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

    def __new__(cls, config):
        if config.TRANSFORMATION == TransformType.BASIC_TRANSFORM:

            image_transform = Compose([
                IAAAdditiveGaussianNoise(scale=config.GAUSSIAN_NOISE_SCALE),
                Blur(blur_limit=config.BLUR_LIMIT),
                Rescale(config.OUTPUT_SIZE),
                RandomCrop(config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            ])

        elif config.TRANSFORMATION == TransformType.EXTENSIVE_TRANSFORM:

            image_transform = Compose([
                HorizontalFlip(),
                Rotate(**config.ROTATE),
                RandomBrightness(),
                RandomContrast(),
                IAAAdditiveGaussianNoise(scale=config.GAUSSIAN_NOISE_SCALE),
                Blur(blur_limit=config.BLUR_LIMIT),
                Rescale(config.OUTPUT_SIZE),
                RandomScale(**config.RANDOM_SCALE),
                RandomCrop(config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            ])

        elif config.TRANSFORMATION == TransformType.BASIC_EVAL_TRANSFORM:
            image_transform = Compose([
                Rescale(config.OUTPUT_SIZE),
                CenterCrop(config.OUTPUT_SIZE, config.OUTPUT_SIZE),
            ])

        else:
            raise NotImplementedError(config.TRANSFORMATION)

        return Transform(image_transform, config)

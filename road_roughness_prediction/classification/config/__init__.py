'''Configs'''
from ..datasets.transformations import TransformType


class DefaultConfig:
    '''Config class'''

    OUTPUT_SIZE = 256
    GAUSSIAN_NOISE_SCALE = (0.01 * 255, 0.15 * 255.)
    ROTATE = dict(limit=10, p=0.3)
    RANDOM_SCALE = dict(
        scale_limit=(1.0, 1.5),
        p=0.2,
    )
    BLUR_LIMIT = 4
    IMAGENET_PARAMS = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    NORMALIZE_PARAMS = IMAGENET_PARAMS

    TRANSFORMATION = TransformType.BASIC_TRANSFORM


class Config(DefaultConfig):

    def from_dict(self, config_dict):
        for item in config_dict:
            if hasattr(self, item):
                setattr(self, item, config_dict[item])
            else:
                raise AttributeError(item)

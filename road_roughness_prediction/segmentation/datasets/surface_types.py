'''Surface category type definition'''
from enum import Enum
from enum import unique

import numpy as np

from matplotlib import cm


class SurfaceCategoryBase(Enum):
    '''Surface category base class'''

    @classmethod
    def get_class_names(cls):
        return [x.name for x in cls]

    @classmethod
    def get_class_names_lower(cls):
        return [x.name.lower() for x in cls]


@unique
class BinaryCategory(SurfaceCategoryBase):
    BACKGROUND = 0
    SIDEWALK = 1


@unique
class SimpleRawCategory(SurfaceCategoryBase):
    '''Surface category segmentation label index.
    NOTE: Do not use SIDEWALK label defined here.'''
    BACKGROUND = 0
    SIDEWALK = 1
    ASPHALT = 2
    GRASS = 3
    SOIL = 4
    FLAT_STONES = 5
    PAVING_STONES = 6
    SETT = 7
    BICYCLE_TILES = 8


@unique
class SimpleCategory(SurfaceCategoryBase):
    BACKGROUND = 0
    ASPHALT = 1
    GRASS = 2
    SOIL = 3
    FLAT_STONES = 4
    PAVING_STONES = 5
    SETT = 6
    BICYCLE_TILES = 7


COLORMAP = cm.jet(np.linspace(0, 255, 8).astype(np.uint8))[:, :3]


def convert_mask(mask, category_type):
    '''Convert image category from SimpleRawCategory to target category'''
    img = mask.copy()
    if category_type == BinaryCategory:
        for category in SimpleRawCategory:
            if category == SimpleRawCategory.BACKGROUND:
                img[img == category.value] = 0
            else:
                img[img == category.value] = 1

    elif category_type == SimpleCategory:
        for category in SimpleRawCategory:
            if category in [SimpleRawCategory.BACKGROUND, SimpleRawCategory.SIDEWALK]:
                img[img == category.value] = 0
            elif category == SimpleRawCategory.ASPHALT:
                img[img == category.value] = 1
            elif category == SimpleRawCategory.GRASS:
                img[img == category.value] = 2
            elif category == SimpleRawCategory.SOIL:
                img[img == category.value] = 3
            elif category == SimpleRawCategory.FLAT_STONES:
                img[img == category.value] = 4
            elif category == SimpleRawCategory.PAVING_STONES:
                img[img == category.value] = 5
            elif category == SimpleRawCategory.SETT:
                img[img == category.value] = 6
            elif category == SimpleRawCategory.BICYCLE_TILES:
                img[img == category.value] = 7
    else:
        raise NotImplementedError
    return img

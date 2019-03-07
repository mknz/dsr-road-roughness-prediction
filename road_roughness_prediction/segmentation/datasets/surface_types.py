'''Surface category type definition'''
from enum import Enum


class SurfaceCategoryBase(Enum):
    '''Surface category base class'''

    @classmethod
    def get_class_names(cls):
        return [x.name for x in cls]

    @classmethod
    def get_class_names_lower(cls):
        return [x.name.lower() for x in cls]


class SimpleCategory(SurfaceCategoryBase):
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

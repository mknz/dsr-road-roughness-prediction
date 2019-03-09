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


class SurfaceSimpleCategory(SurfaceCategoryBase):
    '''Surface category, first definition'''
    ASPHALT = 0
    GRASS = 1
    SOIL = 2
    FLAT_STONES = 3
    PAVING_STONES = 4
    SETT = 5
    BICYCLE_TILES = 6


class SurfaceBasicCategory(SurfaceCategoryBase):
    '''Surface category, first definition'''
    ASPHALT = 0
    COMPACTED = 1
    FLAT_STONES = 2
    GRASS = 3
    PAVING_STONES_SMOOTH = 4
    SETT_COARSE = 5
    SETT_FINE = 6
    SETT_REGULAR = 7

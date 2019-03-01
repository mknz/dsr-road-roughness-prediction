'''Surface category type definition'''
from enum import Enum


class SurfaceBasicCategory(Enum):
    '''Surface category, first definition'''
    ASPHALT = 0
    COMPACTED = 1
    FLAT_STONES = 2
    GRASS = 3
    PAVING_STONES_SMOOTH = 4
    SETT_COARSE = 5
    SETT_FINE = 6
    SETT_REGULAR = 7

    @classmethod
    def get_class_names(cls):
        return [x.name for x in cls]

    @classmethod
    def get_class_names_lower(cls):
        return [x.name.lower() for x in cls]

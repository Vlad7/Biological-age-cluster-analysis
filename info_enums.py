from enum import Enum

class Provider(Enum):
    GERONTOLOGY = 1
    KNHANES = 2
    NHANES = 3


class NHANES(Enum):
    NHANES3 = 1
    NHANES3_HDTrain = 2
    NHANES4 = 3


class GERONTOLOGY(Enum):
    NEW = 1
    OLD = 2


class DatasetType(Enum):
    """Type of dataset"""
    Antropometry = 1
    Biochemistry = 2
    Bones = 3
    Gemogramma = 4


class LabelStatus(Enum):
    before = 1
    after = 2


class Sex(Enum):
    Male = 1
    Female = 2
    Both_sexes = 3
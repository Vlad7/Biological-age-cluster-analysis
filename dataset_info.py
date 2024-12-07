from enum import Enum

class Provider(Enum):
    """
        Dataset provider - name from who dataset
    """
    GERONTOLOGY = 1
    KNHANES = 2
    NHANES = 3


class NHANES(Enum):
    """
        NHANES - version of NHANES dataset
    """
    NHANES3 = 1
    NHANES3_HDTrain = 2
    NHANES4 = 3


class GERONTOLOGY(Enum):
    """
        GERONTOLOGY - version of Institute of gerontology dataset
    """
    NEW = 1
    OLD = 2


class DatasetType(Enum):
    """
        Type of data in dataset
    """
    Antropometry = 1
    Biochemistry = 2
    Bones = 3
    Gemogramma = 4



class Sex(Enum):
    """
        Dataset sex of humans on database sheet
    """
    Male = 1
    Female = 2
    Both_sexes = 3
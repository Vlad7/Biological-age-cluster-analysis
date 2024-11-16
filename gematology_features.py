from enum import Enum


features_all=[
    'MCH',
    'MCHC',
    'MCV',
    'MPV',
    'PDW',
    'RDW',
    'Hematocrit',
    'Hemoglobin',
    'Granulocytes',
    'Red blood cells',
    'Leukocytes',
    'Lymphocytes',
    'Monocyte',
    'Thrombocrit',
    'Thrombocytes',
    'ESR']


# Pisaruk, that correlates most with age near 0.15

features_hight_correlation_with_age = ['RDW',
    'Hematocrit',
    'Hemoglobin',
    'Thrombocytes',
    'ESR']



class Feature(Enum):
    MCH = 1
    MCHC = 2
    MCV = 3
    MPV = 4
    PDW = 5
    RDW = 6
    Hematocrit = 7
    Hemoglobin = 8
    Granulocytes = 9
    Red_blood_cells = 10
    Leukocytes = 11
    Lymphocytes = 12
    Monocyte = 13
    Thrombocrit = 14
    Thrombocytes = 15
    ESR = 16
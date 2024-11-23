from enum import Enum


features_gematology_all=[
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

features_bones_all=['BMI',
                    'FRAX-all',
                    'FRAX-hip',
                    'Lumbar spine',
                    'Right femoral neck',
                    'Left femoral neck',
                    'Right proximal part of the femor',
                    'Left proximal part of the femor',
                    'Radius',
                    'TBS'
]

features_biochemistry_all=['Alkaline phosphatase',
                           'AlT',
                           'AsT',
                           'Cholesterol',
                           'Cholesterol-HDL',
                           'Cholesterol-LDL',
                           'Cholesterol-VLDL',
                           'Creatinine',
                           'Glucose in plasma basal',
                           'Glucose in plasma after 2 hours SGTT',
                           'HOMA index',
                           'Insuline basal',
                           'Triglycerides',
                           'Urea'
]


# Pisaruk, that correlates most with age near 0.15

features_gematology_hight_correlation_with_age = ['RDW',
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

#features1 = [for e in Feature]
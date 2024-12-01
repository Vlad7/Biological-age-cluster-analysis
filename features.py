from enum import Enum

# GERONTOLOGY DATASET BIOMARKERS

gerontology_gematology_all=[
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

gerontology_bones_all=['BMI',
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

gerontology_biochemistry_all=['Alkaline phosphatase',
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

gerontology_features_gematology_hight_correlation_with_age = ['RDW',
    'Hematocrit',
    'Hemoglobin',
    'Thrombocytes',
    'ESR']

# NHANES DATASET FEATURES

features_NHANES3_HDTrain_all = [
    'sampleID',
    'year',
    'wave',
    'gender',
    'age',
    'annual_income',
    'income_recode',
    'education',
    'edu',
    'ethnicity',
    'race',
    'poverty_ratio',
    'pregnant',
    'adl',
    'bmi',
    'height',
    'waist',
    'weight',
    'btotpf',
    'ffm',
    'fm',
    'fev',
    'fev_1000',
    'grip_r',
    'grip_l',
    'grip_d',
    'grip_scaled',
    'health',
    'lnwalk',
    'vomeas',
    'albumin',
    'albumin_gL',
    'alp',
    'lnalp',
    'bun',
    'lnbun',
    'creat',
    'creat_umol',
    'lncreat',
    'lncreat_umol',
    'glucose',
    'glucose_mmol',
    'ttbl',
    'uap',
    'lnuap',
    'bap',
    'basopa',
    'eosnpa',
    'lymph',
    'mcv',
    'monopa',
    'neut',
    'rbc',
    'rdw',
    'wbc',
    'cadmium',
    'crp',
    'crp_cat',
    'lncrp',
    'cyst',
    'dbp',
    'sbp',
    'meanbp',
    'pulse',
    'ggt',
    'glucose_fasting',
    'insulin',
    'phpfast',
    'hba1c',
    'lnhba1c',
    'hdl',
    'ldl',
    'trig',
    'totchol',
    'vitaminA',
    'vitaminE',
    'vitaminB12',
    'vitaminC',
    'eligstat',
    'status',
    'ucod_leading',
    'diabetes',
    'hyperten',
    'permth_int',
    'time',
    'kdm0',
    'kdm_advance0',
    'phenoage0',
    'phenoage_advance0'
]

features_NHANES3_HDTrain_biochemistry_all=[
    'albumin',
    'albumin_gL',
    'alp',
    'bun',
    'creat',
    'creat_umol',
    'glucose',
    'glucose_mmol',
    'ttbl',
    'glucose_fasting',
    'insulin',
    'phpfast',
    'hdl',
    'ldl',
    'trig',
    'totchol'
]

NHANES3_HDTrain_biochemistry_selected=[
    'alp',
    'bun',
    'creat_umol',
    'glucose_mmol',
    'phpfast',
    'hdl',
    'ldl',
    'trig',
    'totchol'
]

features_NHANES3_HDTrain_antropometry=[
    'bmi',
    'height',
    'waist',
    'weight',
]



features_NHANES3_HDTrain_gematology=[
    'lymph',
    'mcv',
    'monopa',
    'rbc',
    'rdw',
    'wbc',
]

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

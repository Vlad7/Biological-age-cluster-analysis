import numpy as np
import pandas as pd
from openpyxl import load_workbook
import features as ft
import sys

from enum import Enum

class Dataset(Enum):
    GERONTOLOGY = 1
    KNHANES = 2
    NHANES = 3

datasets = {GERONTOLOGY: 'gematology',}
class DatasetNHANES(Enum):
    NHANES3 = 1
    NHANES3_HDTrain = 2
    NHANES4 = 3

class DatasetType(Enum):
    """Type of dataset"""
    Antropometry = 1
    Biochemistry = 2
    Bones = 3
    Gematology = 4

class Sex (Enum):
    """Sex of persons in dataset"""
    Both = 1
    Female = 2
    Male = 3

def load_dataset(dataset_provider, dataset_name_input, dataset_type, sex):



    dataset_attributes = None

    if dataset == DatasetNHANES.NHANES3_HDTrain:
        dataset_attributes = []
        dataset_attributes.extend(ft.features_NHANES3_HDTrain_all)

        print("All features: " + str(ft.features_NHANES3_HDTrain_all))

    sheet_name = sex.name
    if sex == Sex.Both and dataset == DatasetNHANES.NHANES3_HDTrain:
        sheet_name = "Worksheet"

    else:
        return 0

    try:
        dataframe = pd.read_excel('../datasets/'+ provider + '/Excel/' + dataset.name + '.xlsx',
                                  sheet_name=sheet_name,
                                  names=dataset_attributes)
        print('Data was imported!')

        return dataframe

    except FileNotFoundError:
        print('File was not found!')
        sys.exit(0)

def select_biomarkers(dataframe, features_set):

    ft.features_NHANES3_HDTrain_biochemistry
    #Age not age for gerontology
    selected_biomarkers = ['age']
    selected_biomarkers.extend(features_set)
    dataframe = dataframe.loc[:, selected_biomarkers]
    return dataframe

def separator(dataset_provider, dataset_version, feature_set):

    folder=""
    filename=""
    feature_set=None
    selected_feature_set=None


    dataset_name = ""

    if dataset_provider == Dataset.GERONTOLOGY:
        folder = Dataset.GERONTOLOGY.name

        filename = feature_set.name

        if feature_set == DatasetType.Biochemistry:
            feature_set = ft.gerontology_features_biochemistry_all
        elif feature_set == DatasetType.Bones:
            feature_set = ft.gerontology_features_bones_all
        elif feature_set == DatasetType.Gematology:
            feature_set = ft.gerontology_features_gematology_all

        selected_feature_set = feature_set
    elif dataset_provider == Dataset.KNHANES:
        folder = Dataset.KNHANES.name

        filename = dataset_version.name
    elif dataset_provider == Dataset.NHANES:
        folder = Dataset.NHANES.name

        filename = dataset_version.name


dataframe = load_dataset(Dataset.NHANES, DatasetNHANES.NHANES3_HDTrain)
selected_biomarkers_dataset = select_biomarkers(dataframe, )
selected_by_ages = select_ages(selected_biomarkers_dataset, 20, 30)
filled_by_polinomial = fill_empty_by_polynomial(selected_by_ages)
print_dataset(filled_by_polinomial)

# Пишем DataFrame на существующий лист
dataframe.to_excel("../datasets/"+Dataset.NHANES.name+"/Excel/Filled empty/"+ "NHANES3_HDTrain_biochemical_filled_empty_by_polynomial_method.xlsx",
                    sheet_name='Worksheet', index=False)

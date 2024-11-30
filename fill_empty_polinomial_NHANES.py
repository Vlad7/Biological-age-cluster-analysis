import numpy as np
import pandas as pd
from openpyxl import load_workbook
import features as ft
import sys

from enum import Enum

import pca_lib as pl

class Dataset(Enum):
    GERONTOLOGY = 1
    KNHANES = 2
    NHANES = 3

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

class LabelStatus(Enum):
    before = 1
    after = 2

def load_dataset(dataset_provider, dataset_name, attributes, sex):

    dataset_attributes = []

    dataset_attributes.extend(attributes)
    print("All features: " + str(attributes))

    try:
        dataframe = pd.read_excel('../datasets/'+ dataset_provider +'/'+dataset_name + '/Excel/' + dataset_name + ' (biochemistry).xlsx',
                                  sheet_name=sex,
                                  names=dataset_attributes)
        print('Data was imported!')

        return dataframe

    except FileNotFoundError:
        print('File was not found!')
        sys.exit(0)

def select_biomarkers(dataframe, features_set):

    #Age not age for gerontology
    selected_biomarkers = ['age']
    selected_biomarkers.extend(features_set)
    dataframe = dataframe.loc[:, selected_biomarkers]
    return dataframe

######################################################################################################################
#################################### FILL EMPTY CORNERS WITH POLYNOMIAL INTERPOLATION ################################
######################################################################################################################

def print_title(dataframe, sheet_name='MALE', status="before"):

    print('------------INFO DATAFRAME '+sheet_name.upper(), end="")
    if status=="after":
        print('('+status+')', end="")

    print('------------')

    print('SIZE: ', dataframe.size)
    print('ROWS: ', dataframe.shape[0])
    print('ISNULL: ', dataframe.isnull().sum().sum())


def select_ages(dataframe, age_lower=20, age_upper=30):

    print_title(dataframe, "MALE", LabelStatus.before.name)

    dataframe = dataframe.dropna(subset=['age'])

    dataframe = dataframe.drop(dataframe[dataframe['age'] < age_lower].index)
    dataframe = dataframe.drop(dataframe[dataframe['age'] > age_upper].index)

    dataframe = dataframe.dropna(thresh=8)

    dataframe = dataframe.sort_values(by='age')

    return dataframe
    

def fill_empty_by_polynomial(dataframe):


    dataframe.interpolate(method='polynomial', order=2)       # !!!!!!!!!!!!!!
    dataframe = dataframe.ffill()
    dataframe = dataframe.bfill()
    dataframe = dataframe.sort_index()
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe = dataframe.reset_index(drop=True)

    dataframe = dataframe.sort_values(by='age')                 # !!!!!!!!!!!!!!!

    print_title(dataframe, "MALE", LabelStatus.after.name)

    return dataframe

def print_dataset(dataframe):
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    print(dataframe)


dataframe = load_dataset(Dataset.NHANES.name, DatasetNHANES.NHANES3_HDTrain.name, , 'Male')
selected_biochemical_dataset = select_biomarkers(dataframe, ft.features_NHANES3_HDTrain_biochemistry_selected)
selected_by_ages = select_ages(selected_biochemical_dataset, 20, 30)
filled_by_polinomial = fill_empty_by_polynomial(selected_by_ages)

print_dataset(filled_by_polinomial)

# Пишем DataFrame на существующий лист
dataframe.to_excel("../datasets/"+Dataset.NHANES.name+'/'+DatasetNHANES.NHANES3_HDTrain.name+"/Excel/Filled empty/"+ "NHANES3_HDTrain_biochemical_filled_empty_by_polynomial_method_male.xlsx",
                    sheet_name='Male', index=False)



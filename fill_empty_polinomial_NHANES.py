import numpy as np
import pandas as pd
from openpyxl import load_workbook
import features as ft
import sys

from enum import Enum

import pca_lib as pl

class Dataset(Enum):
    IGerontology = 1
    NHANES_HDTrain = 2
    KNHANES = 3

class DatasetType(Enum):
    """Type of dataset"""
    Biochemistry = 1
    Bones = 2
    Gematology = 3
    Antropometry = 4


dataset_type = None


dataset_attributes = []

dataset_attributes.extend((ft.features_NHANES3_HDTrain_all))
print("All features: " + str(ft.features_NHANES3_HDTrain_all))


def load_dataset(folder, dataset):
    try:
        dataframe = pd.read_excel('../datasets/'+ folder + dataset,
                                  sheet_name='Worksheet',
                                  names=dataset_attributes)

        print('Data was imported!')

        return dataframe

    except FileNotFoundError:
        print('File was not found!')
        sys.exit(0)

def select_biochemical_biomarkers(dataframe):
    selected_biomarkers = ['age']
    selected_biomarkers.extend(ft.features_NHANES3_HDTrain_biochemistry)
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

    print_title(dataframe, "MALE", "before")

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

    print_title(dataframe, "MALE", "after")

    return dataframe

def print_dataset(dataframe):
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    print(dataframe)


folder = None
filename = None

dataset = Dataset.NHANES_HDTrain
dataset_type = DatasetType.Biochemistry


if dataset == Dataset.NHANES_HDTrain:
    folder = 'From internet/Excel/'
    filename = 'NHANES3_HDTrain.xlsx'

def create_database(folder, filename):
    dataframe = load_dataset(folder, dataset)
    selected_biomarkers_dataset = select_biochemical_biomarkers(dataframe)
    selected_by_ages = select_ages(selected_biomarkers_dataset, 20, 30)
    filled_by_polinomial = fill_empty_by_polynomial(selected_by_ages)
    print_dataset((filled_by_polinomial))




# Загрузка существующего Excel-файла
#path = "datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx"
#book = load_workbook(path)

dataset_type = 'biochemical'
# Пишем DataFrame на существующий лист
dataframe.to_excel("../datasets/From internet/Excel/Filled empty/"+dataset_type+"NHANES_filled_empty_by_polynomial_method_3.xlsx",
                    sheet_name='Worksheet', index=False)


dataframe = pd.read_excel('../datasets/From internet/Excel/NHANES3_HDTrain.xlsx',
                                  sheet_name='Worksheet',
                                  names=dataset_attributes)
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import features as ft
import sys
import info_enums as ie
import features_determinator as fd
from enum import Enum

import pca_lib as pl



class LabelStatus(Enum):
    before = 1
    after = 2



def load_dataset(dataset_provider, dataset_name, dataset_type, attributes, sex):

    dataset_attributes = ['age']
    dataset_attributes.extend(attributes)
    print("All features: " + str(attributes))

    try:
        dataframe = pd.read_excel('../datasets/'+ dataset_provider +'/'+dataset_name + '/Excel/' + dataset_type + '.xlsx',
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

def print_title(dataframe, sheet_name="", status="before"):

    print('------------INFO DATAFRAME '+sheet_name.upper(), end=" ")

    if status=="after":
        print('('+status+')', end="")

    print('------------')

    print('SIZE: ', dataframe.size)
    print('ROWS: ', dataframe.shape[0])
    print('ISNULL: ', dataframe.isnull().sum().sum())


def select_ages(dataframe, sheet_name="", age_lower=20, age_upper=30):


    dataframe = dataframe.dropna(subset=['age'])

    print_title(dataframe, sheet_name, LabelStatus.before.name)

    dataframe = dataframe.drop(dataframe[dataframe['age'] < age_lower].index)
    dataframe = dataframe.drop(dataframe[dataframe['age'] > age_upper].index)

    dataframe = dataframe.dropna(thresh=8)   # используется в pandas для удаления строк из DataFrame, которые имеют меньше заполненных  (не NaN) значений, чем указанное пороговое значение (thresh=8).

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


    return dataframe

def print_dataset(dataframe):
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    print(dataframe)

def create_filled_polynomial(provider, version, type, sex, age_lower, age_upper):
    features_all, features_selected = fd.determine_features_all_and_features_selected(version, type)
    dataframe = load_dataset(provider.name, version.name, type.name.lower(), features_all, sex.name)
    selected_biochemical_dataset = select_biomarkers(dataframe, features_selected)
    selected_by_ages = select_ages(selected_biochemical_dataset, sex.name, age_lower, age_upper)
    filled_by_polinomial = fill_empty_by_polynomial(selected_by_ages)
    print_title(dataframe, sex.name, LabelStatus.after.name)
    print_dataset(filled_by_polinomial)

    # Пишем DataFrame на существующий лист
    filled_by_polinomial.to_excel("../datasets/"+provider.name+'/'+version.name+"/Excel/Filled empty/"+ type.name.lower() + "_filled_empty_by_polynomial_method_"+sex.name.lower()+".xlsx",
                    sheet_name=sex.name, index=False)

"""
provider = ie.Provider.NHANES           #May be GERONTOLOGY, KNHANES or NHANES
version = ie.NHANES.NHANES3_HDTrain     # 'None' for institute of gerontology, except for old or new biochemical database
type = ie.DatasetType.Biochemistry
sex = ie.Sex.Male
age_lower = 20
age_upper = 30
"""

provider = ie.Provider.GERONTOLOGY
version = ie.GERONTOLOGY.NEW
type = ie.DatasetType.Gemogramma
sex = ie.Sex.Male
age_lower = 23
age_upper = 90

create_filled_polynomial(provider, version, type, sex, age_lower, age_upper)





import numpy as np
import pandas as pd
from openpyxl import load_workbook
import features as ft
import sys
import dataset_info as di
import features_determinator as fd
from enum import Enum
import os

import pca_lib as pl

class TitleStatus(Enum):
    before = 1
    after = 2


def load_dataset(dataset_provider, dataset_version, dataset_type, attributes, sex):
    """
        Load dataset function

        input:
            dataset_provider - see dataset_info.py Provider enum
            dataset_version  = see dataset_info.py NHANES or GERONTOLOGY
            dataset_type     - see dataset_info.py DatasetType enum
            attributes       - see features.py - biomarkers of ageing for every dataset
            sex              - see dataset_info.py Sex enum
    """

    dataset_attributes = ['age']
    dataset_attributes.extend(attributes)

    print("All features: " + str(attributes))

    path = '../datasets/'+ dataset_provider +'/'+dataset_version + '/Excel/' + dataset_type + '.xlsx'

    try:
        dataframe = pd.read_excel(path,
                                  sheet_name=sex,
                                  usecols=dataset_attributes,
                                  header=0)
        print('Data was imported!')
        return dataframe

    except FileNotFoundError:
        print(f'File not found at path: {path}')
        sys.exit(1)

    except ValueError as e:
        print(f'ValueError: {e}. Check the sheet name or column names.')
        sys.exit(1)

"""
def get_age_attribute_case(dataset_provider):
    
    # Function for determining the case of age (uppercase or lowercase) in dataset
    
    if dataset_provider == di.Provider.GERONTOLOGY:
        return 'Age'

    elif dataset_provider == di.Provider.NHANES:
        return 'age'

    elif dataset_provider == di.Provider.KNHANES:
        return 'age'

    else:
        raise ValueError(dataset_provider)
"""
"""
def select_biomarkers(dataframe, features_set):
    
    # Select biomarkers method for selecting from dataframe subset of biomarkers
    
    #Age not age for gerontology
    selected_biomarkers = ['age']
    selected_biomarkers.extend(features_set)
    dataframe = dataframe.loc[:, selected_biomarkers]
    return dataframe
"""

def print_title(dataframe, sheet_name="", status=""):
    """
        Print title and information about dataframe
    """
    print('------------INFO DATAFRAME '+sheet_name.upper(), end=" ")

    if status=="after":
        print('('+status+')', end="")

    print('------------')

    print('SIZE: ', dataframe.size)                   # Size rows * columns
    print('ROWS: ', dataframe.shape[0])               # Rows number
    print('ISNULL: ', dataframe.isnull().sum().sum()) # Number of null elements


def select_ages(dataframe, sheet_name="", age_lower=None, age_upper=None):

    """
        Remove ages with empty values
        Remove rows with ages lover selected and upper selected
        Sort dataset by age
    """

    dataframe = dataframe.dropna(subset=['age']) # Remove rows, where age is missing value corner

    print_title(dataframe, sheet_name, TitleStatus.before.name) # Print title and information about dataframe

    dataframe = dataframe.drop(dataframe[dataframe['age'] < age_lower].index) # Remove lower selected age
    dataframe = dataframe.drop(dataframe[dataframe['age'] > age_upper].index) # Remove upper selected age


    dataframe = dataframe.dropna(thresh=8)  #  Используется в pandas для удаления строк из DataFrame,
                                            #  которые имеют меньше заполненных  (не NaN) значений,
                                            #  чем указанное пороговое значение (thresh=8).

    dataframe = dataframe.sort_values(by='age')

    return dataframe
    


def fill_empty(dataframe, method='polynomial'):
    """Fill empty corners with polynomial interpolation"""

    if method == 'linear':

        # print(dataframe[dataframe.columns].duplicated())
        # Check for duplicate indices
        # print(dataframe.index.duplicated())

        # Сохраняем первый столбец отдельно
        first_column = dataframe.iloc[:, 0]
        # Применяем интерполяцию ко всем остальным столбцам
        dataframe.iloc[:, 1:] = dataframe.iloc[:, 1:].interpolate(method='linear')

        # Объединяем первый столбец обратно
        dataframe.iloc[:, 0] = first_column

        # Если необходимо заполнить пропуски в начале или конце
        # df_male.iloc[:, 1:] = df_male.iloc[:, 1:].ffill().bfill()
        dataframe = dataframe.ffill()
        dataframe = dataframe.bfill()

    elif method == 'polynomial':
        dataframe.interpolate(method=method, order=2)  # !!!!!!!!!!!!!!

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

def create_filled(provider, version, type, sex, age_lower, age_upper):
    features_all, features_selected = fd.determine_features_all_and_features_selected(version, type)
    dataframe = load_dataset(provider.name, version.name, type.name.lower(), features_selected, sex.name)
    #selected_biochemical_dataset = select_biomarkers(dataframe, features_selected)
    selected_by_ages = select_ages(dataframe, sex.name, age_lower, age_upper)

    filled_empty = None
    filled_type = ""
    if provider == di.Provider.GERONTOLOGY and version == di.GERONTOLOGY.NEW and type==di.DatasetType.Biochemistry:
        filled_empty = fill_empty(selected_by_ages, 'linear')
        filled_type = 'linear'
    else:
        filled_empty = fill_empty(selected_by_ages, 'polynomial')
        filled_type = 'polynomial'
    print_title(dataframe, sex.name, TitleStatus.after.name)
    print_dataset(filled_empty)

    # Путь к файлу
    file_path = f"../datasets/{provider.name}/{version.name}/Excel/Filled empty/{type.name.lower()}_filled_empty_by_{filled_type}_method.xlsx"

    if os.path.exists(file_path):
        print("Файл существует.")


        # Читаем файл, чтобы добавить новый лист
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            filled_empty.to_excel(writer, sheet_name=sex.name, index=False)


    else:
        print("Файл не существует.")

        # Пишем DataFrame на существующий лист
        filled_empty.to_excel(file_path, sheet_name=sex.name, index=False)

    """
    # Пишем DataFrame на существующий лист
    with pd.ExcelWriter("datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    writer.book = df_male
    df_male.to_excel(writer, sheet_name='Male', index=False)"""

    """# Загрузка существующего Excel-файла
        path = "datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx"
        book = load_workbook(path)

        df_male.to_excel("datasets/biochemistry_filled_empty_by_polynomial_method_3.xlsx",
             sheet_name='Biochemistry', index=False)"""






def test_1():

    provider = di.Provider.NHANES           #May be GERONTOLOGY, KNHANES or NHANES
    version = di.NHANES.NHANES3_HDTrain     # for institute of gerontology, except for old or new biochemical database
    type = di.DatasetType.Biochemistry
    sex = di.Sex.Male
    age_lower = 20
    age_upper = 30

    create_filled(provider, version, type, sex, age_lower, age_upper)

def test_2():

    provider = di.Provider.GERONTOLOGY
    version = di.GERONTOLOGY.NEW
    type = di.DatasetType.Gemogramma
    sex = di.Sex.Male
    age_lower = 23
    age_upper = 90

    create_filled(provider, version, type, sex, age_lower, age_upper)

def test_3():

    provider = di.Provider.GERONTOLOGY
    version = di.GERONTOLOGY.NEW
    type = di.DatasetType.Gemogramma
    sex = di.Sex.Female
    age_lower = 18
    age_upper = 86

    create_filled(provider, version, type, sex, age_lower, age_upper)

def test_4():

    provider = di.Provider.GERONTOLOGY
    version = di.GERONTOLOGY.NEW
    type = di.DatasetType.Biochemistry
    sex = di.Sex.Both_sexes
    age_lower = 20
    age_upper = 79

    create_filled(provider, version, type, sex, age_lower, age_upper)


test_1()


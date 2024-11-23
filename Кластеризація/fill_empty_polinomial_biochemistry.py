import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import features as ft

dataset_attributes = ['Age']

dataset_attributes.extend((ft.features_biochemistry_all))
print("All features: " + str(ft.features_biochemistry_all))



df_male = pd.read_excel('datasets/biochemistry_sorted.xlsx',
                          sheet_name='Biochemistry',
                          names=dataset_attributes)

print('Data was imported')

######################################################################################################################
#################################### FILL EMPTY CORNERS WITH POLYNOMIAL INTERPOLATION ################################
######################################################################################################################

print('------------INFO DATAFRAME MALE------------')
print('SIZE: ', df_male.size)
print('ROWS: ', df_male.shape[0])
print('ISNULL: ', df_male.isnull().sum().sum())

df_male = df_male.dropna(subset=['Age'])


df_male = df_male.drop(df_male[df_male['Age'] < 20].index)
df_male = df_male.drop(df_male[df_male['Age'] > 80].index)
print(df_male.shape[0])
df_male = df_male.dropna(thresh=8)

df_male = df_male.sort_values(by='Age')

#print(df_male[df_male.columns].duplicated())
# Check for duplicate indices
#print(df_male.index.duplicated())
#print('Stop')

# Сохраняем первый столбец отдельно
#first_column = df_male.iloc[:, 0]
# Применяем интерполяцию ко всем остальным столбцам
#df_male.iloc[:, 1:] = df_male.iloc[:, 1:].interpolate(method='linear')

df_male.interpolate(method='linear')       # !!!!!!!!!!!!!!
# Если необходимо заполнить пропуски в начале или конце
#df_male.iloc[:, 1:] = df_male.iloc[:, 1:].ffill().bfill()
df_male = df_male.ffill()
df_male = df_male.bfill()
# Объединяем первый столбец обратно
#df_male.iloc[:, 0] = first_column
df_male = df_male.sort_index()
df_male = df_male.sample(frac=1).reset_index(drop=True)
df_male = df_male.reset_index(drop=True)

df_male = df_male.sort_values(by='Age')                 # !!!!!!!!!!!!!!!

print(df_male)

print('------------INFO DATAFRAME FEMALE (after)------------')
print('SIZE: ', df_male.size)
print('ROWS: ', df_male.shape[0])
print('ISNULL: ', df_male.isnull().sum().sum())

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Загрузка существующего Excel-файла
path = "datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx"
book = load_workbook(path)

df_male.to_excel("datasets/biochemistry_filled_empty_by_polynomial_method_3.xlsx",
             sheet_name='Biochemistry', index=False)



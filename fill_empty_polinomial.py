import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





df_male = pd.read_excel('datasets/gemogramma_sorted_biomarker_columns_2.xlsx',
                          sheet_name='Male',
                          names=['Age',
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
                                 'ESR'])

print('Data was imported')

######################################################################################################################
#################################### FILL EMPTY CORNERS WITH POLYNOMIAL INTERPOLATION ################################
######################################################################################################################

print('------------INFO DATAFRAME MALE------------')
print('SIZE: ', df_male.size)
print('ROWS: ', df_male.shape[0])
print('ISNULL: ', df_male.isnull().sum().sum())

df_male = df_male.dropna(subset=['Age'])

df_male = df_male.drop(df_male[df_male['Age'] < 23].index)
df_male = df_male.drop(df_male[df_male['Age'] > 90].index)

df_male = df_male.dropna(thresh=8)

df_male = df_male.sort_values(by='Age')


df_male.interpolate(method='polynomial', order=2)       # !!!!!!!!!!!!!!
df_male = df_male.ffill()
df_male = df_male.bfill()
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

df_male.to_excel("datasets/gemogramma_filled_empty_by_polynomial_method [3].xlsx",
             sheet_name='Male', index=False)

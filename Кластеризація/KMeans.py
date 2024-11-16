import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering


df_male = pd.read_excel('datasets/gemogramma.xlsx',
                          sheet_name='male',
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

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
XPCAreduced = pca.fit_transform(transpose(df_male))
print ('Sklearn reduced X: \n', XPCAreduced)




# Scatterplot of two parameters
plt.scatter(df_male['Hemoglobin'],df_male['Leukocytes'], color=(1,0,0))
plt.xlim(75, 180)
plt.ylim(0, 50)
plt.show()





fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

sequence_containing_x_vals = list(range(220, 400))
sequence_containing_y_vals = list(range(0, 180))
sequence_containing_z_vals = list(range(0, 1000))



ax.scatter(df_male['MCH'],df_male['MCHC'],df_male['MCV'])
plt.show()

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

print('------------INFO DATAFRAME FEMALE (after)------------')
print('SIZE: ', df_male.size)
print('ROWS: ', df_male.shape[0])
print('ISNULL: ', df_male.isnull().sum().sum())

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


#######################################################################################################################
################################################# NORMALIZATION #######################################################
#######################################################################################################################

# Шаг 1: Создание объекта MinMaxScaler
scaler = MinMaxScaler()

# Шаг 2: Применение нормализации к данным
df_normalized = pd.DataFrame(scaler.fit_transform(df_male), columns=df_male.columns)

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

#####################################################################################################################
########################################  Завантаження EXCEL файлу ##################################################
#####################################################################################################################

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


#######################################################################################################################
############################################### DISTANCE MATRIX #######################################################
#######################################################################################################################

distance_matrix_euclidean = pdist(df_normalized, metric='euclidean')
distance_matrix_manhetten = pdist(df_normalized, metric='cityblock')
distance_matrix_mahalonobis = pdist(df_normalized, metric='mahalanobis')

#######################################################################################################################
################################################ CLASTERIZATION #######################################################
#######################################################################################################################
# Шаг 4: Агломеративная кластеризация
Z = linkage(distance_matrix_euclidean, method='ward')
Z2 = linkage(distance_matrix_manhetten, method='ward')
Z3 = linkage(distance_matrix_mahalonobis, method='ward')
# Шаг 5: Построение дендрограммы
plt.figure(figsize=(10, 7))
dendrogram(Z)

plt.show()

dendrogram(Z2)
plt.show()
dendrogram(Z3)
plt.show()


"""
# Шаг 6: Определение кластеров
max_d = 50  # Уровень среза дендрограммы
clusters = fcluster(Z, max_d, criterion='distance')
"""

K = 5   #Clusters number

hierarchical_cluster = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(df_normalized)

print(df_normalized.value_counts())

n = 100

matrix = np.zeros((n,K), dtype=float)
def division_coefficient():

    result = 0

    for j in range (1, K):
        for i in range (1, n):

            result+=(matrix[i][j]^2)/n
            pass


print(labels)
#plt.scatter(df_male['MCH'], df_male['MCHC'], c=labels)
#plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

sequence_containing_x_vals = list(range(220, 400))
sequence_containing_y_vals = list(range(0, 180))
sequence_containing_z_vals = list(range(0, 1000))



ax.scatter(df_male['MCH'],df_male['MCHC'],df_male['MCV'], c=labels)
plt.show()

# Вывод кластеров
#df_male['Cluster'] = clusters
#print(df_male)

# Результат
print(df_normalized)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import numpy as np
from sklearn.decomposition import PCA

x = np.arange(1,11)
y = np.array([2.73446908, 4.35122722, 7.21132988, 11.24872601, 9.58103444,
              12.09865079, 13.78706794, 13.85301221, 15.29003911, 18.0998018])
X = np.vstack((x,y))
print(X)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(X)

print(scaled_df)

pca = PCA(n_components=1)
XPCAreduced = pca.fit_transform(np.transpose(scaled_df))
print(XPCAreduced)
print(pca.mean_)
print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





df_male = pd.read_excel('datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx',
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

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(df_male)

print(scaled_df)
"""

print (X)

Xcentered = (X[0] - x.mean(), X[1] - y.mean())
m = (x.mean(), y.mean())
print (Xcentered)
print ("Mean vector: ", m)

covmat = np.cov(Xcentered)
print (covmat, "\n")
print ("Variance of X: ", np.cov(Xcentered)[0,0])
print ("Variance of Y: ", np.cov(Xcentered)[1,1])
print ("Covariance X and Y: ", np.cov(Xcentered)[0,1])
"""

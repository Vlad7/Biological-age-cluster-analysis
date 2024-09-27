import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""from sklearn import datasets
wine_data = datasets.load_wine(as_frame=True)"""
#

"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])"""

#Dataset

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

#Biomarkers

features = ['MCH',
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
            'ESR']

# Separating out the features
x = df_male.loc[:, features].values
print(x)

# Separating out the ages
y = df_male.loc[:,['Age']].values

# Scaling

std_scaler = StandardScaler()
scaled_df = std_scaler.fit_transform(x)
print(scaled_df)

# Установить параметр для вывода всех строк
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 30)
pd.set_option('display.min_columns', 30)





# Principal component analisys for 3 components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(scaled_df)

# Create dataframe with principal components
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

print(principalDf)

# Classification with only one class
target = pd.DataFrame(data=np.array(['0']*len(y)).transpose(), columns = ['Age category'])
print(target)
print(len(target))

finalDf = pd.concat([principalDf, target], axis = 1)

#finalDf.index = np.arange(1, len(finalDf) + 1)

print(finalDf)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)

#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#colors = ['r', 'g', 'b']

# Classes

targets = ['0']
colors = ['b']




for target, color in zip(targets,colors):
   
    # Select all indexes of humans with targeting classes
    
    indicesToKeep = finalDf['Age category'] == target
 
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

plt.show()
"""
df = wine_data.data
print(df)

print(df.shape)
print(df.info())


X = np.array([[0,0],
              [1,1],
              [2,2],
              [3,3],
              [4,4]])





print(std_scaler.mean_)
print(std_scaler.transform(X))

print(scaled_df)
"""


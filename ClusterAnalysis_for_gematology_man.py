import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from enum import Enum
import webcolors as wc
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split

class Feature(Enum):
    MCH = 1
    MCHC = 2
    MCV = 3
    MPV = 4
    PDW = 5
    RDW = 6
    Hematocrit = 7
    Hemoglobin = 8
    Granulocytes = 9
    Red_blood_cells = 10
    Leukocytes = 11
    Lymphocytes = 12
    Monocyte = 13
    Thrombocrit = 14
    Thrombocytes = 15
    ESR = 16

class ClusterAnalysis:
    
    def __init__ (self, path, sex):

        self.df_male = pd.read_excel(path,
                          sheet_name=sex,
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
        
        #features1 = [for e in Feature]
        features1=[
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
                                 'ESR']

        print (features1)

        #Pisaruk

        
        features2 = ['RDW',
            'Hematocrit',
            'Hemoglobin',
            'Thrombocytes',
            'ESR']

        features = features1

        self.data = self.df_male
       

        self.split_on_train_and_test_datasets()
        
        # Separating out the features
        self.Features = self.train_data.loc[:, features]
        print(self.Features)

        # Separating out the ages
        self.Ages = self.train_data.loc[:,['Age']]
        print(self.Ages)

        
    def split_on_train_and_test_datasets(self):

        # Разбиваем на возрастные бины
        bins = [20, 30, 40, 50, 60, 70, 80, 90]
        labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
        self.data['AgeBin'] = pd.cut(self.data['Age'], bins=bins, labels=labels)

       
        # Пустые DataFrame'ы для тренировочной и тестовой выборки
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        # Пропорциональное разбиение для каждого бина
        for bin_label in labels:
            bin_data = self.data[self.data['AgeBin'] == bin_label]

            # Пропорциональный размер тестового набора зависит от количества данных в бине
            if len(bin_data) > 1:  # Проверяем, что есть больше одного элемента для разделения
                test_size = min(0.3, 1 / len(bin_data))  # Чем меньше данных, тем меньший тестовый набор
                print(test_size)
                train_bin, test_bin = train_test_split(bin_data, test_size=test_size, random_state=42)
            
                train_data = pd.concat([train_data, train_bin], axis=0)
                test_data = pd.concat([test_data, test_bin], axis=0)
           

        # Сбрасываем индексы
        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        print("Training data:")
        print(train_data)
        print("\nTest data:")
        print(test_data)

        #print(self.data['AgeBin'])

        #X_train, X_test, y_train, y_test = train_test_split(self.Features, self.Ages.iloc[:,0],
        #                                                    test_size=0.2, random_state=42, stratify=self.Ages.iloc[:,0])
        #print(X_train)




    def ages_distribution(self):

        # Припустимо, що у вас є колонка 'Age' з віковими даними
        self.data['Age'].hist(bins=10)
        plt.title('Розподіл вікових груп')
        plt.xlabel('Вік')
        plt.ylabel('Кількість')
        plt.show()

        # !!! Треба по групам, а не просто по вікам.
        age_distribution = self.data['Age'].value_counts()
        print(age_distribution)

    def biological_age (self, analysis):

        # Add normalisation for biomarkers !!!!!!!!!!!!!!!!!!!!!!
        min_dist = 100000000000
        min_index = 0
        for index, row in self.Features.iterrows():
            dist = np.linalg.norm(row-analysis)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        print(self.Ages.values[min_index])
    def scale (self):
        
        # Scaling

        std_scaler = StandardScaler()
        self.scaled_df = std_scaler.fit_transform(self.Features.values)
        
        print(self.scaled_df)


        
    def plot_pca(self, classes_number, labels):

      
        # Principal component analisys for 3 components
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(self.scaled_df)

        # Create dataframe with principal components
        principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

        print(labels)

        # Classification with only one class
        data = np.array(labels)
        print(data)
       
        target = pd.DataFrame(data=data, columns = ['Age category'])
        #target = pd.DataFrame(data=np.array(['0']*len(self.Ages.values)).transpose(), columns = ['Age category'])
        print(target)

        print(len(target))

        finalDf = pd.concat([principalDf, target], axis = 1)

        #finalDf.index = np.arange(1, len(finalDf) + 1)

        print(finalDf)
        
        print(principalDf)

        # Установить параметр для вывода всех строк
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        








        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(projection='3d') 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_zlabel('Principal Component 3', fontsize = 15)
        ax.set_title('3 component PCA', fontsize = 20)

        #targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        #colors = ['r', 'g', 'b']

        # Classes

        targets = []
        for x in labels:
            if x not in targets:
                targets.append(x)
        print(targets)

        colors = []
        targets = [0, 1, 2, 3, 4]
        colors = ['b', 'y','r','g','c']
      
      

        for k in targets:
            colors.append(wc.rgb_to_hex((int(255 * k / len (targets)), int(255 * k / len (targets)), int(255 * k / len (targets)))))
       


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

        ax.set_xlim([-12, 12])
        ax.set_ylim([-12, 12])
        ax.set_zlim([-12, 12])
        plt.show()

        print(pca.explained_variance_ratio_)

    def plot_pca_cumulative(self):
        # Principal component analisys for 3 components

        pca = PCA(n_components=len(features))
        principalComponents = pca.fit_transform(scaled_df)
        explained = pca.explained_variance_ratio_

        exp_var_pca = pca.explained_variance_ratio_
        #
        # Cumulative sum of eigenvalues; This will be used to create step plot
        # for visualizing the variance explained by each principal component.
        #
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)

        plt.bar(range(1,len(exp_var_pca) + 1), exp_var_pca, alpha=0.5, align='center', label='Індивідуальна пояснена дисперсія')
        plt.step(range(1,len(cum_sum_eigenvalues) + 1), cum_sum_eigenvalues, where='mid',label='Кумулятивна пояснена дисперсія')
        plt.ylabel('Частка поясненої дисперсії')
        plt.xlabel('Індекс головної компоненти')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid()
        plt.show()




    


    def kmeans_clustering(self):


        from sklearn.cluster import KMeans

        x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
        y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
        data = self.Features
        #data = list(zip(x, y))
        print(data)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(data)

        print (kmeans.labels_)

        

        

        # Mean ariphmetic by each cluster bio age
        clusters_pacient_indexes = {}
        for i in range(len(kmeans.labels_)):
            label = kmeans.labels_[i]
            if label in clusters_pacient_indexes.keys():
                clusters_pacient_indexes[label].append(i)
            else:
                clusters_pacient_indexes[label] = [i]

        clusters_bio_age = {}
        
        for cluster_number in clusters_pacient_indexes.keys():

            summ = 0

            for pacient_index in clusters_pacient_indexes[cluster_number]:

                summ += self.Ages[pacient_index]

            summ = summ / len( clusters_pacient_indexes[cluster_number])

            clusters_bio_age[cluster_number] = summ

        print(clusters_bio_age)
            
      
        print(self.Ages.values)
        
        self.plot_pca(2, kmeans.labels_)
        #plt.scatter(self.data['MCH'], self.df_male['MCHC'], c=kmeans.labels_)
        #plt.show()


    def cmeans_clustering(self):

        classes_number = 5
        
        x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
        y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


        data = self.data
        #data = list(zip(x, y))

        
        
        print(data)
        
        data = np.array(data)



        # C-means clustering
        # m - fuzziness parameter
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        np.transpose(data), c=classes_number, m=3, error=0.005, maxiter=1000, init=None)    
        


        # Assign pacients to clusters based on maximum membership,
        # Result: len(clusters) == pacients count, element at index in "clusters" == cluster number
        # All works correctly
        
        clusters = np.argmax(u, axis=0)
      
        # Plot assigned clusters

        """
        for j in range(classes_number):
             
            # : (двоеточие) — означает, что мы берем все строки (или весь диапазон данных по первой оси).
            plt.plot(data[:, 0][clusters == j], data[:, 1][clusters == j], 'o', label=f'cluster{j}')

        plt.legend()
        plt.show()
        """
        
        self.plot_pca(classes_number, clusters)
        #plt.scatter(self.data['MCH'], self.data['MCHC'], c=kmeans.labels_)
        #plt.show()
    
if __name__ == '__main__':
    """
    import matplotlib.pyplot as plt

    x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    plt.scatter(x, y)
    plt.show()

    

    data = list(zip(x, y))
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    """
    

    ClAnalysis = ClusterAnalysis(r'datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx', 'Male')
    ClAnalysis.ages_distribution()
  
    ClAnalysis.scale()
    ClAnalysis.kmeans_clustering()
    ClAnalysis.plot_pca()
    #ClAnalysis.kmeans_clustering()
    
    #ClAnalysis.biological_age((24.1, 391, 78, 9.7, 14.4, 15.7, 0.45, 149, 66.4,	5.13, 8, 29.9, 3.7, 0.218, 226, 5))









    
"""from sklearn import datasets
wine_data = datasets.load_wine(as_frame=True)"""
#

"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])"""

#Dataset





 












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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler    #Standardize features by removing the mean and scaling to unit variance.

#import webcolors as wc
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

import features
import features as ft
import sys
from enum import Enum

import pca_lib as pl



class Dataset(Enum):
    """Type of dataset"""
    Biochemistry = 1
    Bones = 2
    Gematology = 3
    NHANESBiochemistry = 4

class Sex (Enum):
    """Sex of persons in dataset"""
    Both = 1
    Female = 2
    Male = 3



class ClusterAnalysis:
    
    def __init__ (self, path, sex, hight_correlated_features=None, datasettype=Dataset.Biochemistry):
        """Constructor for cluster analysis

        :param path: path to file with database
        :param sex:  sex of persons in database
        :param is_hight_correlated_features:
        :param datasettype: type of dataset
        """
        
        #First attribute - Age
        dataset_attributes = ['Age']
        
        if datasettype == Dataset.Biochemistry:
            # Add all attributes from biochemistry
            dataset_attributes.extend((ft.features_biochemistry_all))
            
            print("All features: " + str(ft.features_biochemistry_all))
            
        elif datasettype == Dataset.Bones:
            # Add all attributes from bones
            dataset_attributes.extend((ft.features_bones_all))
                                      
            print("All features: " + str(ft.features_bones_all))
            
        elif datasettype == Dataset.Gematology:
            # Add all attributes from gematology            
            dataset_attributes.extend(ft.features_gematology_all)
            
            print("All features: " + str(ft.features_gematology_all))

        elif datasettype == Dataset.NHANESBiochemistry:
            # Add all attributes from NHANES biochemistry
            dataset_attributes.extend(ft.features_NHANES3_HDTrain_biochemistry)

            print("All features: " + str(ft.features_NHANES3_HDTrain_biochemistry))

        sexs = ""

        if sex == Sex.Both:
            sexs = "Both sexes"
        elif sex == Sex.Female:
            sexs = "Female"
        elif sex == Sex.Male:
            sexs = "Male"

        try:
            self.data = pd.read_excel(path,
                            sheet_name=sexs,
                            names=dataset_attributes)

            print('Data was imported!')

        except FileNotFoundError:
            print('File was not found!')
            sys.exit(0)

        #Selected biomarkers
        selected_biomarkers = None

        if hight_correlated_features != None:
            # Select feature labels that hight correlates with age
            selected_biomarkers = ['Age']
            selected_biomarkers.extend(hight_correlated_features)
        else:
            selected_biomarkers = dataset_attributes

        selected_data = self.data.loc[:, selected_biomarkers]

        #Split dataset on train and test datasets with ages accordingly
        self.train_data, self.test_data, self.train_ages, self.test_ages = (
            self.split_on_train_and_test_datasets(selected_data, age_bins=True))

        # Print train data with selected features
        print("Training data:")
        print(self.train_data)

        # Train ages
        print("Training ages:")
        print(self.train_ages)


        # Biomarkers sfs = selected feature set
        self.train_data_scaled = self.scale(self.train_data.values)
        self.test_data_scaled = self.scale(self.test_data.values)

        # train_ages_scaled = std_scaler.fit_transform(self.train_ages.values)
        # test_ages_scaled = std_scaler.fit_transform(self.test_ages.values)
        #self.train_data_sfs_scaled, self.test_data_sfs_scaled, self.train_ages_scaled, self.test_ages_scaled = (
        #    self.scale(self.train_data, self.test_data, self.train_ages, self.test_ages))


    def scale (self, dataframe):
        """Scaling method

        :param dataframe: dataframe to scale
        :return: scaled_data
        """
        # Scaling

        std_scaler = StandardScaler()
        data_scaled = std_scaler.fit_transform(dataframe)

        return data_scaled
        
    def move_age_bin_column_to_after_age_position(self, data):
        """Move age bin column from the end to position after age

        :param data: data - dataframe
        :return: data - datafreme with moved column
        """
        # Впевнимося, що Age_bin переміщається одразу після Age
        columns = list(data.columns)
        age_index = columns.index("Age")
        columns.remove("AgeBin")  # Прибираємо Age_bin з поточної позиції
        columns.insert(age_index + 1, "AgeBin")  # Вставляемо Age_bin одразу після Age

        # Переставляємо колонки
        data = data[columns]

        return data

    def split_on_train_and_test_datasets_based_on_age_bins(self, data):
        """
            Розділити вибірку на тренувальну та тестову на основі вікових бінів.
            Вікові біни - це як би вікові групи
        :param data: input data
        :return: train_data, test_data - dataframes
        """
        # Порожні data'и для тренувальної та тестової вибірок
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        train_ages = pd.DataFrame()
        test_ages = pd.DataFrame()



        # Розбиваємо на вікові біни
        bins = [20, 30, 40, 50, 60, 70, 80, 90]
        labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
        data['AgeBin'] = pd.cut(data['Age'], bins=bins, labels=labels)

        data = self.move_age_bin_column_to_after_age_position(data)

        # Пропорциональное разбиение для каждого бина
        for bin_label in labels:
            bin_data = data[data['AgeBin'] == bin_label]

            # Пропорциональный размер тестового набора зависит от количества данных в бине
            if len(bin_data) > 1:  # Проверяем, что есть больше одного элемента для разделения

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!errors?
                test_size = min(0.3,
                                1 / len(bin_data))  # Чем меньше данных, тем меньший тестовый набор до определенного
                # предела 3 обьекта в бине, потом, при большем количестве данных будет меньше тестовый набор
                print("Test size in age bin " + str(test_size))

                # Maybe make with y! stratify maybe
                train_bin, test_bin = train_test_split(bin_data, test_size=test_size, random_state=42)

                train_data = pd.concat([train_data, train_bin], axis=0)
                test_data = pd.concat([test_data, test_bin], axis=0)

        # Вхідні дані (біомаркери)
        Xtrain = train_data.drop(['Age', 'AgeBin'], axis=1)
        Xtest = test_data.drop(['Age', 'AgeBin'], axis=1)
        # Цільова змінна (вік)
        ytrain = train_data['Age']
        ytest = test_data['Age']

        return Xtrain, Xtest, ytrain, ytest

    def split_on_train_and_test_datasets_without_bins(self, data):

        # Пустые data'ы для тренировочной и тестовой выборки
        train_data = pd.data()
        test_data = pd.data()

        train_ages = pd.data()
        test_ages = pd.data()

        # Вхідні дані (біомаркери)
        X = data.drop('Age', axis=1)

        # Цільова змінна (вік)
        y = data['Age']

        # Розбиваємо вибірку на навчальний та тестовий набори
        # test_size=0.1 означає, що 10% даних піде у тестовий набір
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Перевірка розмірів вибірок
        print(f'Розмір навчального набору: {X_train.shape}')
        print(f'Розмір тестового набору: {X_test.shape}')

        train_data = pd.concat([train_data, X_train], axis=0)
        test_data = pd.concat([test_data, X_test], axis=0)

        train_ages = pd.concat([train_ages, y_train], axis=0)
        test_ages = pd.concat([test_ages, y_test], axis=0)
        ##############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ########## Зробити повернення вікових даних

        return train_data, test_data, train_ages, test_ages

    def split_on_train_and_test_datasets(self, data, age_bins=True):

        # Пустые data'ы для тренировочной и тестовой выборки
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        train_ages = pd.DataFrame()
        test_ages = pd.DataFrame()
        
        if age_bins:
           train_data, test_data, train_ages, test_ages = self.split_on_train_and_test_datasets_based_on_age_bins(data)

        else:

            train_data, test_data, train_ages, test_ages = self.split_on_train_and_test_datasets_without_bins(data)

        # Сбрасываем индексыб, не нужно!!!
        #train_data = train_data.reset_index(drop=True)
        #test_data = test_data.reset_index(drop=True)

        train_data = train_data.sort_index()
        test_data = test_data.sort_index()

        train_ages = train_ages.sort_index()
        test_ages = test_ages.sort_index()



        print("Training data:")
        print(train_data)
        print("\nTest data:")
        print(test_data)

        return train_data, test_data, train_ages, test_ages


        #print(self.data['AgeBin'])

        #X_train, X_test, y_train, y_test = train_test_split(self.train_data_selected_features, self.train_ages.iloc[:,0],
        #                                                    test_size=0.2, random_state=42, stratify=self.train_ages.iloc[:,0])
        #print(X_train)






    def ages_distribution(self, rounded_intervals=True):

        if rounded_intervals:
            # Приклад даних
            data = self.data['Age']
            data = pd.DataFrame(data)

            # Визначаємо інтервали та мітки
            bins = [20, 30, 40, 50, 60, 70, 80, 90]
            labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']

            # Створюємо нову колонку з інтервалами
            data['AgeBin'] = pd.cut(data['Age'], bins=bins, labels=labels)

            # Побудова гістограми
            plt.hist(data['Age'], bins=bins, edgecolor='black')
            plt.xlabel('Age Groups')
            plt.ylabel('Frequency')
            plt.title('Age Distribution in 10-Year Intervals')
            plt.grid(True, which='both', axis='x')
            plt.xticks(bins)  # Встановлюємо мітки на осі X для інтервалів
            plt.show()

            # Виводимо data для перевірки
            #print("Ages distribution data: " + data)

            # Обчислюємо частотність для кожної групи
            age_distribution = data['AgeBin'].value_counts().sort_index()

            # Вивід результату
            #print("Ages distribution: " + age_distribution)

        else:

            # Припустимо, що у нас є колонка 'Age' з віковими даними
            n, bins, patches = plt.hist(self.data['Age'], bins=10)
            plt.title('Розподіл вікових груп')
            plt.xlabel('Вік')
            plt.ylabel('Кількість')
            # Додаємо сітку по границям бінів
            plt.grid(True, which='both', axis='x')

            plt.xticks(bins)  # Встановлюємо мітки по границям бінів
            # Створюємо гістограму та отримуємо дані про бінінг

            plt.show()

            # Можливо зробити, щоб виводилися дані про частотність у цьому випадку.



    def biological_age (self, analysis):

        # Add normalisation for biomarkers !!!!!!!!!!!!!!!!!!!!!!
        min_dist = 100000000000
        min_index = 0
        for index, row in self.train_data_selected_features.iterrows():
            dist = np.linalg.norm(row-analysis)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        print(self.train_ages.values[min_index])

    ###################################################################
    ###################################################################
    ###################################################################
   










    ###################################################################
    ###################################################################
    ###################################################################    

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


    ####################################################################
    ####################################################################
    ####################################################################

    def kmeans_clustering(self, data, clusters_number):

        """ k-means clusterning

            input: features data
                   clusters_number

            output: kmeans.cluster_centers_ - centers of clusters as result of kmeans
                    kmeans.labels_          - labels of clusters for each person

            Example:
                x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
                y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
                data = list(zip(x, y))
        """

        # K-means++ - покращений алгоритм K-means з ініціалізацією центрів

        kmeans = KMeans(n_clusters=clusters_number, init='k-means++')
        kmeans.fit(data)

        return kmeans.cluster_centers_, kmeans.labels_



    def kmeans_clustering_factory(self):
        """OK"""

        """Доробити алгоритм k-means"""

        data = self.train_data_scaled

        clasters_number = 0

        while clasters_number == 0:
            try:
                clasters_number = int(input("Enter clusters number: "))
                print("Clasters number: ", clasters_number)

            except ValueError:
                print("Введите лучше число!")



        centers, labels = self.kmeans_clustering(data, clasters_number)

        """
        classes_number = len(set(labels))
        persons_number = len(labels)

        u = np.zeros((classes_number, persons_number))

        for i in range(len(labels)):
            u[labels[i]][i] = 1
        """


        indexes = self.clusters_patient_indexes(labels)
        clusters_bio_age = self.clusters_bio_age(self.train_ages, indexes)

        print(indexes)
        print(clusters_bio_age)

        pl.plot_pca(data, labels, centers, show_ages=True)

        # indexes = self.clusters_patient_indexes(kmeans.labels_)
        # self.clusters_bio_age(self.train_ages, indexes)

        # plt.scatter(self.data['MCH'], self.df_male['MCHC'], c=kmeans.labels_)
        # plt.show()



    def elbow(self):
     
        from yellowbrick.cluster import KElbowVisualizer      

        # Instantiate the clustering model and visualizer
        km = KMeans(random_state=42, init='k-means++')
        visualizer = KElbowVisualizer(km, k=(2,10))
         
        visualizer.fit(self.train_data_selected_features_set_scaled)        # Fit the data to the visualizer
        visualizer.show()        # Finalize and render the figure


    ####################################################################################################################
    ########################################## K-means biological age detection ########################################
    ####################################################################################################################

    def clusters_patient_indexes (self, labels):

        """ Find list of human indexes for each cluster

            input:
                - labels - list with class labels for each human as index

            output:

                - dictionary with indexes of human in each class
        """

        unique_classes = np.unique(labels)

        clusters_patient_indexes = {key: [] for key in dict.fromkeys(unique_classes)}

        for index, claster_number in enumerate(labels):

            clusters_patient_indexes[claster_number].append(index)

        return clusters_patient_indexes


    def clusters_bio_age(self, train_ages_dataframe, indexes_of_persons_in_clusters):
        
        """ Mean ariphmetic by each cluster bio age """
        
        clusters_bio_age = {}
        
        for cluster_number in indexes_of_persons_in_clusters.keys():

            summ = 0

            persons_indexes = indexes_of_persons_in_clusters[cluster_number]

            for person_index in persons_indexes:
                 #3                summ +=t rain_ages_dataframe['Age'].values[person_index]
                summ +=train_ages_dataframe.values[person_index]

            summ = summ / len(persons_indexes)

            clusters_bio_age[cluster_number] = summ


        return clusters_bio_age
       
            
      


    ###################################################################################################################
    ######################################## Initialising centers for cmeans ##########################################
    ###################################################################################################################


    def initialize_centers_kmeans_pp(self, data, num_clusters):
        centers = []
        centers.append(data[np.random.randint(0, len(data))])

        for _ in range(1, num_clusters):
            distances = np.array([min(np.linalg.norm(x - center) ** 2 for center in centers) for x in data])
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()

            for idx, prob in enumerate(cumulative_probs):
                if r < prob:
                    centers.append(data[idx])
                    break

        return np.array(centers)

    def calculate_initial_membership_matrix(self, X, centers, m):
        # Число точек данных и число кластеров
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]

        # Вычисляем расстояния от каждой точки до каждого центра

        distances = cdist(centers, X)

        # Инициализируем матрицу принадлежности
        U = np.zeros((n_clusters, n_samples))

        # Заполняем матрицу принадлежности по формуле FCM
        for i in range(n_clusters):
            for j in range(n_samples):
                if distances[i, j] == 0:
                    # Если точка совпадает с центром кластера, принадлежит ему на 100%
                    U[:, j] = 0
                    U[i, j] = 1
                    break
                else:
                    # Стандартный расчет при ненулевом расстоянии
                    denominator = sum((distances[i, j] / distances[k, j]) ** (2 / (m - 1)) for k in range(n_clusters) if
                                      distances[k, j] != 0)
                    U[i, j] = 1 / denominator

        return U

    def cmeans_clustering(self, dataset, clasters_number):

        # x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
        # y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
        # data = np.random.rand(100, 2)  # Данные
        # data = list(zip(x, y))

        print(dataset)

        data = np.array(dataset)

        # Пример использования

        m = 3  # Параметр "размытия"

        # Initializing centers matrix by k-means++ method
        centers = self.initialize_centers_kmeans_pp(data, clasters_number)

        print(centers)

        # Начальная матрица приналежности
        membership_matrix = self.calculate_initial_membership_matrix(data, centers, m)

        # C-means clustering
        # m - fuzziness parameter
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            np.transpose(data), c=clasters_number, m=3, error=0.005, maxiter=1000, init=membership_matrix)

        return cntr, u

    def cmeans_factory(self):

        data = self.train_data_scaled

        clasters_number = 0

        while clasters_number == 0:
            try:
                clasters_number = int(input("Enter clusters number: "))
                print("Clasters number: ", clasters_number)

            except ValueError:
                print("Введите лучше число!")

        # u - Степени принадлежности каждого объекта к каждому кластеру
        cntr, u = self.cmeans_clustering(data, clasters_number)

        pl.plot_pca(data, u, cntr, show_indexes=False, show_ages=False)

        # Assign pacients to clusters based on maximum membership,
        # Result: len(clusters) == pacients count, element at index in "clusters" == cluster number
        # All works correctly

        #labels = np.argmax(u, axis=0)
        #indexes = self.clusters_patient_indexes(labels)
        #indexes = self.clusters_patient_indexes_2(u)
        clusters_bio_age = self.clusters_bio_age_c_means(self.train_ages, u)

        #print(indexes)
        print(clusters_bio_age)



    """
    def clusters_bio_age_c_means(self, train_ages_dataframe, indexes_of_persons_in_clusters, u):

        ###Mean ariphmetic by each cluster bio age 

        clusters_bio_age = {}

        for cluster_number in indexes_of_persons_in_clusters.keys():

            summ = 0

            persons_indexes = indexes_of_persons_in_clusters[cluster_number]

            for person_index in persons_indexes:
                summ += train_ages_dataframe['Age'].values[person_index] * u[cluster_number][person_index]

            summ = summ / np.sum(u[cluster_number])

            clusters_bio_age[cluster_number] = summ

        return clusters_bio_age
    """

    def clusters_bio_age_c_means(self, train_ages_dataframe, u):

        """Mean ariphmetic by each cluster bio age """

        clusters_bio_age = {}

        for cluster_number in range(len(u)):

            summ = 0

            for person_index, age in enumerate(train_ages_dataframe.values):

            #for person_index, age in enumerate(train_ages_dataframe['Age'].values):
                summ += age * u[cluster_number][person_index]

            summ = summ / np.sum(u[cluster_number])    ### !!!!!!!!!!!!!!!

            clusters_bio_age[cluster_number] = summ

        return clusters_bio_age






    def minimal_spanning_tree_clustering(self):

        import seaborn as sns; sns.set()

        # matplotlib 1.4 + numpy 1.10 produces warnings; we'll filter these
        import warnings; warnings.filterwarnings('ignore', message='elementwise')


        def plot_mst(model, cmap='rainbow'):
           
            """Utility code to visualize a minimum spanning tree"""
            X = model.X_fit_
            fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
            for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
                segments = model.get_graph_segments(full_graph=full_graph)
                print(segments)
                axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
                axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
                axi.axis('tight')


            ax[0].set_title('Full Minimum Spanning Tree', size=16)
            ax[1].set_title('Trimmed Minimum Spanning Tree', size=16);
            plt.show()

        """
               def plot_mst(model, cmap='rainbow'):

                   #Utility code to visualize a minimum spanning tree
                   X = model.X_fit_

                                        # Применяем PCA для проекции в пространство трех главных компонент
                                pca = PCA(n_components=3)
                                   X_pca = pca.fit_transform(X)

                    # Создаем 3D-график
                   fig = plt.figure(figsize=(16, 8))
                   ax1 = fig.add_subplot(121, projection='3d' )
                   ax2 = fig.add_subplot(122, projection='3d')

                   #fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
                   for axi, full_graph, colors in zip([ax1, ax2], [True, False], ['lightblue', model.labels_]):
                       segments = model.get_graph_segments(full_graph=full_graph)
                       print(segments)
                       #axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
                       #axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
                       #axi.axis('tight')
                        # Для каждого сегмента рисуем линию в пространстве PCA
                       for seg in segments:
                           p1 = X_pca[seg[0]]
                           p2 = X_pca[seg[1]]
                           ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-k', zorder=1, lw=1)

                       # Рисуем точки
                       scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap=cmap, zorder=2)
                       ax.set_title('Full MST' if full_graph else 'Trimmed MST', size=16)

                       #ax[0].set_title('Full Minimum Spanning Tree', size=16)
                       #ax[1].set_title('Trimmed Minimum Spanning Tree', size=16);
                   plt.show()
               """

        from sklearn.datasets import make_blobs
        
        #X, y = make_blobs(200, centers=4, random_state=42)
        #X, y = make_blobs(n_samples=200, centers=4, n_features=16 ,
        #          random_state=42)
        #self.plot_pca(0, [0])
        
        






        from mst_clustering import MSTClustering

        model = MSTClustering(cutoff_scale=7, approximate=False)

        labels = model.fit_predict(self.train_data_selected_features_set_scaled)
        
        #plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow');
        #plt.show()




        #plot_mst(model)

        #print(labels)
        #self.plot_pca(len(np.unique(labels)), labels)
        #print(model.get_graph_segments(True))
        # Допустим, segments содержит две строки — начало и конец рёбер в 16D пространстве:
        # segments[0] — это координаты начала рёбер, segments[1] — координаты конца рёбер


        segments = model.get_graph_segments(True)

        def segments_from_features_space_to_principal_component_space (segments):            
            start_points_list = []
            end_points_list = []

    
            for i in range(len(segments)):
                start_points_list.append(segments[i][0])
                end_points_list.append(segments[i][1])

            """import numpy as np
                a = np.array((1,2,3))
                b = np.array((2,3,4))
                np.column_stack((a,b))
                array([[1, 2],
                        [2, 3],
                        [3, 4]])"""

            start_points = np.column_stack(start_points_list)  # Начальные точки рёбер
            end_points = np.column_stack(end_points_list)      # Конечные точки рёбер

            # Объединяем начало и конец рёбер для общей проекции
            all_points = np.vstack([start_points, end_points])  # Все точки для PCA

            # Применяем PCA для проекции из 16D в 3D
            pca = PCA(n_components=3)
            all_points_pca = pca.fit_transform(all_points)

            # Разделяем обратно на начальные и конечные точки рёбер в новом 3D-пространстве
            start_points_pca = all_points_pca[:len(start_points)]
            end_points_pca = all_points_pca[len(start_points):]

            return start_points_pca, end_points_pca

        start_points_pca, end_points_pca = segments_from_features_space_to_principal_component_space(segments)

      
        # Создаем 3D-график
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Проходим по всем сегментам и рисуем линии в 3D-пространстве PCA
        for i in range(len(start_points_pca)):
            x_start, y_start, z_start = start_points_pca[i]
            x_end, y_end, z_end = end_points_pca[i]
            
            # Рисуем ребро между начальной и конечной точкой сегмента
            ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], 'k-', lw=1)

        # Добавляем точки для визуализации вершин
        ax.scatter(start_points_pca[:, 0], start_points_pca[:, 1], start_points_pca[:, 2], c='r', s=50)
        ax.scatter(end_points_pca[:, 0], end_points_pca[:, 1], end_points_pca[:, 2], c='b', s=50)

        # Настраиваем оси
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')

        plt.show()



        
        # create some data with four clusters
        X, y = make_blobs(200, centers=4, random_state=42)
        
        # predict the labels with the MST algorithm
        model = MSTClustering(cutoff_scale=2)
        labels = model.fit_predict(X)
        print(labels)
        
        # plot the results
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
        plt.show()

    
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

    path_to_biochemistry_dataset = r'../datasets/biochemistry_filled_empty_by_polynomial_method_3.xlsx'
    path_to_bones_dataset = r'../datasets/bones_filled_empty_by_polynomial_method_3.xlsx'
    path_to_gematology_dataset = r'../datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx'
    path_to_NHANES_biochemistry = r'../datasets/biochemistryNHANES_filled_empty_by_polynomial_method_3.xlsx'

    ClAnalysisGematologyMale = ClusterAnalysis(path_to_gematology_dataset, Sex.Male,
                                     features.features_gematology_hight_correlation_with_age, Dataset.Gematology)


    #ClAnalysisGematologyMale.ages_distribution()
    #ClAnalysisGematologyMale.scale()
    pl.plot_pca(ClAnalysisGematologyMale.train_data_scaled)
    #ClAnalysisMale.elbow()
    #ClAnalysisMale.kmeans_clustering_factory()
    #ClAnalysisMale.cmeans_factory()


    #ClAnalysisBonesFemale = ClusterAnalysis(path_to_bones_dataset, Sex.Female,
    #                                        None, Dataset.Bones)
    #ClAnalysisBonesFemale.scale()
    #pl.plot_pca(ClAnalysisBonesFemale.train_data_scaled)
    #ClAnalysisMale.elbow()
    #ClAnalysisMale.kmeans_clustering_factory()
    #ClAnalysisMale.cmeans_factory()

    #ClAnalysisBiochemistryBoth = ClusterAnalysis(path_to_biochemistry_dataset, Sex.Both,
    #                                             None, Dataset.Biochemistry)
    #ClAnalysisBiochemistryBoth.scale()
    #pl.plot_pca(ClAnalysisBiochemistryBoth.train_data_scaled)
    #print(ClAnalysis.train_data_selected_features_set_scaled)
    #ClAnalysisMale.minimal_spanning_tree_clustering()

    #ClAnalysisNHANESBiochemistry = ClusterAnalysis(path_to_NHANES_biochemistry, Sex.Both, None, Dataset.NHANESBiochemistry)
    #ClAnalysisFemale = ClusterAnalysis(r'datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx', 'Female')
    #ClAnalysisFemale.ages_distribution()
    #ClAnalysisFemale.scale()
    #pl.plot_pca(ClAnalysisNHANESBiochemistry.train_data_scaled)
    #ClAnalysisFemale.elbow()
    #ClAnalysisNHANESBiochemistry.kmeans_clustering_factory()
    #ClAnalysisNHANESBiochemistry.cmeans_factory()

    #ClAnalysis.kmeans_clustering()
    
    #ClAnalysis.biological_age((24.1, 391, 78, 9.7, 14.4, 15.7, 0.45, 149, 66.4,	5.13, 8, 29.9, 3.7, 0.218, 226, 5))









    
"""from sklearn import datasets
wine_data = datasets.load_wine(as_frame=True)"""
#

"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas data
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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler    #Standardize features by removing the mean and scaling to unit variance.
from sklearn.decomposition import PCA
#import webcolors as wc
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import gematology_features as gf
import sys

class ClusterAnalysis:
    def __init__(self, path, sex, is_hight_correlated_features=True):
        dataset_attributes = ['Age']
        dataset_attributes.extend(gf.features_all)

        try:
            self.data = pd.read_excel(path,
                                      sheet_name=sex,
                                      names=dataset_attributes)

            print('data was imported!')

        except FileNotFoundError:
            print('File was not found!')
            sys.exit(0)

        print("All features: " + str(gf.features_all))

        # Split dataset on train and test datasets with ages accordingly
        self.train_data, self.test_data, self.train_ages, self.test_ages = (
            self.split_on_train_and_test_datasets(self.data, True))

        # Select feature labels that hight correlates with age
        features = gf.features_hight_correlation_with_age

        self.train_data_selected_features = None
        self.test_data_selected_features = None

        if is_hight_correlated_features:
            # Separating out the features of interest from train_data
            self.train_data_selected_features = self.train_data.loc[:, gf.features_hight_correlation_with_age]
            self.test_data_selected_features = self.test_data.loc[:, gf.features_hight_correlation_with_age]

        else:
            self.train_data_selected_features = self.train_data.loc[:, gf.features_all]
            self.test_data_selected_features = self.test_data.loc[:, gf.features_all]

        # Print train data with selected features
        print(self.train_data_selected_features)

        # Train ages
        print(self.train_ages)

        # Biomarkers
        self.scale()

# print(len(self.train_data_selected_features))

    def scale(self):

        # Scaling

        std_scaler = StandardScaler()
        self.train_data_selected_features_set_scaled = std_scaler.fit_transform(
            self.train_data_selected_features.values)

    def move_age_bin_column_to_after_age_position(self, data):

        # Впевнимося, що Age_bin перемыщається одразу після Age
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
        # train_data = train_data.reset_index(drop=True)
        # test_data = test_data.reset_index(drop=True)

        train_data = train_data.sort_index()
        test_data = test_data.sort_index()

        train_ages = train_ages.sort_index()
        test_ages = test_ages.sort_index()

        print("Training data:")
        print(train_data)
        print("\nTest data:")
        print(test_data)

        return train_data, test_data, train_ages, test_ages

        # print(self.data['AgeBin'])

        # X_train, X_test, y_train, y_test = train_test_split(self.train_data_selected_features, self.train_ages.iloc[:,0],
        #                                                    test_size=0.2, random_state=42, stratify=self.train_ages.iloc[:,0])
        # print(X_train)




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

    ClAnalysisMale = ClusterAnalysis(r'datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx', 'Male')
    # ClAnalysisMale.ages_distribution()
    # ClAnalysisMale.scale()
    # ClAnalysisMale.plot_pca(ClAnalysisMale.train_data_selected_features_set_scaled)
    # ClAnalysisMale.elbow()
    # ClAnalysisMale.kmeans_clustering_factory()
    # ClAnalysisMale.cmeans_factory()

    input = ("")

    # print(ClAnalysis.train_data_selected_features_set_scaled)
    # ClAnalysisMale.minimal_spanning_tree_clustering()

    # ClAnalysisFemale = ClusterAnalysis(r'datasets/gemogramma_filled_empty_by_polynomial_method_3.xlsx', 'Female')
    # ClAnalysisFemale.ages_distribution()
    # ClAnalysisFemale.scale()
    # ClAnalysisFemale.plot_pca(ClAnalysisFemale.train_data_selected_features_set_scaled)
    # ClAnalysisFemale.elbow()
    # ClAnalysisFemale.kmeans_clustering_factory()
    # ClAnalysisFemale.cmeans_factory()

    # ClAnalysis.kmeans_clustering()

    # ClAnalysis.biological_age((24.1, 391, 78, 9.7, 14.4, 15.7, 0.45, 149, 66.4,	5.13, 8, 29.9, 3.7, 0.218, 226, 5))


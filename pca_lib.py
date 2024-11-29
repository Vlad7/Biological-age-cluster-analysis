from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pca(features_set, n_components_=2):
    """ PCA

        input:
            - features_set - data with features
            - n_components - number of principal components
        output:
            principal components
    """

    # Principal component analisys for 3 components
    pca = PCA(n_components=n_components_)
    principalComponents = pca.fit_transform(features_set)

    sufixes = []

    if n_components_ == 2:
        sufixes.append("2nd")
    elif n_components_ == 3:
        sufixes.append("3rd")
    else:
        raise Exception("Number of components must be 2 or 3!")

    print("PCA explained variance ratio (1st, " + ', '.join(sufixes) + "): ", pca.explained_variance_ratio_)

    return principalComponents

def plot_pca(features_set, membership_matrix=None, centers=None, show_indexes=False, show_ages=False):
    """ Plot clustered data in the space of principal components

        input:
            - features_set - data with features
            - membership_matrix - matrix with membership of person to cluster
            - centers - centers of clusters
            - show_indexes - show texts of indexes near data points on plot
            - show_ages - show texts of ages near data points on plot

        output:
            None

        !!! Method completed! Maybe make same scale of different axis
    """
    # Create data with principal components
    principalDf = pd.DataFrame(data=pca(features_set, 3),
                               columns=['principal component 1', 'principal component 2', 'principal component 3'])

    labels = None

    #- labels = labels from classified class
    if membership_matrix is None:

        # Transform labels list to np.array and numeration from 1
        labels = np.array([0] * len(features_set))

    elif membership_matrix.ndim == 1:

        labels = membership_matrix

    else:

        labels = np.argmax(membership_matrix, axis=0)

    labels = np.array(labels) + 1

    # Create target data with one column with labels and named 'Age category
    target = pd.DataFrame(data=labels, columns=['Age category'])

    """
    #Classification with only one class
    #target = pd.data(data=np.array(['0']*len(self.train_ages.values)).transpose(), columns = ['Age category'])
    """

    # Create data from concatenation of two along x axis
    finalDf = pd.concat([principalDf, target], axis=1)

    """
    #finalDf.index = np.arange(1, len(finalDf) + 1)
    """

    # Установить параметр для вывода всех строк
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Отображение данных на графике
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    """
    #targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    #colors = ['r', 'g', 'b']
    """

    # Alphabeta of classes

    # targets = np.unique(labels)
    targets = set(labels)

    """
    #targets = [0, 1, 2, 3, 4]
    #colors = ['b', 'y','r','g','c']
    """

    # Взять из палитры len(targets) цветов.
    colors = plt.get_cmap('tab10', len(targets)).colors  # Используем палитру 'tab10'

    """
    #for class_ in targets:
    #    colors.append(wc.rgb_to_hex((int(255 * class_ / len (targets)), int(255 * class_ / len (targets)), int(255 * class_ / len (targets)))))
    """
    print(targets)

    for target, color in zip(targets, colors):

        # Select all indexes of humans with targeting classes

        indicesToKeep = finalDf['Age category'] == target

        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c=color
                   , s=50)

        if (show_indexes):
            # Добавляем метки с номерами объектов рядом с точками
            for i in finalDf[indicesToKeep].index:
                ax.text(finalDf.loc[i, 'principal component 1'],
                        finalDf.loc[i, 'principal component 2'],
                        finalDf.loc[i, 'principal component 3'],
                        str(i),  # Здесь str(i) будет выводить номер объекта (индекс)
                        fontsize=9, color='black')

        """
        if (show_ages):
            # Добавляем метки с возрастами объектов рядом с точками
            for i in finalDf[indicesToKeep].index:
                 ax.text(finalDf.loc[self.test_ages[i], 'principal component 1'],
                        finalDf.loc[self.test_ages[i], 'principal component 2'],
                        finalDf.loc[self.test_ages[i], 'principal component 3'],
                str(i),  # Здесь str(i) будет выводить номер объекта (индекс)
                fontsize=9, color='black')
        """

        """
        # Mark the center of each fuzzy cluster
        if centers is not None:

            for pt in centers:
                pca_pt = self.pca(pt, 3)
                ax.plot(pca_pt[0, 0], pca_pt[0, 1], pca_pt[0, 2], 'rs')

        """

    """
    for i, point in enumerate(features_set):
        # Определение цвета на основе принадлежности кластерам
        color = np.dot(u[i],
                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # RGB на основе степеней принадлежности
        ax.plot(point[0], point[1], marker='o', markersize=5, color=color)

    # Отображение центров кластеров
    ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='black', label='Кластерные центры')
    plt.legend()
    """

    # Генерируем подписи для каждого класса с использованием list comprehension
    if len(targets) > 1:
        labels = [f'{target} class' for target in targets]
        ax.legend(labels)

    """
          for j in range(classes_number):

              # : (двоеточие) — означает, что мы берем все строки (или весь диапазон данных по первой оси).
              plt.plot(data[:, 0][clusters == j], data[:, 1][clusters == j], 'o', label=f'cluster{j}')

          plt.legend()

          """

    # plt.scatter(self.data['MCH'], self.data['MCHC'], c=kmeans.labels_)
    # plt.show()

    ax.grid()

    ax.set_xlim([finalDf['principal component 1'].min(), finalDf['principal component 1'].max()])
    ax.set_ylim([finalDf['principal component 2'].min(), finalDf['principal component 2'].max()])
    ax.set_zlim([finalDf['principal component 3'].min(), finalDf['principal component 3'].max()])

    plt.show()

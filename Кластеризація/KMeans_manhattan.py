import numpy as np

def manhattan_distance(a, b):
        """Вычисление Манхэттенского расстояния между двумя точками."""
        
        return np.sum(np.abs(a - b), axis=1)
        

def initialize_centroids(X, k, kmeans_pp=True):

    if kmeans_pp==False:
        """Случайная инициализация центроидов."""
        np.random.seed(42)  # Для воспроизводимости
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]
    else:
        """Инициализация центров по методу K-means++ с Манхэттенской метрикой."""
        n_samples = X.shape[0]
        centers = []

        # 1. Случайный выбор первого центра
        first_center = X[np.random.randint(0, n_samples)]
        centers.append(first_center)

        # 2. Выбор оставшихся центров
        for _ in range(1, k):
            # Вычисление расстояний до ближайшего уже выбранного центра
            distances = np.min([manhattan_distance(X, center) for center in centers], axis=0)

            # Вероятностное распределение для выбора следующего центра
            probs = distances / np.sum(distances)

            # Выбор следующего центра на основе распределения вероятностей
            next_center = X[np.random.choice(n_samples, p=probs)]
            centers.append(next_center)

        return np.array(centers)


def initialize_centers(X, k):




def update_centroids(X, labels, k):
    """Обновление центроидов как медианы каждой координаты."""
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.median(cluster_points, axis=0)
    return centroids

def assign_labels(X, centroids):
    """Назначение меток для каждой точки на основе ближайшего центроида."""
    distances = np.array([manhattan_distance(X, c) for c in centroids]).T
    return np.argmin(distances, axis=1)

def k_means_manhattan(X, k, max_iters=100, tol=1e-4):
    """Основной цикл алгоритма k-means с Манхэттенской метрикой."""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_labels(X, centroids)
        centroids = update_centroids(X, labels, k)
            
        # Проверка сходимости
        if np.allclose(centroids, old_centroids, atol=tol):
            break
    return centroids, labels

# Пример использования
if __name__ == "__main__":
    # Создаем случайные данные
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 8], [9, 10], [10, 12]])
    k = 2  # Количество кластеров

    centroids, labels = k_means_manhattan(X, k)
    print("Центроиды:", centroids)
    print("Метки кластеров:", labels)

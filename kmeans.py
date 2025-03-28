import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=4, init_method='random', max_iter=100):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.centroids = None
    
    def initialize_centroids(self, X):
        """
        Inicializa los centroides según el método especificado.
        """
        if self.init_method == 'random':
            min_vals, max_vals = X.min(axis=0), X.max(axis=0)
            self.centroids = np.random.uniform(min_vals, max_vals, (self.n_clusters, X.shape[1]))
        elif self.init_method == 'dataset':
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[indices]
        else:
            raise ValueError("Método de inicialización no válido")
    
    def euclidean_distance(self, a, b):
        """
        Calcula la distancia euclidiana entre un punto y todos los centroides.
        """
        return np.linalg.norm(a - b, axis=1)
    
    def fit(self, X):
        """
        Ajusta el modelo K-Means a los datos.
        """
        self.initialize_centroids(X)
        clusters = np.zeros(len(X), dtype=int)
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)
        
        for _ in range(self.max_iter):
            new_clusters = np.zeros(len(X), dtype=int)
            
            for i, x in enumerate(X):
                distances = self.euclidean_distance(x, self.centroids)
                min_dist = np.min(distances)
                closest_clusters = np.where(distances == min_dist)[0]
                
                # Si hay más de un cluster con la misma distancia mínima, elegir el de menor tamaño
                if len(closest_clusters) > 1:
                    chosen_cluster = min(closest_clusters, key=lambda c: cluster_sizes[c])
                else:
                    chosen_cluster = closest_clusters[0]
                
                new_clusters[i] = chosen_cluster
            
            # Recalcular los centroides como la media de los puntos asignados
            for i in range(self.n_clusters):
                points_in_cluster = X[new_clusters == i]
                cluster_sizes[i] = len(points_in_cluster)
                if len(points_in_cluster) > 0:
                    self.centroids[i] = points_in_cluster.mean(axis=0)
            
            # Verificar convergencia
            if np.array_equal(clusters, new_clusters):
                break
            clusters = new_clusters
        
        self.labels_ = clusters
         # Visualización
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis', s=10)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100, label='Centroides')
        plt.xlabel("Eixo X")
        plt.ylabel("Eixo Y")
        plt.title("Clusters obtidos con K-Means")
        plt.legend()
        plt.savefig('clusters.png')
        plt.show()
    


if __name__ == '__main__':
    # Generación de datos sintéticos
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=4, init_method='random', max_iter=100)
    kmeans.fit(X)
    labels = kmeans.labels_

   
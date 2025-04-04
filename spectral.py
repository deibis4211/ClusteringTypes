import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class SpectralClusteringModel:
    def __init__(self, n_clusters=2, sigma=1.0):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.labels = None

    def fit(self, X, feature_names):
        """Ajusta el modelo de clustering espectral a los datos"""
        # Paso 1: Construir la matriz de afinidad (grafo de similitud)
        affinity_matrix = self._compute_affinity_matrix(X)

        # Paso 2: Calcular la Laplaciana del grafo
        laplacian = self._compute_laplacian(affinity_matrix)

        # Paso 3: Calcular los vectores propios
        _, eigenvectors = np.linalg.eigh(laplacian)

        # Seleccionar los primeros n_clusters vectores propios
        selected_eigenvectors = eigenvectors[:, :self.n_clusters]

        # Paso 4: Aplicar K-Means a los vectores propios seleccionados
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(selected_eigenvectors)

        self.plot_clusters(X, feature_names)

    def fit_predict(self, X, feature_names):
        """Ajusta el modelo y devuelve las etiquetas predichas"""
        self.fit(X, feature_names)
        return self.labels

    def plot_clusters(self, X, feature_names):
        """Método para graficar los clusters"""
        if self.labels is None:
            raise ValueError("Ejecuta 'fit' o 'fit_predict' antes de graficar")

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            cluster_points = X[self.labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label + 1}', s=30)

        plt.title("Spectral Clustering")
        # Usar los nombres de las características si se proporcionan
        print(f"Feature names: {feature_names}")
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def _compute_affinity_matrix(self, X):
        """Calcula la matriz de afinidad usando un kernel Gaussiano"""
        n_samples = X.shape[0]
        affinity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    distance = np.linalg.norm(X[i] - X[j])
                    affinity_matrix[i, j] = np.exp(-distance**2 / (2 * self.sigma**2))
        return affinity_matrix

    def _compute_laplacian(self, affinity_matrix):
        """Calcula la matriz Laplaciana normalizada"""
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        laplacian = degree_matrix - affinity_matrix
        return laplacian

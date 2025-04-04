import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, points, affinity):
        self.points = points  # Aquí `points` será una lista de índices
        self.affinity = affinity

    def distance(self, other, linkage):
        """Calcula la distancia entre este clúster y otro clúster."""
        if self.affinity == 'euclidean':
            distances = [
                np.linalg.norm(np.array(p1) - np.array(p2))
                for p1 in self.points
                for p2 in other.points
            ]
        else:
            raise ValueError(f"Affinity '{self.affinity}' no está soportada.")

        if linkage == 'single':
            return min(distances)  # Distancia mínima (Single Linkage)
        elif linkage == 'complete':
            return max(distances)  # Distancia máxima (Complete Linkage)
        elif linkage == 'average':
            return np.mean(distances)  # Promedio de distancias (Average Linkage)
        else:
            raise ValueError(f"Linkage '{linkage}' no está soportado.")

    def merge(self, other):
        """Fusiona dos clústeres y devuelve un nuevo clúster."""
        new_points = self.points + other.points
        return Cluster(new_points, self.affinity)


class agglomerative_clustering:
    def __init__(self, n_clusters=8, affinity='euclidean', linkage='single'):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.clusters = None
        self.labels = None

    def fit(self, X, feature_names):
        # Inicializar los clústeres con índices en lugar de puntos
        self.clusters = [Cluster([i], self.affinity) for i in range(len(X))]
        while len(self.clusters) > self.n_clusters:
            closest_clusters = self._find_closest_clusters(X)
            self._merge_clusters(*closest_clusters)

        # Asignar etiquetas a los puntos
        self.labels = np.zeros(len(X), dtype=int)
        for i, c in enumerate(self.clusters):
            for idx in c.points:  # `c.points` ahora contiene índices
                self.labels[idx] = i

        self.plot_clusters(X, feature_names)  # Llamar al método para graficar los clústeres

    def _find_closest_clusters(self, X):
        closest_distance = float('inf')
        closest_clusters = None
        for i, c1 in enumerate(self.clusters):
            for j, c2 in enumerate(self.clusters):
                if i == j:
                    continue
                distance = c1.distance(c2, self.linkage)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_clusters = (i, j)
        return closest_clusters

    def _merge_clusters(self, i, j):
        new_cluster = self.clusters[i].merge(self.clusters[j])
        del self.clusters[j]
        self.clusters[i] = new_cluster

    def fit_predict(self, X, feature_names=None):
        self.fit(X, feature_names)
        return self.labels

    def plot_clusters(self, X, feature_names=None):
        """Método para graficar los clústeres."""
        if self.labels is None:
            raise ValueError("Debe ejecutar 'fit' o 'fit_predict' antes de graficar.")

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(self.labels)  # Obtener etiquetas únicas
        for label in unique_labels:
            cluster_points = X[self.labels == label]  # Filtrar puntos del clúster
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label + 1}', s=30)

        plt.title("Agglomerative Clustering")
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

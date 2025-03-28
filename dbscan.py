import numpy as np
import matplotlib.pyplot as plt


class DBScan:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.clusters = None
        self.visited = None
        self.noise = None
        self.data = None

    def fit(self, data, feature_names=None):
        self.data = np.array(data)
        self.clusters = [0] * len(data)
        self.visited = [False] * len(data)
        self.noise = []
        c = 0
        for i in range(len(data)):
            if not self.visited[i]:
                self.visited[i] = True
                neighbours = self.region_query(i)
                if len(neighbours) < self.min_pts:
                    self.noise.append(i)
                else:
                    c += 1
                    self.expand_cluster(i, neighbours, c)
        self.plot_clusters(feature_names)  # Llamar al método para graficar los clústeres
        return self.clusters

    def region_query(self, i):
        neighbours = []
        for j in range(len(self.data)):
            if np.linalg.norm(self.data[i] - self.data[j]) < self.eps:
                neighbours.append(j)
        return neighbours

    def expand_cluster(self, i, neighbours, c):
        self.clusters[i] = c
        j = 0
        while j < len(neighbours):
            n = neighbours[j]
            if not self.visited[n]:
                self.visited[n] = True
                neighbours_ = self.region_query(n)
                if len(neighbours_) >= self.min_pts:
                    neighbours += neighbours_
            if self.clusters[n] == 0:
                self.clusters[n] = c
            j += 1

    def plot_clusters(self, feature_names=None):
        """Método para graficar los clústeres."""
        clusters = np.array(self.clusters)
        unique_clusters = set(clusters)

        plt.figure(figsize=(8, 6))
        for cluster in unique_clusters:
            if cluster == 0:  # Puntos de ruido
                color = 'k'
                label = 'Noise'
            else:
                color = plt.cm.jet(float(cluster) / max(unique_clusters))  # Colores únicos por clúster
                label = f'Cluster {cluster}'

            cluster_points = self.data[clusters == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=label, s=30)

        plt.title("DBSCAN Clustering")
        # Usar los nombres de las características si se proporcionan
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
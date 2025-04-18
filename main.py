import dbscan
import agglomerative
import spectral
import kmeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def import_iris(type="normal"):
    from sklearn import datasets
    if type == "normal":
        return datasets.load_iris()
    elif type == "pca":
        iris = datasets.load_iris()
        pca = PCA(n_components=2)
        data = pca.fit_transform(iris.data)
        return {"data": data, "target": iris.target}  # Devuelve datos y etiquetas

def evaluate_clustering(true_labels, predicted_labels, method_name):
    """Evalúa el rendimiento del clustering usando ARI y NMI."""
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f"Evaluación para {method_name}:")
    print(f"  Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi_score:.4f}\n")


def call_dbscan(data, feature_names, true_labels, class_labels):
    dbscan_model = dbscan.DBScan(eps=0.5, min_pts=5)
    predicted_labels = dbscan_model.fit(data, feature_names, class_labels)
    evaluate_clustering(true_labels, predicted_labels, "DBSCAN")


def call_agglomerative(data, feature_names, true_labels, class_labels):
    agglomerative_model = agglomerative.agglomerative_clustering(n_clusters=3, affinity='euclidean', linkage='average')
    predicted_labels = agglomerative_model.fit_predict(data, feature_names, class_labels)
    evaluate_clustering(true_labels, predicted_labels, "Agglomerative Clustering")


def call_spectral(data, feature_names, true_labels, class_labels):
    spectral_model = spectral.SpectralClusteringModel(n_clusters=3)
    predicted_labels = spectral_model.fit_predict(data, feature_names, class_labels)
    evaluate_clustering(true_labels, predicted_labels, "Spectral Clustering")

def call_kmeans(data, feature_names, true_labels):
    kmeans_model = kmeans.KMeans(n_clusters=3, init_method='random', max_iter=100)
    predicted_labels = kmeans_model.fit(data)
    evaluate_clustering(true_labels, predicted_labels, "K-Means Clustering")



if __name__ == '__main__':
    # Problema con el PCA arreglado :)
    type = "pca"
    iris = import_iris(type)
    class_labels = {0: "Setosa", 1: "Versicolor", 2: "Viriginica"}
    if type == "normal":
        data = iris.data  # Extraer las características del conjunto de datos
        true_labels = iris.target
        feature_names = iris.feature_names
    elif type == "pca":
        data = iris["data"]  # Extraer las características transformadas por PCA
        true_labels = iris["target"]
        feature_names = ["PCA1", "PCA2"]

    # Llamadas a las funciones
    call_dbscan(data, feature_names, true_labels, class_labels)
    call_agglomerative(data, feature_names, true_labels, class_labels)
    call_spectral(data, feature_names, true_labels, class_labels)
    call_kmeans(data, feature_names, true_labels)

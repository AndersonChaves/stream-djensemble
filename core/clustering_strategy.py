import numpy as np
import core.categorization as ct
from sklearn.cluster import Birch

class ClusteringStrategy():
    def __init__(self, clustering_technique, embedding_method, clustering_algorithm = "birch"):
        self.birch = Birch(branching_factor=3, n_clusters=None, threshold=1)
        self.clustering_technique = clustering_technique
        self.embedding_method = embedding_method
        self.clustering_frame = None

    def initialize_clustering(self, dataset):
        return self.update_clustering(dataset, self.embedding_method)

    def update_clustering(self, dataset, embedding_method):
        emb_list = ct.get_embedded_series_representation(dataset, method=embedding_method)
        ct.normalize_embedding_list(emb_list)
        clustering = self.cluster_using_birch(emb_list)
        return emb_list, clustering

    def cluster_using_birch(self, clustering_items: np.array):
        self.birch = self.birch.partial_fit(clustering_items)
        clustering = self.birch.predict(clustering_items)
        return clustering

    def get_clustering(self):
        try:
            assert(type(self.clustering_frame) != type(None))
        except:
            raise("Clustering Strategy Error: Clustering strategy not initialized")
        return self.clustering_frame
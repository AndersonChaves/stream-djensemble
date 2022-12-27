import numpy as np
import core.categorization as ct
from sklearn.cluster import Birch

class ClusteringStrategy():
    def __init__(self, clustering_algorithm = "birch", clustering_technique = "yolo"):
        self.birch = Birch(branching_factor=3, n_clusters=None, threshold=1)
        self.clustering_frame = None

    def initialize_clustering(self, dataset):
        return self.update_clustering(dataset)

    def update_clustering(self, dataset):
        gld_list = ct.get_gld_series_representation_from_dataset(dataset)
        #gld_list = np.rint(np.random.rand(dataset.shape[1] * dataset.shape[2], 4) * 10) # --todo

        ct.normalize_embedding_list(gld_list)
        clustering = self.cluster_using_birch(gld_list)

        return gld_list, clustering

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
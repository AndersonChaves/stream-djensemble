import numpy as np
from core.config_manager import ConfigManager
from core.clustering_strategy import ClusteringStrategy
import core.categorization as ct
import core.utils as ut
import time

class ContinuousQuery(ConfigManager):
    def __init__(self, config_file_path, query_id):
        super().__init__(config_file_path)
        if "query_region" in self.config_parameters.keys():
            d = eval(self.config_parameters["query_region"])
            self.config_parameters["x1"] = d["lat"][0], d["long"][0]
            self.config_parameters["x2"] = d["lat"][1], d["long"][1]
            self.x1, self.x2 = self.get_config_value("x1"), self.get_config_value("x2")
            self.query_shape = self.x2[0] - self.x1[0], self.x2[1] - self.x1[1]

        self.temporal_models_path = self.config_parameters["temporal_models_path"]
        self.convolutional_models_path = self.config_parameters["convolutional_models_path"]

        self.query_id = query_id
        self.clustering_frame = None
        self.clustering_strategy = None
        self.tiling_is_updated = False
        self.rmse_history = []

    def get_query_endpoints(self):
        self.x1, self.x2 = self.get_config_value("x1"), self.get_config_value("x2")
        return self.x1, self.x2

    def get_candidate_models_list(self):
        temporal_models_names = ut.get_names_of_models_in_dir(self.temporal_models_path)
        convolutional_models_names = ut.get_names_of_models_in_dir(self.convolutional_models_path)
        return temporal_models_names + convolutional_models_names

    def is_clustering_initialized(self):
        if self.clustering_strategy == None:
            return False
        return True

    def initialize_clustering(self, dataset, clustering_method):
        self.clustering_strategy = ClusteringStrategy("birch", clustering_method)

        embedding, clustering = self.clustering_strategy.initialize_clustering(dataset)
        _, lat, long = dataset.shape
        self.embedding_frame = np.reshape(embedding, (lat, long, embedding.shape[-1]))
        self.clustering_frame = np.reshape(clustering, newshape=(lat, long))
        self.tiling_is_updated = False

    def update_clustering(self, dataset):
        benchmarks = {}
        method = self.get_config_value("embedding_method")
        start_clustering_time = time.time()
        if not self.is_clustering_initialized():
            self.initialize_clustering(dataset, clustering_method=method)
        else:
            embedding, clustering = self.clustering_strategy.update_clustering(dataset, method)
            _, lat, long = dataset.shape
            self.intracluster_variance = self.__calculate_intracluster_variance(embedding, clustering)
            self.embedding_frame = np.reshape(embedding, (lat, long, embedding.shape[-1]))  # Number of GLD parameters = 4
            self.clustering_frame = np.reshape(clustering, newshape=(lat, long))

        benchmarks["LOCAL_CLUSTERING_METHOD"] = method
        benchmarks["CLUSTERING_TIME"] = time.time() - start_clustering_time

        benchmarks["N_CLUSTERS"] = len(np.unique(self.clustering_frame))
        start_tilinging_time = time.time()
        self.perform_tiling(dataset)
        benchmarks["TILING_TIME"] = time.time() - start_tilinging_time
        benchmarks["TILING_METHOD"] = self.config_parameters["tiling_method"]
        benchmarks["N_TILES"] = len(self.get_tiling_metadata().keys())
        self.tiling_is_updated = True
        return benchmarks

    def get_current_clustering(self):
        if self.is_clustering_initialized():
            return self.clustering_frame
        else:
            raise Exception("Query Error: Clustering is not initialized.")

    def get_current_series_embedding(self):
        if self.is_clustering_initialized():
            return self.embedding_frame
        else:
            raise("Query Error: Clustering is not initialized.")

    def perform_tiling(self, dataset): # Returns a dict with information about tiling: bounds and centroid
        series_embedding = self.get_current_series_embedding()
        clustering = self.get_current_clustering()
        p = eval(self.config_parameters["min_tiling_purity_rate"])
        tiling_method = self.config_parameters["tiling_method"]
        self.tiling, self.tiling_metadata = ct.perform_tiling(dataset,
                              series_embedding, clustering, method=tiling_method, min_purity_rate=p)
        self.tiling_is_updated = True

    def get_tiling_metadata(self) -> dict:
        return self.tiling_metadata

    def get_tiling(self):
        return self.tiling

    def set_predicted_series(self, predicted_series):
        self.predicted_series = predicted_series

    def get_predicted_series(self):
        try:
            return self.predicted_series
        except:
            raise("Query error: Query has not been executed yet")

    def update_rmse(self, real_next_frame):
        predicted = self.get_predicted_series()[-1]
        query_rmse = np.average(np.sqrt(np.square(predicted - real_next_frame)))
        self.rmse_history.append(query_rmse)
        self.config_parameters["Error History"] = str(self.rmse_history)
        self.config_parameters["Average RMSE"] = str(sum(self.rmse_history) / len(self.rmse_history))

    def get_current_number_of_tiles(self):
        return len(self.get_tiling_metadata().keys())

    def get_error_history(self, type="rmse"):
        if type == "rmse":
            return self.rmse_history

    def __calculate_intracluster_variance(self, embedding, clustering):
        pass
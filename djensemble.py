import numpy as np
import time

import core.categorization as categorization
from core.config_manager import ConfigManager
from core.query_manager import QueryManager
from core.dataset_manager import DatasetManager
from core.cluster_manager import ClusterManager
from core.models_manager import ModelsManager
import core.view

MAXIMUM_COST = 999999

class DJEnsemble:
    # --------------------------------------------------------------
    # Main High Level Functions ------------------------------------
    # --------------------------------------------------------------

    def __init__(self, config_manager: ConfigManager, notifier_list: list = None):
        self.query_manager = None
        self.figures_directory = "figures/"
        self.config_manager = config_manager
        self.notifier_list = [] if notifier_list is None else notifier_list
        self.start_modules()

    # --------------------------------------------------------------
    # Model initialization functions -------------------------------
    # --------------------------------------------------------------
    def start_modules(self):
        self.start_query_manager()
        self.start_dataset_manager()
        self.start_models_manager()
        self.start_cluster_manager()

    def start_config_manager(self, config_file_path):
        self.config_manager = ConfigManager(config_file_path)

    def start_query_manager(self):
        if self.config_manager == None:
            raise ("Query Manager Initialization Error: configurations not defined")
        query_dir = self.config_manager.get_config_value("query_directory")
        self.query_manager = QueryManager(query_dir)

    def start_dataset_manager(self):
        if self.config_manager == None:
            raise ("Dataset Manager Initialization Error: No configurations defined")
        ds_path = self.config_manager.get_config_value("dataset_path")
        self.dataset_manager = DatasetManager(ds_path)
        self.dataset_manager.loadDataset(ds_attribute=self.config_manager.get_config_value("target_attribute"))

    def start_models_manager(self):
        if self.config_manager == None:
            raise ("Models Manager Initialization Error: No configurations defined")
        self.models_manager = ModelsManager(self.config_manager)

    def start_cluster_manager(self):
        if self.config_manager == None:
            raise ("Cluster Manager Initialization Error: No configurations defined")
        self.cluster_manager = ClusterManager(self.config_manager)

    # --------------------------------------------------------------
    # Main Steps ---------------------------------------------------
    # --------------------------------------------------------------

    def run_offline_step(self):
        self.log("Executing offline stage... ")
        self.start_offline = time.time()
        noise_level_for_cef = eval(self.config_manager.get_config_value("noise_level_for_cef"))
        self.update_cost_estimation_function(noise_level_for_cef)
        self.initialized = True
        self.log("End of offline stage, total time: " + str(time.time() - self.start_offline))

    def run_online_step(self, single_iteration=True):
        t_start = eval(self.config_manager.get_config_value("dataset_start"))
        clustering_offset = eval(self.config_manager.get_config_value("clustering_offset"))
        window_size = eval(self.config_manager.get_config_value("window_size"))
        predictive_length = eval(self.config_manager.get_config_value("predictive_length"))
        self.clustering_behavior = self.config_manager.get_config_value("series_clustering_behavior")
        self.start_online = time.time()
        self.log("Measuring online statistics... ")
        data_stream_active = True
        self.t_current = t_start
        self.data_buffer = np.empty(self.get_buffer_shape())
        while data_stream_active:
            self.log("--- Reading new Record t: " + str(self.t_current))
            data_matrix = self.read_record(self.t_current)
            self.data_buffer = np.concatenate((self.data_buffer, np.expand_dims(data_matrix, axis=0)), axis=0)
            self.update_frame_visualization()
            if len(self.data_buffer) >= window_size:
                self.update_clusters(self.data_buffer)
                self.perform_continuous_queries(self.query_manager, self.data_buffer,
                                                self.models_manager, self.dataset_manager)
                self.evaluate_error(self.read_record(self.t_current+1), self.query_manager)
                self.update_window_visualization()
                self.data_buffer = np.empty(self.get_buffer_shape())
                if single_iteration:
                    data_stream_active = False
            self.t_current += 1
        self.log("Iteration took " + str(time.time() - self.start_online) + "secs")

    def get_buffer_shape(self):
        return (0,) + self.dataset_manager.get_spatial_shape()

    # --------------------------------------------------------------
    # Offline Functions --------------------------------------------
    # --------------------------------------------------------------
    def update_cost_estimation_function(self, noise_level_for_cef):
        self.log("Updating Cost Estimation Function")
        self.models_manager.update_cef(noise_level_for_cef)

    # --------------------------------------------------------------
    # Online Functions ---------------------------------------------
    # --------------------------------------------------------------
    def categorize_dataset(self, dataset: np.array):
        return categorization.categorize_dataset(dataset)

    def read_record(self, t_current):
        return self.dataset_manager.read_instant(t_current)

    def update_clusters(self, data_buffer: np.array):
        if self.cluster_manager.clustering_behavior == 'query':
            self.cluster_manager.update_clustering_local(data_buffer, self.query_manager)

    def perform_continuous_queries(self, query_manager, data_buffer, models_manager, dataset_manager):
        query_manager.execute_queries(data_buffer, models_manager, dataset_manager)
        pass

    def evaluate_error(self, real_next_frame, query_manager: QueryManager):
        for query_id in self.query_manager.get_all_query_ids():
            continuous_query = self.query_manager.get_continuous_query(query_id)
            query_next_frame = self.dataset_manager.filter_frame_by_query_region(real_next_frame, continuous_query)
            continuous_query.update_rmse(query_next_frame)

    def set_config_manager(self, config_manager):
        self.config_manager = config_manager

    def log(self, msg):
        for notifier in self.notifier_list:
            notifier.notify(msg)

    def update_window_visualization(self):
        query_ids = list(self.query_manager.get_all_query_ids())
        continuous_query = self.query_manager.get_continuous_query(query_ids[0])
        self.log("Generating visualization: clustering")
        clustering = continuous_query.get_current_clustering()
        core.view.save_figure_from_matrix(clustering, "clustering",
                                          parent_directory = 'figures/', write_values = True)

        self.log("Generating visualization: tiling")
        tiling = continuous_query.get_tiling()
        core.view.save_figure_from_matrix(tiling, "tiling",
                                          parent_directory='figures/', write_values=True)

        self.log("Generating visualization: current frame")
        core.view.save_figure_from_matrix(self.data_buffer[-1],
                                          self.figures_directory +"current-frame")

        real_next_frame = self.read_record(self.t_current+1)
        query_next_frame = self.dataset_manager.filter_frame_by_query_region(real_next_frame, continuous_query)

        self.log("Generating visualization: predicted frame")
        core.view.save_figure_from_matrix(continuous_query.get_predicted_series()[-1],
                                          self.figures_directory + "predicted-frame",
                                          write_values=True)

        self.log("Generating visualization: next real frame")
        core.view.save_figure_from_matrix(query_next_frame,
                                          self.figures_directory + "next-frame-for-query",
                                          write_values=True)

        self.log("Generating visualization: error graph")
        core.view.plot_graph_line(continuous_query.get_error_history(),
                                  self.figures_directory + "rmse-graph")

        core.view.save_figure_from_matrix(query_next_frame,
                                          self.figures_directory + "next-frame-for-query",
                                          write_values=True)
        self.log("Images Updated")


    def update_frame_visualization(self):
        self.log("Generating visualization: current frame")
        core.view.save_figure_from_matrix(self.data_buffer[-1],
                                          self.figures_directory + "current-frame")

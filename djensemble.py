import numpy as np
import time, datetime
from core.config_manager import ConfigManager
from core.query_manager import QueryManager
from core.dataset_manager import DatasetManager
from core.cluster_manager import ClusterManager
from core.models_manager import ModelsManager
import core.view
import core.utils as ut

MAXIMUM_COST = 999999

class DJEnsemble:
    # --------------------------------------------------------------
    # Main High Level Functions ------------------------------------
    # --------------------------------------------------------------
    def __init__(self, config_manager: ConfigManager, notifier_list: list = None,
                     results_directory = ""):
        self.query_manager = None
        self.results_directory = results_directory
        self.figures_directory = self.results_directory + "figures/"
        ut.create_directory_if_not_exists(self.results_directory)
        ut.create_directory_if_not_exists(self.figures_directory)
        self.config_manager = config_manager
        self.notifier_list = [] if notifier_list is None else notifier_list
        self.start_modules()
        self.start_time = str(datetime.datetime.now())

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
        self.query_manager = QueryManager(query_dir, self.notifier_list)

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
        self.cluster_manager = ClusterManager(self.config_manager, notifier_list=self.notifier_list)

    # --------------------------------------------------------------
    # Main Steps ---------------------------------------------------
    # --------------------------------------------------------------

    def run_offline_step(self):
        self.log("Executing offline stage... ")

        self.log("Calculating Cost Estimation Functions... ")
        self.start_cef = time.time()
        noise_level_for_cef = eval(self.config_manager.get_config_value("noise_level_for_cef"))
        self.update_cost_estimation_function(noise_level_for_cef)
        self.log("CEFs Calculated, total time: " + str(time.time() - self.start_cef))

        if self.config_manager.get_config_value("clusterization_mode") == 'static':
            static_clusterization_range = eval(self.config_manager.get_config_value("static_clusterization_range"))
            self.log("Performing Global Clustering and Tiling: Frame" + str(0) + " to " + str(static_clusterization_range))
            self.start_global_clustering = time.time()
            # PERFORM STATIC CLUSTERING - GLOBAL
            if self.config_manager.get_config_value("clusterization_mode") == 'static':
                clusterization_window = self.dataset_manager.read_window(0, static_clusterization_range)
                self.cluster_manager.perform_global_static_clustering(clusterization_window, label="time0to" + str(static_clusterization_range))
                self.cluster_manager.perform_global_static_tiling(
                    self.dataset_manager.read_window(0, static_clusterization_range))
                self.log("Global Clustering and Tiling Performed, total time: " + str(
                    time.time() - self.start_global_clustering))
                self.log("Number of Tiles: " + str(self.cluster_manager.get_current_number_of_tiles()))
            else:
                # PERFORM STATIC CLUSTERING - LOCAL
                print("Error: Local static clustering not yet implemented")
                raise(Exception("Error: Local static clustering not yet implemented"))


        self.initialized = True
        self.log("End of offline stage, total time: " + str(time.time() - self.start_cef))

    def run_online_step(self, single_iteration=True, t_start=-1):
        if t_start == -1:
            t_start = eval(self.config_manager.get_config_value("dataset_start"))
        window_size = eval(self.config_manager.get_config_value("window_size"))
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

            start_frame_visualization = time.time()
            self.log("Time generating visualization: " + str(time.time()-start_frame_visualization))

            if len(self.data_buffer) >= window_size:
                #self.update_frame_visualization()
                if self.config_manager.get_config_value("clusterization_mode") == 'dynamic':
                    start_clusterization = time.time()
                    self.update_clusters_online(self.data_buffer)
                    self.log("Time for Clusterization: " + str(time.time() - start_clusterization))

                self.perform_continuous_queries(self.query_manager, self.data_buffer,
                                                self.models_manager, self.dataset_manager)
                self.evaluate_error(self.read_record(self.t_current+1), self.query_manager)
                #if self.t_current % 200 == 0:
                #self.update_window_visualization()
                self.data_buffer = np.empty(self.get_buffer_shape())
                if single_iteration:
                    data_stream_active = False
            self.t_current += 1
        self.log("Iteration took " + str(time.time() - self.start_online) + "secs")
        self.save_configurations()

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

    def read_record(self, t_current):
        return self.dataset_manager.read_instant(t_current)

    def update_clusters_online(self, data_buffer: np.array):
        if self.cluster_manager.clustering_behavior == 'local':
            self.cluster_manager.update_clustering_local(data_buffer, self.query_manager)
        else:
            self.cluster_manager.update_global_clustering(data_buffer) # Not tested yet

    def perform_continuous_queries(self, query_manager, data_buffer, models_manager, dataset_manager):
        self.log("Executing query: Window Allocation t=" + str(self.t_current))
        query_manager.execute_queries(data_buffer, models_manager, dataset_manager, self.cluster_manager)

    def evaluate_error(self, real_next_frame, query_manager: QueryManager):
        for query_id in self.query_manager.get_all_query_ids():
            continuous_query = self.query_manager.get_continuous_query(query_id)
            x1, x2 = continuous_query.get_query_endpoints()
            query_next_frame = self.dataset_manager.filter_frame_by_query_region(real_next_frame, x1, x2)
            continuous_query.update_rmse(query_next_frame)

            self.log("Query error updated: " + str(query_id) +
                     str(continuous_query.config_parameters["Error History"]))
            self.log("Average RMSE: " + str(query_id) +
                     str(continuous_query.config_parameters["Average RMSE"]))


    def set_config_manager(self, config_manager):
        self.config_manager = config_manager

    def log(self, msg):
        for notifier in self.notifier_list:
            notifier.notify(msg)

    def update_window_visualization_for_clustering_and_tiling(self, continuous_query):
        self.log("Generating visualization: clustering")
        if self.cluster_manager.is_global_clustering():
            clustering = self.cluster_manager.clustering
        else:
            clustering = continuous_query.get_current_clustering()
        if len(clustering) > 10:
           f_size = 1
        else:
           f_size = 15
        core.view.save_figure_from_matrix(clustering, "clustering",
                                         parent_directory = self.figures_directory,
                                         write_values = True, font_size = f_size)

        self.log("Generating visualization: tiling")
        if self.cluster_manager.is_global_clustering():
            tiling = self.cluster_manager.tiling
        else:
            start_tiling = time.time()
            tiling = continuous_query.get_tiling()
        core.view.save_figure_from_matrix(tiling, "tiling",
                                         parent_directory=self.figures_directory,
                                         write_values=True, font_size = f_size)

        self.log("Generating visualization: current frame")
        core.view.save_figure_from_matrix(self.data_buffer[-1],
                                         self.figures_directory +"current-frame")

    def update_window_visualization(self):
        query_ids = list(self.query_manager.get_all_query_ids())
        continuous_query = self.query_manager.get_continuous_query(query_ids[0])
        #if self.config_manager.get_config_value("clusterization_mode") == 'dynamic':
        #    self.update_window_visualization_for_clustering_and_tiling(continuous_query)

        real_next_frame = self.read_record(self.t_current+1)
        x1, x2 = continuous_query.get_query_endpoints()
        query_next_frame = self.dataset_manager.filter_frame_by_query_region(real_next_frame, x1, x2)

        self.log("Generating visualization: predicted frame")
        core.view.save_figure_from_matrix(continuous_query.get_predicted_series()[-1],
                                          self.figures_directory + "predicted-frame_" + str(self.t_current),
                                          write_values=True)

        self.log("Generating visualization: next real frame")
        core.view.save_figure_from_matrix(query_next_frame,
                                          self.figures_directory + "next-frame-for-query_" + str(self.t_current),
                                          write_values=True)

        self.log("Generating visualization: error graph")
        core.view.plot_graph_line(continuous_query.get_error_history(),
                                  self.figures_directory + "rmse-graph")

        self.log("Images Updated")


    def update_frame_visualization(self):
        self.log("Generating visualization: current frame")
        core.view.save_figure_from_matrix(self.data_buffer[-1],
                                          self.figures_directory + "current-frame")

    def save_configurations(self):
        self.config_manager.save_config_file(self.results_directory + "configurations")
        self.query_manager.save_query_configurations(self.results_directory)
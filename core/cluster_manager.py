from numpy import reshape
from sklearn.cluster import Birch
import core.categorization as ct
import numpy as np
from .config_manager import ConfigManager
from .query_manager import QueryManager
#from config_manager import ConfigManager
#from query_manager import QueryManager
#from dataset_manager import DatasetManager
from .series_generator import SeriesGenerator


class ClusterManager():
    def __init__(self, config_manager: ConfigManager, n_clusters=4):
        self.config_manager = config_manager
        self.clustering_algorithm = config_manager.get_config_value("clustering_algorithm")
        self.embedding_method = config_manager.get_config_value("embedding_method")
        self.clustering_behavior = self.config_manager.get_config_value("series_clustering_behavior")
        if self.clustering_algorithm=="birch":
            #self.birch = Birch(branching_factor=6, n_clusters=2, threshold=15)
            self.birch = Birch(n_clusters=n_clusters)

    def update_clustering_local(self, update_dataset: np.array,
                                query_manager: QueryManager):
        for query_id in query_manager.get_all_query_ids():
            continuous_query = query_manager.get_continuous_query(query_id)
            x1, x2 = continuous_query.get_query_endpoints()
            query_dataset = update_dataset[:, x1[0]:x2[0], x1[1]:x2[1]]
            continuous_query.update_clustering(query_dataset)

    def update_clustering(self, update_dataset: np.array, gld_list=None):
        if gld_list is None:
            gld_list = ct.get_embedded_series_representation(update_dataset, method=self.embedding_method)
            ct.normalize_embedding_list(gld_list)
        self.clustering = self.cluster_glds_using_birch(gld_list)
        return gld_list, self.clustering

    def cluster_glds_using_birch(self, gld_list: np.array):
        self.birch = self.birch.partial_fit(gld_list)
        clustering = self.birch.predict(gld_list)
        return clustering

    def perform_global_clustering(self, data_frame_series: np.array):
        gld_list, clustering = ct.cluster_dataset(data_frame_series)
        return gld_list, clustering

    def clusters_from_series(self, data_series: np.array):
        gld_list = ct.get_gld_series_representation_from_dataset(data_series)
        ct.normalize_embedding_list(gld_list)
        clustering = self.birch.predict(data_series)
        return clustering

def test_prepare_dummy_array():
    # Prepare dummy array
    s1 = np.array([i + 1 for i in range(10)] * 10)
    s2 = np.array([i + 1 if i % 2 == 0 else -i for i in range(10)] * 10)
    dummy_array = np.array((
        (s1, s1, s1, s1, s1),
        (s1, s1, s1, s1, s1),
        (s2, s2, s2, s2, s2),
        (s2, s2, s2, s2, s2),
        (s2, s2, s2, s2, s2)
    ))
    dummy_array = np.moveaxis(dummy_array, 2, 0)
    return dummy_array

def test_perform_global_clustering(dummy_array):
    gld_list, global_clustering = ClusterManager(
        ConfigManager("../experiment-metadata/djensemble-exp1.config")).perform_global_clustering(dummy_array[:10])
    return global_clustering

def test_prepare_dataset_for_online_clustering(dummy_array):
    # -- Perform online continuous Clustering
    # Turing dataset into windows
    frame_window_series, _ = SeriesGenerator().split_series_into_tumbling_windows(
        dummy_array, 10, n_steps_out=0)

    return frame_window_series

def test_dummy_array():
    dummy_array = test_prepare_dummy_array()
    # global_clustering = test_perform_global_clustering(dummy_array)
    frame_window_series = test_prepare_dataset_for_online_clustering(dummy_array)
    local_cls_manager = ClusterManager(ConfigManager("../experiment-metadata/djensemble-exp1.config"))

    local_clustering = []
    history_gld_list = np.empty((0, 4))
    for i, fs in enumerate(frame_window_series):
        gld_list, local_clustering = local_cls_manager.update_clustering(fs)

        history_gld_list = np.concatenate((history_gld_list, gld_list), axis=0)
        history_clustering = local_cls_manager.birch.predict(history_gld_list)

        x = history_gld_list[:, 0]
        y = history_gld_list[:, 2]
        view.save_figure_as_scatter_plot(x, y, history_clustering, "clustering-" + str(i))
        view.save_figure_from_matrix(np.reshape(local_clustering, (5, 5)), "matrix-" + str(i))

        # # Annotate gld series positions
        # for j, point in enumerate(gld_list):
        #     plt.annotate(str(j), (point[0], point[1]), fontsize=5)
        # # Print leaves
        #
        # leaves = local_cls_manager.birch.subcluster_centers_
        # plt.ylim(0, 100)
        # plt.xlim(0, 100)
        # plt.scatter(leaves[:, 0], leaves[:, 1], facecolors='none', edgecolors='black')
        #
        # plt.savefig(str(i) + ".png")
        # plt.clf()

    # print((global_clustering - local_clustering).sum()) # noqa
    print("local cl: ", local_clustering)
    print("global cl: ", global_clustering)
    coincidence_1 = [i for i, x in enumerate(local_clustering[:10]) if x in local_clustering[10:]]
    coincidence_2 = [i for i, x in enumerate(local_clustering[10:]) if x in local_clustering[:10]]
    print("1: ", coincidence_1)
    print("2: ", coincidence_2)

    import categorization
    # categorization.print_array(gld_list)

    # print("Intersection: ", inter, "Intersection length", len(inter))
    # print("Group: ", )
    # incorrect = [g for g in local_clustering[10:] if g in group_1]
    # print("Group Repetition: ", incorrect, "Length: ", len(incorrect))

if __name__ == '__main__':
    test_dummy_array()


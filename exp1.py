from pyparsing import unicode_set
import core.utils as ut
from core.config_manager import ConfigManager
from core.cluster_manager import ClusterManager
from core.dataset_manager import DatasetManager
from core import view
import core.categorization as ct
from sklearn.metrics import silhouette_score
import numpy as np
from datetime import datetime
from time import time
from core.utils import create_directory_if_not_exists
import sys

filter_query_region = False

# Global Configurations
global_configurations_path = "experiment-metadata/djensemble-exp1.config"
config_manager = ConfigManager(global_configurations_path)

cur_time = str(datetime.now())
if filter_query_region:
    results_directory = "results/cfsr-static_clustering_dyn_silhouette-10:14/" + cur_time + "/"
else:
    results_directory = "results/cfsr-static_clustering_dyn_silhouette/" + cur_time + "/"
embedding_directory = "results/clustering/"

def save_embedding(embedding, base_directory, dataset_name, embedding_method, time_start, time_end):
    file_name = base_directory + dataset_name + "-" + \
        embedding_method + "-" + \
        "time" + str(time_start) + "to" + str(time_end) + \
        ".embedding.npy"
    np.save(file_name, embedding)

def load_embedding_if_exists(base_directory,
                             dataset_name,
                             embedding_method,
                             time_start, time_end):
    file_name = base_directory + dataset_name + "-" + \
        embedding_method + "-" + \
        "time" + str(time_start) + "to" + str(time_end) + \
        ".embedding.npy"
    embedding = None
    if ut.file_exists(file_name):
        embedding = np.load(file_name)
    return embedding

def calculate_silhouette_through_time(initial_instant, window_size,
                                        file_name, embedding_method, clustering_type = "dynamic",
                                      clustering = None, silhouete_name=""):
    if clustering_type == "static" and clustering is None:
        raise Exception("Clustering is static, but empty")

    history_gld_list = []
    t_instant = initial_instant
    i = 0
    while (t_instant < ((365 * 4) + window_size)):
        with open(file_name, "a") as f:
            start = time()
            window_series = ds_manager.read_window(t_instant, t_instant + window_size)
            if window_series.shape[0] < window_size:
                break
            embedding_list = load_embedding_if_exists(
                base_directory=embedding_directory,
                dataset_name=ds_name,
                embedding_method=embedding_method,
                time_start=0,
                time_end=t_instant + window_size
            )
            if embedding_list is None:
                embedding_list = ct.get_embedded_series_representation(window_series, method=embedding_method)
                save_embedding(
                    embedding_list,
                    base_directory=embedding_directory,
                    dataset_name=ds_name,
                    embedding_method=embedding_method,
                    time_start=t_instant,
                    time_end=t_instant + window_size
                )
                history_gld_list.append(embedding_list)

            local_dynamic_silhouette = silhouette_score(embedding_list, clustering)

            f.write("Frame " + str(t_instant) + ": \n")
            f.write("----Clustering: " + str(clustering) + ": \n")
            f.write(" ".join(str(item) for item in clustering)+ ": \n")
            f.write(silhouete_name + ":" + str(local_dynamic_silhouette) + "\n")
            local_static_clustering_time = str(time() - start)
            f.write("----Silhouette Time: " + str(local_static_clustering_time) + "\n")
            t_instant += window_size
    return history_gld_list

if __name__ == '__main__':
    create_directory_if_not_exists(results_directory)
    sys.argv = ["exp1.py", "parcorr", "1"]
    print(sys.argv)
    if len(sys.argv) < 2:
        raise Exception("Inform the embedding method: gld or parcorr")
    if len(sys.argv) < 3:
        raise Exception("Inform the window size, in weeks (1 to 52)")

    embedding_method = sys.argv[1]
    window_size = int(sys.argv[2]) * 4 * 7

    # Load the Dataset
    ds_path = config_manager.get_config_value("dataset_path")
    ds_name = ds_path.split('/')[-1]
    ds_manager = DatasetManager(ds_path)

    ds_manager.loadDataset(ds_attribute=config_manager.get_config_value("target_attribute"))

    if filter_query_region:
        ds_manager.filter_by_region((10, 14), (10, 14))

    # ******************** FIRST TEST: PERFORM GLOBAL CLUSTERING **************

    start = time()
    all_data = ds_manager.read_all_data()
    embedding_list = load_embedding_if_exists(
                         base_directory=embedding_directory,
                         dataset_name=ds_name,
                         embedding_method=embedding_method,
                         time_start=0,
                         time_end=len(all_data)
                     )
    embedding_list, global_clustering, global_silhouette = ct.cluster_dataset(
                                                                    all_data,
                                                                    embedding_method=embedding_method,
                                                                    series_embedding_matrix=embedding_list
                                                           )
    save_embedding(
        embedding_list,
        base_directory=embedding_directory,
        dataset_name=ds_name,
        embedding_method=embedding_method,
        time_start=0,
        time_end=len(all_data)
    )
    global_number_of_groups = len(set(global_clustering))
    global_clustering_time = str(time() - start)

    file_name = results_directory + "results" + cur_time + ".txt"
    with open(file_name, "w") as f:
        f.write("----Number of Groups: " + str(global_number_of_groups) + "\n")
        f.write("----Global Clustering: " + str(global_clustering) + "\n")
        f.write(" ".join(str(item) for item in global_clustering)+ ": \n")
        f.write("----Global Silhouette Score: " + str(global_silhouette) + "\n")
        f.write("----Global Clustering Time: " + str(global_clustering_time) + "\n")

    with open(file_name, "a") as f:
        f.write("*************Calculating Silhouette Trough Time: Global Clustering" "\n")
    history_embedded_series_representations = calculate_silhouette_through_time(
                        0,
                        embedding_method=embedding_method,
                        window_size=window_size,
                        file_name=file_name,
                        clustering_type="static",
                        clustering=global_clustering,
                        silhouete_name="Silhuette Config 1 Method" + embedding_method
    )

    # ******************** SECOND TEST: PERFORM LOCAL CLUSTERING, ANALYZE SILHOUETE Along windows **************
    silhouete_name = "Silhuette Config 2"
    start = time()
    embedding_list = load_embedding_if_exists(
        base_directory=embedding_directory,
        dataset_name=ds_name,
        embedding_method=embedding_method,
        time_start=0,
        time_end= 28 * 4
    )
    if not (embedding_list is None):
        embedding_list, clustering, local_dynamic_silhouette = ct.cluster_dataset(ds_manager.read_window(0, 28 * 4),
                                                                                  embedding_method=embedding_method,
                                                                                  series_embedding_matrix=embedding_list)
        save_embedding(
            embedding_list,
            base_directory=embedding_directory,
            dataset_name=ds_name,
            embedding_method=embedding_method,
            time_start=0,
            time_end=28*4
        )


    local_static_number_of_groups = len(set(clustering))
    local_static_clustering_time = str(time() - start)

    file_name = results_directory + "results" + cur_time + ".txt"
    with open(file_name, "a") as f:
        f.write("----Frame 0: ""\n")
        f.write("----Number of Groups: " + str(local_static_number_of_groups) + "\n")
        f.write(silhouete_name +": "+ str(clustering) + "\n")
        f.write(" ".join(str(item) for item in clustering)+ ": \n")
        f.write("----Local Initial Dynamic Silhouette Score: " + str(local_dynamic_silhouette) + "\n")
        f.write("----Local Static Clustering Time: " + str(local_static_clustering_time) + "\n")

    t_instant = 0
    it_gld_list = iter(history_embedded_series_representations)
    while (t_instant < ((365 * 4) + window_size)):
        with open(file_name, "a") as f:
            start = time()
            window_series = ds_manager.read_window(t_instant, t_instant+window_size)
            if window_series.shape[0] < window_size:
                break
            #gld_list = ct.calculate_gld_list_from_dataset(window_series)
            embedding_list = load_embedding_if_exists(
                base_directory=embedding_directory,
                dataset_name=ds_name,
                embedding_method=embedding_method,
                time_start=0,
                time_end=t_instant + window_size
            )
            #ct.normalize_gld_list(gld_list)
            local_dynamic_silhouette = silhouette_score(embedding_list, clustering)

            f.write("----STATIC CLUSTERING, DYNAMIC SILHOUETTE Frame " + str(t_instant) + ": \n")
            f.write("----Number of Groups: " + str(local_static_number_of_groups) + "\n")
            f.write("----Clustering: " + str(clustering) + "\n")
            f.write(" ".join(str(item) for item in clustering)+ ": \n")
            f.write(silhouete_name +": " + str(local_dynamic_silhouette) + "\n")
            local_static_clustering_time = str(time() - start)
            f.write("frame:" + str(t_instant) + "----Silhouette Time: " + str(local_static_clustering_time) + "\n")
            t_instant += window_size

    # ******************** THIRD TEST: PERFORM DYNAMIC CLUSTERING, ANALYZE SILHOUETTE Along windows **************
    # Perform local dynamic Clustering
    silhouete_name = "Silh TEST3"
    local_clustering = []
    history_gld_list = np.empty((0, 141, 153, 5))
    t_instant = 0
    it_gld_list = iter(history_embedded_series_representations)
    while(t_instant < ((365*4)+window_size)):
        with open(file_name, "a") as f:
            start = time()
            frame_window_series = ds_manager.read_window(t_instant, t_instant+window_size)
            if frame_window_series.shape[0] < window_size:
                break
            embedding_list = load_embedding_if_exists(
                base_directory=embedding_directory,
                dataset_name=ds_name,
                embedding_method=embedding_method,
                time_start=0,
                time_end=t_instant + window_size
            )
            local_cls_manager = ClusterManager(config_manager, n_clusters=local_static_number_of_groups)
            local_cls_manager.update_global_clustering(frame_window_series, embedding_list)
            embedding_list = local_cls_manager.global_series_embedding
            local_clustering = local_cls_manager.clustering
            f.write("----Test 3: PERFORM DYNAMIC CLUSTERING, ANALYZE SILHOUETTE Along windows\n")
            f.write("----Local Clustering t = " + str(t_instant) + " to " + str(t_instant + window_size) + ": " + "\n")
            f.write(" ".join(str(item) for item in local_clustering) + "\n")
            if np.max(local_clustering) == np.min(local_clustering):
                f.write(silhouete_name +": " + "++++Local Silhouette Score: " + "n/a" + "\n")
            else:
                f.write(
                    silhouete_name + ": " + "++++Local Silhouette Score: " +
                     str(
                         silhouette_score(
                             np.reshape(embedding_list, (-1, embedding_list.shape[-1])),
                             local_clustering.flatten() #np.reshape(local_clustering, (-1, local_clustering.shape[-1]))
                         )
                     ) + "\n"
                )
            f.write("++++Local Clustering Time: " + str(time() - start) + "\n")

            # history_gld_list = np.concatenate((
            #     history_gld_list,
            #     np.reshape(gld_list, (1, *gld_list.shape))
            # ), axis=0)
            # history_clustering = local_cls_manager.birch.predict(history_gld_list[-1])
            #
            # x = history_gld_list[:, 0]
            # y = history_gld_list[:, 2]
            # #view.save_figure_as_scatter_plot(x, y, history_clustering, results_directory + "clustering-" + str(t_instant), annotate=False)
            # print("Generating Figure")
            # if t_instant % (window_size * 4) == 0:
            #   view.save_figure_from_matrix(np.reshape(local_clustering, frame_window_series.shape[1:]),
            #                                results_directory + "matrix-" + str(t_instant))
            t_instant += window_size




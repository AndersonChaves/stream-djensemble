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

# Global Configurations
global_configurations_path = "experiment-metadata/djensemble-exp1.config"
config_manager = ConfigManager(global_configurations_path)

cur_time = str(datetime.now())
results_directory = "results/cfsr-11:16/" + cur_time + "/"

if __name__ == '__main__':
    create_directory_if_not_exists(results_directory)

    # Load the Dataset
    ds_manager = DatasetManager(config_manager.get_config_value("dataset_path"))
    ds_manager.loadDataset(ds_attribute=config_manager.get_config_value("target_attribute"))
    ds_manager.filter_by_region((11, 16), (11, 16))

    # Perform Global Clustering
    start = time()
    gld_list, global_clustering, global_silhouette = ct.cluster_dataset(ds_manager.read_all_data())
    global_number_of_groups = len(set(global_clustering))
    global_clustering_time = str(time() - start)

    # Prepare local cluster manager
    local_cls_manager = ClusterManager(config_manager, n_clusters=global_number_of_groups)

    np.save("global-gld_list-region", gld_list)
    np.save("global-clustering-region", global_clustering)
    print("----Global Clustering: ", global_clustering)
    print("----Global Silhouette Score: ", global_silhouette)
    print("----Global Clustering Time: " + global_clustering_time)
    np.save(results_directory + "gld_list-full", gld_list)
    np.save(results_directory + "clustering-full", global_clustering)


    file_name = results_directory + "results" + cur_time + ".txt"
    with open(file_name, "w") as f:
        f.write("----Number of Groups: " + str(global_number_of_groups) + "\n")
        f.write("----Global Clustering: " + str(global_clustering) + "\n")
        f.write(" ".join(str(item) for item in global_clustering))
        f.write("----Global Silhouette Score: " + str(global_silhouette) + "\n")
        f.write("----Global Clustering Time: " + str(global_clustering_time) + "\n")

    # Perform Local Clustering
    local_clustering = []
    history_gld_list = np.empty((0, 4))
    t_instant = 0
    while(t_instant < ((365*4)+28)):
        with open(file_name, "a") as f:
            start = time()
            frame_window_series = ds_manager.read_window(t_instant, t_instant+28)
            if frame_window_series is not None:
                gld_list, local_clustering = local_cls_manager.update_clustering(frame_window_series)
            else:
                break
            np.save(results_directory + "gld_list-t=" + str(t_instant), gld_list)
            np.save(results_directory + "clustering-t=" + str(t_instant), local_clustering)

            print("----Local Clustering: ", local_clustering)
            print("----Local Silhouette Score: ", silhouette_score(gld_list, local_clustering))
            f.write("----Local Clustering t = " + str(t_instant) + " to " + str(t_instant + 28) + ": " + "\n")
            f.write(" ".join(str(item) for item in local_clustering) + "\n")
            if len(set(local_clustering)) == 1:
                f.write("++++Local Silhouette Score: " + "n/a" + "\n")
            else:
                f.write("++++Local Silhouette Score: " + str(silhouette_score(gld_list, local_clustering)) + "\n")
            f.write("++++Local Clustering Time: " + str(time() - start) + "\n")

            history_gld_list = np.concatenate((history_gld_list, gld_list), axis=0)
            history_clustering = local_cls_manager.birch.predict(history_gld_list)

            x = history_gld_list[:, 0]
            y = history_gld_list[:, 2]
            view.save_figure_as_scatter_plot(x, y, history_clustering, results_directory + "clustering-" + str(t_instant), annotate=False)
            view.save_figure_from_matrix(np.reshape(local_clustering, frame_window_series.shape[1:]), results_directory + "matrix-" + str(t_instant))
            t_instant += 28




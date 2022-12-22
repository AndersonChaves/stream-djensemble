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
results_directory = "results/cfsr-full/"

if __name__ == '__main__':
    # Prepare cluster managers
    local_cls_manager = ClusterManager(config_manager)

    # Load the Dataset
    ds_manager = DatasetManager(config_manager.get_config_value("dataset_path"))
    ds_manager.loadDataset(ds_attribute=config_manager.get_config_value("target_attribute"))
    #ds_manager.filter_by_region((11, 16), (11, 16))

    # Perform Global Clustering
    start = time()
    # global_cls_manager.perform_global_clustering(data_frame_series=ds_manager.read_all_data())
    gld_list, global_clustering, global_silhouette = ct.cluster_dataset(ds_manager.read_all_data())

    np.save("global-gld_list-region", gld_list)
    np.save("global-clustering-region", global_clustering)
    print("----Global Clustering: ", global_clustering)
    print("----Global Silhouette Score: ", global_silhouette)
    print("----Global Clustering Time: ", time() - start)

    create_directory_if_not_exists(results_directory)
    file_name = results_directory + "results" + str(datetime.now()) + ".txt"
    with open(file_name, "w") as f:
        f.write("----Global Clustering: " + str(global_clustering) + "\n")
        f.write("----Global Silhouette Score: " + str(global_silhouette) + "\n")


    # Perform Local Clustering

    local_clustering = []
    history_gld_list = np.empty((0, 4))
    t_instant = 0
    while(t_instant < ((365*4)+28)):
        with open(file_name, "a") as f:
            start = time()
            frame_window_series = ds_manager.read_window(t_instant, t_instant+28)
            gld_list, local_clustering = local_cls_manager.update_clustering(frame_window_series)
            np.save("gld_list-t=" + str(t_instant), gld_list)
            np.save("clustering-t=" + str(t_instant), local_clustering)

            print("----Local Clustering: ", local_clustering)
            print("----Local Silhouette Score: ", silhouette_score(gld_list, local_clustering))
            f.write("----Local Clustering t = " + str(t_instant) + " to " + str(t_instant + 28) + ": " + str(local_clustering) + "\n")
            f.write("++++Local Silhouette Score: " + str(silhouette_score(gld_list, local_clustering)) + "\n")
            f.write("++++Local Clustering Time: " + str(time() - start) + "\n")

            history_gld_list = np.concatenate((history_gld_list, gld_list), axis=0)
            history_clustering = local_cls_manager.birch.predict(history_gld_list)

            x = history_gld_list[:, 0]
            y = history_gld_list[:, 2]
            view.save_figure_as_scatter_plot(x, y, history_clustering, results_directory + "clustering-" + str(t_instant))
            view.save_figure_from_matrix(np.reshape(local_clustering, frame_window_series.shape[1:]), results_directory + "matrix-" + str(t_instant))

            t_instant += 28




from datetime import datetime
from core.utils import create_directory_if_not_exists
from core.dataset_manager import DatasetManager
from core.config_manager import ConfigManager
import core.categorization as ct
import core.view as view
from time import time
import numpy as np

config_manager = ConfigManager("experiment-metadata/exp3-clustering-rain-radar-france.config")
embedding_method = config_manager.get_config_value("embedding_method")
cur_time = str(datetime.now())
ds_name = config_manager.get_config_value("ds_name")
#clustering_region = (10, 14) # For both Lat and Long
clustering_region = (-1, -1) # if the whole dataset, uncomment

# Clustering period CFSR
#clustering_start, window_size = int((1460 / 4) * 0), int(1460 / 4) # summer south hemisphere
# clustering_start, window_size = int(0), int(1460) # whole dataset
#clustering_start, window_size = int((1460 / 4) * 2), int(1460 / 4) # winter south hemisphere

# Clustering period Radar-France
#clustering_start, window_size = int((1460 / 4) * 0), int(1460 / 4) # summer south hemisphere
clustering_start, window_size = int(0), int(17275) # whole dataset
#clustering_start, window_size = int((1460 / 4) * 2), int(1460 / 4) # winter south hemisphere


results_directory = "results/clustering/" + ds_name + str(clustering_region) + \
                    "-start=" + str(clustering_start)  + "-wsize=" + \
                    str(window_size) + '-' + cur_time + "/"

#clustering_time   = ('2014-01-01', '2014-06-30')
#clustering_time   = ('2014-07-01', '2014-12-31')


if __name__ == '__main__':
    create_directory_if_not_exists(results_directory)

    # Load the Dataset
    ds_manager = DatasetManager(config_manager.get_config_value("dataset_path"))

    # Filter Time
    ds_manager.loadDataset(ds_attribute=config_manager.get_config_value("target_attribute"))

    # Filter Region
    if clustering_region != (-1, -1):
        ds_manager.filter_by_region(clustering_region, clustering_region)

    # ******************** FIRST TEST: PERFORM GLOBAL CLUSTERING **************
    start = time()
    #ds = ds_manager.read_all_data() # All dataset
    ds = ds_manager.read_window(clustering_start, window_size)
    ds = np.nan_to_num(ds, nan=0, posinf=0, neginf=0)

    shp = ds.shape
    gld_list, global_clustering, global_silhouette = ct.cluster_dataset(ds, embedding_method=embedding_method,
                                                                        min_clusters=3)
    global_number_of_groups = len(set(global_clustering))
    global_clustering_time = str(time() - start)


    global_clustering = np.reshape(global_clustering, newshape=shp[1:])

    file_name = results_directory + "results" + cur_time + ".txt"
    with open(file_name, "w") as f:
        f.write("----Number of Groups: " + str(global_number_of_groups) + "\n")
        f.write("----Global Silhouette Score: " + str(global_silhouette) + "\n")
        f.write("----Global Clustering Time: " + str(global_clustering_time) + "\n")

    view.save_figure_from_matrix(global_clustering, "clustering",
                                 parent_directory=results_directory)

    np.save(results_directory + "Clustering.npy", global_clustering)
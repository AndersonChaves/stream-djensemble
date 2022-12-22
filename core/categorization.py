from .series_generator import SeriesGenerator
import numpy as np
from gldpy import GLD
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import floor, ceil
from itertools import cycle
import multiprocessing

def print_array(array_list):
    from matplotlib import pyplot as plt

    if isinstance(array_list, list):
        n = len(array_list)
        f, axis = plt.subplots(1, n)
        title_list = ["Clustering", "Tiling"]
        for array, i, title in zip(array_list, range(n), title_list):
            axis[i].set_title(title)
            #axis[i].imshow(array, cmap="tab20", vmin=0, vmax=array_list[i].max())
            axis[i].imshow(array, cmap="viridis", vmin=0, vmax=array.max())
    else:
        plt.imshow(array_list, cmap='gray', vmin=0, vmax=array_list[1].max())#, interpolation='nearest')
    plt.show()

def create_tiles_dictionary_mock(shape):
    _, lat, long = shape
    d = {}
    d[0] = {"lat": (0,                  floor(lat / 2)), "long": (0, floor(long / 2))}
    d[1] = {"lat": (floor(lat / 2)    , ceil(lat / 2)), "long": (0, floor(long / 2))}
    d[2] = {"lat": (0,                  floor(lat / 2)), "long": (floor(long / 2), ceil(long / 2))}
    d[3] = {"lat": (floor(lat / 2)    , ceil(lat / 2)), "long": (floor(long / 2), ceil(long / 2))}
    return d

def expand_tile(tiling: np.array, clustering: np.array, start: tuple,
                tile_id, max_impurity_rate = 0.05):
    impurity = 0

    lat, long = clustering.shape
    tile_cluster = clustering[start]
    cursor = end = start
    tiling[cursor] = tile_id

    dir_remaining = 4
    directions = cycle([(1, 0), (0, 1), (-1, 0), (0, -1)])
    skip_list = []
    max_impurity = 1
    for dir_step in directions:
        if len(skip_list) == 4:
            break
        if dir_step in skip_list:
            continue
        updates = True
        current_impurity = impurity
        #---EXPAND RIGHT---------------------------
        if dir_step == (1, 0) and end[1]+1 < long: #todo: Create Tile Objetct for Expansion
            end = end[0], end[1]+1
            x_start, y_start = start[0], end[1]
            x_end, y_end = end
            for i in range(x_start, x_end+1):
                cursor = i, y_start
                if tiling[cursor] != -1:
                    end = end[0], end[1] - 1
                    updates = False
                    break
                elif clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        end = end[0], end[1] - 1 # Undo Expantion
                        updates=False
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(x_start, x_end + 1):
                    cursor = i, y_start
                    tiling[cursor] = tile_id
        # ---EXPAND DOWN---------------------------
        elif dir_step == (0, 1) and end[0]+1 < lat:
            end = end[0]+1, end[1]
            x_start, y_start = end[0], start[1]
            x_end, y_end = end
            for i in range(y_start, y_end + 1):
                cursor = x_start, i
                if tiling[cursor] != -1:
                    updates = False
                    end = end[0] - 1, end[1]
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        updates=False
                        end = end[0] - 1, end[1]
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(y_start, y_end + 1):
                    cursor = x_start, i
                    tiling[cursor] = tile_id
        # ---EXPAND LEFT---------------------------
        elif dir_step == (-1, 0) and start[1]-1 >= 0:
            start = start[0], start[1]-1
            x_start, y_start = start
            x_end, y_end = end[0], start[1]+1
            for i in range(x_start, x_end + 1):
                cursor = i, y_start
                if tiling[cursor] != -1:
                    start = start[0], start[1] + 1
                    updates = False
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        start = start[0], start[1]+1
                        updates = False
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(x_start, x_end + 1):
                    cursor = i, y_start
                    tiling[cursor] = tile_id

        # ---EXPAND UP---------------------------
        elif dir_step == (0, -1) and start[0]-1 >= 0:
            start = start[0]-1, start[1]
            x_start, y_start = start
            x_end, y_end = start[0], end[1]
            for i in range(y_start, y_end + 1):
                cursor = x_start, i
                if tiling[cursor] != -1:
                    updates = False
                    start = start[0] + 1, start[1]
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        updates=False
                        start = start[0] + 1, start[1]
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(y_start, y_end + 1):
                    cursor = x_start, i
                    tiling[cursor] = tile_id
        #----------------------------------------
        else:
            updates = False
        if updates:
            impurity = current_impurity
            tile_size = (abs(start[0] - end[0])+1) * (abs(start[1] - end[1])+1)
            max_impurity = round((tile_size*max_impurity_rate), 0)
            skip_list = []
        else:
            skip_list.append(dir_step)
            dir_remaining -= 1
    return {"start": start, "end": end}

def create_tiling(clustering: np.array):
    shp = clustering.shape
    lat, long = shp
    tiling = np.full(shp, -1)
    x = 0
    tile_id = 1
    tile_dict = {}
    while x < lat:
        y = 0
        while y < long:
            if tiling[x, y] == -1:
                tile_dict[tile_id] = expand_tile(tiling, clustering, (x, y), tile_id)
                tile_id += 1
            y += 1
        x += 1
    return tiling, tile_dict

def calculate_gld_list_from_dataset(target_dataset):
    time, lat, long = target_dataset.shape
    # print("Calculating GLD")
    gld_list = np.empty((0, 4), float)

    # Create Series List
    series_list = []
    for i in range(lat):
        print("Calculating GLDs for lat", i)
        for j in range(long):
            cut_start, cut_ending = 0, time
            X, _ = SeriesGenerator().manual_split_series_into_sliding_windows(
                target_dataset[cut_start:cut_ending, i, j], time, n_steps_out=0)
            X = X.reshape((len(X[0])))
            series_list.append(X)

    gld_list = np.empty((0, 4))
    with multiprocessing.Pool() as pool:
        for result in pool.map(calculate_gld_estimator_using_fmkl, series_list):
            gld_estimators = result
            gld_estimators = np.reshape(gld_estimators, (1, 4))
            gld_list = np.append(gld_list, gld_estimators, axis=0)
    print("GLDs calculated")
    return gld_list

def calculate_gld_estimator_using_gpd(X: np.array):
    # Call GPD from R -------------------------------
    from rpy2.robjects.packages import importr
    gld = importr('gld')

    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    # Create a converter that starts with rpy2's default converter
    # to which the numpy conversion rules are added.
    np_cv_rules = default_converter + numpy2ri.converter

    with localconverter(np_cv_rules) as cv:
        #r_series = rpy2.robjects.r(X)
        # todo: gld using r's gpd is not working
        gld_results = gld.fit_gpd(X)
    return gld_results[:, 0]

def calculate_gld_estimator_using_fmkl(X: np.array):
    # GLD Using python library
    gld = GLD('FMKL')
    param_MM = gld.fit_MM(X, [0.5, 1], bins_hist=20, maxiter=1000, maxfun=1000, disp_fit=False)
    indexes = np.array((range(len(X))))
    while (True):
        guess = [random.randint(-2, 2), random.randint(-2, 2)]
        #guess = (1, 1)
        try:
            param_MM = gld.fit_curve(indexes, X, initial_guess=guess, N_gen=1000,
                                     optimization_phase=False, shift=True, disp_fit=False)
            return param_MM[0]
        except Exception as e:
            #print(e)
            #print("GLD Error: changing initial guess...")
            continue

def cluster_dataset(target_dataset):
    gld_list = calculate_gld_list_from_dataset(target_dataset)
    normalize_gld_list(gld_list)
    # Cluster using k-means into n clusters
    number_of_clusters = 1
    best_silhouete = -2
    kmeans_best_clustering = [0 for _ in range(len(gld_list))]
    for number_of_clusters in range(2, 5+1):
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
        kmeans_labels = kmeans.fit_predict(gld_list)
        silhouette_avg = silhouette_score(gld_list, kmeans_labels)
        if silhouette_avg > best_silhouete:
            kmeans_best_clustering = kmeans_labels
            best_silhouete = silhouette_avg
    print("KMeans best clustering: ", kmeans_best_clustering)
    print("KMeans best silhouete: ", best_silhouete)
    return gld_list, kmeans_best_clustering, best_silhouete

def normalize_gld_list(gld_list):
    for att in range(4):
        max_value = max(gld_list[:, att])
        scale_factor = 100 / max_value if max_value != 0 else 0
        for i, el in enumerate(gld_list):
            gld_list[i, att] = el[att] * scale_factor
    return gld_list

def calculate_centroid(clustering, start, end):
    tile_glds = clustering[start[0]:end[0]+1, start[1]:end[1]+1]
    t_shp = tile_glds.shape
    centroid_gld = [np.average(tile_glds[:, :, p]) for p in range(4)]

    from sklearn.metrics import pairwise_distances_argmin_min
    c, _ = pairwise_distances_argmin_min(np.reshape(centroid_gld, (1, -1)),
                                     np.reshape(tile_glds, (t_shp[0] * t_shp[1], 4)))
    print("Centroid: ", c)
    centroid = c // t_shp[1], c % t_shp[1]
    # Returns the centroid position relative to the tile
    return centroid

def categorize_dataset(target_dataset):
    time, lat, long = target_dataset.shape
    gld_list, kmeans_best_clustering = cluster_dataset(target_dataset)
    clustering_reshaped = np.reshape(kmeans_best_clustering, newshape=(lat, long))
    perform_yolo_tiling(target_dataset, clustering_reshaped)
    #clustering_reshaped = kmeans_best_clustering

def perform_yolo_tiling(gld_frame, clustering_frame):
    tiling, tile_dict = create_tiling(clustering_frame)
    #print_array([clustering_frame, tiling])
    save_clustering(clustering_frame)
    tile_dim_dict = {}
    for tile_id, value in tile_dict.items():
        tile_dim_dict[tile_id] = {}
        start, end = value['start'], value['end']
        tile_dim_dict[tile_id]["lat"] = (start[0], end[0])
        tile_dim_dict[tile_id]["long"] = (start[1], end[1])
        tile_dim_dict[tile_id]["centroid"] = calculate_centroid(gld_frame,
                                                                start, end)
    return tiling, tile_dim_dict

def save_clustering(clustering):
    np.save("results/clustering.npy", clustering)

def test_1():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (0, 0), 1))

def test_2():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (0, 1), 1))

def test_3():
    tiling = np.full((5, 5), -1)
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0]])
    print(expand_tile(tiling, clustering, (1, 1), 1))

def test_clustering():
    dataset = np.full((10, 5, 5), 7)
    clustering = cluster_dataset(dataset)
    print_array(np.reshape(clustering, (5, 5)))


def test_tiling():
    clustering = np.array([[1, 1, 1, 1, 1],
                           [1, 0, 1, 0, 0],
                           [1, 1, 1, 1, 0],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0]])
    create_tiling(clustering)

def custom_made_tiling():
    tiling = [[0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1],
              [3, 3, 3, 3, 3, 3, 3],
              [3, 3, 3, 3, 3, 3, 3]]
    tile_dict = {1: {'start': (5, 0), 'end': (6, 6)},
                 2: {'start': (0, 0), 'end': (4, 3)},
                 3: {'start': (0, 4), 'end': (4, 6)}, }
    return tiling, tile_dict

def test_categorization():
    array = DatasetManager().synthetize_dataset(shape=(10, 10, 10))
    print(categorize_dataset(array))

if __name__ == "__main__":
    test_categorization()
    #print_array()
    # test_clustering()
    # print(gld)

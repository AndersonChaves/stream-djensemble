from core import categorization as ct
import numpy as np
from core import utils as ut
import core.view as view
import time

#base_path = "results/exp1/cfsr/fatnode/CFSR-2014.nc(-1, -1)2023-01-19 14:57:50.701534-GLD-FULL"
#base_path = "results/exp1/cfsr/fatnode/CFSR-2014.nc(-1, -1)2023-01-19 14:57:50.701534-parcorr-full"
#base_path = "results/exp1/radar-france/gld-parcorr-radar-france"
base_path = 'results/exp1/alerta-rio/gld-parcorr-alerta-rio'

dirs = ut.list_all_sub_directories(base_path)
for d in dirs:
    f_list = ut.list_all_files_in_dir(d, extension='npy')
    for f_array in f_list:
        for method in ['yolo', 'quadtree']:
            clustering = np.load(d + '/' + f_array)
            start_tiling = time.time()
            tiling, tiling_metadata = ct.perform_tiling(None, clustering, method, 0.8)
            end_tiling = time.time()
            number_of_tiles = len(set(tiling.ravel()))
            with open(d + '/' + method + '-number_of_tiles=' + str(number_of_tiles) + '.tile', "w") as f:
                f.write(method + '-number of tiles: ' + str(number_of_tiles) + '\n')
                print(str(end_tiling - start_tiling))
                f.write(method + '-time: ' + str(end_tiling - start_tiling)  + '\n')
            view.save_figure_from_matrix(tiling, method + "tiling",
                                         parent_directory=d + '/')

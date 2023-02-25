import numpy as np
from core.tile import Tile
import os, time
from core.continuous_query import ContinuousQuery
from core.learner import Learner

class QueryManager():
    continuous_query = {}

    def execute_queries(self, data_buffer, models_manager,
                           dataset_manager, cluster_manager):
        execution_benchmarks = {}
        self.cluster_manager = cluster_manager
        for query_id, query in self.continuous_query.items():
            query_execution_benchmarks = self.perform_continuous_query(query, data_buffer, models_manager, dataset_manager)
            for key, value in query_execution_benchmarks.items():
                execution_benchmarks[query_id + "_" + key] = value
        return execution_benchmarks

    def __init__(self, query_dir, notifier_list: list = None):
        files = os.listdir(query_dir)
        for file_name in files:
            query_id, file_extension = os.path.splitext(file_name)
            if file_extension == '.query':
                self.continuous_query[query_id] = ContinuousQuery(query_dir + file_name, query_id)
        self.notifier_list = [] if notifier_list is None else notifier_list

    def perform_continuous_query(self, query: ContinuousQuery,
                                 data_buffer: np.array,
                                 models_manager, dataset_manager):
        query_benchmarks = {}
        start_query = time.time()
        # 1. Get spatial region from the query
        x1, x2 = query.get_query_endpoints()
        chk_1 = time.time()

        # 2. Get corresponding series from data buffer
        if not self.cluster_manager.is_global_clustering():
            target_data_window = data_buffer[:, x1[0]:x2[0], x1[1]:x2[1]]
        else:
            target_data_window = data_buffer
        chk_2 = time.time()

        # 3. Get corresponding clustering from data buffer (gld window)
        # tiling is a dict {tile_id: {'start': (x1). 'end': (x2), centroid: <centroid-coordinate>}
        if self.cluster_manager.is_global_clustering():
            tiling_metadata = self.cluster_manager.get_tiling_metadata()
            intersecting_tiles = self.get_intersecting_tiles(tiling_metadata, x1, x2)
            all_tiles_keys = list(tiling_metadata.keys())
            for tile_id in all_tiles_keys:
                if tile_id not in intersecting_tiles:
                    tiling_metadata.pop(tile_id)
        else:
            tiling_metadata = query.get_tiling_metadata()
        self.log("TILING: Query has " + str(len(tiling_metadata.keys())) + \
                   " Tiles. (Query " + query.query_id + ")")
        chk_3 = time.time()

        # 4. For each tile, get its representing series
        error_estimative = self.get_error_estimative(data_buffer, tiling_metadata,
                                                       models_manager, dataset_manager, query)
        chk_4 = time.time()

        # 5. Get best model for each tile
        self.log("--------------------- CALCULATE ALLOCATION COSTS -------------------------------")
        ensemble = self.get_lower_cost_combination(error_estimative)
        i = 0
        for key, value in ensemble.items():
            query_benchmarks["TILE_" + str(i) + "_MODEL"] = str(value[0])
            query_benchmarks["TILE_" + str(i) + "_EST_ERROR"] = str(value[1])
            i += 1

        self.log("Best Ensemble: " + str(ensemble))
        chk_5 = time.time()

        # 6. Perform model prediction for each tile
        prediction_length = eval(query.get_config_value("prediction-length"))

        query_size_lat, query_size_lon = x2[0] - x1[0], x2[1] - x1[1]
        tile_prediction = {}
        i = 0
        for tile_id, model in ensemble.items():
            model_name = ensemble[tile_id][0]
            self.log("Model: " + model_name)
            self.log("==>Evaluating error for tile " +str(tile_id) + ": " + str(i + 1) + "of " + str(len(ensemble.keys())))
            learner = models_manager.get_model_from_name(model_name)
            tile_metadata = tiling_metadata[tile_id]
            pred = self.perform_prediction(data_buffer, learner, tile_metadata)
            #print("DEBUG: Prediction Len is ", len(pred))
            tile_prediction[tile_id] = pred
            i += 1

        self.log("Recomposing data series")
        query_predicted_series = np.zeros((prediction_length, query_size_lat, query_size_lon))
        local_clustering = not self.cluster_manager.is_global_clustering()
        for tile_id, value in tile_prediction.items():
            self.compose_predicted_frame(query_predicted_series, query, tiling_metadata[tile_id],
                                         tile_prediction[tile_id], tiles_relative_to_query=local_clustering)
        self.log("--------------------- Query Executed -------------------------------")

        # 7. Recompose data series
        query.set_predicted_series(query_predicted_series)
        chk_6 = time.time()

        # Print Times
        query_id = query.query_id
        self.log(query_id + ": Total Query Execution Time: " + str(round(chk_6 - start_query, 5)))
        self.log("1.--Time to get query parameters: " + str(round(chk_1 - start_query, 5)))
        self.log("2.--Time to filter query region: " + str(round(chk_2 - chk_1, 5)))
        self.log("3.--Time to get tiling: " + str(round(chk_3 - chk_2, 5)))
        self.log("4.--Time to get tile centroid series: " + str(round(chk_4 - chk_3, 5)))
        self.log("5.--Time to select best model: " + str(round(chk_5 - chk_4, 5)))
        self.log("6.--Time to Perform model prediction: " + str(round(chk_6 - chk_5, 5)))

        query_benchmarks["QUERY_MODEL_PREDICTION_TIME"] = str(round(chk_6 - chk_5, 5))
        query_benchmarks["QUERY_EXECUTION_TIME"] = str(round(chk_6 - start_query, 5))
        return query_benchmarks

    def perform_prediction(self, data_window: np.array, learner: Learner, tile_metadata: dict):
        tile_lat, tile_long = list(tile_metadata["lat"]), list(tile_metadata["long"])
        tile_lat[1] += 1
        tile_long[1] += 1
        input_dataset = data_window[:, tile_lat[0]: tile_lat[1], tile_long[0]:tile_long[1]]
        return learner.invoke_on_dataset(input_dataset)

    def compose_predicted_frame(self, resulting_array, query, tile_metadata: dict,
                                tile_prediction: np.array, tiles_relative_to_query):

        query_endpoints = query.get_query_endpoints()
        tile_lat = list(tile_metadata["lat"])
        tile_long = list(tile_metadata["long"])
        # x1 = query_endpoints[0]
        # x1_lat, x1_lon =
        query_lat  = query_endpoints[0][0], query_endpoints[1][0]-1#query endpoints are always open interval, so -1
        query_long = query_endpoints[0][1], query_endpoints[1][1]-1 # If change this must change result declaration

        # Transform tile coordinates into absolute coordinates
        if tiles_relative_to_query:
            tile_lat[0]  = tile_lat[0] + query_lat[0]
            tile_lat[1]  = tile_lat[1] + query_lat[0]
            tile_long[0] = tile_long[0] + query_long[0]
            tile_long[1] = tile_long[1] + query_long[0]
        # Get from prediction, area corresponding to query
        ##1. Get intersecting coordinates relative to tile
        intersection_lat = max(tile_lat[0] , query_lat[0]), min(tile_lat[1] , query_lat[1])
        intersection_long = max(tile_long[0], query_long[0]), min(tile_long[1], query_long[1])

        ##2. Get data corresponding to coordinates found
        i_lat = intersection_lat[0] - tile_lat[0], intersection_lat[1] - tile_lat[0]
        i_lon = intersection_long[0] - tile_long[0], intersection_long[1] - tile_long[0]

        #print("DEBUG Compose: shape of tile_prediction is ", tile_prediction.shape)
        if len(tile_prediction.shape) == 3:
            intersecting_data = tile_prediction[-1, i_lat[0]:i_lat[1] + 1, i_lon[0]:i_lon[1] + 1]
        else:
            intersecting_data = tile_prediction[i_lat[0]:i_lat[1]+1, i_lon[0]:i_lon[1]+1]

        ##3. Get intersecting coordinates corresponding to query
        lat = intersection_lat[0] - query_lat[0], intersection_lat[1] - query_lat[0]
        lon = intersection_long[0] - query_long[0], intersection_long[1] - query_long[0]

        ##5. Attribute data to query array in corresponding coordinates
        resulting_array[:, lat[0]:lat[1]+1, lon[0]:lon[1]+1] = intersecting_data

    def get_continuous_query(self, query_id) -> ContinuousQuery:
        return self.continuous_query[query_id]

    def get_all_query_ids(self):
        return self.continuous_query.keys()

    def get_error_estimative(self, data_window,
                             tile_metadata, models_manager, dataset_manager, query):
        error_estimative = {}
        for tile_id in tile_metadata.keys():
            self.log("----Estimating error for tile "+ str(tile_id))
            t = Tile(tile_id, tile_metadata[tile_id]) # Self.tiles == tile_dim_dict
            candidate_models_names = query.get_candidate_models_list()
            candidate_models = models_manager.get_models_from_list(candidate_models_names)
            error_estimative[tile_id] = models_manager.get_error_estimative_ranking(
                  data_window, t, dataset_manager, candidate_models)
        return error_estimative

    def get_intersecting_tiles(self, tiling_metadata, x1, x2):
        # Query ex. {"lat": (2, 5), "long": (3, 5)}
        tiles_list = tiling_metadata
        query = {"lat": (x1[0], x2[0]), "long": (x1[1], x2[1])}
        intersecting_tiles = list(tiles_list.keys())
        for dim, value in query.items():
            q_min, q_max = value
            for tile_id in tiles_list.keys():
                tile = tiles_list[tile_id]
                tile_lower_bound = tile[dim][0]
                tile_upper_bound = tile[dim][1]
                if q_min > tile_upper_bound or \
                        q_max < tile_lower_bound:
                    if tile_id in intersecting_tiles:
                        intersecting_tiles.remove(tile_id)
                    # break
        return intersecting_tiles

    def get_lower_cost_combination(self, error_estimative):
        ensemble = {}
        for tile_id in error_estimative.keys():
            best_model, best_error = 'x', float('inf')
            for model_name, error in error_estimative[tile_id].items():
                if error < best_error:# and model_name not in self.exclude_models_list:
                    best_model = model_name
                    best_error = error
            ensemble[tile_id] = (best_model, best_error)
        return ensemble

    def save_query_configurations(self, base_directory):
        for id, query in self.continuous_query.items():
            query.save_config_file(base_directory + id + ".query")

    def log(self, msg):
        for notifier in self.notifier_list:
            notifier.notify(msg)

# Test query manager
if __name__ == '__main__':
    query_dir = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/queries/"
    q = QueryManager(query_dir)
    config = q.get_continuous_query("query-alerta-rio")
    print(config.get_config_value("dataset_path"))
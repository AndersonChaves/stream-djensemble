import numpy as np
import core.categorization as ct
from core.tile import Tile
import os, time
from core.continuous_query import ContinuousQuery

class QueryManager():
    continuous_query = {}

    def execute_queries(self, data_buffer, models_manager, dataset_manager):
        for id, query in self.continuous_query.items():
            self.perform_continuous_query(query, data_buffer, models_manager, dataset_manager)

    def __init__(self, query_dir):
        files = os.listdir(query_dir)
        for file_name in files:
            query_id, file_extension = os.path.splitext(file_name)
            if file_extension == '.query':
                self.continuous_query[query_id] = ContinuousQuery(query_dir + file_name)

    def perform_continuous_query(self, query: ContinuousQuery,
                                 data_buffer: np.array,
                                 models_manager, dataset_manager):
        # 1. Get spatial region from the query
        x1, x2 = query.get_query_endpoints()
        # 2. Get corresponding series from data buffer
        data_window = data_buffer[:, x1[0]:x2[0], x1[1]:x2[1]]

        # 3. Get corresponding clustering from data buffer (gld window)
        # tiling is a dict {tile_id: {'start': (x1). 'end': (x2), centroid: <centroid-coordinate>}
        tiling = query.get_tiling_metadata()

        # 4. For each tile, get its representing series
        error_estimative = self.get_error_estimative(data_window,
                                                      tiling, models_manager, dataset_manager)
        # 5. Get best model for each tile
        print("--------------------- CALCULATE ALLOCATION COSTS -------------------------------")
        ensemble = self.get_lower_cost_combination(error_estimative)

        # 6. Perform model prediction for each tile
        prediction_length = eval(query.get_config_value("prediction-length"))
        predicted_series = np.empty((prediction_length,) + data_window.shape[1:])
        for tile_id, model in ensemble.items():
            model_name = ensemble[tile_id][0]
            #print("==>Evaluating error for tile ", tile_id, ": ", i + 1, "of ", len(query_tiles))
            print("Model: ", model_name)
            learner = models_manager.get_model_from_name(model_name)
            lat, long = tiling[tile_id]["lat"], tiling[tile_id]["long"]
            prediction = learner.invoke_on_dataset(data_window)
            lat_size, long_size = lat[1] - lat[0], long[1] - long[0]
            predicted_series[:, lat[0]:lat[1]+1, long[0]:long[1]+1] = prediction[:lat_size+1, :long_size+1]
        print("--------------------- Query Executed -------------------------------")
        query.set_predicted_series(predicted_series)

    def get_continuous_query(self, query_id) -> ContinuousQuery:
        return self.continuous_query[query_id]

    def get_all_query_ids(self):
        return self.continuous_query.keys()

    def get_error_estimative(self, data_window,
                                tile_bounds, models_manager, dataset_manager):
        error_estimative = {}
        for tile_id in tile_bounds.keys():
            print("Estimating error for tile ", tile_id)
            t = Tile(tile_id, tile_bounds[tile_id]) # Self.tiles == tile_dim_dict
            error_estimative[tile_id] = models_manager.get_error_estimative_ranking(data_window,
                                                                                    t, dataset_manager)
        return error_estimative

    def get_lower_cost_combination(self, error_estimative):
        ensemble = {}
        for tile_id in error_estimative.keys():
            best_model, best_error = 'x', float('inf')
            for model_name, error in error_estimative[tile_id].items():
                if error < best_error:
                    best_model = model_name
                    best_error = error
            ensemble[tile_id] = (best_model, best_error)
        return ensemble

    def compose_predicted_frame(self, tile_bounds, prediction_by_tile):
        pass


# Test query manager
if __name__ == '__main__':
    query_dir = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/queries/"
    q = QueryManager(query_dir)
    config = q.get_continuous_query("query-alerta-rio")
    print(config.get_config_value("dataset_path"))
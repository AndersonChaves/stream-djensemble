class Tile():

    bounds = {}
    def __init__(self, id, dim_dict):
        self.id = id
        self.bounds = dim_dict
        self.centroid = dim_dict["centroid"]
        self.centroid_series = dim_dict["centroid_series"]

    def get_start_coordinate(self):
        return self.bounds["lat"][0], self.bounds["long"][0]

    def get_end_coordinate(self):
        return self.bounds["lat"][1], self.bounds["long"][1]

    def set_centroid(self, centroid):
        self.centroid = centroid

    def get_centroid_coordinates(self):
        return self.centroid

    def get_centroid_series(self):
        return self.centroid_series
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from functools import reduce
import dtw

from .noise_generator import NoiseGenerator
from .series_generator import SeriesGenerator
from .simple_regressor import LinearRegressor
from .convolutional_model_invoker import ConvolutionalModelInvoker
from .categorization import calculate_centroid
from .categorization import calculate_gld_list_from_dataset
from .dataset import Dataset
from abc import ABC, abstractmethod

class Learner(ABC):
    model = None
    _is_temporal_model = False
    temporal_length = 10

    def __init__(self, model_directory, model_name,
                 is_temporal_model=False):
        self.model_directory = model_directory
        self.model_name = model_name
        self.number_of_training_samples = 10
        self._is_temporal_model = is_temporal_model
        self.load_model(model_directory, model_name)
        super().__init__()

    def load_model(self, model_directory, model_name):
        self.name = model_name
        self.model_full_path = model_directory + model_name
        # Loads metadata from json file
        json_file = open(model_directory + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # Load weights into new model
        model.load_weights(model_directory + model_name + '.h5')
        # Loads metadata from database
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model = model
        self.reference_dataset = None
        self.centroid = (-1, -1)

    def invoke(self, X):
        #from time import time
        #x = time()
        temp = self.get_model().predict(X)
        #print("Tempo de invocação: ", time() - x)
        return temp

    def get_model(self):
        return self.model

    def get_shape(self):
        return self.model.input.shape

    def set_model(self, model):
        self.model = model

    def get_name(self):
        return self.name

    def is_temporal_model(self):
        return self._is_temporal_model

    def get_reference_dataset(self):
        if self.reference_dataset is None:
            dataset_full_path = self.model_directory + self.model_name + '.npy'
            self.reference_dataset = (np.load(dataset_full_path))
        return self.reference_dataset

    def get_reference_dataset_mock(self):
        #1. Get the learner's spatial coordinates
        #2. Obtain the reference dataset from stream (e.g. directly from kafka)
        # - Use data aggregator? Another Class? Where does DJEnsemble access the stream from?
        #todo - Swap Mock
        dataset_mock = np.full((100, 10, 10), 10.0)
        return dataset_mock

    @abstractmethod
    def evaluate(self, reference_dataset: np.array):
        pass

    @abstractmethod
    def invoke_on_dataset(self, target_dataset):
        pass

    def characterize_dataset_time_series(self, target_dataset):
        print("Calculating GLDs...")
        time, lat, long = target_dataset.shape
        gld_list = calculate_gld_list_from_dataset(target_dataset)
        gld_list = np.reshape(gld_list, (lat, long, 4))  # Number of GLD parameters = 4
        print("Calculating Centroid...")
        return calculate_centroid(gld_list, (0, 0), (lat, long))

    def compare_dataset_distances(self, original_dataset, compared_dataset, c1, c2):
        # 2. Identify the centroid time series C1 and C2 based on the resulting parameters
        s1 = original_dataset[:, c1[0], c1[1]]
        s2 = compared_dataset[:, c2[0], c2[1]]
        # 3. Determine the distance between C1 and C2 (e.g. using euclidian dist between vectors)
        output = dtw.dtw(s1, s2)
        return output.distance

    def calculate_centroid_coordinate(self, target_dataset):
        # Characterize each dataset time series (e.g. using GLD or Autoencoder)
        print("Characterizing time series for dataset of shape ", target_dataset.shape)
        return self.characterize_dataset_time_series(target_dataset)

    def get_reference_dataset_centroid_coordinate(self):
        return (0, 0) # todo change for correct procedure
        #if self.centroid == (-1, -1):
        #    self.centroid = self.calculate_centroid_coordinate(self.get_reference_dataset())
        #return self.centroid

    def update_eef(self, noise_level_for_cef):
        print("Updating EEF - Model " + self.model_name)
        reference_dataset = self.get_reference_dataset()
        noise_dataset = reference_dataset.copy()

        # Measures model performance
        distances = []
        error = []
        centroid = self.get_reference_dataset_centroid_coordinate()
        for i in range(noise_level_for_cef):
            print("Evaluating model", self.model_name, " on noise dataset ", i)
            error.append(self.evaluate(noise_dataset[:50]))
            distances.append(self.compare_dataset_distances(reference_dataset, noise_dataset,
                                                                centroid, centroid))
            NoiseGenerator().add_noise(noise_dataset)

        # Fit line
        r = LinearRegressor(distances, error)
        r.train()
        self.r = r

    def execute_eef(self, dataset, centroid):
        reference_dataset = self.get_reference_dataset()
        ref_centroid = self.get_reference_dataset_centroid_coordinate()
        x = self.compare_dataset_distances(reference_dataset, dataset, ref_centroid, centroid)
        return self.r.predict(x)

class UnidimensionalLearner(Learner):
    def evaluate(self, target_dataset: np.array):
        time, lat, long = target_dataset.shape
        rmse_vector = []
        for i in range(lat):
            for j in range(long):
                cut_start = 0
                cut_ending = self.temporal_length + self.number_of_training_samples

                X, y = SeriesGenerator().manual_split_series_into_sliding_windows(
                    target_dataset[cut_start:cut_ending, i, j], self.temporal_length, 1)
                # X = X.reshape((self.number_of_samples, self.temporal_length, 1))
                output = self.invoke(X)
                region_rmse = tf.sqrt(tf.math.reduce_mean(tf.losses.mean_squared_error(output, y)))
                rmse_vector.append(region_rmse.numpy())
        average_rmse = reduce(lambda a, b: a + b, rmse_vector) / len(rmse_vector)
        return average_rmse

    def invoke_on_dataset(self, target_dataset):
        time, lat, long = target_dataset.shape
        output = np.empty((time, lat, long))
        for i in range(lat):
            for j in range(long):
                output[:, i, j] = self.invoke(target_dataset[:, i, j])
        return output

class MultidimensionalLearner(Learner):
    def evaluate(self, target_dataset: np.array):
        invoker = ConvolutionalModelInvoker()
        return invoker.evaluate_convolutional_model(target_dataset, self)

    def invoke_on_dataset(self, target_dataset):
        invoker = ConvolutionalModelInvoker()
        new_shape = (1,) + target_dataset.shape + (1,)
        extended_target_dataset = np.reshape(target_dataset, new_shape)
        output, _ = invoker.invoke_candidate_model(self, extended_target_dataset)
        return output[0, -1, :, :, 0]

def test_unidimensional_layer():
    directory = "/home/anderson/Dropbox/Doutorado/Tese/Fermatta/DJEnsemble/models/rain/temporal/"
    name = "best_model_C4"
    l = UnidimensionalLearner(directory, name, is_temporal_model=True)
    l.update_eef(1)
    print(l.execute_eef(NoiseGenerator().add_noise(l.get_reference_dataset()), (0, 0)) )

def test_multidimensional_layer():
    directory = "/home/anderson/Dropbox/Doutorado/Tese/Fermatta/DJEnsemble/models/rain/convolutional/"
    name = "best_DJ1_v1_upd"
    l = MultidimensionalLearner(directory, name, is_temporal_model=False)
    l.update_eef(1)
    print(l.execute_eef(NoiseGenerator().add_noise(l.get_reference_dataset())))

if __name__ == "__main__":
    test_unidimensional_layer()
    test_multidimensional_layer()

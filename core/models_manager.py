from .learner import UnidimensionalLearner, MultidimensionalLearner
import fnmatch
import os
from .dataset_manager import DatasetManager

class ModelsManager():
    temporal_models      = []
    convolutional_models = []

    def __init__(self, config_manager):
        self.conv_models_path = config_manager.get_config_value("convolutional_models_path")
        self.temp_models_path = config_manager.get_config_value("temporal_models_path")
        self.include_convolutional_models_from_directory(self.conv_models_path)
        self.include_temporal_models_from_directory(self.temp_models_path)
        self.load_models()

    def load_models(self):
        print("Loading temporal models")
        self.temporal_models      = self.load_all_temporal_models()
        print("Loading convolutional models")
        self.convolutional_models = self.load_all_convolutional_models() 
        
    def load_all_temporal_models(self):
        model_list = []
        for file in os.listdir(self.temp_models_path):
            if file.endswith(".h5"):
                model_name = file[:-3]
                model_list.append(self.load_temporal_model(model_name))
        return model_list

    def load_all_convolutional_models(self):
        models_names = self.get_names_of_models_in_dir(self.conv_models_path)
        models_list = [self.load_convolutional_model(model) for model in models_names]
        return models_list

    def load_model(self, model_name):
        if model_name.startswith('best_model'):
            return self.load_temporal_model(model_name)
        else:
            return self.load_convolutional_model(model_name)

    def load_temporal_model(self, model_name):
        return UnidimensionalLearner(self.temp_models_path,
                                     model_name,
                                     is_temporal_model=True)

    def load_convolutional_model(self, model_name):
        return MultidimensionalLearner(self.convolutional_models_directory, model_name, is_temporal_model=False)

    def include_convolutional_models_from_directory(self, models_path):
        self.convolutional_models_directory = models_path

    def include_temporal_models_from_directory(self, models_path):
        self.temp_models_path = models_path

    def get_names_of_models_in_dir(self, models_path):
        models = fnmatch.filter(os.listdir(models_path), '*.h5')
        for i, m in enumerate(models):
            models[i] = m.split('.h5')[0]
        return models

    def get_latitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[2])

    def get_longitude_input_size(self, model_name):
        model = self.get_model_from_name(model_name)
        if model.is_temporal_model:
            return 1
        else:
            return int(model.get_model().input.shape[3])

    def get_model_from_name(self, model_name):
        all_learners = self.temporal_models + self.convolutional_models
        if all_learners == []:
            print("No models loaded. Inform models directory and run function load_models")
            raise(Exception)
        for learner in all_learners:
            name = learner.get_name()
            if name == model_name:
                return learner
        print("Model not found")
        raise(Exception)

    def get_models_from_list_of_names(self, best_allocation):
        models_list = []
        for model_name in best_allocation:
            models_list.append(self.get_model_from_name(model_name))
        return models_list

    def get_models(self):
        return self.temporal_models + self.convolutional_models

    def get_error_estimative_ranking(self, dataset,
                                       tile, dataset_manager):
        error_estimative = {}
        tile_dataset = dataset_manager.get_data_from_tile(dataset, tile)
        for learner in self.get_models():
            error_estimative[learner.name] = learner.execute_eef(tile_dataset, tile.get_centroid())[0][0]
        return error_estimative

    def update_cef(self, noise_level_for_cef):
        for learner in self.get_models():
            learner.update_eef(noise_level_for_cef)

if __name__ == "__main__":
    print("Teste")
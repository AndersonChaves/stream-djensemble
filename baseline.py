import numpy as np
import pandas as pd
from core.config_manager import ConfigManager
from core.dataset_manager import DatasetManager
from core.models_manager import ModelsManager
from datetime import datetime
from time import time
from core.utils import create_directory_if_not_exists
from core.learner import Learner

# global Configurations
cur_time = str(datetime.now())
embedding_directory = "results/clustering/"
results_directory = "results/baseline/"

def perform_prediction(self, data_window: np.array, learner: Learner, tile_metadata: dict):
    tile_lat, tile_long = list(tile_metadata["lat"]), list(tile_metadata["long"])
    tile_lat[1] += 1
    tile_long[1] += 1
    input_dataset = data_window[:, tile_lat[0]: tile_lat[1], tile_long[0]:tile_long[1]]
    return learner.invoke_on_dataset(input_dataset)

def get_history_rmse(learner, ds_manager):
    t_current = 0
    history_rmse = []
    print("Evaluating model", learner.get_name())
    while (t_current < ((365 * 4) + window_size)):
        print("Evaluating frame t = ", t_current, ": ")
        window_series = ds_manager.read_window(t_current, window_size)

        if window_series.shape[0] < window_size:
            break
        real_next_frame = ds_manager.read_instant(t_current + window_size + 1)
        predicted_frame = learner.invoke_on_dataset(window_series)

        model_rmse = np.average(
            np.sqrt(
                np.square(
                    np.reshape(
                        predicted_frame,
                        predicted_frame.shape[-2:]
                    )
                    - real_next_frame
                )
            )
        )
        print("RMSE ", ": ", model_rmse)
        history_rmse.append(model_rmse)
        t_current += 100
    return history_rmse

def get_model_stacking(learner_list, ds_manager):
    # Load pre-trained models
    model1 = learner_list[0]
    model2 = learner_list[1]
    model3 = learner_list[2]

    # Prepare your dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions using each of your pre-trained models on the testing set
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    # Combine the predictions of your pre-trained models using a meta-model
    meta_model = Sequential()
    meta_model.add(Dense(64, input_shape=(3,)))
    meta_model.add(Activation('relu'))
    meta_model.add(Dense(1))
    meta_model.compile(optimizer='adam', loss='mse')

    # Use the predictions of the pre-trained models as input features for the meta-model
    X_meta_train = np.column_stack((y_pred1, y_pred2, y_pred3))
    X_meta_test = np.column_stack((model1.predict(X_test), model2.predict(X_test), model3.predict(X_test)))

    # Train the meta-model on the training set
    meta_model.fit(X_meta_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the performance of your stacked model on the testing set
    mse = meta_model.evaluate(X_meta_test, y_test)
    print("MSE:", mse)

if __name__ == '__main__':
    create_directory_if_not_exists(results_directory)

    window_size, output_size = 10, 1

    # Load the Dataset
    ds_path = "datasets/CFSR-2014.nc"
    ds_name = ds_path.split('/')[-1]
    ds_manager = DatasetManager(ds_path)
    ds_manager.loadDataset(ds_attribute="TMP_L100")
    ds_manager.filter_by_region(
        (86, 117+1),
        (6, 24+1)
    )
    # Load pre-trained models
    global_configurations_path = "experiment-metadata/djensemble-exp1.config"
    config_manager = ConfigManager(global_configurations_path)
    models_manager = ModelsManager(config_manager)
    #models_manager.load_models()
    learner = models_manager.get_models()[0]

    # ******************** LOOP START: PERFORM GLOBAL CLUSTERING **************
    prediction_data = {}
    for learner in models_manager.get_models():
        history_rmse = get_history_rmse(learner, ds_manager)
        average_rmse = sum(history_rmse)/len(history_rmse)
        prediction_data[learner.name] = history_rmse


    df = pd.DataFrame(prediction_data, index=range(0, 1500, 100))
    df.to_csv(
        results_directory + "models_results: " + cur_time + '.csv',
        index = True,
        sep = ";",
        decimal= ","
    )

    get_model_stacking(models_manager.get_models(), ds_manager)
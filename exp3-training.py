import xarray as xr
import core.model_training as model_training
from pandas import datetime
from core.lstm_learner import LstmLearner
import numpy as np

retrain_model = True
series_size = 10

ds_dir = "datasets/"
ds_variable = 'TMP_L100'
#ds_name = "curated.chuva.alertario-malha.2020.nc" # Rio - Pluviométricos
ds_name = "CFSR-2014.nc" # CFSR - Temperature

x0, x1 = (53, 1), (53, 1)
model_name = ds_name + '-x0=' + str(x0) + "-1x1"
models_dir = "results/models-trained/"
# date-time parsing function for loading the dataset
def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')

# load dataset
ds = xr.open_dataset(ds_dir + ds_name)

filtered_dataset = ds.sel(time=slice('2014-01-01 00:00:00', '2014-03-30 23:45:00'), drop=True)
train = filtered_dataset.isel(lat=0, lon=0)
test  = filtered_dataset.isel(lat=0, lon=0)

if retrain_model:
  lstm_model = LstmLearner("", models_dir + model_name, auto_loading=False)
  lstm_model.update_architecture(neurons=32, nb_epochs=100,
                                 batch_size=1, number_of_hidden_layers=2)
  np_train = train[ds_variable].to_numpy()
  lstm_model.train(np_train, series_size)
  np.save(models_dir + model_name + ".npy", np_train)
  model_training.save_model_as_h5(lstm_model.get_model(), models_dir + model_name)
else:
  lstm_model = LstmLearner("", models_dir + model_name, auto_loading=True)

supervised_test = model_training.transform_supervised(test[ds_variable].to_numpy(), series_size)
results = lstm_model.predict(lstm_model.get_model(), supervised_test)

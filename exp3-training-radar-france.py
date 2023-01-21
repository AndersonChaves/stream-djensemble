import xarray as xr
import core.model_training as model_training
from pandas import datetime
from core.lstm_learner import LstmLearner
import numpy as np

retrain_model = True
series_size = 10

ds_dir = "datasets/"
ds_variable = 'rain'
#ds_name = "curated.chuva.alertario-malha.2020.nc" # Rio - Pluviométricos
#ds_name = "CFSR-2014.nc" # CFSR - Temperature
ds_name = "radar-2016-0to17275.npy" # RFRANCE - Rain

#summer = '2014-01-01 00:00:00', '2014-03-30 23:45:00'
#winter = '2014-07-01 00:00:00', '2014-09-31 23:45:00'

winter = (0, 17276//4)
summer = (17276//4 *2, 17276)

p = "summer"
if p == "summer":
  period = summer
else:
  period = winter

#Models 1x1 - CFSR
#x0, x1 = (53, 1), (53, 1) #C0
#x0, x1 = (1, 1), (1, 1) #C1
#x0, x1 = (113, 28), (113, 28) #C2

#Models 1x1 - RFRANCE
x0, x1 = (55, 59), (55, 59) #C0
#x0, x1 = (110, 8), (110, 8) #C1
#x0, x1 = (30, 67), (30, 67) #C2

model_name = ds_name + '-x0=' + str(x0) + "-1x1" + "-" + p

models_dir = "results/models-trained/"
# date-time parsing function for loading the dataset
def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')

# load dataset
ds = np.load(ds_dir + ds_name)

filtered_dataset = ds[period[0]:period[1]]
filtered_dataset = filtered_dataset[x0[0]: x1[0], x0[1]: x1[1]]

train = filtered_dataset
test  = filtered_dataset

if retrain_model:
  lstm_model = LstmLearner("", models_dir + model_name, auto_loading=False)
  lstm_model.update_architecture(neurons=32, nb_epochs=100,
                                 batch_size=1, number_of_hidden_layers=2)

  lstm_model.train(train, series_size)
  np.save(models_dir + model_name + ".npy", train)
  model_training.save_model_as_h5(lstm_model.get_model(), models_dir + model_name)
else:
  lstm_model = LstmLearner("", models_dir + model_name, auto_loading=True)

supervised_test = model_training.transform_supervised(test, series_size)
results = lstm_model.predict(lstm_model.get_model(), supervised_test)

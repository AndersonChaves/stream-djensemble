import xarray as xr
import core.model_training as model_training
from pandas import datetime
from core.lstm_learner import LstmLearner
from core.conv_lstm_learner import ConvLstmLearner
import numpy as np

retrain_model = False
series_size = 10

# DS_NAME
#ds_name = "CFSR-2014.nc" # CFSR - Temperature
#ds_name = "curated.chuva.alertario-malha.2020.nc" # Rio - Pluviométricos
ds_name = "radar-2016-0to17275.npy" # RFRANCE - Rain

# DS DIR
ds_dir = "datasets/"
if ds_name.startswith("CFSR"):
    ds_variable = 'TMP_L100'
else:
    ds_variable = 'rain'

model_type = "conv_lstm" # or lstm

#summer = '2014-01-01 00:00:00', '2014-03-30 23:45:00'
#winter = '2014-07-01 00:00:00', '2014-09-31 23:45:00'

if ds_name == "CFSR-2014.nc":
  summer_training = '2014-01-01 00:00:00', '2014-03-30 23:45:00'
  summer_testing = '2014-04-01 00:00:00', '2014-06-30 23:45:00'
  winter_training = '2014-07-01 00:00:00', '2014-09-30 23:45:00'
  winter_testing = '2014-10-01 00:00:00', '2014-12-31 23:45:00'
else:
  winter_training = (0, 17276 // 4)
  winter_testing = ((17276 // 4) * 1, (17276 // 4) *2)
  summer_training = ((17276 // 4) * 2, (17276 // 4) * 3)
  summer_testing = (17276 // 4 * 3, 17276)

p = "winter"
if p == "summer":
  p_training = summer_training
  p_testing = summer_testing
else:
  p_training = winter_training
  p_testing = winter_testing

#Models 1x1 - CFSR
#x0, x1 = (53, 1), (53, 1) #C0
#x0, x1 = (1, 1), (1, 1) #C1
#x0, x1 = (113, 28), (113, 28) #C2

#Models 3x3 - CFSR
#x0, x1 = (55, 2), (57, 4) #C0
#x0, x1 = (3, 3), (5,5) #C1
#x0, x1 = (115, 129), (117,131)  #C2

#Models 5x5 - CFSR
#x0, x1 = (65, 4), (69, 8) #C0
#x0, x1 = (7, 7), (11,11) #C1
#x0, x1 = (120, 131), (124,135)  #C2

#Models 7x7 - CFSR
#x0, x1 = (80, 9), (86, 15) #C0
#x0, x1 = (10, 8), (16,14) #C1
#x0, x1 = (130, 132), (136,138)  #C2

#Models 1x1 - RFRANCE
#x0, x1 = (55, 59), (55, 59) #C0
#x0, x1 = (110, 8), (110, 8) #C1
#x0, x1 = (30, 67), (30, 67) #C2

#Models 3x3 - RFRANCE
x0, x1 = (56, 61), (58, 63)
#(112, 10) a (114, 12)
#(31, 68) a (33, 70)



lts, lns = x1[0] - x0[0]+1, x1[1] - x0[1]+1
str_model_size = str(lts) + "x" + str(lns)
model_name = ds_name + '-x0=' + str(x0) + "-" + str(lts) + "x" +\
             str(lns) + "-" + str(p_training) + "-" + p

models_dir = "results/models-trained/rfrance/" + str_model_size + "/" + p + "/"
# date-time parsing function for loading the dataset
def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')

# load dataset

def load_dataset_rfrance(ds_dir, ds_name, period, x0, x1):
  ds = np.load(ds_dir + ds_name)
  ds = np.nan_to_num(ds, nan=0, posinf=0, neginf=0)
  filtered_dataset = ds[period[0]:period[1]]
  filtered_dataset = filtered_dataset[:, x0[0]: x1[0]+1, x0[1]: x1[1]+1]
  shp = filtered_dataset.shape
  filtered_dataset = np.reshape(filtered_dataset, newshape=(shp[0], 1))
  return filtered_dataset

def load_dataset_cfsr(ds_dir, ds_name, period, x0, x1):
  ds = xr.open_dataset(ds_dir + ds_name)
  filtered_dataset = ds.sel(time=slice(*period), drop=True)
  filtered_dataset = filtered_dataset.isel(lat=slice(x0[0], x1[0]+1), lon=slice(x0[1], x1[1]+1))
  np_ds = filtered_dataset["TMP_L100"].to_numpy()
  return np_ds

if ds_name == "CFSR-2014.nc":
  train = load_dataset_cfsr(ds_dir, ds_name, p_training, x0, x1)
  test = load_dataset_cfsr(ds_dir, ds_name, p_testing, x0, x1)
else:
  train = load_dataset_rfrance(ds_dir, ds_name, p_training, x0, x1)
  test  = load_dataset_rfrance(ds_dir, ds_name, p_testing, x0, x1)

def train_test_lstm():
  if retrain_model:
    lstm_model = LstmLearner("", models_dir + model_name, auto_loading=False)
    lstm_model.update_architecture(neurons=32, nb_epochs=100,
                                   batch_size=100, number_of_hidden_layers=2)
    lstm_model.train(train, series_size)
    np.save(models_dir + model_name + ".npy", train)
    model_training.save_model_as_h5(lstm_model.get_model(), models_dir + model_name)
  else:
    lstm_model = LstmLearner("", models_dir + model_name, auto_loading=True)

  supervised_test = model_training.transform_supervised(test, series_size)
  results = lstm_model.predict(lstm_model.get_model(), supervised_test)


def train_test_conv_lstm():
  if retrain_model:
    conv_lstm_model = ConvLstmLearner("", models_dir + model_name, auto_loading=False)
    conv_lstm_model.update_architecture(neurons=32, nb_epochs=100,
                                   batch_size=100, number_of_hidden_layers=2)
    conv_lstm_model.train(train, series_size)
    np.save(models_dir + model_name + ".npy", train)
    model_training.save_model_as_h5(conv_lstm_model.get_model(), models_dir + model_name)
  else:
    conv_lstm_model = ConvLstmLearner("", models_dir + model_name, auto_loading=True)
  return conv_lstm_model


#supervised_test = model_training.transform_supervised(test, series_size)
#supervised_test = model_training.build_dataset(test, series_size)

if model_type == "lstm":
  model = train_test_lstm()
  supervised_test = model_training.transform_supervised(test, series_size)
else:
  model = train_test_conv_lstm()
  x_train, x_val, y_train, y_val, train_dataset, val_dataset = model_training.conv_transform_supervised(test, series_size)
  supervised_test = [x_train, y_train]

results = model.predict(model.get_model(), supervised_test)

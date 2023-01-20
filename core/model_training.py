from pandas import DataFrame
from pandas import Series
from pandas import concat
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
from matplotlib import pyplot
import numpy


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in reversed(range(1, lag+1))]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(ds, scaler=None):
    if scaler is None:
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(ds)

    # transform train
    ds = ds.reshape(ds.shape[0], ds.shape[1])
    ds_scaled = scaler.transform(ds)
    return ds_scaled, scaler

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, is_stateful=False,
             number_of_hidden_layers=1):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(layers.Input(shape=(None, 1)))
    for _ in range(number_of_hidden_layers):
        model.add(layers.LSTM(neurons, stateful=is_stateful,  return_sequences=True))
    model.add(layers.LSTM(neurons, stateful=is_stateful, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if not is_stateful:
        model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True)
    else:
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
            model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, len(X), 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

def transform_supervised(raw_series, series_size, differentiate=False):
    time_series = raw_series

    # transform data to be stationary
    if differentiate:
        time_series = difference(time_series, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(time_series, lag=series_size)
    supervised_values = supervised.values

    return supervised_values

def save_model_as_h5(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")

def load_model_from_h5(model_directory, model_name):
    # Loads metadata from json file
    json_file = open(model_directory + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(model_directory + model_name + '.h5')
    # Loads metadata from database
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

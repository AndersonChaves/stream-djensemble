from pandas import DataFrame
from pandas import Series
from pandas import concat
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy

from core.learner import UnidimensionalLearner
import core.model_training as mt


class LstmLearner(UnidimensionalLearner):
    def __init__(self, model_directory, model_name,
                  auto_loading = False):
        is_temporal_model = True
        self.series_size = -1
        super().__init__(model_directory, model_name,
                          is_temporal_model=is_temporal_model, auto_loading=auto_loading)
        self.update_training_options()
        self.update_arquitecture()

    def update_training_options(self, differentiate=False,
              train_as_stateful=False, scale_data=False):
        self.differentiate     = differentiate
        self.train_as_stateful = train_as_stateful
        self.scale_data        = scale_data
        self.scaler            = None

    def update_arquitecture(self, neurons=32, nb_epochs=100):
        self.neurons   = neurons
        self.nb_epochs = nb_epochs

    def train(self, training_series, series_size):
        self.series_size = series_size
        supervised_values = mt.transform_supervised(training_series, series_size)
        if self.scale_data:
            # transform the scale of the data
            train_scaled, scaler = mt.scale(supervised_values)
        else:
            train_scaled, scaler = supervised_values, None

        # fit the model
        lstm_model = mt.fit_lstm(train_scaled, batch_size=1, nb_epoch=self.nb_epochs,
                              neurons=self.neurons, is_stateful=self.train_as_stateful)

        if self.train_as_stateful:
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=1)
        self.model = lstm_model

    def predict(self, lstm_model, raw_testing_series, differentiate=False, scaler=None):
        testing_series = raw_testing_series
        len_series = len(testing_series[0, 0:-1])
        if scaler is not None:
            X = testing_series.reshape(len(testing_series), 1)
            testing_series = scaler.transform(X)

        # walk-forward validation on the test data
        predictions = list()
        for i in range(len_series, len(testing_series)):
            # make one-step forecast
            X, y = testing_series[i, 0:-1], testing_series[i, -1]
            yhat = mt.forecast_lstm(lstm_model, 1, X)

            if differentiate:
                yhat = mt.invert_scale(scaler, X, yhat)
                yhat = mt.inverse_difference(raw_testing_series.values, yhat, len(testing_series) + 1 - i)

            # store forecast
            predictions.append(yhat)
            expected = y
            print('Indexs=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

            # report performance
        rmse = sqrt(mean_squared_error(testing_series[len_series:, -1], predictions))
        print('Test RMSE: %.3f' % rmse)
        # line plot of observed vs predicted
        pyplot.plot(testing_series[len_series:, -1])
        pyplot.plot(predictions)
        pyplot.show()
import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegressor:
    def __init__(self, X, y):
        # self.X = X # np.array([[x] for x in X])
        self.X = np.array([[x] for x in X])
        #self.X = np.array(X).reshape((len(X[0]), 1))
        #self.y = y # np.array([[_y] for _y in y])
        self.y = np.array([[_y] for _y in y])

    def train(self):
        self.regression_line = LinearRegression().fit(self.X, self.y)

    # Expected 2d numpy array
    def predict(self, x):
        x = np.array([[_x] for _x in [x]])
        return self.regression_line.predict(x)

class FrameRegressor:
    def __init__(self, X):
        self.X = X

        shape = X.shape
        seq_size = shape[0]
        self.regressor_matrix = []
        for i in range(shape[1]):
            regressor_row = []
            for j in range(shape[2]):
                l = LinearRegressor([range(seq_size)], X[:, i, j])
                regressor_row.append(l)
            self.regressor_matrix.append(regressor_row)

    def train(self):
        for row in self.regressor_matrix:
            for regressor in row:
                regressor.train()

    # Expected 3d numpy array
    def predict(self, size_ahead):
        shape = self.X.shape
        start = shape[0] + 1
        x = range(start, start + size_ahead)
        y = np.zeros(shape = (size_ahead, self.X.shape[1], self.X.shape[2]))

        for i in range(shape[1]):
            for j in range(shape[2]):
                regressor = self.regressor_matrix[i][j]
                line = regressor.predict(x)
                y[:, i, j] = line.reshape(line.shape[0])
        return y




# Example
# X = np.array([[1], [3], [5], [7]])
# y = np.array([[10], [30], [50], [70]])
# r = LinearRegressionStrategy(X, y)
# r.train()
# print(r.predict(np.array([[20]])))

# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
# print(reg.predict(np.array([[5]])))
import math
import numpy as np
import h5py
import xarray as xr
from .tile import Tile
import os.path
import netCDF4 as nc

class DatasetManager:
    def __init__(self, data_path):
        self.data_path = data_path

    def loadDataset(self, start = None, end = None, ds_attribute=None):
        data_path = self.data_path
        if not os.path.isfile(data_path):
            raise Exception("Data Path " + data_path + " is empty.")

        elif data_path[-4:] == '.npy':
            self.ds = np.load(data_path, mmap_mode='c')
            self.accessor = AccessorNumpy()
            #self.ds = np.nan_to_num(self.ds, nan=0, posinf=0, neginf=0)
        elif data_path[-3:] == '.nc':
            self.ds = nc.Dataset(data_path)
            self.ds = np.array(self.ds[ds_attribute])#[:100])
            self.accessor = AccessorNumpy()
            #self.ds = np.nan_to_num(self.ds, nan=0, posinf=0, neginf=0)
        else:
            self.ds = xr.load_dataset(data_path).sortby('time')
            self.accessor = AccessorXArray()

    def read_instant(self, t_instant):
        return self.accessor.read_instant(self.ds, t_instant)

    def read_window(self, t_instant, window_size):
        return self.accessor.read_window(self.ds, t_instant, window_size)

    def read_all_data(self):
        return self.ds

    def filter_by_region(self, x, y):
        self.ds = self.accessor.filter_by_region(self.ds, x, y)

    def filter_by_date(self, ds, lower_date, upper_date):
        filtered_dataset = ds.sel(time=slice(lower_date, upper_date), drop=True)
        filtered_dataset['rain'] = filtered_dataset['rain'].fillna(0)
        return filtered_dataset

    def loadTemperatureDataset(self, dataPath):
        with h5py.File(dataPath) as f:
            dataset = f['real'][...]
        return dataset

    def get_data_from_tile(self, dataset: np.array, tile: Tile):
        sx, sy = tile.get_start_coordinate()
        ex, ey = tile.get_end_coordinate()
        return dataset[:, sx:ex+1, sy:ey+1]

    def get_spatial_shape(self):
        return self.accessor.get_spatial_shape(self.ds)

    def filter_frame_by_query_region(self, dataset, x1, x2):
        data_window = dataset[x1[0]:x2[0], x1[1]:x2[1]]
        return data_window

class Accessor():
    def read_instant(self, ds, t_instant):
        pass

    def get_spatial_shape(self, ds):
        pass

    def read_window(self, ds, t_instant, window_size):
        pass

class AccessorNumpy(Accessor):
    def read_instant(self, ds, t_instant, region: tuple =None):
        if region == None:
            return ds[t_instant]
        else:
            x, y = region
            return ds[t_instant, x, y]

    def read_window(self, ds, t_instant, window_size):
        return ds[t_instant:t_instant+window_size]

    def filter_by_region(self, ds, x, y):
        return ds[:, x[0]:x[1], y[0]:y[1]]

    def get_spatial_shape(self, ds):
        return ds.shape[1:]

class AccessorNetcdf(Accessor):
    def read_instant(self, ds, t_instant):
        return ds[t_instant]

    def get_spatial_shape(self, ds):
        return ds.shape[1:]

class AccessorXArray(Accessor):
    def read_instant(self, ds, t_instant):
        return ds[t_instant]


def extend_dataset(x: int, y: int, data: np.array):
    '''
    Extends the dataset to a multiple of the specified spatial
    coordinates x and y.

    First it calculates ext, the size of the dimension that must be extended
    so that the final size is multiple of the models input.
    # Ex. mx=3, tx = 10: ceil(10/3)*3-10=4*3-10=2
    Then duplicates the last column of the data to fit the models input size
    # Ex. x y ===> x y y
    #     z w      z w w

    :param x: x target coordinate multiple
    :param y: y target coordinate multiple
    :param data: A dataset of shape (1, t, xref, yref, 1)
    :return: Extended dataset
    '''
    while not (data.shape[2] % x == 0 and data.shape[3] % y == 0):
        if data.shape[2] % x != 0:
            ext_x = abs(math.ceil(data.shape[2] / x) * x - data.shape[2])
            data = np.concatenate((data, data[:, :, -ext_x:, :, :]), axis=2)

        if data.shape[3] % y != 0:
            ext_y = abs(math.ceil(data.shape[3] / y) * y - data.shape[3])
            data = np.concatenate((data, data[:, :, :, -ext_y:, :]), axis=3)
    return data


# ----------------------------------
# For tests
def synthetize_dataset(shape, noise=0):
    time, lat, long = shape

    series = []
    for i in range(lat // 3 * long):
        series.append([k for k in range(time)])
    for i in range(lat // 3 * long):
        series.append([3 - (k % 3) for k in range(time)])
    for i in range((lat // 3 + lat % 3) * long):
        series.append([10 for _ in range(time)])

    array = np.reshape(np.array(series), (lat, long, time))
    array = np.swapaxes(array, 0, 2)

    if noise > 0:
        noise = np.random.normal(0, noise, array.shape)
        array = array + noise
    return array


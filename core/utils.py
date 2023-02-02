import os
from os.path import exists
import fnmatch

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_all_sub_directories(path):
    return [x[0] for x in os.walk(path)]

def list_all_directories(path):
    return next(os.walk(path))[1]

def get_names_of_models_in_dir(models_path):
    models = fnmatch.filter(os.listdir(models_path), '*.h5')
    for i, m in enumerate(models):
        models[i] = m.split('.h5')[0]
    return models

def list_all_files_in_dir(path, extension=''):
    res = []
    for file in os.listdir(path):
        if file.endswith(extension) or file.endswith('.' + extension):
            res.append(file)
    return(res)

def file_exists(path_to_file):
    return exists(path_to_file)

def get_file_name_from_path(full_path_to_file: str):
    return full_path_to_file.split("/")[-1]
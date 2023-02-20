import pickle

import h5py
import numpy as np


def read_h5(path):
    """Read from simulation dictionary in a .h5 file

    Args:
        path: file path to data
    Returns:
        data_dictionary: dictionary that contains the items in the h5 file
    """
    data_dict = {}
    with h5py.File(path, "r") as hf:
        for k in hf.keys():
            data_dict[k] = hf.get(k)[()]
    return data_dict


def unpickle_file(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    return data, file


def pickle_file(file_name, data):
    file = open(file_name, "wb")
    pickle.dump(data, file)
    file.close()
    return

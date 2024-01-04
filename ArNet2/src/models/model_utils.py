import os
import pathlib
import pickle
import numpy as np
import h5py

from xgboost import XGBClassifier
from OneDCNN import OneDCNN
from ResNet import ResNet
from ArNet2.src.models.ArNet2 import ArNet2
from CRNN import CRNN
from RNN import RNN

class_funcs = {
    'XGB': XGBClassifier,
    '1D-CNN': OneDCNN,
    'ResNet': ResNet,
    'ArNet2': ArNet2,
    'CRNN': CRNN,
    'RNN': RNN
}

use_history = {
    'XGB': False,
    '1D-CNN': False,
    'ResNet': False,
    'ArNet2': True,
    'CRNN': True,
    'RNN': True
}

def save_model(final_dict, clf, path, algo):
    # Check if the directory exists, create it otherwise
    directory, filename = os.path.split(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Add increment if the file already exists
    incr = 0
    while os.path.exists(path):
        incr += 1
        path = pathlib.PurePath(directory) / (filename.split(".")[0].split("_")[0] + f"_{str(incr)}.pkl")

    # Save model
    with open(path, 'wb') as file:
        if "XGB" in algo:
            final_dict['classifier'] = clf
        else:
            final_dict['classifier'] = clf.get_state_dict()
        pickle.dump(final_dict, file)


def load_model(path, algo, path_feature_extractor=None):
    with open(path, 'rb') as file:
        model_dict = pickle.load(file)
        if algo != 'XGB':
            hypercomb = model_dict["hyperparameters"]
            if algo=="ArNet2":
                model = class_funcs[algo](**hypercomb, path_feature_extractor=path_feature_extractor)
            else:
                model = class_funcs[algo](**hypercomb)
            model.set_state_dict(model_dict['classifier'])
            model_dict['classifier'] = model
    return model_dict


def save_df(df, path):
    directory, filename = os.path.split(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(path)

def save_dict_to_hdf5(dic, filename):

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')



def recursively_save_dict_contents_to_group( h5file, path, dic):

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        print(key)
        key = str(key)
        if isinstance(item, list):
            # item = np.array(item)
            item = {i: k for i, k in enumerate(item)}
            print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32,int)):
            #print( 'here' )
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key].value, item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        # else:
            #print(item)
            # raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group( h5file, path):

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

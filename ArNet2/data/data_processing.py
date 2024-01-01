import itertools
import numpy as np
import copy
import pandas as pd
import math
import utils.consts as cts
import utils.graphics as graph


def hyperparamaters_comb(dict_vals):
    """ This function returns all the possible combinations of hyperparameters.
    Returns a list of dictionaries. Each dict contains the name and the value of the hyperparameter
    for the given combination.
    :param dict_vals: Dictionnary containing as keys the different hyperparameter names and as values the different values they take.
    :returns dict_comb: The output list."""
    hyper_names = list(dict_vals.keys())
    n_hyper = len(hyper_names)
    combinations = list(itertools.product(*list(dict_vals.values())))
    dict_comb = [{hyper_names[i]: comb[i] for i in range(n_hyper)} for comb in combinations]
    return dict_comb


def is_integer(float_num):

    """ This function returns if the given number is an integer
    :param float_num: the input number
    :returns bool: boolean, True if the input is an integer, False otherwise."""
    return math.ceil(float_num) == float_num

def cumsum_reset(np_arr):

    """ This function performs the 'cumsum' operation on an array, resetting itself each time a zero is
        found in the array.
        :param np_arr: Numpy Array containing the input.
        :returns res: Numpy Array containing the cumulative sum."""

    return np.array(pd.DataFrame(np_arr).astype(int).apply(lambda x: x.groupby((~x.astype(bool)).cumsum()).cumsum())).reshape(-1)

def resample_by_interpolation(signal, input_fs, output_fs):

    """ This function interpolates a signal from an original to a desired sampling frequency.
        :param signal: The input signal.
        :param input_fs: The original sampling frequency.
        :param output_fs: The desired sampling frequency.
        :returns resampled_signal: The signal resampled to output_fs."""

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def normalize_data(data_train, data_test):

    """ This functions normalizes train and test sets (Z-normalization) based on the parameters (mean and std.)
        derived from the training set.
        :param data_train: Numpy array containing the training data (dimensions: (n_samples, n_features)).
        :param data_test: Numpy array containing the test data (dimensions: (n_samples, n_features))."""

    mean_train = data_train.mean(axis=0)
    std_train = data_train.std(axis=0)
    data_train = (data_train - mean_train) / std_train
    data_train[np.isnan(data_train)] = 0.0
    if len(data_test) > 0:
        data_test = (data_test - mean_train) / std_train
        data_test[np.isnan(data_test)] = 0.0
    return data_train, data_test, mean_train, std_train


def fillna(X):

    """ Fills the input array colum-wise using the average value over the column.
        :param X: the input array.
        :returns X_new: the imputed array.
        :returns means: the means over the different columns."""

    if len(X) > 0:
        X_new = copy.deepcopy(X)
        means = np.array(X_new.shape[1])
        for i in range(X_new.shape[1]):
            if np.any(np.isnan(X_new[:, i])):
                if means is None:
                    curr_mean = np.nanmean(X_new[:, i])
                    X_new[np.isnan(X_new[:, i]), i] = curr_mean
                    means[i] = curr_mean
                else:
                    X_new[np.isnan(X_new[:, i]), i] = means[i]

        return X_new, means
    else:
        return X, np.array([])


def check_stratification(X_train, X_test, y_train, y_test, plot=False, feats_name=None, n_points=50):

    """ This function verifies the stratification between train and test data
        for each one of the different features (verification of the distributions) and for the labels
        (checks if the proportions for each class is similar)
        :param X_train: The training dataset.
        :param X_test: The test dataset.
        :param y_train: The training labels.
        :param y_test: The test labels.
        :param plot (optional): Boolean value to indicate if the features histograms need to be plotted.
        :param feats_name (optional): The name of the different features ordered in a list.
        :param n_points (optional): Number of points required for the histograms binning."""

    if len(y_train) == 0 or len(y_test) == 0:
        return 
    else:
        for index in range(X_train.shape[1]):

            data_train, data_test = X_train[:, index], X_test[:, index]
            max = np.max([data_train.max(), data_test.max()])
            min = np.min([data_train.min(), data_test.min()])
            if plot:
                fig, axes = graph.create_figure()
                labs = np.unique(y_train)
                for lab in labs:
                    weights_train = np.ones_like(data_train[y_train == lab]) / len(data_train[y_train == lab])
                    weights_test = np.ones_like(data_test[y_test == lab]) / len(data_test[y_test == lab])
                    axes[0][0].hist(data_train[y_train == lab], np.linspace(min, max, n_points), color=cts.COLORS[lab],
                                    label='Label: ' + str(lab), weights=weights_train, rwidth=1 - lab / (len(labs)))
                    axes[0][0].hist(data_test[y_test == lab], np.linspace(min, max, n_points), color=cts.COLORS[lab + 2],
                                    label='Label: ' + str(lab), weights=weights_test, rwidth=1 - lab / (len(labs)))
                if feats_name is not None:
                    graph.complete_figure(fig, axes, suptitle=feats_name[index])
                else:
                    graph.complete_figure(fig, axes)

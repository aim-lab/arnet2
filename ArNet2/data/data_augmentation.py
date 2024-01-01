"""Drawn from two papers : """
""" This is a simple example to apply data augmentation to time-series data (e.g. wearable sensor data). If it helps your research, please cite the below paper.
T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.
https://dl.acm.org/citation.cfm?id=3136817
https://arxiv.org/abs/1706.00527
@inproceedings{TerryUm_ICMI2017, author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana}, title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks}, booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction}, series = {ICMI 2017}, year = {2017}, isbn = {978-1-4503-5543-8}, location = {Glasgow, UK}, pages = {216--220}, numpages = {5}, doi = {10.1145/3136755.3136817}, acmid = {3136817}, publisher = {ACM}, address = {New York, NY, USA}, keywords = {Parkinson\&#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor}, }
You can freely modify this code for your own purpose. However, please leave the above citation information untouched when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.
"""
"""
B. K. Iwana and S. Uchida, "An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks," arXiv, 2020.
https://github.com/uchidalab/time_series_augmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import utils.consts as cts
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


###############
##### TIME TRANSFORMATION
###############
def jitter(X, sigma=0.05):
    """ Add Gaussian noise to the input rr windows."""

    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def permutation(x, max_segments=5, seg_mode="equal"):
    """ Split the window into sub-windows and permute them."""

    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(3, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def stretching(x, y, sigma=0.5):
    """ Scale the window relatively to its mean. Stretch if it is AF, and reduce otherwise."""

    # Remove the mean
    means = np.mean(x, axis=1, keepdims=True)
    x_zero_mean = x - means

    # Stretch if it is AF, and reduce otherwise, with an amplitude factor depending on sigma
    sign = -1 + 2 * (y==True)
    factors = 1 + sign * np.abs(np.random.normal(loc=0, scale=sigma, size=(x.shape[0],)))
    x_zero_mean_scaled = np.multiply(x_zero_mean, factors[:, np.newaxis])

    # Add the mean
    x_new = x_zero_mean_scaled + means
    return x_new


def magnitude_warp(x, sigma=0.2, knot=4):
    """ Magnitude warping around the mean axis (positive stretching)."""

    orig_steps = np.arange(x.shape[1])
    means = np.mean(x, axis=1, keepdims=True)
    x_zero_mean = x - means
    x_zero_mean = x_zero_mean[:, :, np.newaxis]

    random_warps = 1 + np.abs(np.random.normal(loc=0, scale=sigma, size=(x_zero_mean.shape[0], knot + 2, x_zero_mean.shape[2])))
    warp_steps = (np.ones((x_zero_mean.shape[2], 1)) * (np.linspace(0, x_zero_mean.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x_zero_mean)
    for i, pat in enumerate(x_zero_mean):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x_zero_mean.shape[2])]).T
        ret[i] = pat * warper

    ret = ret[:, :, 0]
    x_new = ret + means

    return x_new


def flipping(x):
    """ Flipping around the mean axis."""

    means = np.mean(x, axis=1, keepdims=True)
    x_zero_mean = x - means
    x_zero_mean_flipped = - x_zero_mean
    x_new = x_zero_mean_flipped + means
    return x_new


def sub_flipping(x, min_win_len=20):
    """ Flipping of random-sized sub-windows."""

    x_new = x.copy()
    for i in range(len(x)):
        a = np.random.randint(0, x.shape[1]-min_win_len)
        b = np.random.randint(a+min_win_len, x.shape[1])
        xx_sub = x[i, a:b]
        xx_sub_flipped = -(xx_sub - np.mean(xx_sub)) + np.mean(xx_sub)
        x_new[i, a:b] = xx_sub_flipped
    return x_new


def simple_overlapping_windows(X, y):
    """ Add overlapping windows between AF episodes, giving the label AF only if the overlapping window is between two AF windows."""

    rr_new = []
    for i in range(len(X) - 1):
        xx = X[i]
        yy = y[i]
        xx_ = X[i + 1]
        yy_ = y[i + 1]
        if (xx[-1] == xx_[-1]) and (yy == True) and (
                yy_ == True):  # 2 consecutive windows from the same patient and both AF
            rr_new.append(np.concatenate((xx[30:60], xx_[:30])))
    rr_new = np.array(rr_new)
    return rr_new


###############
##### PATTERN MIXING
###############
def smote(X, y, method="simple"):
    """ Apply SMOTE in the input space."""
    if method == "simple":
        oversample = SMOTE()
        X_smote, y_smote = oversample.fit_resample(X, y)
    elif method == "undersample":
        over = SMOTE(sampling_strategy=0.2)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_smote, y_smote = pipeline.fit_resample(X, y)
    elif method == "borderline":
        oversample = BorderlineSMOTE()
        X_smote, y_smote = oversample.fit_resample(X, y)
    elif method == "adasyn":
        oversample = ADASYN()
        X_smote, y_smote = oversample.fit_resample(X, y)
    return X_smote, y_smote


###############
##### MAIN FUNCTIONS
###############
def augment_funcs(method_name, rr, y):
    if method_name == "sub_flipping":
        return sub_flipping(rr, min_win_len=20)
    elif method_name == "flipping":
        return flipping(rr)
    elif method_name == "jittering":
        return jitter(rr, sigma=0.1)
    elif method_name == "stretching":
        return stretching(rr, y, sigma=0.1)
    elif method_name == "permutation":
        return permutation(rr, max_segments=6, seg_mode="equal")
    elif method_name == "magnitude_warp":
        return magnitude_warp(rr, sigma=0.2, knot=4)

def augment(data_train, method_name="sub_flipping"):
    """ Perform data_augmentation on a training set.
    Only the two main methods were kept, but you can add anything you want based on the above implemented functions."""
    X_train, y_train, t_s = data_train
    rr_train = X_train[:, :-3].astype('float32')

    if method_name == 'LatentSMOTE':
        # LatentSMOTE
        # folder_date_deepsmote = "2021-08-11T15:50:26"
        path_models_deepsmote = cts.REPO_DIR / "data" / "splits" / "model_input" / "LatentSMOTE"
        X_gen, y_gen, t_s_gen = np.load(path_models_deepsmote / "X_gen_extra_25.npy", allow_pickle=True), np.load(path_models_deepsmote / "y_gen.npy", allow_pickle=True), np.load(path_models_deepsmote / "t_s_gen.npy", allow_pickle=True)

    elif method_name == 'LatentSMOTE+sub_flipping':
        path_models_deepsmote = cts.REPO_DIR / "data" / "splits" / "model_input" / "LatentSMOTE"
        X_gen_SMOTE, y_gen_SMOTE, t_s_gen_SMOTE = np.load(path_models_deepsmote / "X_gen_extra_25.npy", allow_pickle=True), np.load(path_models_deepsmote / "y_gen.npy", allow_pickle=True), np.load(path_models_deepsmote / "t_s_gen.npy", allow_pickle=True)
        rr_gen = augment_funcs('sub_flipping', rr_train, y_train)
        X_gen_flip = np.concatenate((rr_gen, X_train[:, -3:]), axis=1)
        y_gen_flip, t_s_gen_flip = y_train, t_s

        X_gen, y_gen, t_s_gen = np.concatenate((X_gen_SMOTE, X_gen_flip), axis=0), np.concatenate((y_gen_SMOTE, y_gen_flip),
                                                                                         axis=0), np.concatenate((t_s_gen_SMOTE, t_s_gen_flip), axis=0)
    else:
        # Transformation based DA
        rr_gen = augment_funcs(method_name, rr_train, y_train)
        X_gen = np.concatenate((rr_gen, X_train[:, -3:]), axis=1)
        y_gen, t_s_gen = y_train, t_s

    # Filter the generated data on the AF patients only
    X_gen, y_gen, t_s_gen = X_gen[X_gen[:, -2] >= 1], y_gen[X_gen[:, -2] >= 1], t_s_gen[X_gen[:, -2] >=1]

    # Add the generated data to the training data
    X_train, y_train, t_s = np.concatenate((X_train, X_gen), axis=0), np.concatenate((y_train, y_gen), axis=0), np.concatenate((t_s, t_s_gen), axis=0)
    data_train = (X_train, y_train, t_s)

    return data_train
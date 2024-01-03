# The main packages used all along the project are loaded here and
# divided by categories
# This script also adds to PATH the different important
# path containing the basic repo directories (utils and parsing)

try:
    import utils.consts as cts
    import utils.graphics as graph
    import utils.feature_comp as fc
    import utils.data_processing as dp
    import utils.in_out as i_o
    import utils.dat_reader as dr
except ModuleNotFoundError:
    import consts as cts
    import graphics as graph
    import feature_comp as fc
    import data_processing as dp
    import in_out as i_o
    import dat_reader as dr

# system management operations
import os
import shutil
import sys
import warnings
import re
import glob
import pathlib
from dataclasses import dataclass

# Data Processing
import numpy as np
import wfdb
import wfdb.processing as processing
import pandas as pd
import multiprocessing
import subprocess
import itertools
import math
import random
import scipy
import scipy.signal as signal
from scipy.signal import savgol_filter, butter, sosfreqz, sosfiltfilt
import scipy.interpolate as interp

import mne

# Graphics
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# I/O
import csv
import xlrd
import xlsxwriter
import scipy.io as sio
import pyedflib
import joblib
import zipfile
import io
import copy
import pickle
import mat73
from scipy.io import savemat

# time-related functions
import datetime as dt
import time
from dateutil.parser import parse

#  machine learning
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, \
    accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(cts.SEED)
random.seed(cts.SEED)
warnings.filterwarnings('ignore')


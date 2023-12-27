# The main packages used all along the project are loaded here and
# divided by categories
# This script also adds to PATH the different important
# path containing the basic repo directories (utils and parsing)

try:
    import utils.consts as cts
except ModuleNotFoundError:
    import consts as cts

# management operations
import os
import shutil
import sys
import warnings
import re
import glob
import pathlib

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

# time-related functions
import datetime as dt
import time

#  machine learning
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree

np.random.seed(cts.SEED)
random.seed(cts.SEED)
warnings.filterwarnings('ignore')


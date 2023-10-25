# The main packages used all along the project are loaded here and
# divided by categories
# This script also adds to PATH the different important
# path containing the basic repo directories (utils and parsing)

try:
    import utils.consts as cts
except ModuleNotFoundError:
    import consts as cts
import os
import shutil
import sys
import warnings

# Data Processing
import numpy as np
import pandas as pd
import scipy.signal as signal
import multiprocessing
import subprocess
import itertools
import math

# Graphics
import matplotlib.pyplot as plt

# I/O
import csv
import xlrd
import xlsxwriter
import scipy.io as sio
import wfdb
import pyedflib
import joblib
import zipfile
import io
import copy
import random
import pickle
import re
import matplotlib.ticker as tick

sys.path.append('C:\\Users\\ShanyBiton\\Documents\\repos\\Generalization\\parsing\\')
sys.path.append('C:\\Users\\ShanyBiton\\Documents\\repos\\Generalization\\utils\\')
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')
np.random.seed(cts.SEED)
random.seed(cts.SEED)
warnings.filterwarnings('ignore')


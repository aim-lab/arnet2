import pathlib
import os
import numpy as np

# TODO: remove consts which were inherited from Armand and are not in use
# Paths definition. 'nt' refers to Windows, 'posix' as Linux (triton01 GPU cluster)
if os.name == 'nt':
    BASE_DIR = pathlib.PurePath('V:\\AIMLab')
    REPO_DIR = pathlib.PurePath('C:\\Users\\ShanyBiton\\Documents\\repos\\Shany_Repo')
    REPO_DIR_POSIX = pathlib.PurePath(str(REPO_DIR).replace('\\', '/').replace('C:', '/mnt/c'))
    N_PROCESSES = 8
elif os.name == 'posix':
    BASE_DIR = pathlib.PurePath('/MLAIM')
    REPO_DIR = pathlib.PurePath('/home/shanybiton/repos/Shany_Repo')
    REPO_DIR_POSIX = REPO_DIR
    N_PROCESSES = 15

# Paths definition
DATA_DIR = BASE_DIR / "databases"
CLASSIFIERS_DIR = BASE_DIR / "AIMLab" / "Shany" / "Classifiers"
SNAPSHOTS_DIR = BASE_DIR / "AIMLab" / "Shany" / "Snapshots"
MATLAB_TEST_VECTORS_DIR = BASE_DIR / "AIMLab" / "Shany" / "Matlab_test_vectors"
EPLTD_PROG_DIR = BASE_DIR / "AIMLab" / "Shany" / "epltd" / "epltd_all"
WQRS_PROG_DIR = pathlib.PurePath('/usr/local/bin/wqrs')
PARSING_PROJECT_DIR = REPO_DIR_POSIX / "parsing"
ERROR_ANALYSIS_DIR = BASE_DIR / "AIMLab" / "Shany" / "ErrorAnalysis"
PREPROCESSED_DATA_DIR = BASE_DIR / "AIMLab" / "Shany" / "PreprocessedDatabases"
GEN_ANN_DIR = BASE_DIR / "AIMLab" / "Shany" / "Annotations"
REANNOTATION_DIR = BASE_DIR / "AIMLab" / 'Shany' / 'medAIM'
MODEL_DIR = REPO_DIR / "saved_models"

# Labels definition
PATIENT_LABEL_UNKNOWN = 5
PATIENT_LABEL_NON_AF = 0
PATIENT_LABEL_AF_MILD = 1
PATIENT_LABEL_AF_MODERATE = 2
PATIENT_LABEL_AF_SEVERE = 3
PATIENT_LABEL_OTHER_CVD = 4

PATIENT_LABELS = np.array([PATIENT_LABEL_NON_AF, PATIENT_LABEL_AF_MILD, PATIENT_LABEL_AF_MODERATE,
                           PATIENT_LABEL_AF_SEVERE, PATIENT_LABEL_OTHER_CVD, PATIENT_LABEL_UNKNOWN])

WINDOW_LABEL_UNKNOWN = -1
WINDOW_LABEL_NON_AF = 0
WINDOW_LABEL_AF_RB = [147, 148]
WINDOW_LABEL_AF = 1
WINDOW_LABEL_OTHER = 2
WINDOW_LABEL_NON_AF_RB = 19

WINDOW_LABELS = np.array([WINDOW_LABEL_NON_AF, WINDOW_LABEL_AF, WINDOW_LABEL_OTHER, WINDOW_LABEL_UNKNOWN])

# Feature names
BASELINE_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medFreq'])
HRV_FEATURES = np.array(['AVNN', 'medRR', 'minRR', 'maxRR', 'VarRR', 'VardRR', 'VarddRR', 'CoeffVarRR', 'RMSSD'])
POINCARE_FEATURES = np.array(['poinc_sd1', "poinc_sd2", "poinc_ratio"])
IMPLEMENTED_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medHR',
                                 'SDNN', 'SEM', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2', 'sq_map_intercept',
                                 'sq_map_linear', 'sq_map_quadratic', 'PIP', 'IALS', 'PSS', 'PAS'])
SELECTED_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medHR',
                              'SDNN', 'SEM', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2', 'PIP', 'IALS', 'PSS',
                              'PAS'])
ANNOTATION_TYPES = np.array(['epltd0', 'xqrs', 'gqrs', 'wqrs', 'jqrs'])

# Label names used in graphs
BASE_WINDOWS = np.arange(60, 121, 10)
COLORS = np.array(['dodgerblue', 'orange', 'red', 'purple', 'brown', 'yellow'])
GLOBAL_LAB_TITLES = np.array(['$Non-AF$', '$AF_{Mild}$', '$AF_{Mod}$', '$AF_{Se}$', '$O$', '$Unknown$'])
BINARY_TITLES = np.array(['$Non-Prominent-AF$', '$Prominent-AF$'])
WINDOW_LAB_TITLES = np.array(['$N$', '$AF$', '$O$', '$Unknown$'])

# Constants definitions (time related)
N_S_IN_HOUR = 3600
N_MS_IN_S = 1000
N_HOURS_IN_DAY = 24
N_MIN_IN_HOUR = 60
N_SEC_IN_MIN = 60

# Parameters used for data filtering in the different scripts
RR_OUTLIER_THRESHOLD_SUP = 10  # Sup Threshold to exclude a RR interval window (excluded if one RR exceeds this
# value, in seconds)
RR_OUTLIER_THRESHOLD_INF = 0  # Inf Threshold to exclude a RR interval window (excluded if one RR is below this
# value, in seconds)
RR_FILE_THRESHOLD = 3 * N_S_IN_HOUR  # Criterion for exclusion of a whole recording (Above 3 Hours of corrupted data,
# file is excluded)
AF_MILD_THRESHOLD = 30  # Threshold on AF Burden to define a patient as Mild AF patient. CAREFUL ! TIME IN SECONDS HERE
AF_MODERATE_THRESHOLD = 0.04  # Threshold on AF Burden to define a patient as Moderate AF patient
AF_SEVERE_THRESHOLD = 0.8  # Threshold on AF Burden to define a patient as Severe AF patient
AF_PERSISTENT_THRESHOLD = 0.99  # Threshold on AF burden to define patient as Persistent AF patient
SQI_FILE_THRESHOLD = 0.75  # Threshold on the number of corrupted windows (based on bsqi criterion) to exclude a file
SQI_WINDOW_THRESHOLD = 0.8  # Threshold on the bsqi criterion to exclude a window
MISSING_ANN_THRESHOLD = 0.75  # Threshold on the percentage of missing annotations to exclude a file.
OSA_AHI_THRESHOLD = 15  # Threshold on the Apnea Hypopnea Index (AHI) to define a patient as OSA
FRAGMENTATION_LIM_SMALL_SEG = 3  # The limit to set a segment as short for fragmentation features
LOW_AGE = 60
HIGH_AGE = 75

START_NIGHT = 22  # Considering night starts at 10 P.M.
END_NIGHT = 5  # Considering night starts at 5 A.M.

EPLTD_FS = 200  # Sampling frequency of the signals which can be analyzed by EPLTD detector for peak detection

# Sign definitions for weak classifiers
INF = -1
SUP = 1

# Seed for pseudo-random number generation
SEED = 42

# Metrics computed all over the scripts
METRICS = np.array(['Accuracy', 'Fb-Score', 'Se', 'Sp', 'PPV', 'NPV', 'AUROC'])
print_metrics = ['Fb-Score', 'AUROC', 'Se', 'Sp', 'PPV', 'E_AF']

# result visualization and evaluation consts
score = 'Fb-Score'
db_name_dict = {'UVAFDB': 'UVAF', 'JPAFDB': 'SHDB', 'RBAFDB': 'RBDB', 'CPSCDB': 'CPSC', 'AFDB': 'AFDB',
                'LTAFDB': 'LTAFDB'}
models = ['XGB', 'ArNet', 'ArNet2']  # , 'CRNN(noHT)', "CRNN+DA(noHT)"]  # Not sure about CRNN results, be wary
evaluation_sets = ['metrics_train', 'metrics_test_UVAFDB', 'metrics_test_JPAFDB', 'metrics_test_RBAFDB',
                   'metrics_test_CPSCDB', 'metrics_test_all', 'metrics_Female_group', 'metrics_Male_group',
                   'metrics_low_age_group', 'metrics_mid_age_group', 'metrics_high_age_group']
evaluation_sets_name_list = ['UVAF train', 'UVAF test', 'SHDB', 'RBDB', 'CPSC', 'Combined \n test set', 'Female',
                             'Male',
                             f'${{\leq}}${LOW_AGE}', f'{LOW_AGE} to {HIGH_AGE}', f'${{\geq}}${HIGH_AGE}']
test_set_list = ['JPAFDB', 'RBAFDB', 'CPSCDB', 'UVAFDB']  # choose databases to combine for overall performance eval
country_of_origin = {'UVAF train': ' (USA)', 'UVAF test': ' (USA)', 'SHDB': ' (Japan)', 'RBDB': ' (Israel)',
                     'CPSC': ' (China)',
                     'Combined \n test set': '', 'Female': '', 'Male': '', f'${{\leq}}${LOW_AGE}': '',
                     f'{LOW_AGE} to {HIGH_AGE}': '', f'${{\geq}}${HIGH_AGE}': ''}
# consts relative to models
path_models = {'XGB': MODEL_DIR / "Others" / "XGB.pkl",
               'AFEv': MODEL_DIR / "Others" / "weak_afev.pkl",
               '1D-CNN': MODEL_DIR / "ArNet" / "1D-CNN.pkl",
               '1D-CNN+DA': MODEL_DIR / "ArNet" / "1D-CNN_DA.pkl",
               'ArNet': MODEL_DIR / "ArNet" / "ArNet2.pkl",
               'ArNet+DA': MODEL_DIR / "ArNet" / "ArNet2_DA.pkl",
               'ResNet': MODEL_DIR / "ArNet2" / "ResNet.pkl",
               'ArNet2': MODEL_DIR / "ArNet2" / "ArNet2.pkl",
               'ResNet+DA': MODEL_DIR / "ArNet2" / "ResNet_DA.pkl",
               'ArNet2+DA': MODEL_DIR / "ArNet2" / "ArNet2_DA.pkl",
               }

hypercomb = {"XGB": {'n_estimators': 453, 'max_depth': 2},
             "1D-CNN": {'n_filters_start': 104, 'len_sub_window': 4, 'dropout': 0.47459329380222043,
                        'n_hidden_start': 68},
             "ArNet": {'time_history': 12, 'extract_level': 'dense_2', 'n_units': 22, 'dropout': 0.7194263073722404,
                       'learning_rate': 0.00010390447220815073, 'af_weight': 3},  # the best hypercomb for "ArNet",
             # with 1D-CNN as feature extractor
             "ResNet": {'n_blocks': 6, 'n_filters_start': 63, 'filter_length': 4, 'dropout_conv': 0.3325515664976955,
                        'n_hidden_start': 188, 'dropout_fcn': 0.16157369665369958,
                        'learning_rate': 0.0012962004356218155,
                        'af_weight': 4},
             "ArNet2": {'time_history': 6, 'extract_level': 'dense_2', 'n_units': 35, 'dropout': 0.8,
                        'learning_rate': 0.0007941175938452183, 'af_weight': 1},
             # the best hypercomb for "ResNet+RNN", with ResNet as feature extractor
             "CRNN": {"window_size": 60, "time_history": 20, "n_blocks": 6, "n_filters_start": 32, "filter_length": 5,
                      "activation": "relu", "dropout_conv": 0.2, "n_units": 8, "n_dense": 2, "dropout_fcn": 0.4,
                      "learning_rate": 0.001, "af_weight": 3},  # not tuned yet
             "RNN": {"window_size": 60, "time_history": 20, "activation": "relu", "n_units": 64, "n_dense": 2,
                     "dropout_fcn": 0.4, "learning_rate": 0.001, "af_weight": 3}  # not tuned yet
             }

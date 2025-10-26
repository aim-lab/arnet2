import pathlib
import os
REPO_DIR = pathlib.PurePath(os.path.dirname(os.path.abspath(__file__)))

# consts relative to models
path_models = {'ResNet': REPO_DIR / "model" / "ResNet.pkl",
               'ArNet2': REPO_DIR / "model" / "ArNet2.pkl",
               }

# Labels definition
PATIENT_LABEL_UNKNOWN = 5
PATIENT_LABEL_NON_AF = 0
PATIENT_LABEL_AF_MILD = 1
PATIENT_LABEL_AF_MODERATE = 2
PATIENT_LABEL_AF_SEVERE = 3
PATIENT_LABEL_OTHER_CVD = 4
AF_MILD_THRESHOLD = 30                  # Threshold on AF Burden to define a patient as Mild AF patient. CAREFUL ! TIME IN SECONDS HERE
AF_MODERATE_THRESHOLD = 0.04            # Threshold on AF Burden to define a patient as Moderate AF patient
AF_SEVERE_THRESHOLD = 0.8               # Threshold on AF Burden to define a patient as Severe AF patient

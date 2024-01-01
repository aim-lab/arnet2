import sys
sys.path.append("/afib/parser/utils")
import pandas as pd
import numpy as np
import pathlib
import consts as cts
exam_C3 = pd.read_csv(cts.REPO_DIR / "class_splits" / "_exam_C3.csv")
# C3_morphological = pd.read_csv(REPO_DIR / "class_splits" / "morphological_exam_C3.csv")
sqi_df = pd.read_csv(cts.REPO_DIR / "class_splits" / "_sqi_df.csv")
ids = np.load("/afib/C3_ids.npy", allow_pickle=True)
nb_files = len(ids)
nb_process = 20
step = nb_files // nb_process
df = pd.DataFrame([])
for start_index in range(1, nb_files-step, step):
    naming = "test_C3_add_temp_morfological_features" + str(start_index) + ".csv"
    filename = cts.REPO_DIR / "add_features" / naming
    temp = pd.read_csv(filename, header=None)
    df = df.append(temp)
print(len(df))
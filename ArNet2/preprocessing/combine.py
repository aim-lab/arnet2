import pandas as pd
import numpy as np
import consts as cts

exam_C3 = pd.read_csv(cts.REPO_DIR / "class_splits" / "_exam_C3.csv")
C3_morphological = pd.read_csv(cts.REPO_DIR / "class_splits" / "morphological_exam_C3.csv")
sqi_df = pd.read_csv(cts.REPO_DIR / "class_splits" / "_sqi_df.csv")

C3_3 = pd.read_csv(cts.REPO_DIR / "C3_3_morfological_features.csv", header=None)
C3_morph = C3_morph.append(C3_5)

temp = exam_C3.loc[exam_C3.status.eq("use"), "id_exam"]

id_exam = C3_morph[C3_morph.columns[-1]]
# id_exam = C3_morph["id_exam"]
C3_morph.drop(labels=C3_morph.columns[-1], axis=1, inplace=True)
# C3_morph.drop(labels=['id_exam'], axis=1, inplace=True)
C3_morph.insert(0, 'id_exam', id_exam)
C3_morph["sqi"] = -1
C3_morph["sqi"] = C3_morph.id_exam.map(sqi_df.set_index("id_exam").sqi)
C3_morph.columns = C3_morphological.columns
C3_morphological_new = C3_morphological.append(C3_morphological)

ids = exam_C3[~exam_C3.id_exam.isin(C3_morphological_new.id_exam)].id_exam.values

C3_morphological_new = C3_morph[C3_morph.id_exam.isin(temp)]
C3_morphological_new = C3_morphological_new.append(C3_morphological)
C3_morphological_new["sqi"] = -1
C3_morphological_new["sqi"] = C3_morphological_new.id_exam.map(sqi_df.set_index("id_exam").sqi)

C3_morphological_new.to_csv(cts.REPO_DIR / "class_splits" / "morphological_exam_C3.csv", index=False)
np.save("/home/shanybiton/AIMLabProjects/afib-prediction-lab/C3_ids.npy", ids)





# for 132.68.176.113
import numpy as np
import h5py
import pathlib
REPO_DIR = pathlib.PurePath('C:\\Users\\shanybiton\\PycharmProjects\\afib-morph-local')
BASE_DATA_DIR = pathlib.PurePath('J:')
DATA_DIR = BASE_DATA_DIR / "tnmg"
# DATA_DIR = BASE_DIR /"databases" / "tnmg"
ECG_DATA_DIR = DATA_DIR /"ecg-traces" / "ecg-traces"

subset_ids = np.load(REPO_DIR / "C3_ids_113.npy", allow_pickle=True)
path_to_hdf5 = ECG_DATA_DIR / "preprocessed" / "traces.hdf5"
path_to_csv = ECG_DATA_DIR / "annotations.csv"
input_dir = "add_features"
dataset_name= "signal"
# input_dir = 'C:\\Users\\Shany\\Documents\\AIMLabProjects\\afib-prediction'
# Get tracings
f = h5py.File(path_to_hdf5, "r")
x = f[dataset_name]
traces_ids = np.array(f['id_exam'])

sub_loc = np.argwhere(np.isin(traces_ids, subset_ids)).flatten()
f_temp = x[sub_loc]
f2 = h5py.File('sub_set_2.hdf5', 'w')
f2.create_dataset('signal', data=f_temp)
f2.create_dataset('id_exam', data=subset_ids)

#dset.write_direct(f_temp)
f2.close()





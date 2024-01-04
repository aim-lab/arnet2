import numpy as np
import pandas as pd
import consts as cts
import os
import pathlib
import h5py
from tnmg_parser import TNMGDB_Parser

if __name__ == '__main__':
    db = TNMGDB_Parser()
    DIR = cts.REPO_DIR / "add_features"
    exams = pd.read_csv(cts.REPO_DIR / "features" / 'exam_C0.csv')
    feats = np.load(cts.REPO_DIR / "features" / "for_use" / "feats.npy", allow_pickle=True).item()
    sqi_df = pd.read_csv(DIR / "C0_sqi_file_2.csv")
    features_C0 = pd.read_csv(cts.REPO_DIR / "features" / 'features_C0.csv')
    reindex_feats = np.load(cts.REPO_DIR / "features" / "reindex_feats.npy", allow_pickle=True)
    df = pd.DataFrame([])

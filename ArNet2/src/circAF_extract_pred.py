# General imports
import os
import pathlib
import sys

import numpy as np
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import warnings
from simulating_scenario import _af_pat_lab

sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/src')
sys.path.append('/home/shanybiton/repos/Generalization/src/models')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU you wish to use here
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import utils.consts as cts
import data.data_processing as dp
from data import data_augmentation as da
from parsing.db_loader import *
import models.model_utils as model_utils


def create_pat_list(exclude_low_sqi=False):
    RBAF_mdclone_info = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info_mdclone.xlsx")  # All patients obtained from rambam
    RBAF_holter_info = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info.xlsx")  # Patients obtained from rambam with holter recordings
    reports_file = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / 'documentation' / 'reports' / 'RBAF_reports.xlsx')  # Patients obtained from rambam with holter reports
    NLP_file = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / 'documentation' / 'reports' / 'Holter_labels_final.xlsx')  # Final labeling per recording according to the reports
    medAIM_ann_file = pd.read_excel(cts.BASE_DIR / 'Shany' / 'medAIM/GS' / 'annotation_progress_file.xlsx')
    ids_test = medAIM_ann_file.loc[medAIM_ann_file.db.eq('RBAFDB'), 'id'].to_list()
    RBAF_mdclone_info.drop_duplicates(subset=['db_id'], inplace=True)
    print(
        f'{len(RBAF_mdclone_info.db_id.unique())} patients in cardiology at Rambam HCC between January 1st 2011 and '
        f'October 1st 2021')

    print('Excluding no holter monitoring or Holter reports:')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(reports_file.holter_id) | RBAF_holter_info.holter_id.isin(ids_test)]
    print(RBAF_holter_info.holter_id.nunique())
    RBAF_holter_info.loc[
        RBAF_holter_info.holter_id.isin(NLP_file.loc[NLP_file.AFIB.eq(1) | NLP_file.AFL.eq(1), 'Holter_id']), 'AFIB'] = 1
    RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(NLP_file.loc[NLP_file.AFL.eq(1), 'Holter_id']), 'AFL'] = 1
    RBAF_holter_info.AFL.fillna(0, inplace=True)
    RBAF_holter_info.AFIB.fillna(0, inplace=True)
    RBAF_holter_info.drop_duplicates('holter_id', inplace=True)

    print('Excluding patients younger then 18:')
    RBAF_holter_info = RBAF_holter_info.loc[
        RBAF_holter_info.age_at_recording.ge(18) | RBAF_holter_info.age_at_recording.isnull()]
    print(RBAF_holter_info.holter_id.nunique())

    print('Excluding patients with pacemaker:')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.pacemaker_at_holter.ne(1)]
    print(RBAF_holter_info.holter_id.nunique())

    print('Excluding corrupted ecg patients: ')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(db.non_corrupted_ecg_patients())]
    print(RBAF_holter_info.holter_id.nunique())

    if exclude_low_sqi:
        db.recompute_sqi(win_thresh=0.75)
        print('Excluding patients with bsqi < 0.75: ')
        RBAF_holter_info = RBAF_holter_info.loc[~RBAF_holter_info.holter_id.isin(db.low_sqi)]
        print(RBAF_holter_info.holter_id.nunique())
    return RBAF_holter_info


def return_data(db, pat_list, exclude_low_sqi_win, win_thresh):
    ids = db.return_patient_ids(pat_list=pat_list, exclude_low_sqi_win=exclude_low_sqi_win, win_thresh=win_thresh)
    rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=pat_list, return_binary=True, exclude_low_sqi_win=exclude_low_sqi_win, win_thresh=win_thresh)
    prec = db.return_preceeding_windows(pat_list=pat_list, exclude_low_sqi_win=exclude_low_sqi_win, win_thresh=win_thresh)
    feats, _, glob_lab = db.return_features(pat_list=pat_list, feats_list=['medHR', 'sqi'], return_global_label=True, exclude_low_sqi_win=exclude_low_sqi_win, win_thresh=win_thresh)
    X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    final = tuple()
    timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    final = final + (X, y, timestamp)
    return feats, final, win_lab


def create_af_label_df(pat_df, pred_df, X, y, afb_thresh=0.04):
    patient_file = pd.read_excel(cts.REPO_DIR / 'patient_file.xlsx')
    pat_df['AF'] = pat_df[['AFIB', 'AFL']].all(axis=1)
    pat_df.loc[pat_df.holter_id.isin(patient_file.loc[patient_file.lab.ge(1), 'id']), 'AF'] = 1
    pred_afb, pred_af_lab, true_afb = {}, {}, {}
    print(f'reprorting metrices for threshold: {afb_thresh}')
    for i, pat in enumerate(pat_df.holter_id):
        # print(int(100 * i / len(test_pat)))
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat]
        y_pred_pat = pred_df.loc[pred_df.id.eq(pat), 'proba'].values > pred_df['decision_th'].unique()[0]
        rr, glob_lab = X_pat[:, :-3], X_pat[0, -2]
        if glob_lab == 4:
            glob_lab = 0
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        pred_afb[pat] = pred_af_burden
        pred_af_lab[pat] = _af_pat_lab(pred_af_burden, thresh=afb_thresh)
    pat_df['pred_afb'] = pat_df['holter_id'].map(pred_afb)
    pat_df[f'AF_lab_thresh_{afb_thresh}'] = pat_df['holter_id'].map(pred_af_lab)
    return pat_df[['holter_id', 'db_id', 'recording_date', 'birth_date',
       'age_at_recording', 'sex', 'pacemaker_at_holter', 'AF', 'pred_afb', 'AF_lab_thresh_0.04']]


if __name__ == '__main__':
    db = RBAFDB_Parser(load_on_start=True)

    # load data
    pat_df = create_pat_list(False)
    feats, final, win_lab = return_data(db, pat_df, exclude_low_sqi_win=False, win_thresh=0.75)
    X, y, t_s = final

    # run model
    algo_py = 'ArNet2'
    feature_extractor_path = cts.path_models['ResNet']
    model_dict = model_utils.load_model(cts.path_models[algo_py], algo=algo_py,
                                        path_feature_extractor=feature_extractor_path)
    model = model_dict['classifier']
    probas = model.predict_proba(X)[:, 1]
    df = pd.DataFrame(columns=['id', 'proba', 'decision_th', 'pred', 'lab', 'start_time', 'end_time', 'start_recording'])
    df['id'] = [i.split('_')[0] for i in X[:, -1]]
    df['proba'] = probas
    df['decision_th'] = model_dict['best_th']
    df['pred'] = (probas > model_dict['best_th'])
    df['start_time'] = t_s[:, 0]
    df['end_time'] = t_s[:, -1]
    df['medHR'] = feats[:, 0]
    df['sqi'] = feats[:, -1]
    df['lab'] = False
    for id in df.id:
        df.loc[df.id.eq(id), 'start_recording'] = db.circadian_dict[id]['start_recording']
    df_labels = create_af_label_df(pat_df, df, X, y, afb_thresh=0.04)
    df.loc[df.sqi.ge(0.75)].to_csv('/home/shanybiton/repos/CircadianAF/ArNet2_circAF_pred_RBDB_latest2.csv')
    df_labels.to_csv('/home/shanybiton/repos/CircadianAF/RBDB_patient_list_labels_latest.csv')

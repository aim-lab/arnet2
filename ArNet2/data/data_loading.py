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

sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/src')
sys.path.append('/home/shanybiton/repos/Generalization/src/models')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import utils.consts as cts
import data.data_processing as dp
from data import data_augmentation as da
from parsing.db_loader import *
import models.model_utils as model_utils

def get_label(val:int):
    Diagnosis_dict = {cts.PATIENT_LABEL_NON_AF: 'Non-AF',
                      cts.PATIENT_LABEL_AF_MILD: 'Paroxysmal AF',
                      cts.PATIENT_LABEL_AF_SEVERE: 'Persistent AF',
                      cts.PATIENT_LABEL_OTHER_CVD: 'Other cvd'}
    return Diagnosis_dict[val]

def print_diagram(db):
    baseline = db.parsed_patients()
    print('Diagram of the UVAF stratified train-test split.')
    print("n: {}".format(len(baseline)))
    print('Remove corrupted recordings')
    baseline = np.intersect1d(baseline, db.non_corrupted_ecg_patients())
    print('n: {}'.format(len(baseline)))
    print('Remove low quality recordings')
    baseline = np.intersect1d(baseline, db.high_sqi_patients())
    print('n: {}'.format(len(baseline)))
    print('Remove < 18 years old')
    baseline = np.intersect1d(baseline, db.over_18_patients)
    print('n: {}'.format(len(baseline)))
    print('Train-Test split')
    print_summary(db, baseline, 'UVAFDB')

def print_summary(db, pat_set, set_name=None):
    """ Print a summary of the characteristics of the database.
    :param db: The database parser.
    :param pat_set: The list for which the data should be generated.
    :param set_name: The naming of the summary"""
    patient_file = pd.read_excel(cts.REPO_DIR / 'patients_df.xlsx')
    if set_name in ['train', 'test', 'val']:
        patient_file = patient_file.loc[patient_file.db.eq(db.name) & patient_file.set.eq(set_name)]
    else:
        patient_file = patient_file.loc[patient_file.db.eq(db.name)]
    for id in pat_set:
        db.record_diagnosis(id, db.window_size)
    AFsev_strat = np.array(list(db.features_dict[id][db.window_size]['diagnosis'] for id in pat_set))
    print('Summary for ' + set_name)
    (AF_labels, counts_AFsev_strat) = np.unique(AFsev_strat, return_counts=True)
    for lab, c in zip(AF_labels, counts_AFsev_strat):
        print("{}: {}".format(get_label(lab), c))
    pat_ids = []
    for recording_id in pat_set:
        pat_ids.append(db.parse_patient_id(recording_id=recording_id))
    pat_set_age = np.array(patient_file.age)
    age_stats = np.nanpercentile(pat_set_age, [25, 50, 75])
    sex_strat = np.array(patient_file.sex).reshape(1,-1).flatten()
    print("Number of patients: {}".format(len(np.unique(patient_file.id))))
    print("Number of recordings: {}".format(len(np.array(pat_set))))
    print('Number of hours: {:.2f}'.format(db.total_time(pat_set)))
    print("Recording durations are of: {:.2f} ± ({:.2f}) ".format(
        np.mean([db.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_set]),
        np.std([db.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_set])))
    print("Recording durations are of: {:.2f}, ({:.2f}-{:.2f})".format(
        np.median([db.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_set]),
        np.quantile([db.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_set], 0.25),
        np.quantile([db.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_set], 0.75)))
    print('Sampling frequency: {}'.format(db.orig_fs))
    print('Age: {:.2f} ± ({:.2f})'.format(np.nanmedian(pat_set_age), np.nanstd(pat_set_age)))
    print("The median and interquartile age: {:0.1f} ({:0.1f}-{:0.1f})".format(age_stats[1], age_stats[0], age_stats[2]))
    print('Females, n (%): {} ({}%)'.format(len(np.where(sex_strat == 'F')[0]),
                                            round((len(np.where(sex_strat == 'F')[0])/ len(sex_strat))*100, 1)))
    if db.name not in ["UVAFDB", "CPSCDB"]:
        dates = np.array(list(db.circadian_dict[id]['recording_date'] for id in pat_set))
        print('from: {}, {} until {}, {}'.format(np.min(dates).month, np.min(dates).year,
                                                  np.max(dates).month, np.max(dates).year,))


def stratified_split_part_two(db, pat_set, test_part):  # Precise that we removed the "only reann in test set" criteria !!
    """ Perform a multi-criteria stratified train-test split on a set of patients ids.
    It is stratified with respect to the AF severity label, the age and the sex.
    Because of how the sklearn `train_test_split` function is implemented, we need to handle the multi-criteria bins in which there is only one patient separately.

    :param db: the database on which we perform the split
    :param pat_set: a list of patients ids related to the database
    :param test_part: the float percentage of patients in the test set. Must be between 0 and 1.
    :return train_pat, test_pat: the resulting lists of patients ids, after split
    """
    # Extract stratification criteria
    if db.name == "UVAFDB":
        AFsev_strat = np.array(list(db.af_pat_lab_dict[id] for id in pat_set))
    else:
        AFsev_strat = np.array(list(db.features_dict[id][db.window_size]['diagnosis'] for id in pat_set))

    pat_set_age = np.array(list(db.features_dict[id][db.window_size]['Age'] for id in pat_set))
    age_strat = (pat_set_age > np.median(pat_set_age))
    sex_strat = np.array(list(db.features_dict[id][db.window_size]['Sex'] for id in pat_set)).reshape(1,-1).flatten()

    # Remove patients isolated in their multi-criteria bin, and add them to a separate list (sklearn train_test_split doesn't handle 1-sized bins)
    isolated_patients = []
    for AFsev in np.unique(AFsev_strat):  # 5 AF severity categories : NonAF, AFmild, AFmoderate, AFsevere, otherCVD
        for age in range(2):  # 2 age categories : below and above the median age
            for sex in range(2):  # 2 gender categories : woman and man
                in_multi_bin = np.logical_and(np.logical_and(AFsev_strat == AFsev, age_strat == age), sex_strat == sex)
                if np.sum(in_multi_bin) == 1:  # if only one patient in the multi-criteria bin
                    pat_idx = np.where(in_multi_bin)[0][0]  # get its idx in the baseline list
                    isolated_patients.append(pat_set[pat_idx])  # add its id to isolated patients
                    AFsev_strat = np.delete(AFsev_strat,
                                            pat_idx)  # remove it from AFsev, age, gender and baseline before stratification
                    age_strat = np.delete(age_strat, pat_idx)
                    sex_strat = np.delete(sex_strat, pat_idx)
                    pat_set = np.delete(pat_set, pat_idx)
    isolated_patients = np.array(isolated_patients)

    # Split patients ids into train and test sets, with multi-criteria stratification
    train_pat, test_pat = train_test_split(pat_set, test_size=test_part, stratify=np.vstack((AFsev_strat, age_strat, sex_strat)).T, random_state=cts.SEED)

    # Randomly add isolated patients to train and test sets, weighted by the size of the sets
    if len(isolated_patients) > 0:
        print("Adding isolated patients")
        rd = np.random.random(len(isolated_patients))
        print("Patients added in first set : ", isolated_patients[np.where(rd <= 1 - test_part)[0]])
        train_pat = np.append(train_pat, isolated_patients[np.where(rd <= 1 - test_part)[0]])
        print("Patients added in second set : ", isolated_patients[np.where(rd > 1 - test_part)[0]])
        test_pat = np.append(test_pat, isolated_patients[np.where(rd > 1 - test_part)[0]])

    else:
        print("No isolated patient")

    # Plot the train-test distribution
    # plot_train_test_distribution(db, train_pat, test_pat, np.median(pat_set_age), part='train_val', plot_diagnosis=False, savefig=savefig, savedir= cts.REPO_DIR / 'figs' / 'data')

    return train_pat, test_pat


def stratified_split_part_one(db, pat_set, partition): # Used for datasets that are NOT annotated at all !!
    """ Perform a multi-criteria stratified train-test split on a set of patients ids.
    This is the first stratification split needed for the re-annotation process.
    It is stratified with respect to the patient age and the sex for each diagnosis category in the partition dict.

    :param db: the database on which we perform the split
    :param pat_set: a list of patients ids related to the database
    :param partition: dict of shape: {cts.PATIENT_LABEL_NON_AF: int, cts.PATIENT_LABEL_AF_MILD: int, cts.PATIENT_LABEL_AF_SEVERE: int}
    specifying the number of ids to include in each diagnosis category for the re-annotated set
    :return train_pat, test_pat: the resulting lists of patients ids, after split
    """
    # Extract stratification criteria
    for id in pat_set:
        db.record_diagnosis(id, db.window_size)
    Diagnosis_strat = np.array(list(db.features_dict[id][db.window_size]['diagnosis'] for id in pat_set))
    pat_set_age = np.array(list(db.features_dict[id][db.window_size]['Age'] for id in pat_set))
    pat_set_sex = np.array(list(db.features_dict[id][db.window_size]['Sex'] for id in pat_set))
    if db.name != 'UVAFDB':
        pat_set_sex = (pat_set_sex == 'F').astype(int).reshape(1,-1).flatten()
    train_pat = dict.fromkeys(partition.keys())
    test_pat = dict.fromkeys(partition.keys())
    for AFsev in list(partition.keys()):  # 4 AF severity categories : NonAF, persistentAF, paroxysmalAF, otherCVD
        pat_subset_idx = np.where(Diagnosis_strat==AFsev)[0]
        if len(pat_subset_idx)>partition[AFsev]:
            age_strat = (pat_set_age[pat_subset_idx] > np.nanmedian(pat_set_age[pat_subset_idx]))
            sex_strat = pat_set_sex[pat_subset_idx]
            # Split patients ids into train and test sets, with multi-criteria stratification
            train_pat_subset, test_pat_subset = train_test_split(pat_set[pat_subset_idx], test_size=partition[AFsev],
                                                   stratify=np.vstack((age_strat, sex_strat)).T,
                                                   random_state=cts.SEED)
            train_pat[AFsev], test_pat[AFsev] = train_pat_subset, test_pat_subset
            # plot_train_test_distribution(db, train_pat[AFsev], test_pat[AFsev], np.median(pat_set_age[pat_subset_idx]), savefig=savefig, plot_diagnosis=False,
            #                              part=get_label(AFsev), savedir=cts.REPO_DIR / 'figs' / 'data')
        else: # not enough recordings to split, add all to test set
            test_pat[AFsev] = pat_set[pat_subset_idx]

    # Plot the train-test distribution
    # plot_train_test_distribution(db, train_pat, test_pat, np.median(pat_set_age), savefig, savedir= cts.REPO_DIR / 'figs' / 'data')

    return train_pat, test_pat

def group_sex(test_dict):
    patient_file = pd.read_excel(cts.REPO_DIR / 'patients_df.xlsx')
    patient_file = patient_file.loc[patient_file.set.eq('test')]
    M_group = patient_file.loc[patient_file.sex.eq('M'), 'id']
    F_group = patient_file.loc[patient_file.sex.eq('F'), 'id']
    mask_ids = np.array([i.rsplit('_', 1)[0] for i in test_dict[0][:,-1]])
    # create female test dict
    F_ids = np.where(np.isin(mask_ids, F_group))
    F_X = test_dict[0][F_ids]
    F_y = test_dict[1][F_ids]
    F_timestamp = test_dict[2][F_ids]
    final_F = tuple()
    final_F = final_F + (F_X, F_y, F_timestamp)

    # create male test dict
    M_ids = np.where(np.isin(mask_ids, M_group))
    M_X = test_dict[0][M_ids]
    M_y = test_dict[1][M_ids]
    M_timestamp = test_dict[2][M_ids]
    final_M = tuple()
    final_M = final_M + (M_X, M_y, M_timestamp)

    return final_F, final_M


def group_age(test_dict, low_age=cts.LOW_AGE, high_age=cts.HIGH_AGE):
    patient_file = pd.read_excel(cts.REPO_DIR / 'patients_df.xlsx')
    low_group = patient_file.loc[patient_file.age.le(low_age), 'id']
    mid_group = patient_file.loc[patient_file.age.gt(low_age) & patient_file.age.le(high_age), 'id']
    old_group = patient_file.loc[patient_file.age.gt(high_age), 'id']
    mask_ids = np.array([i.rsplit('_', 1)[0] for i in test_dict[0][:, -1]])
    # create low age test dict
    low_group_ids = np.where(np.isin(mask_ids, low_group))
    low_group_X = test_dict[0][low_group_ids]
    low_group_y = test_dict[1][low_group_ids]
    low_group_timestamp = test_dict[2][low_group_ids]
    final_low_group = tuple()
    final_low_group = final_low_group + (low_group_X, low_group_y, low_group_timestamp)

    # create low age test dict
    mid_group_ids = np.where(np.isin(mask_ids, mid_group))
    mid_group_X = test_dict[0][mid_group_ids]
    mid_group_y = test_dict[1][mid_group_ids]
    mid_group_timestamp = test_dict[2][mid_group_ids]
    final_mid_group = tuple()
    final_mid_group = final_mid_group + (mid_group_X, mid_group_y, mid_group_timestamp)
    old_group_ids = np.where(np.isin(mask_ids, old_group))
    old_group_X = test_dict[0][old_group_ids]
    old_group_y = test_dict[1][old_group_ids]
    old_group_timestamp = test_dict[2][old_group_ids]
    final_old_group = tuple()
    final_old_group = final_old_group + (old_group_X, old_group_y, old_group_timestamp)

    return final_low_group, final_mid_group, final_old_group

def Simulating_intended_use_scenario(db_parser, window_size=60):
    db = db_parser(window_size=window_size)
    ann_ids = next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1]
    NLP_file = pd.read_excel(cts.DATA_DIR / db.name.lower() / 'documentation' / 'reports'/ 'Holter_labels_final.xlsx')
    pat_list = NLP_file.loc[~NLP_file.Holter_id.isin(ann_ids), 'Holter_id'].values
    pat_list = pat_list[~np.isin(pat_list, db.corrupted_ecg)]
    ids = db.return_patient_ids(pat_list=pat_list)
    rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=pat_list, return_binary=True)
    prec = db.return_preceeding_windows(pat_list=pat_list)
    feats, y, glob_lab = db.return_features(pat_list=pat_list, feats_list=np.append(cts.SELECTED_FEATURES, 'sqi'), return_global_label=True)

    # Process data
    feats, mean_feats = dp.fillna(feats)
    # Concatenate the features
    X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    final = tuple()
    timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    final = final + (X, y, timestamp)
    return feats, final, win_lab

def load_db(db_parser, train_part=0.6, val_part=0.2, high_sqi_pat_only=True, over_18_pat_only=True, window_size=60, expert_ann=False, reann_pat_only=False):
    """ Load general database and perform a train-val-test split with multi-criteria stratification.

    :param db_parser:
    :param expert_ann:
    :param reann_pat_only:
    :param train_part: the float percentage of patients in the training set. Must be between 0 and 1.
    :param val_part: the float percentage of patients in the val set. Must be between 0 and 1.
    :param high_sqi_pat_only: a boolean indicating whether we want to exclude the low sqi patients
    :param over_18_pat_only: a boolean indicating whether we want to exclude the <18 years old patients
    :param window_size: the size of the RR interval windows
    :return db: the parsed database
    :return train_pat, val_pat, test_pat: the lists of patients ids for each set, after split
    """

    # Extract DB patients
    db = db_parser(window_size=window_size)

    # Exclude low sqi and under 18 patients
    baseline = db.non_corrupted_ecg_patients()
    if high_sqi_pat_only:
        baseline = np.intersect1d(baseline, db.high_sqi_patients())
    if over_18_pat_only:
        baseline = np.intersect1d(baseline, db.over_18_patients)

    # Keep re-annotated patients only
    if reann_pat_only:
        db.extract_reann_pat()
        reann_pat = db.reann_pat
        baseline = np.intersect1d(baseline, reann_pat)

    print("Train-Test split")
    if expert_ann:
        train_val_pat_dict, test_pat_dict = stratified_split_part_one(db, baseline, partition={cts.PATIENT_LABEL_NON_AF: 20, cts.PATIENT_LABEL_AF_MILD: 60, cts.PATIENT_LABEL_AF_SEVERE: 20})
        train_val_pat = []
        test_pat = []
        for keys, values in train_val_pat_dict.items():
            if values is not None:
                [train_val_pat.append(i) for i in values if i]
        for keys, values in test_pat_dict.items():
            if values is not None:
                [test_pat.append(i) for i in values if i]
        print("Train-Val split")
        train_pat, val_pat = stratified_split_part_two(db, train_val_pat, test_part=int(len(train_val_pat) * val_part))
    return db, train_pat, val_pat, test_pat


def return_data(db_parser, pat_list, set, window_size=60):  # TODO : overhaul this function, removing normalization, and taking only one set as input
    """ Load the training and test patients features and concatenate them for the model input.

    :param window_size:
    :param db_parser: the parsed database
    :param pat_list: the list of patients ids
    :param set: weather the pat_list belongs to train-val/test-test_GS sets
    :return: the model input data and xgb features
    """
    db = db_parser(window_size=window_size)
    # Get variables
    if db.name not in  ["UVAFDB", 'CPSCDB', 'AFDB', 'LTAFDB']:
        ann_ids = next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1]
        pat_list = pat_list[np.isin(pat_list, ann_ids)]
        for pat in pat_list:
            print(pat)
            db.parse_elem_data(pat=pat, reannotated=True)
            db._win_lab(id=pat, win=window_size)
            db._af_win_lab(id=pat, win=window_size)
            db._af_pat_lab(id=pat)
            # test_anns = np.setdiff1d(db.annotation_types, db.sqi_ref_ann)
            # for ann_type in test_anns:  # Computing SQI
            #     db._sqi(pat, window_size, test_ann=ann_type)
    if db.name in ["CPSCDB"]:
        for i in pat_list:
            db.rlab_dict[i][db.rlab_dict[i] == db.rhythms_dict['(AFL']] = db.rhythms_dict['(AFIB']

    ids = db.return_patient_ids(pat_list=pat_list)
    if set == '_test':
        rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=pat_list, return_binary=True)
    else:
        rr, rrt, y, _, win_start, win_end = db.return_rr(pat_list=pat_list, return_binary=True, return_global_label=True)
        win_lab = None
    prec = db.return_preceeding_windows(pat_list=pat_list)
    feats, y, glob_lab = db.return_features(pat_list=pat_list, feats_list=np.append(cts.SELECTED_FEATURES, 'sqi'), return_global_label=True)

    # Process data
    feats, mean_feats = dp.fillna(feats)
    # Concatenate the features
    if db.name == 'CPSCDB':
        ids = np.array([id.rsplit('_', 1)[0] for id in ids], dtype=ids.dtype)
    ids = np.array([str(id) + '_' + db.name for id in ids], dtype=ids.dtype)
    X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    final = tuple()
    timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    final = final + (X, y, timestamp)
    return feats, final, win_lab


def create_input_model(db_parser, ids_dir, input_model_dir):
    """ Create files for model training and evaluating

    :param db_parser: the parsed database
    :param ids_dir: directory contains all database ids split into subsets
    :param input_model_dir: where to save the model input files
    """
    if db_parser.__name__ == 'UVAFDB_Parser':
        set_list = ['_train', '_val', '_test']
    else:
        set_list = ['_test']
    db_name = db_parser.__name__.split('_')[0]
    for set in set_list:
        set_ids = np.load(ids_dir / (db_name + set + "_pat.npy"), allow_pickle=True)
        feats, final, win_lab = return_data(db_parser, pat_list=set_ids, set=set, window_size=60)
        with open(input_model_dir / (db_name + set + '_input.pickle'), 'wb') as f:
            pickle.dump(final, f)
        with open(input_model_dir / (db_name + set + '_xgboost_input.pickle'), 'wb') as f:
            pickle.dump(feats, f)
    return

def return_input_model(parser_mapping_dict, test_set_list, regenerate=False, algo='ArNet2'):
    data_test = {}
    x_all, y_all, t_s_all = np.array([], dtype=np.float).reshape(0, 63), np.array([], dtype=np.bool).reshape(1, -1), np.array([], dtype=np.float).reshape(0, 2)
    for parser in parser_mapping_dict.values():
        if regenerate:
            create_input_model(parser, ids_dir=cts.REPO_DIR / 'data/splits/ids', input_model_dir=cts.REPO_DIR / 'data/splits/model_input')
        if parser.__name__=='UVAFDB_Parser':
            data_train, data_val, data_test[parser.__name__.split('_')[0]] = load_data(algo, db_parser=parser.__name__, input_model_dir=cts.REPO_DIR / 'data/splits/model_input')
        # check data size compatibility to algo input
        else:
            data_test[parser.__name__.split('_')[0]] = load_data(algo, db_parser=parser.__name__, input_model_dir=cts.REPO_DIR / 'data/splits/model_input')[0]
        if x_all.shape[1] != data_train[0].shape[1]:
            x_all = np.array([], dtype=np.float).reshape(0, data_train[0].shape[1])
        if parser.__name__.split('_')[0] in test_set_list:
            x_all, y_all, t_s_all = np.vstack([x_all, data_test[parser.__name__.split('_')[0]][0]]),\
                                    np.hstack([y_all, data_test[parser.__name__.split('_')[0]][1].reshape(1, -1)]),\
                                    np.vstack([t_s_all, data_test[parser.__name__.split('_')[0]][2]])
    data_train_val = (np.concatenate((data_train[0], data_val[0]), axis=0), np.concatenate((data_train[1], data_val[1]), axis=0), np.concatenate((data_train[2], data_val[2]), axis=0))

    # Data combined
    test_dict = {'train': data_train_val}
    test_dict.update({'test_' + set_name: v for (set_name, v) in data_test.items()})
    test_dict.update({'test_all': (x_all, y_all.flatten(), t_s_all)})
    # Data Augmentation
    if 'DA' in algo:
        print("Data Augmentation...")
        data_train = da.augment(data_train_val, method_name="sub_flipping")
    return data_train, test_dict


def load_data(algo, db_parser, input_model_dir, append_labs=True):
    """ A general script to load the different datasets used for training and evaluation

    :param input_model_dir: the input files destination
    :param append_labs: boolean. In the generalization paper, [mild, moderate] are the same label (PAF) and [non-AF, other CVD] are non AF
    :param db_parser: database parser to choose from [UVAF, JPAF, RBAF, CPSC, AFDB]
    :param algo: a string representing the model's name
    :return data : a tuple of datasets, each dataset being a tuple (X, y), with additionally timestamps for the RR windows
    """
    data_dict = {}
    if db_parser == 'UVAFDB_Parser':
        set_list = ['_train', '_val', '_test']
    else:
        set_list = ['_test']
    db_name = db_parser.split('_')[0]
    for set in set_list:
        data_dict[db_name + set + '_X'], data_dict[db_name + set + '_y'], data_dict[db_name + set + '_t_s'] = pickle.load(open(input_model_dir / (db_name + set + '_input.pickle'), "rb"))
        data_dict[db_name + set + '_feats'] = pickle.load(open(input_model_dir / (db_name + set + '_xgboost_input.pickle'), "rb"))
        if append_labs:
            # data_dict[db_name + set + '_X'][data_dict[db_name + set + '_X'][:, -2] == cts.PATIENT_LABEL_AF_MODERATE, -2] = cts.PATIENT_LABEL_AF_MILD
            data_dict[db_name + set + '_X'][data_dict[db_name + set + '_X'][:, -2] == cts.PATIENT_LABEL_OTHER_CVD, -2] = cts.PATIENT_LABEL_NON_AF
    if algo=='XGB':
        data = [(np.concatenate((data_dict[db_name + set_ + '_feats'], data_dict[db_name + set_ + '_X'][:, -3:]), axis=1), data_dict[db_name + set_ + '_y'], data_dict[db_name + set_ + '_t_s']) for set_ in set_list]
    else:
        data = [(data_dict[db_name + set_ + '_X'], data_dict[db_name + set_ + '_y'], data_dict[db_name + set_ + '_t_s']) for set_ in set_list]
    return data

def load_all_models(models):
    all_models = {}
    for model in models:
        path = cts.path_models[model]
        if model == 'ArNet':
            model_dict = model_utils.load_model(path, algo='ArNet2',
                                                path_feature_extractor=cts.path_models['1D-CNN'])
        elif model == 'ArNet2':
            model_dict = model_utils.load_model(path, algo='ArNet2',
                                                path_feature_extractor=cts.path_models['ResNet'])
        elif 'DA' in model:
                model_dict = model_utils.load_model(path, algo=model.split("+")[0],
                                                    path_feature_extractor=cts.path_models['ResNet+DA'])
        elif model == 'AFEv':
            model_dict = model_utils.load_model(path, algo='XGB')
        else:
            model_dict = model_utils.load_model(path, algo=model)
        all_models[model] = model_dict
        print('>loaded %s' % model)
    return all_models

def plot_train_test_distribution(db, train_pat, test_pat, median_age, plot_diagnosis, part=None, savefig=True, savedir=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20, 8), ncols=3 if plot_diagnosis else 2)
    # plot age distribution
    age_train = list(
        np.array(list(db.features_dict[id][db.window_size]['Age'] for id in train_pat)) > median_age)
    age_test = list(np.array(list(db.features_dict[id][db.window_size]['Age'] for id in test_pat)) > median_age)
    # ax[0].hist(list(db.features_dict[id][db.window_size]['Age'] for id in train_pat), label='train')
    # ax[0].hist(list(db.features_dict[id][db.window_size]['Age'] for id in test_pat), label='test')
    ax[0].hist([list(db.features_dict[id][db.window_size]['Age'] for id in train_pat), list(db.features_dict[id][db.window_size]['Age'] for id in test_pat)], label=['train', 'test'])
    max_v = max(max(np.histogram(list(db.features_dict[id][db.window_size]['Age'] for id in train_pat))[0]), max(
        np.histogram(list(db.features_dict[id][db.window_size]['Age'] for id in test_pat))[0]))
    ax[0].vlines(median_age, 0, max_v, colors='red', linestyles='--', label='median')
    ax[0].legend()
    ax[0].set_title(
        f'Age train-test distribution \n {100 * age_train.count(0) / (age_train.count(0) + age_test.count(0)):.1f} - {100 * age_train.count(1) / (age_train.count(1) + age_test.count(1)): .1f}')
    ax[0].set_xlabel('Age', fontsize=20)
    ax[0].set_ylabel('Counts', fontsize=20)

    # plot sex distribution
    sex_train = np.array(list(db.features_dict[id][db.window_size]['Sex'] for id in train_pat))
    sex_test = np.array(list(db.features_dict[id][db.window_size]['Sex'] for id in test_pat))
    sex_train = sex_train.reshape(1, -1).flatten()
    sex_test = sex_test.reshape(1,-1).flatten()
    (sex_labels, counts_sex_train) = np.unique(sex_train, return_counts=True)
    (_, counts_sex_test) = np.unique(sex_test, return_counts=True)
    index=np.arange(2)
    bar_width = 0.35
    tr = ax[1].bar(index, counts_sex_train, bar_width, label='train')
    te = ax[1].bar(index+bar_width, counts_sex_test, bar_width, label='test')
    # ax[1].set_title(
    #     f'Sex train-test distribution \n {100 * sex_train.count(sex_labels[0]) / (sex_train.count(sex_labels[0]) + sex_test.count(sex_labels[0])):.1f} - {100 * sex_train.count(sex_labels[1]) / (sex_train.count(sex_labels[1]) + sex_test.count(sex_labels[1])): .1f}')
    ax[1].set_title(
        f'Sex train-test distribution \n {100 * (sex_train == sex_labels[0]).sum() / ((sex_train == sex_labels[0]).sum() + (sex_test == sex_labels[0]).sum()):.1f} - {100 * (sex_train == sex_labels[1]).sum() / ((sex_train == sex_labels[1]).sum() + (sex_test == sex_labels[1]).sum()): .1f}')
    ax[1].set_xlabel('Sex', fontsize=20)
    ax[1].set_ylabel('Counts', fontsize=20)
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels(sex_labels)

    if plot_diagnosis:
        AFsev_train = list(db.features_dict[id][db.window_size]['diagnosis'] for id in train_pat)
        AFsev_test = list(db.features_dict[id][db.window_size]['diagnosis'] for id in test_pat)
        (_, counts_AFsev_train) = np.unique(AFsev_train, return_counts=True)
        (_, counts_AFsev_test) = np.unique(AFsev_test, return_counts=True)
        AF_labels = ['Non-AF', 'Paroxysmal AF', 'Persistent AF', 'Other cvd']
        index = np.arange(len(AF_labels))
        bar_width = bar_width / 2
        ax[2].bar(index, counts_AFsev_train, bar_width, label='train')
        ax[2].bar(index+bar_width, counts_AFsev_test, bar_width, label='test')
        ax[2].set_title(
            f'AFsev train-test distribution \n {100 * AFsev_train.count(0) / (AFsev_train.count(0) + AFsev_test.count(0)):.1f} - {100 * AFsev_train.count(1) / (AFsev_train.count(1) + AFsev_test.count(1)): .1f} - {100 * AFsev_train.count(3) / (AFsev_train.count(3) + AFsev_test.count(3)):.1f} - {100 * AFsev_train.count(4) / (AFsev_train.count(4) + AFsev_test.count(4)):.1f}')
        ax[2].set_xlabel('Diagnosis', fontsize=20)
        ax[2].set_ylabel('Counts', fontsize=20)
        ax[2].set_xticks(index + bar_width / 2)
        ax[2].set_xticklabels(AF_labels)
    fig.tight_layout()
    if savefig:
        lab = db.name + '_train_test_distribution_' + part + '.png'
        fig.savefig(savedir / lab, dpi=400, transparent=True)
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data loading for AF classification')
    parser.add_argument('--parser', choices=PARSER_MAP.keys(),
                        help='parser for creating the input file')
    parser.add_argument('--skip_plots', action='store_true',
                        help='dont generate plots')
    parser.add_argument('--regenerate', action='store_true',
                        help='save input file')
    parser.add_argument('--global_path', default=cts.REPO_DIR / "data" / "splits" / "ids",
                        help='location of db ids')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    if args.regenerate:
        db, train_pat_, val_pat_, test_pat_ = load_db(db_parser=PARSER_MAP[args.parser], train_part=0.6, val_part=0.2, expert_ann=True)
        np.save(args.global_path / (db.name + "_train_pat.npy"), train_pat_)
        np.save(args.global_path / (db.name + "_val_pat.npy"), val_pat_)
        np.save(args.global_path / (db.name + "_test_pat.npy"), test_pat_)
    else:
        db = PARSER_MAP[args.parser]()
        if args.parser == 'UVAFDB_Parser':
            print_diagram(db)
            train_pat_ = np.load(args.global_path / (db.name + "_train_pat.npy"), allow_pickle=True)

            val_pat_ = np.load(args.global_path / (db.name + "_val_pat.npy"), allow_pickle=True)
            print_summary(db, np.append(val_pat_, train_pat_), 'train-val')
        test_pat_ = np.load(args.global_path / (db.name + "_test_pat.npy"), allow_pickle=True)
    print('Summary for db: {}'.format(db.name))
    print_summary(db, test_pat_, 'test')
    # Load data for exploration
    # print("Loading Input Data...")
    # data_train, test_dict = return_input_model(parser_mapping_dict=PARSER_MAP, regenerate=args.regenerate, algo='ArNet2')

    # for pat in test_pat_:
    #     directory = pathlib.PurePath("/MLAIM/AIMLab/Shany/medAIM/GS") / db.name / str(pat)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     db.export_to_physiozoo(pat, directory=directory, export_rhythms=False, force=True, n_leads=db.get_num_leads(pat))

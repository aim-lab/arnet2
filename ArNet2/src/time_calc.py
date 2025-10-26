# General imports
import argparse
import os
import pathlib
import warnings
import numpy as np
import sys
import time
import sys
#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')
sys.path.append('/home/shanybiton/repos/Generalization/src')

import data.data_loading as data_loading
import src.models.metrics as metrics
import utils.consts as cts
import utils.feature_comp as fc
import models.model_utils as model_utils

from parsing.db_loader import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate inference time per window')
    parser.add_argument('--add_age_sex', action='store_false',
                        help='create sex and age test dicts')

    args, unk = parser.parse_known_args()
    if unk:
            warnings.warn("Unknown arguments:" + str(unk) + ".")
    data_train, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list)

    window_size = 60
    n = 10
    th_policy = 'F_beta'
    size = None
    UVAF_db = UVAFDB_Parser(window_size=window_size, load_on_start=False)
    X, y, t_s = test_dict['test_all']
    feats_to_use = np.append(cts.SELECTED_FEATURES, 'sqi')
    UVAF_db.create_pool()
    # Time to compute features
    start_feats = time.time()
    rr = X[:, :-3]
    for _ in range(n):
        feats_arr = np.zeros((len(rr), len(feats_to_use)))
        for i, feat in enumerate(feats_to_use):
            if feat == 'sqi':
                feats_arr[:, i] = 1
            else:
                func = getattr(fc, 'comp_' + feat)
                feats_arr[:, i] = np.array(UVAF_db.pool.starmap(func, zip(rr, )))

    end_feats = time.time()
    time_feats = (end_feats - start_feats) / n
    print("Time for feature computation using " + str(cts.N_PROCESSES) + " processes: " + str(time_feats))

    # Time algos
    algos = cts.models
    times = {}
    for algo in algos:
        if algo == 'ArNet':
            algo_py = 'ArNet2'
            cts.hypercomb['ArNet2'] = {'time_history': 12, 'extract_level': 'dense_2', 'n_units': 22,
                                       'dropout': 0.7194263073722404, 'learning_rate': 0.00010390447220815073,
                                       'af_weight': 3}
            feature_extractor_path = cts.path_models['1D-CNN']
        elif algo == 'ArNet2':
            algo_py = 'ArNet2'
            feature_extractor_path = cts.path_models['ResNet']
        else:
            algo_py = algo
            feature_extractor_path = None
        model_dict = model_utils.load_model(cts.path_models[algo], algo=algo_py, path_feature_extractor=feature_extractor_path)
        model = model_dict['classifier']
        best_th = model_dict['best_th']
        X, y, t_s = test_dict['test_all']
        if model_utils.use_history[algo_py]:
            start = time.time()
            for _ in range(n):
                probas = model.predict_proba(X)[:, 1]
            end = time.time()
        else:
            _, test_dict_XGB = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP,
                                                                    test_set_list=cts.test_set_list, algo='XGB')
            X_XGB, y_XGB, t_s_XGB = test_dict_XGB['test_all']
            rr = X_XGB[:,:-3]
            start = time.time()
            for _ in range(n):
                probas = model.predict_proba(rr)[:, 1]
            end = time.time()
        times[algo] = (end - start)
        if algo == 'XGB':
            print("Time for " + str(algo) + ": " +  str(times[algo] + time_feats))
        else:
            print("Time for " + str(algo) + ": " + str(times[algo]))
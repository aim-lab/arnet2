# General imports
import argparse
import pickle
import warnings
import sys
import numpy as np
import pandas as pd
import sklearn.utils.class_weight as skl_cw
import os
import timeit

# relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')
sys.path.append('/home/shanybiton/repos/Generalization/src')
# Relative imports

import utils.consts as cts
import models.metrics as metrics
import models.model_utils as model_utils
from parsing.db_loader import *
import data.data_loading as data_loading


def train(data_train, hypercomb, algo="1D-CNN", n_epochs=5, validation_data=None, path_feature_extractor=None):
    """ Instantiate and train the chosen model on training data.

    :param data_train: the training data, being a tuple (X_train, y_train)
    :param hypercomb: the dictionary of model's hyperparameters
    :param algo: a string representing the model's name
    :param n_epochs: the number of epochs for training
    :param validation_data: the optional validation data, for the learning curves, being a tuple (X_val, y_val)
    :param path_feature_extractor: if the algo is "ArNet2", the full path of the feature extractor, being either a 1D-CNN or a ResNet
    :return model: the trained ML model
    """

    # Model instantiation
    if algo == "XGB":
        model = model_utils.class_funcs[algo](**hypercomb, n_jobs=-1)
    elif algo == "ArNet2":
        model = model_utils.class_funcs[algo](**hypercomb, path_feature_extractor=path_feature_extractor)
    else:
        model = model_utils.class_funcs[algo](**hypercomb)

    # Model training
    X_train, y_train, _ = data_train
    if validation_data is not None:
        validation_data = validation_data[:2]
    if model_utils.use_history[algo]:
        model.fit(X_train, y_train, validation_data=validation_data, n_epochs=n_epochs)
    else:
        rr_train = X_train[:, :-3].astype('float32')
        if algo == "XGB":
            mean_train, std_train = np.mean(rr_train, axis=0), np.std(rr_train, axis=0)
            rr_train = (rr_train - mean_train) / std_train
            eval_set = [(rr_train, y_train)]
            if validation_data is not None:
                validation_data = ((validation_data[0] - mean_train) / std_train, validation_data[1])
                eval_set.append(validation_data)
            model.fit(rr_train, y_train, sample_weight=skl_cw.compute_sample_weight("balanced", y_train),
                      eval_set=eval_set, eval_metric='logloss', verbose=True)
        else:
            model.fit(rr_train, y_train, validation_data=validation_data, n_epochs=n_epochs)

    return model


def eval(model, algo="1D-CNN", decision_th=None, save=True, path_models=None, hypercomb=None, error_analysis=False,
         **eval_sets):  # Rk : train must be in eval sets, and the first one
    """ Evaluate the trained model on all the input evaluation sets, and optionally save them and perform error analysis.

    :param model: the trained ML model
    :param algo: a string representing the model's name
    :param decision_th: the float decision threshold chosen for the classifier. If None, it is computed to maximize the F1 score on the training set.
    :param save: a boolean to indicate whether we want to save the model
    :param path_models: the path of the folder in which we want to save the model
    :param hypercomb: the dictionary of model's hyperparameters (for saving)
    :param error_analysis: a boolean to indicate whether we want to perform error analysis
    :param eval_sets: a dictionary of datasets of the form (X, y), and possibly a third element t_s representing the timestamps of the windows
    :return metrics_dict: a dictionary of performance metrics, for each dataset
    :return best_th_dict: a dictionary of optimal decision thresholds, for each dataset
    :return mean_abs_afb_error_dict:  a dictionary of mean AFB error, for each dataset
    """

    # Initialize the results dictionaries
    metrics_dict, best_th_dict, mean_abs_afb_error_dict = {}, {}, {}
    model_dict = {'hyperparameters': hypercomb, 'best_th': decision_th}

    # Loop over the different sets to evaluate, and add their results to output dictionaries
    for set_name, set_data in eval_sets.items():
        print(f"evaluating performance on {set_name}...")
        # Predict probas
        X, y, t_s = set_data
        if model_utils.use_history[algo]:
            probas = model.predict_proba(X)[:, 1]
        else:
            rr = X[:, :-3].astype('float32')
            if algo == "XGB":
                if "train" in set_name:
                    mean_train, std_train = np.mean(rr, axis=0), np.std(rr, axis=0)
                try:
                    rr = (rr - mean_train) / std_train
                except NameError:
                    raise NameError(
                        "mean_train undefined. You have to put the training set as the first element of the eval_sets dict")
            probas = model.predict_proba(rr)[:, 1]

        # Compute metrics
        best_th = metrics.maximize_f_beta(probas, y)
        best_th_dict[set_name] = best_th
        if (decision_th is None) and (
                "train" in set_name):  # if the decision threshold is not given, set it to the optimal decision threshold on train
            decision_th = best_th_dict['train']
            model_dict['best_th'] = decision_th
        metrics_dict[set_name] = metrics.model_metrics(probas, y, probas > decision_th, print_metrics=True)
        mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y,
                                                                       probas > decision_th)  # takes a lot of time, so maybe we should comment it for now before making it faster

        # Update model_dict
        model_dict[f'metrics_{set_name}'] = dict(zip(cts.METRICS, metrics_dict[set_name]))
        model_dict[f'metrics_{set_name}']['mean_abs_afb_error'] = mean_abs_afb_error_dict[set_name]

        # Perform error analysis and save the related dataframe
        if error_analysis:
            model_name = list(cts.path_models.keys())[list(cts.path_models.values()).index(path_models)]
            df = pd.DataFrame(
                columns=['id', 'proba', 'decision_th', 'pred', 'lab', 'start_time', 'end_time', 'set_name'])
            df['id'] = X[:, -1]
            df['proba'] = probas
            df['decision_th'] = decision_th
            df['pred'] = (probas > decision_th)
            df['lab'] = y
            df['start_time'] = t_s[:, 0]
            df['end_time'] = t_s[:, -1]
            df['set_name'] = set_name

    # Save model and results
    if save:
        # model_utils.save_df(df, f"/home/shanybiton/repos/Generalization/output/{str(model_name)}_{set_name}_pred.csv")
        model_utils.save_model(model_dict, model, path_models, algo)

    return metrics_dict, best_th_dict, mean_abs_afb_error_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data input for AF classification')
    parser.add_argument('--algo', default='XGB',
                        help='choose: "XGB" / "1D-CNN" / "1D-CNN+DA" / "ArNet" / "ArNet+DA / "ResNet" / "ArNet2" / "ResNet+DA" / "ArNet2+DA" / "CRNN" / "RNN"')
    parser.add_argument('--task', default='train',
                        help='train / eval')
    parser.add_argument('--regenerate', action='store_true',
                        help='create and save new generated model input file')
    parser.add_argument('--add_age_sex', action='store_false',
                        help='create sex and age test dicts')
    parser.add_argument('--exp', default=None,
                        help='choose: None / 0 / 1')

    args, unk = parser.parse_known_args()

    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    if args.exp is not None:
        cts.path_models['ResNet'] = cts.MODEL_DIR / f'experiment_{args.exp}' / 'ResNet.pkl'
        cts.path_models['ArNet2'] = cts.MODEL_DIR / f'experiment_{args.exp}' / 'ArNet2.pkl'
    # Load data
    print("Loading Data...")
    data_train, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, regenerate=args.regenerate,
                                                            algo=args.algo, test_set_list=cts.test_set_list)
    if args.add_age_sex:
        test_dict['Female_group'], test_dict['Male_group'] = data_loading.group_sex(test_dict['test_all'])
        test_dict['low_age_group'], test_dict['mid_age_group'], test_dict['high_age_group'] = data_loading.group_age(
            test_dict['test_all'])
    test_dict['simulating_scenario'] = pickle.load(
        open(cts.REPO_DIR / 'data/splits/model_input' / 'RBAFDB_simulating_scenario_input.pickle', "rb"))
    if args.algo == 'ArNet':
        algo_py = 'ArNet2'
        cts.hypercomb['ArNet2'] = {'time_history': 12, 'extract_level': 'dense_2', 'n_units': 22,
                                   'dropout': 0.7194263073722404, 'learning_rate': 0.00010390447220815073,
                                   'af_weight': 3}
        feature_extractor_path = cts.path_models['1D-CNN']
    elif args.algo == 'ArNet2':
        algo_py = 'ArNet2'
        feature_extractor_path = cts.path_models['ResNet']
    elif 'DA' in args.algo:
        algo_py = args.algo.split("+")[0]
        feature_extractor_path = None
        if algo_py == 'ArNet':
            algo_py = 'ArNet2'
            cts.hypercomb['ArNet2'] = {'time_history': 12, 'extract_level': 'dense_2', 'n_units': 22,
                                       'dropout': 0.7194263073722404, 'learning_rate': 0.00010390447220815073,
                                       'af_weight': 3}
            feature_extractor_path = cts.path_models['1D-CNN+DA']
        elif algo_py == 'ArNet2':
            algo_py = 'ArNet2'
            feature_extractor_path = cts.path_models['ResNet+DA']
    else:
        algo_py = args.algo
        feature_extractor_path = None

    if args.task == "train":
        test_k = test_dict.keys()
        test_k_ = ['train', 'test_JPAFDB', 'test_UVAFDB', 'test_CPSCDB']
        if args.exp == '2':
            test_k_ = ['train', 'test_JPAFDB', 'test_UVAFDB']
            UVAF_reann_pat = np.load(cts.MODEL_DIR / 'experiment_2/UVAF_reann_pat.npy', allow_pickle=True)
            UVAF_reann_pat = np.array([i + '_UVAFDB' for i in UVAF_reann_pat])
            mask_UVAF_reann_pat = np.isin(test_dict['train'][0][:, -1], UVAF_reann_pat)
            test_dict.update({'train': (
                test_dict['train'][0][mask_UVAF_reann_pat], test_dict['train'][1][mask_UVAF_reann_pat],
                test_dict['train'][2][mask_UVAF_reann_pat])})
        X_temp = [test_dict[key][0] for key in test_k_]
        y_temp = [test_dict[key][1] for key in test_k_]
        t_s_temp = [test_dict[key][2] for key in test_k_]
        final = tuple()
        final = final + (np.vstack(X_temp), np.hstack(y_temp), np.vstack(t_s_temp))

        print(f"Training on all dataset except {np.setdiff1d(list(test_k), test_k_)}...")
        print(f"Saved path: {cts.path_models[args.algo]}")

        model = train(final, cts.hypercomb[algo_py], algo=algo_py, n_epochs=5, validation_data=None,
                      path_feature_extractor=feature_extractor_path)
        print("Evaluating performances...")
        eval(model, algo=algo_py, decision_th=None, save=True, path_models=cts.path_models[args.algo],
             hypercomb=cts.hypercomb[algo_py],
             **{'train': final, 'test': test_dict})  # test_dict {'train': data_train, 'test': test_dict}

    elif args.task == 'eval':
        print(f"Loading {args.algo}...")
        model_dict = model_utils.load_model(cts.path_models[args.algo], algo=algo_py,
                                            path_feature_extractor=feature_extractor_path)
        model = model_dict['classifier']
        print(f"Evaluating performances...")
        eval(model, algo=algo_py, path_models=cts.path_models[args.algo], decision_th=model_dict["best_th"],
             error_analysis=False, hypercomb=cts.hypercomb[algo_py], **test_dict, save=False)


    else:
        raise (NotImplementedError(f"The task {args.task} is not implemented."))

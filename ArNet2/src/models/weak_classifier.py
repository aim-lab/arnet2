# General imports
import argparse
import sys
import warnings
import pandas as pd
import numpy as np

#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import utils.consts as cts
from sklearn.metrics import precision_recall_curve
import model_utils as model_utils
import error_analysis_utils as metrics
from parsing.db_loader import *
import data.data_loading as data_loading

class WeakClassifier:

    def __init__(self, opt_metric='f1'):

        self.opt_metric = opt_metric
        self.thresh = None
        self.sign = None
        self.trained = False

    def fit(self, X, y):
        # Returning threshold and sign for the best weak classifier

        # First we check the SUP possibility, i.e. data > threshold
        best_f1_sup, thresh_sup = self.optimize_f1(X, y)

        # First we check the INF possibility, i.e. data < threshold
        best_f1_inf, thresh_inf = self.optimize_f1(-X, y)

        if best_f1_inf >= best_f1_sup:
            self.sign = cts.INF
            self.thresh = -thresh_inf
        else:
            self.sign = cts.SUP
            self.thresh = thresh_sup

        self.trained = True

    def predict(self, X_new):

        predicted = self.sign * X_new >= self.sign * self.thresh
        return predicted

    def predict_proba(self, X_new):
        return X_new

    def optimize_f1(self, X, y):
        precision, recall, thresholds = precision_recall_curve(y, X)
        f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0.0
        best_f1_idx = np.argmax(f1)
        best_f1 = f1[best_f1_idx]
        thresh = thresholds[best_f1_idx]
        return best_f1, thresh

    def eval(self, save=True, **eval_sets):  # Rk : train must be in eval sets, and the first one
        metrics_dict, mean_abs_afb_error_dict, model_dict = {}, {}, {}
        for set_name, set_data in eval_sets.items():
            print(f'evalutaing AFEv classifier on {set_name}...')
            X, y, t_s = set_data
            df_afev = pd.read_csv(cts.REPO_DIR / 'output' / f'AFEv_{set_name}_pred.csv')
            df_afev['pred'] = self.predict(df_afev.proba)
            metrics_dict[set_name] = metrics.model_metrics(df_afev.proba, df_afev.lab, self.predict(df_afev.proba),
                                                           print_metrics=True)
            mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y, self.predict(
                df_afev.proba))  # takes a lot of time, so maybe we should comment it for now before making it faster
            if save:
                df_afev.to_csv(cts.REPO_DIR / 'output' / f'Lorenz_{set_name}_pred.csv')
            model_dict[f'metrics_{set_name}'] = dict(zip(cts.METRICS, metrics_dict[set_name]))
            model_dict[f'metrics_{set_name}']['mean_abs_afb_error'] = mean_abs_afb_error_dict[set_name]
        if save:
            model_utils.save_model(model_dict, self, cts.path_models['AFEV'], 'XGB')
        return model_dict, metrics_dict, mean_abs_afb_error_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and store weak classifier based on lorenz plot')
    parser.add_argument('--regenerate', action='store_false',
                        help='train and save new classifier')


    args, unk = parser.parse_known_args()

    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    print("Loading Data...")
    _, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, algo='ArNet2', test_set_list=cts.test_set_list)
    test_dict['Female_group'], test_dict['Male_group'] = data_loading.group_sex(test_dict['test_all'])
    test_dict['low_age_group'], test_dict['mid_age_group'], test_dict['high_age_group'] = data_loading.group_age(
        test_dict['test_all'])
    df_train = pd.read_csv(cts.REPO_DIR / 'output' / 'AFEv_train_pred.csv')

    # train
    clf = WeakClassifier()
    clf.fit(df_train.proba, df_train.lab)

    #eval
    model_dict, metrics_dict, mean_abs_afb_error_dict = clf.eval(save=args.regenerate, **test_dict)


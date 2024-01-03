# General imports
import argparse
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# relative paths
from src.models import metrics

sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import src.models.model_utils as model_utils
import utils.consts as cts
import data.data_loading as data_loading
from parsing.db_loader import *
import src.models.metrics as metrics


class CostumedVotingClassifier(VotingClassifier):
    """
    Class defining a voting classifier.
    This class offers the possibility to load the classifiers onto the system.
    It further predicts probabilities based on the classifiers' threshold.
    """

    def _predict(self, X):
        """
        Collect results from clf.predict_proba calls and further predicts labels based on best_th.
        """
        stackX = None
        for est in self.estimators_:
            decision_th = est['best_th']
            probas = est['classifier'].predict_proba(X)[:, 1]
            yhat = probas > decision_th
            if stackX is None:
                stackX = yhat
            else:
                stackX = np.dstack((stackX, yhat))
        return stackX[0]

    def _collect_probas(self, X):
        """
        Collect results from clf.predict_proba calls.
        """
        stackX = None
        for est in self.estimators_:
            probas = est['classifier'].predict_proba(X)[:, 1]
            if stackX is None:
                stackX = probas
            else:
                stackX = np.dstack((stackX, probas))
        return stackX[0]


if __name__ == '__main__':
    # Load data
    print("Loading Data...")
    _, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list)

    print("Loading voting classifier...")
    members = data_loading.load_all_models(models=['ArNet', 'ArNet2', 'ArNet2+DA'])
    eclf = CostumedVotingClassifier(estimators=[(key, model['classifier']) for key, model in members.items()],
                                    weights=[1, 2, 1.5])
    _, y, _ = test_dict['test_JPAFDB']
    eclf.estimators_ = list(members.values())
    eclf.le_ = LabelEncoder().fit(y)
    eclf.classes_ = eclf.le_.classes_

    error_analysis = True
    metrics_dict, best_th_dict, mean_abs_afb_error_dict = {}, {}, {}
    for set_name, set_data in test_dict.items():
        print(f"evaluating performance on {set_name}...")
        X, y, t_s = set_data
        # Without calling fit
        yhat = eclf.predict(X)
        probas = eclf._collect_probas(X)
        metrics_dict[set_name] = metrics.model_metrics(np.zeros(shape=np.shape(y)), y, yhat, print_metrics=True)
        mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y,
                                                                       yhat)  # takes a lot of time, so maybe we should comment it for now before making it faster

        if error_analysis:
            df = pd.DataFrame([])
            df['id'] = X[:, -1]
            df = df.join(pd.DataFrame.from_dict(dict(zip(list(members.keys()), probas.T)), orient='columns'))
            df = df.join(pd.DataFrame.from_dict(dict(zip([member + ' decision threshold' for member in members.keys()],
                                                         np.tile([model['best_th'] for model in members.values()],
                                                                 np.shape(y.reshape(-1, 1))).T)), orient='columns'))
            df['yhat'] = yhat
            df['lab'] = y
            df['start_time'] = t_s[:, 0]
            df['end_time'] = t_s[:, -1]
            df['set_name'] = set_name
            model_utils.save_df(df,
                                f"/home/shanybiton/repos/Generalization/output/VotingClassifier_{set_name}_pred.csv")

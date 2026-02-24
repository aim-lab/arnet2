# from base_packages import *
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, \
    accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc

def print_met(accuracy, fbeta, se, sp, PPV, NPV, AUROC, AUCPR, beta):
    """
     This function prints the different metrics reeived as input.
    :param accuracy: The accuracy measure.
    :param fbeta: The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :param se: The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param sp: The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param PPV: The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Precision_and_recall)
    :param NPV: The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    :param AUROC: The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :param AUCPR: The Area Under the PR Curve. (https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/)
    :param beta: Index for the F-beta measure.
    """
    print("Accuracy: " + str(accuracy))
    print("F" + str(beta) + "-Score: " + str(fbeta))
    print("Sensitivity: " + str(se))
    print("Specificity: " + str(sp))
    print("PPV: " + str(PPV))
    print("NPV: " + str(NPV))
    print("AUROC: " + str(AUROC))
    print("AUCPR: " + str(AUCPR))


def model_metrics(X, y, y_true, print_metrics=True, beta=1):
    """
    This function returns different statistical binary metrics based on the data (output score/probabilities),
    the predicted and the actual labels. Function established for binary classification only.
    :param X: The output score/probabilities of the algorithm.
    :param y: The actual labels of the examples.
    :param y_hat: The predicted labels of the examples.
    :param print_metrics: Boolean value to print or not the metrics. Default is True
    :param beta: Index for the F-beta measure.
    """
    AUROC = roc_auc_score(y, X)
    precision_, recall_, thresholds = precision_recall_curve(y, X)
    accuracy = accuracy_score(y, y_true)
    TN, FP, FN, TP = confusion_matrix(y, y_true).ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    AUCPR = auc(recall_, precision_)

    if np.isnan(precision):
        precision = sys.float_info.epsilon
    if np.isnan(recall):
        recall = sys.float_info.epsilon

    sensitivity = recall
    specificity = TN / (TN + FP)
    NPV = TN / (TN + FN)
    PPV = precision
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    if np.isnan(fbeta):
        fbeta = sys.float_info.epsilon
    if print_metrics:
        print_met(accuracy, fbeta, sensitivity, specificity, PPV, NPV, AUROC, AUCPR, beta)
        print(confusion_matrix(y, y_true))
    return accuracy, fbeta, sensitivity, specificity, PPV, NPV, AUROC, AUCPR


def eval(clf, X_new, y_new, sign=1, print_metrics=True, threshold=None, beta=1):
    """
    This function evaluates the performance statistics of a given classifier and returns them.
    The classifier is assumed to implement the interface of sklearn classifiers (object which
    should have the following methods: predict, predict_proba).
    :param clf: The input classifier already trained.
    :param X_new: The raw data on which the classifier has been trained.
    :param y_new: The actual labels of the samples.
    :param sign: The direction of the decision function ( '<=' or '>=' for weak classifiers).
    :param beta: Index for the F-beta measure computation.
    :param print_metrics: Boolean value to print or not the mtrics. Default is True
    :param threshold: Threshold on the decision scores (output of clf.predict_proba) for the positive class. If None, set at 0.5
    :param beta: Index for the F-beta measure.
    :returns: evaluation metrics returned by model_metrics().
    """
    if threshold is None:
        predicted = clf.predict(X_new)
    else:
        predicted = clf.predict_proba(X_new)[:, 1] > threshold
    pred_score = clf.predict_proba(X_new)[:, -1]
    return model_metrics(sign * pred_score, y_new, predicted, print_metrics, beta)


def maximize_f_beta(probas, y_true, beta=1):
    """
    This function returns the decision threshold which maximizes the F_beta score.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns best_th: The threshold which optimizes the F_beta score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    if np.any(np.isnan(fbeta)):
        fbeta[np.isnan(fbeta)] = sys.float_info.epsilon
    best_th = thresholds[np.argmax(fbeta)]
    return best_th


def maximize_Se_plus_Sp(probas, y_true):
    """
    This function returns the decision threshold which maximizes the Se + Sp Measure.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :returns best_th: The threshold which optimizes the F_beta score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    se, sp = tpr, 1 - fpr
    best_th = thresholds[np.argmin(np.abs(se - sp))]
    return best_th


def mean_abs_afb_error(X, y, y_true, window_size=None, sampling_rate=200):
    """
    This function returns average absolute AF Burden.
    Works with both RR intervals (ArNet2) and raw ECG signals (ArNetECG).
    
    :param X: The raw data on which the classifier has been trained.
              - For ArNet2 (RR intervals): format [RR_data (60), prec_windows, global_label, ids]
              - For ArNetECG (ECG): format [ECG_data (6000), prec_windows, succ_windows, global_label, ids]
    :param y: The predicted labels.
    :param y_true: The actual labels.
    :param window_size: Optional. Window size (60 for RR, 6000 for ECG). If None, auto-detects from data.
    :param sampling_rate: Sampling rate for ECG data (default 200 Hz). Only used for ECG data.
    :returns mean_abs_error_af_burden: The mean absolute error in AF burden estimation.
    """
    pat_list = np.unique(X[:, -1])
    mean_abs_error_af_burden = 0
    
    # Auto-detect data type if window_size not provided
    if window_size is None:
        # Check if it's ECG (large window size) or RR intervals (small window size)
        # ArNet2: last 3 columns are metadata -> data starts at column 0, ends at -3
        # ArNetECG: last 4 columns are metadata -> data starts at column 0, ends at -4
        n_cols = X.shape[1]
        
        # Try ArNetECG format first (has succ_windows, so 4 metadata columns)
        if n_cols >= 6004:  # At least 6000 + 4 metadata columns
            window_size = n_cols - 4
            is_ecg = True
        # Try ArNet2 format (3 metadata columns)
        elif n_cols >= 63:  # At least 60 + 3 metadata columns
            window_size = n_cols - 3
            is_ecg = False
        else:
            # Default: assume RR intervals if window size is small
            window_size = n_cols - 3
            is_ecg = window_size > 100  # If window size > 100, likely ECG
    else:
        # Use provided window_size to determine data type
        is_ecg = window_size > 100  # ECG windows are typically 6000, RR windows are 60
    
    # Determine metadata column positions
    if is_ecg:
        # ArNetECG format: [ECG_data, prec_windows, succ_windows, global_label, ids]
        # Last 4 columns are metadata
        data_cols_end = -4
        window_duration = window_size / sampling_rate  # Duration in seconds (e.g., 6000/200 = 30s)
    else:
        # ArNet2 format: [RR_data, prec_windows, global_label, ids]
        # Last 3 columns are metadata
        data_cols_end = -3
        window_duration = None  # Will be calculated from RR intervals
    
    for i, pat in enumerate(pat_list):
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat].astype(np.float32)
        y_pred_pat = y_true[X[:, -1] == pat].astype(np.float32)
        
        if is_ecg:
            # For ECG: use constant window duration
            # Each window has the same duration (e.g., 30 seconds)
            window_durations = np.full(len(X_pat), window_duration, dtype=np.float32)
        else:
            # For RR intervals: calculate duration from sum of RR intervals
            rr = X_pat[:, :data_cols_end].astype(np.float32)
            window_durations = np.sum(rr, axis=1).astype(np.float32)  # Duration of each window in seconds
        
        # Calculate total duration
        total_duration = np.sum(window_durations)
        
        if total_duration == 0:
            continue  # Skip patients with zero duration
        
        # Calculate AF burden: (time_in_AF / total_time) * 100
        true_af_burden = 100 * (np.sum(window_durations * y_pat) / total_duration)
        pred_af_burden = 100 * (np.sum(window_durations * y_pred_pat) / total_duration)
        error_af_burden = pred_af_burden - true_af_burden
        mean_abs_error_af_burden += abs(error_af_burden) / len(pat_list)
    
    return mean_abs_error_af_burden


def minimize_err_AFB(probas, y_true, ids, rr_len):
    """
    This function returns the decision threshold which minimizes the mean error on the AF Burden.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param ids: The ids of the patients.
    :param rr_len: The lengths of the corresponding rr_intervals (in seconds).
    :returns best_th: The threshold which optimizes the mean AFB error.
    """
    # Creating a DataFrame and collecting all the possibilities.

    id_df = pd.DataFrame({'id': ids, 'len_rr': rr_len, 'probas': probas, 'label': y_true})
    id_df['time_in_af'] = id_df['label'] * id_df['len_rr']
    for i in np.arange(0.0, 1.001, 0.01):
        id_df['time_in_af_' + str(i)] = (id_df['probas'] > i) * id_df['len_rr']  # To sum to eventually obtain the
        # Predicted AF Burden.

    time_in_af = id_df.groupby('id').agg('sum')
    af_burdens = time_in_af.copy()
    for i in np.arange(0.0, 1.001, 0.01):
        af_burdens[str(i)] = ((time_in_af['time_in_af_' + str(i)] - time_in_af['time_in_af']) / time_in_af['len_rr']).apply(np.abs)

    final_res = af_burdens[[str(i) for i in np.arange(0.0, 1.001, 0.01)]].agg('mean')
    best_th = float(final_res.argmin() / 100)
    return best_th


def afb_f_beta_curve(y_pred, y_true, ids, rr_len, pat_labels, beta=1):
    """
    This function returns the decision threshold which maximizes the F_beta score on the AF_Burden estimation for the
    input patients.
    :param y_pred: The labels returned by the model.
    :param y_true: The actual labels.
    :param ids: The ids of the patients.
    :param rr_len: The lengths of the corresponding rr_intervals (in seconds).
    :param pat_labels: Dictionary containing the different global labels for the patients.
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns thresholds: All the possible thresholds.
    :returns F_betas: All the possible values for the F_beta score.
    """
    id_df = pd.DataFrame({'id': ids, 'len_rr': rr_len, 'y_pred': y_pred, 'y_true': y_true})
    id_df['time_in_af_pred'] = id_df['y_pred'] * id_df['len_rr']
    res = id_df.groupby('id').agg('sum')
    af_burdens = {pat: res.loc[pat]['time_in_af_pred'] / res.loc[pat]['len_rr'] for pat in pat_labels.keys()}
    afb, glob_lab = np.array(list(pat_labels.values())), np.array(list(af_burdens.values()))
    precision, recall, thresholds = precision_recall_curve(afb, glob_lab)
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    if np.any(np.isnan(fbeta)):
        fbeta[np.isnan(fbeta)] = sys.float_info.epsilon
    return thresholds, precision[:-1], recall[:-1], fbeta[:-1]

def compute_af_event_statistics(rec_length, events):
    """
    Computes statistical metrics for AF events.
    
    :param rec_length: Overall time of recording in seconds.
    :param events: DataFrame of AF events with 'Beginning' and 'End' columns.
    :returns: Tuple of (n_events, max_event_duration, min_event_duration, af_burden).
    """
    if len(events) == 0:
        return 4*[0]
    events['duration'] = events['End'] - events['Beginning']
    n_events = len(events)
    max_event = max(events["duration"])
    min_event = min(events["duration"])
    afb = events['duration'].sum() / rec_length
    print(f'Number of AF events: {n_events},\n'
          f'Longest event: {round(max_event, 2)} sec,\n'
          f'Shortest event: {round(min_event, 2)} sec,\n'
          f'AF-burden: {round(100*afb, 2)}%')

    return n_events, max_event, min_event, afb
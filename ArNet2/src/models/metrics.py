import sys
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from scipy.stats import ttest_rel, kruskal
from statsmodels.stats.proportion import proportions_ztest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
import scikit_posthocs as sp

import utils.consts as cts
import utils.graphics as graph

font = {'weight': 'normal',
        # 'family' : 'normal',
        'size': 22}

mpl.rc('font', **font)
cmap = get_cmap("tab10")
colors = cmap.colors

def print_met(accuracy, fbeta, se, sp, PPV, NPV, AUROC, AUCPR, beta):
    """ This function prints the different metrics reeived as input.
    :param accuracy:            The accuracy measure.
    :param fbeta:               The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :param AUROC:               The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :param sensitivity:         The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param specificity:         The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :param PPV:                 The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Precision_and_recall)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """
    print("Accuracy: " + str(accuracy))
    print("F" + str(beta) + "-Score: " + str(fbeta))
    print("Sensitivity: " + str(se))
    print("Specificity: " + str(sp))
    print("PPV: " + str(PPV))
    print("NPV: " + str(NPV))
    print("AUROC: " + str(AUROC))
    print("AUCPR: " + str(AUCPR))


def model_metrics(X, y, y_hat, print_metrics=True, beta=1):
    """ This function returns different statistical binary metrics based on the data (output score/probabilities),
        the predicted and the actual labels. Function established for binary classification only.
    :param X:                   The output score/probabilities of the algorithm.
    :param y:                   The actual labels of the examples.
    :param y_hat:               The predicted labels of the examples.
    :param beta:                Index for the F-beta measure.
    :param print_metrics:       Boolean value to print or not the mtrics. Default is True
    :returns accuracy:          The accuracy measure.
    :returns fbeta:             The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :returns AUROC:             The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :returns sensitivity:       The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns specificity:       The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns PPV:               The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """
    AUROC = roc_auc_score(y, X)
    precision_, recall_, thresholds = precision_recall_curve(y, X)
    accuracy = accuracy_score(y, y_hat)
    TN, FP, FN, TP = confusion_matrix(y, y_hat).ravel()
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
        print(confusion_matrix(y, y_hat))
    return accuracy, fbeta, sensitivity, specificity, PPV, NPV, AUROC


def eval(clf, X_new, y_new, sign=1, print_metrics=True, threshold=None, beta=1):

    """ This function evaluates the performance statistics of a given classifier and returns them.
        The classifier is assumed to implement the interface of sklearn classifiers (object which
        should have the following methods: predict, predict_proba).

    :param clf:                 The input classifier already trained.
    :param X_new:               The raw data on which the classifier has been trained (numpy array with dimensions (n_samples, n_features).
    :param y_new:               The actual labels of the samples.
    :param sign:                The direction of the decision function ( '<=' or '>=' for weak classifiers).
    :param beta:                Index for the F-beta measure computation.
    :param print_metrics:       Boolean value to print or not the mtrics. Default is True
    :param threshold:           Threshold on the decision scores (output of clf.predict_proba) for the positive class. If None, set at 0.5
    :returns accuracy:          The accuracy measure.
    :returns fbeta:             The F-beta measure (https://en.wikipedia.org/wiki/F1_score)
    :returns AUROC:             The Area Under the ROC Curve. (https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/)
    :returns sensitivity:       The Sensitivity (or Recall) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns specificity:       The Specificity (or False Positive Rate) of the algorithm. (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    :returns PPV:               The Positive Predictive Value (or Precision) of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    :returns NPV:               The Negative Predictive Value of the algorithm. (https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """

    if threshold is None:
        predicted = clf.predict(X_new)
    else:
        predicted = clf.predict_proba(X_new)[:, 1] > threshold
    pred_score = clf.predict_proba(X_new)[:, -1]
    return model_metrics(sign * pred_score, y_new, predicted, print_metrics, beta)

def mean_abs_afb_error(X, y, yhat):
    pat_list = np.unique(X[:, -1])
    mean_abs_error_af_burden = 0
    for i, pat in enumerate(pat_list):
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat]
        y_pred_pat = yhat[X[:, -1] == pat]
        rr = X_pat[:, :-3]
        true_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pat) / np.sum(rr))
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        error_af_burden = pred_af_burden - true_af_burden
        mean_abs_error_af_burden += abs(error_af_burden)/len(pat_list)
    return mean_abs_error_af_burden

def maximize_f_beta(probas, y_true, beta=1):
    """ This function returns the decision threshold which maximizes the F_beta score.
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


def maximize_Se_plus_Sp(probas, y_true, ids, rr_len, beta=1):
    """ This function returns the decision threshold which maximizes the Se + Sp Measure.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param ids: The ids of the patients.
    :param rr_len: The lengths of the corresponding rr_intervals (in seconds).
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns best_th: The threshold which optimizes the F_beta score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    se, sp = tpr, 1 - fpr
    best_th = thresholds[np.argmin(np.abs(se - sp))]
    return best_th


def minimize_err_AFB(probas, y_true, ids, rr_len, beta=1):
    """ This function returns the decision threshold which minimizes the mean error on the AF Burden.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param ids: The ids of the patients.
    :param rr_len: The lengths of the corresponding rr_intervals (in seconds).
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns best_th: The threshold which optimizes the mean AFB error.
    """
    # Creating a DataFrame and collecting all the possibilities.

    id_df = pd.DataFrame({'id': ids, 'len_rr': rr_len, 'probas': probas, 'label': y_true})
    id_df['time_in_af'] = id_df['label'] * id_df['len_rr']
    for i in np.arange(0.0, 1.001, 0.01):
        id_df['time_in_af_' + str(i)] = (id_df['probas'] > i) * id_df['len_rr']     # To sum to eventually obtain the Predicted AF Burden.

    time_in_af = id_df.groupby('id').agg('sum')
    af_burdens = time_in_af.copy()
    for i in np.arange(0.0, 1.001, 0.01):
        af_burdens[str(i)] = ((time_in_af['time_in_af_' + str(i)] - time_in_af['time_in_af']) / time_in_af['len_rr']).apply(np.abs)

    final_res = af_burdens[[str(i) for i in np.arange(0.0, 1.001, 0.01)]].agg('mean')
    best_th = float(final_res.argmin() / 100)
    return best_th

def plot_E_AF_burden(errors_af_burden_dict, set, model, format='png', savefig=False, savedir=None, dpi=400):

    # General Plot
    all_errors = np.concatenate(tuple([np.abs(x) for x in errors_af_burden_dict.values()]))
    std_error = np.std(all_errors, ddof=1)
    # Q = np.nanpercentile(all_errors, [25, 50, 75])

    final_labels = np.array([
        f"{cts.GLOBAL_LAB_TITLES[i]} (n= {len(errors_af_burden_dict[i])}) \n$|E_{{AF}}| (\%)$: "
        f"{np.median(np.abs(errors_af_burden_dict[i])):.1f} "
        f"({np.quantile(np.abs(errors_af_burden_dict[i]), 0.25):.1f}-"
        f"{np.quantile(np.abs(errors_af_burden_dict[i]), 0.75):.1f})"
        for i in range(len(errors_af_burden_dict))])
    plt.style.use('seaborn-white')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=dpi)
    bins = np.arange(-100, 100, 5)
    # bins = np.arange(0, 100, 5)
    axes[0][0].hist(errors_af_burden_dict[0], bins=bins, rwidth=0.8, label=final_labels[0], color=colors[0])
    axes[0][1].hist(errors_af_burden_dict[1], bins=bins, rwidth=0.8, label=final_labels[1], color=colors[1])
    axes[1][0].hist(errors_af_burden_dict[2], bins=bins, rwidth=0.8, label=final_labels[2], color=colors[2])
    axes[1][1].hist(errors_af_burden_dict[3], bins=bins, rwidth=0.8, label=final_labels[3], color=colors[3])
    # for i in range(len(axes)):
    #     for j in range(len(axes[0])):
    #         axes[i][j].axvline(x=Q[0], linestyle='dashed', color='black', linewidth=3)
    #         axes[i][j].axvline(x=Q[2], linestyle='dashed', color='black', linewidth=3)

    graph.complete_figure(fig, axes, x_titles=[['', ''], ['$E_{AF} (\%)$', '$E_{AF} (\%)$']],
                          y_titles=[['Number of Patients', ''],
                                    ['Number of Patients', '']], savefig=False,
                          main_title=f'Predicted_AF_burden_{set}_{model}',
                          xticks_fontsize=24, yticks_fontsize=24, xlabel_fontsize=24, ylabel_fontsize=24,
                          legend_fontsize=18, format=format)
    axes[0][0].legend(loc="best", handlelength=0, handletextpad=0, fancybox=True, fontsize=18)
    axes[0][1].legend(loc="best", handlelength=0, handletextpad=0, fancybox=True, fontsize=18)
    axes[1][0].legend(loc="best", handlelength=0, handletextpad=0, fancybox=True, fontsize=18)
    axes[1][1].legend(loc="best", handlelength=0, handletextpad=0, fancybox=True, fontsize=18)
    if savefig:
        fig.tight_layout()
        plt.savefig(str(savedir) + "/" + f'Predicted_AF_burden_{set}_{model}.' + format, dpi=dpi,
                    transparent=True, format=format)
    else:
        plt.show()

def print_afb_error_summary(models, members, data, sets=['test_all'], plot_E_AF=False, savedir=None, format='png'):
    errors_af_burden_dict = {}
    for set in sets:
        errors_af_burden_dict[set] = {}
        print('> reporing |E(AF)| for dataset: %s' % set)
        X, y, t_s = data[set]
        pat_list = np.unique(X[:, -1])
        for model in models:
            print('> reporing |E(AF)| for model: %s' % model)
            pred_df = pd.read_csv(cts.REPO_DIR / 'output' / (model + '_' + set + '_pred.csv'), index_col=0)
            probas = pred_df[['id', 'proba']]
            best_th = members[model]['best_th']
            errors_af_burden_dict[set][model] = afb_errors(pat_list, X, y, probas, best_th)
            if plot_E_AF:
                plot_E_AF_burden(errors_af_burden_dict[set][model], set=set, model=model, savedir=savedir, savefig=True, dpi=400, format=format)
    return errors_af_burden_dict

def afb_f_beta_curve(y_pred, y_true, ids, rr_len, pat_labels, beta=1):
    """ This function returns the decision threshold which maximizes the F_beta score on the AF_Burden estimation for the input patients.
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


def afb_errors(pat_list, X, y, probas, best_th):
    errors_af_burden_dict = {0: [], 1: [], 2: [], 3: []}
    for i, pat in enumerate(pat_list):
        # print(int(100 * i / len(test_pat)))
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat]
        y_pred_pat = probas.loc[probas.id.eq(pat), 'proba'].values > best_th
        if len(y_pred_pat) == 0:
            y_pred_pat = [False]*len(probas.loc[probas.id.eq(pat), 'proba'].values)
        rr, glob_lab = X_pat[:, :-3], X_pat[0, -2]
        if glob_lab == 4:
            glob_lab = 0
        true_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pat) / np.sum(rr))
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        error_af_burden = pred_af_burden - true_af_burden
        errors_af_burden_dict[glob_lab].append(error_af_burden)

    errors_af_burden_all = np.hstack(errors_af_burden_dict.values())
    if(len(errors_af_burden_all)==0):
        print('No E_AF for this group')
    else:
        print(f"for all patients: absolute: Min-Q1-Med-Q3-Max={np.min(np.abs(errors_af_burden_all)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_all), 0.25):.2f}"
              f"-{np.median(np.abs(errors_af_burden_all)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_all), 0.75):.2f}"
              f"-{np.max(np.abs(errors_af_burden_all)):.2f}")

    errors_af_burden_AF = np.hstack([errors_af_burden_dict[i] for i in range(1,4)])
    if(len(errors_af_burden_AF)<=1):
        print('No E_AF for AF patients in this group')
    else:
        print(f"for AF patients: absolute: Min-Q1-Med-Q3-Max={np.min(np.abs(errors_af_burden_AF)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_AF), 0.25):.2f}"
              f"-{np.median(np.abs(errors_af_burden_AF)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_AF), 0.75):.2f}"
              f"-{np.max(np.abs(errors_af_burden_AF)):.2f}")
    return errors_af_burden_dict

def plot_afb(X, af):
    rrs, ids = X[:, :-3], X[:, -1]
    pat_list = set(ids)
    af_burdens = []
    times_in_af = []
    for i, pat in enumerate(pat_list):
        print(int(100 * i / len(pat_list)))
        rr_pat = rrs[ids==pat]
        af_pat = af[ids==pat]
        time_in_af = np.sum(np.sum(rr_pat, axis=1) * af_pat)
        times_in_af.append(time_in_af)
        af_burden = 100 * (time_in_af / np.sum(rr_pat))
        af_burdens.append(af_burden)
    n_af = np.sum(np.array(times_in_af) >= cts.AF_MILD_THRESHOLD)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(af_burdens, bins=10)
    ax.set_xlabel('AFB (%)')
    ax.set_ylabel('Number of Patients')
    ax.set_title(f"#AF Patients = {n_af}/{len(pat_list)} (time in AF >= 30s) \n (threshold for AF window = 0.40)")
    plt.show()


def plot_model_history(model, algo, folder_date):
    def plot_one_history(H, title=''):
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.legend()
        plt.title(title)
        plt.show()
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.legend()
        plt.title(title)
        plt.show()
        plt.plot(H.history["auc"], label="train_auc")
        plt.plot(H.history["val_auc"], label="val_auc")
        plt.legend()
        plt.title(title)
        plt.show()

    if algo == "ArNet2":
        for (key, H) in model.histories.items():
            plot_one_history(H, title=algo+'_'+str(key)+'_'+str(folder_date))
    else:
        H = model.history
        plot_one_history(H, title=algo+'_'+str(folder_date))


def plot_proba_histogram(probas, best_th, y_true):

    # Plot predicted proba distribution
    fig, ax = plt.subplots(figsize=(5, 5))
    n_probs, n_corrects = [], []
    for i1 in np.arange(0, 1, 0.1):
        i2 = i1 + 0.1
        n_prob = np.sum((np.logical_and(i1 <= probas, probas <= i2)))
        n_correct = np.sum(np.logical_and(np.logical_and(i1 <= probas, probas <= i2),
                                          ((probas > best_th) == y_true)))
        n_probs.append(n_prob)
        n_corrects.append(n_correct)
    ax.bar(np.arange(0, 1, 0.1), n_probs, 0.2, label='n_prob')
    ax.bar(np.arange(0, 1, 0.1), n_corrects, 0.2, label='n_corrects')
    ax.set_title('Probas LTAF ' + f'Best Threshold Train = {best_th:.2f}')
    ax.legend(loc='upper right')
    plt.show()

    # Plot the accuracy vs the confidence, bin per bin
    plt.plot(np.arange(0, 1, 0.1), np.array(n_corrects) / np.array(n_probs))
    plt.show()

def paired_T_test(errors_af_burden_dict, model_1, model_2, set='test_all'):
    errors_af_burden_all_1 = np.hstack(errors_af_burden_dict[set][model_1].values())
    errors_af_burden_all_2 = np.hstack(errors_af_burden_dict[set][model_2].values())

    mean = np.mean(np.abs(errors_af_burden_all_1) - np.abs(errors_af_burden_all_2))
    std = np.std(np.abs(errors_af_burden_all_1) - np.abs(errors_af_burden_all_2), ddof=1)
    T = mean / (std / np.sqrt(len(errors_af_burden_all_2)))
    (stat, pval) = ttest_rel(errors_af_burden_all_1, errors_af_burden_all_2)
    print(f"Statistical testing (T-test) {model_1} vs {model_2}:")
    print(f"The t-values is: {T}\n")
    print('|EAF (%)| was {} with p-value of {}<0.0001'.format(
        'statistically significant' if pval < 0.0001 else 'not statistically significant', pval))

def propotional_T_test(model_1, model_2, set='test_all'):
    df_model_1 = pd.read_csv(cts.REPO_DIR / 'output' / f'{model_1}_{set}_pred.csv')
    df_model_2 = pd.read_csv(cts.REPO_DIR / 'output' / f'{model_2}_{set}_pred.csv')
    sample_success_model_1, sample_size_model_1 = (np.count_nonzero(df_model_1.pred), len(df_model_1))
    sample_success_model_2, sample_size_model_2 = (np.count_nonzero(df_model_2.pred), len(df_model_2))
    successes = np.array([sample_success_model_1, sample_success_model_2])
    samples = np.array([sample_size_model_1, sample_size_model_2])
    (stat, pval) = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')
    print(f"Statistical testing (propotional_T_test) {model_1} vs {model_2}:")
    print('F1-score was {} with p-value of {}<0.0001'.format(
        'statistically significant' if pval < 0.0001 else 'not statistically significant', pval))

def statistical_post_hoc_test(df, val_col_kruskal, val_col, group_col):
    for group in df[group_col].unique():
        stat, p = kruskal(df.loc[df[group_col].eq(val_col_kruskal), val_col],
                         df.loc[df[group_col].eq(group), val_col])
        print(f'{val_col_kruskal} vs {group}:')
        print('Statistics=%.3f, p=%f' % (stat, p))
    psthoc_df = sp.posthoc_dunn(df, val_col=val_col, group_col=group_col, p_adjust='holm')
    return psthoc_df

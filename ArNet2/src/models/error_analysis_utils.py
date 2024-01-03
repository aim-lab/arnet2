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


def _af_pat_lab(afb, thresh=cts.AF_MODERATE_THRESHOLD):
    """
    Computes the AF Burden and the global label for a given patient. As oppose to _af_pat_lab() implemented in
    base_parser.py, this function does not condition on time, as for some of the databases, the label per beat is
    unknown.
    The different categories of patients are: Severe AF (AFB between 80% and 100%), Moderate AF (AFB between
    thresh and 80%) and Non-AF otherwise. Assumption: mild AF and moderate AF are combined.
    :param afb: The patient AF Burden (afb). Computed as the time spent on AF divided by the total time of the recording
    :param thresh: afb threshold setting moderate label.
    """
    if afb > 100 * cts.AF_SEVERE_THRESHOLD:  # Assessing the class according to the guidelines
        af_pat_lab = cts.PATIENT_LABEL_AF_SEVERE
    elif afb > 100 * thresh:
        af_pat_lab = cts.PATIENT_LABEL_AF_MODERATE
    else:
        af_pat_lab = cts.PATIENT_LABEL_NON_AF
    return af_pat_lab



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
                plot_E_AF_burden(errors_af_burden_dict[set][model], set=set, model=model, savedir=savedir, savefig=True,
                                 dpi=400, format=format)
    return errors_af_burden_dict


def afb_errors(pat_list, X, y, probas, best_th):
    errors_af_burden_dict = {0: [], 1: [], 2: [], 3: []}
    for i, pat in enumerate(pat_list):
        # print(int(100 * i / len(test_pat)))
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat]
        y_pred_pat = probas.loc[probas.id.eq(pat), 'proba'].values > best_th
        if len(y_pred_pat) == 0:
            y_pred_pat = [False] * len(probas.loc[probas.id.eq(pat), 'proba'].values)
        rr, glob_lab = X_pat[:, :-3], X_pat[0, -2]
        if glob_lab == 4:
            glob_lab = 0
        true_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pat) / np.sum(rr))
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        error_af_burden = pred_af_burden - true_af_burden
        errors_af_burden_dict[glob_lab].append(error_af_burden)

    errors_af_burden_all = np.hstack(errors_af_burden_dict.values())
    if (len(errors_af_burden_all) == 0):
        print('No E_AF for this group')
    else:
        print(f"for all patients: absolute: Min-Q1-Med-Q3-Max={np.min(np.abs(errors_af_burden_all)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_all), 0.25):.2f}"
              f"-{np.median(np.abs(errors_af_burden_all)):.2f}"
              f"-{np.quantile(np.abs(errors_af_burden_all), 0.75):.2f}"
              f"-{np.max(np.abs(errors_af_burden_all)):.2f}")

    errors_af_burden_AF = np.hstack([errors_af_burden_dict[i] for i in range(1, 4)])
    if (len(errors_af_burden_AF) <= 1):
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
        rr_pat = rrs[ids == pat]
        af_pat = af[ids == pat]
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
            plot_one_history(H, title=algo + '_' + str(key) + '_' + str(folder_date))
    else:
        H = model.history
        plot_one_history(H, title=algo + '_' + str(folder_date))


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

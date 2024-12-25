# General imports
import argparse
import os
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap

import ArNet2.data.data_loading
import models.metrics as metrics
import numpy as np
import sys

#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import models.model_utils as model_utils
import utils.consts as cts
from ArNet2.data.data_loading import *
from parsing.db_loader import *
from models.weak_classifier import WeakClassifier

font = {'weight' : 'normal',
        # 'family' : 'normal',
        'size'   : 22}

mpl.rc('font', **font)
cmap = get_cmap("tab10")
colors = cmap.colors

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def generate_frame(parts, color, pos):
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("#0D232E")
        vp.set_linewidth(0.7)
    if len(color) == 1:
        color = [color] * len(pos)
    for i in range(len(pos)):
        parts['bodies'][i].set_facecolor(color[i])
        # parts['bodies'][i].set_edgecolor("#0D232E")
        parts['bodies'][i].set_alpha(0.8)
    return

def print_performance_summary(performance_dict, ordered_db_dict, print_metrics, models, savedir, file_name):
    df=pd.DataFrame([])
    for model in models:
        df_model = pd.DataFrame(performance_dict[model], list(np.arange(len(cts.evaluation_sets))))
        df_model.rename(columns={'mean_abs_afb_error': 'E_AF'}, inplace=True)
        df_model.insert(0, "model", model, True)
        df_model.insert(1, "db", ordered_db_dict, True)
        df = pd.concat([df, df_model])
    df = df.round(2).sort_index().sort_values(['db'])
    with open(f'{str(savedir)}/performace_{file_name}.tex', 'w') as tf:
        tf.write(df[['db', 'model']+print_metrics].to_latex())

def plot_performance_bars(models, metric_dict, db_metrics, db_list, score, width=0.1, savefig=False, savedir=None,
                          format='png'):
    """
    plot performace distriubtion.
    :param models: models to plot
    :param db_metrics: list of dataset metrics to plot
    :param db_list: list of x-labels
    :param score: performance score to use (example: Fb-score)
    :param width: the width of the bars
    :param savefig: True/False
    :param savedir: full path to save location
    :return: None
    """
    x = np.arange(len(models))  # the label locations
    fig, ax = plt.subplots(figsize=(24, 8))
    for k, model in enumerate(models):
        ([ax.bar(k + db_set * width, 100 * round(metric_dict[model][db_set][score], 3), width, label=set_lab,
                 color=colors[db_set]) for
          (db_set, set_lab) in zip(np.arange(len(db_metrics)), db_list)])
        # ax.bar_label(rects, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if score=='Fb-Score':
        ax.set_title('F1-score by architecture'.format(score))
        ax.set_ylabel('F1-score')
    else:
        ax.set_title('{} by architecture'.format(score))
        ax.set_ylabel(score)
    ax.set_xticks(x - (width / 2) + (width * len(models)) / 2)
    ax.set_xticklabels(models)
    # ax.legend(db_list, loc='upper left')
    ax.legend(db_list, loc='center left', bbox_to_anchor=(1, 0.5))

    score_values = []
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=15)
        score_values.append(container.datavalues[0])

    fig.tight_layout()
    plt.ylim(min(score_values) - 1, max(score_values) + 2)
    # plt.show()
    if savefig:
        plt.tight_layout()
        lab = score + '_distribution.' + format
        plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True, format=format)
    return

def plot_performace_dot(ax, models, metric_dict, evaluation_sets_dict, evaluation_sets_plot,
                        test_dict, score, savefig=False, savedir=None, extension='all_db', format='png'):
    """
    plot performance with dotted line.
    :param models: the evaluated models (i.e XGB, ArNet, ArNet2).
    :param metric_dict: models metrics statistics for different evaluation set.
    keys: models. values: lists of statistics for each evaluation set.
    :param evaluation_sets_dict: names of evaluation sets ordered as in metric_dict.
    :param evaluation_sets_plot: the reduced list of evaluation_sets desired for plotting.
    :param test_dict: dictionary with sets evaluation as keys and number of examples as values.
    :param score: evaluation metric to plot.
    :param savefig: True/False
    :param savedir: path to save th figure to.
    :param extension: added string to the naming of the figure.
    :return: None
    """
    plt.style.use('seaborn-white')
    y = np.empty(shape=[len(models), len(evaluation_sets_dict)])
    for k, model in enumerate(models):
        y[k] = np.array([100 * round(metric_dict[model][db_set][score], 3) for
                         db_set in np.arange(len(evaluation_sets_dict))])

    idx = np.argwhere(np.isin(list(evaluation_sets_dict.keys()), evaluation_sets_plot)).flatten()

    scatter = [ax.scatter(np.arange(len(idx)) + 1, y[i][idx], color=colors[i], s=95, label=model[i], zorder=i) for i in range(len(models))]
    [ax.plot(np.arange(len(idx)) + 1, y[i][idx], color=colors[i], linewidth=2, alpha=0.2, mec='k', label=model[i],
             zorder=i) for i in range(len(models))]
    [ax.plot([], [], 'o', linestyle = 'None', color=colors[i], label=model[i]) for i in range(len(models))]
    ax.set_xticks(np.arange(len(idx)) + 1)
    ax.set_xticklabels([evaluation_sets_dict[set_name] + cts.country_of_origin[evaluation_sets_dict[set_name]]
                        + '\n' + str(test_dict[set_name])
                        + ' patients' for set_name in evaluation_sets_plot], rotation=45)
    ax.tick_params(axis="both", which="major", labelsize=18, length=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([75, 100])

    plt.tight_layout()
    if savefig:
        ax.legend(models, loc='best')
        lab = score + extension + '_points.' + format
        plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True, format=format)
        plt.close()
    return

def plot_violin_probas(nonAF, AF, set, thresh=None, savefig=False, savedir=None, dpi=400, format='png'):
        plt.style.use('seaborn-white')
        plt.style.use('seaborn-white')
        fig, axes = plt.subplots(figsize=(8, 8), dpi=dpi)
        pos = [1, 2]
        parts = axes.violinplot([nonAF.proba, AF.proba],
                                positions=pos, showmeans=False,
                                showmedians=True, showextrema=True)
        generate_frame(parts, color=colors, pos=pos)

        axes.set_xticks(pos)
        axes.set_xticklabels(["non-AF" + '\nwindows= ' + str(len(nonAF)), "AF" + '\nwindows= ' + str(len(AF)),], fontsize=18)
        axes.set_ylabel("Probability", fontsize=24)
        axes.tick_params(axis="y", labelsize=18)
        if thresh is not None:
            axes.axhline(thresh, ls='--', c='r', linewidth=1)
        fig.tight_layout()
        if savefig:
            plt.tight_layout()
            lab = set + '_violin_proba.' + format
            plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True, format=format)
        else:
            plt.show()
        plt.close()
        return

def plot_by_pathology(FP, savefig=False, savedir=None, thresh=None, dpi=400, format='png'):
    NSR = FP.loc[FP.NSR.eq(60)]
    other = FP.loc[~FP.NSR.eq(60)]
    (labels_FP, counts_strat_FP) = np.unique(other.lab_rhythm.values, return_counts=True)
    counts_strat_FP = counts_strat_FP[labels_FP != 16]
    labels_FP = labels_FP[labels_FP != 16]
    labs = [list(cts.rhythms_dict.keys())[list(cts.rhythms_dict.values()).index(i)] for i in labels_FP]
    labs = labs[1:] + ['mixed labels']
    lNSR = len(NSR)
    counts_strat_FP = list(counts_strat_FP)[1:] + list(counts_strat_FP)[:1]
    counts_strat_FP = [lNSR] + counts_strat_FP
    labs = ['Other rhythms']+labs
    percentage = 100 * ((len(FP) - lNSR) / len(FP))
    n_dict = {list(cts.rhythms_dict.keys())[list(cts.rhythms_dict.values()).index(i)]: np.round(
        100 * (len(FP.loc[FP.lab_rhythm.eq(i)]) / len(FP)), 2)
     for i in labels_FP[1:]}
    n_dict['Other rhythms'] = np.round(100 * (lNSR / len(FP)), 2)
    n_dict['mixed labels'] = np.round(100*(counts_strat_FP[-1]/len(FP)), 2)
    print("Overall %.2f percent of FP had another documented cardiac abnormality" % percentage)
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax1 = fig.add_subplot(111)
    pos = np.arange(len(labs))+1
    parts = ax1.violinplot([NSR.proba.values] +
                            [FP.loc[FP.lab_rhythm.eq(i), 'proba'].values for i in labels_FP[1:]] +
                            [other.proba.values], positions=pos,
                            showmeans=False,
                            showmedians=True, showextrema=True)

    generate_frame(parts, color=[colors[0]], pos=pos)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labs, fontsize=18)
    ax1.tick_params(axis="both", which="major", labelsize=18, length=10)
    ax1.set_ylabel("Probability", fontsize=24)
    ax1.set_xlabel("Window label", fontsize=24)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2 = ax1.twiny()
    ax2.set_xticks(pos)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(["w=" + str(counts_strat_FP[i]) + '\n(%.2f' % n_dict[col] + '%)' for i, col in enumerate(labs)], fontsize=18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    if thresh is not None:
        ax1.axhline(thresh, ls='--', c='r', linewidth=1)
    fig.tight_layout()
    if savefig:
        fig.savefig(savedir / ('other_labels.' + format), dpi=dpi, transparent=True, bbox_inches='tight', format=format)
    else:
        plt.show()
    plt.close()
    return

def create_figure(models, metric_dict, evaluation_sets_dict,
                        test_dict, score, format='png', **kwargs):
    plt.style.use('seaborn-white')
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, width_ratios=[3, 1.5, 2], wspace=0.25, bottom=0.3, right=0.95,left=0.05)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(25, 8))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    plot_performace_dot(ax1, models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
                        evaluation_sets_plot=list(evaluation_sets_dict.keys())[:6],
                        test_dict=test_dict, score=score, format=format)

    # ax2 = fig.add_subplot(gs[0, 1], sharey = ax1)  # row 0, col 1
    plot_performace_dot(ax2, models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
                        evaluation_sets_plot=list(evaluation_sets_dict.keys())[6:8], test_dict=test_dict,
                        score=score, format=format)

    # ax3 = fig.add_subplot(gs[0, 2], sharey = ax1)  # row 1, span all columns
    plot_performace_dot(ax3, models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
                        evaluation_sets_plot=list(evaluation_sets_dict.keys())[8:], test_dict=test_dict,
                        score=score, format=format)
    handles, labels = ax3.get_legend_handles_labels()
    ax1.set_ylabel('F1')
    fig.legend(handles, models, ncol=2)
    if kwargs.get('savefig', False):
        fig.tight_layout()
        lab = '/performance_dots.' + format
        plt.savefig(str(kwargs.get('savedir')) + lab, dpi=400, transparent=True, format=format)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model results for AF detection')
    parser.add_argument('--fig_path', default=cts.REPO_DIR / 'figs',
                        help= 'where to save figures')
    parser.add_argument('--add_age_sex', action='store_false',
                        help='create sex and age test dicts')

    args, unk = parser.parse_known_args()
    if unk:
            warnings.warn("Unknown arguments:" + str(unk) + ".")

    evaluation_sets_dict = dict(zip(cts.evaluation_sets, cts.evaluation_sets_name_list))
    print("Loading Models...")

    members = data_loading.load_all_models(models=['AFEv'] + cts.models)
    metric_dict = {model: list(map(members[model].get, cts.evaluation_sets)) for model in
                   ['AFEv'] + cts.models}
    print("Loading Data...")
    data_train, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list)
    if args.add_age_sex:
        test_dict['Female_group'], test_dict['Male_group'] = data_loading.group_sex(test_dict['test_all'])
        test_dict['low_age_group'], test_dict['mid_age_group'], test_dict['high_age_group'] = data_loading.group_age(test_dict['test_all'])
    test_dict_N = {}
    for set_, vals in test_dict.items():
        test_dict_N['metrics_' + set_] = len(np.unique(np.array([id.rsplit('_', 1)[0] for id in vals[0][:,-1]])))

    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_performace_dot(ax, cts.models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
    #                     evaluation_sets_plot=['metrics_train', 'metrics_test_UVAFDB', 'metrics_test_all'], test_dict=test_dict_N,
    #                     score=cts.score, savefig=True, savedir=args.fig_path/'performance/',
    #                     extension='_reduced_ethnicity_db_latest', format='eps')
    #
    # plot across ethnicity
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_performace_dot(ax, cts.models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
    #                     evaluation_sets_plot=list(evaluation_sets_dict.keys())[:6], test_dict=test_dict_N,
    #                     score=cts.score, savefig=True, savedir=args.fig_path/'performance/',
    #                     extension='_ethnicity_latest', format='eps')
    #
    # # plot across sex
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_performace_dot(ax, cts.models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
    #                     evaluation_sets_plot=list(evaluation_sets_dict.keys())[6:8], test_dict=test_dict_N,
    #                     score=cts.score, savefig=True, savedir=args.fig_path/'performance/',
    #                     extension='_sex_latest', format='eps')

    # # plot across age
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_performace_dot(ax, cts.models, metric_dict=metric_dict, evaluation_sets_dict=evaluation_sets_dict,
    #                     evaluation_sets_plot=list(evaluation_sets_dict.keys())[8:], test_dict=test_dict_N,
    #                     score=cts.score, savefig=True, savedir=args.fig_path/'performance/',
    #                     extension='_age_latest', format='eps')

    # plot all together
    create_figure(['AFEv'] + cts.models, metric_dict, evaluation_sets_dict,
                  test_dict_N, cts.score, savefig=True, savedir=args.fig_path/'performance/', format='png')

    # errors_af_burden_dict = metrics.print_afb_error_summary(cts.models, members, data=test_dict, sets=list(test_dict.keys())[5:], plot_E_AF=False, savedir=args.fig_path/'E_AF/', format='eps')
    # print_performance_summary(metric_dict, cts.evaluation_sets, models=cts.models, print_metrics=cts.print_metrics, savedir=cts.REPO_DIR/'output', file_name='performances', format='eps')
    #
    # metrics.paired_T_test(errors_af_burden_dict, 'ArNet2', 'ArNet')
    # metrics.propotional_T_test('ArNet2', 'ArNet')
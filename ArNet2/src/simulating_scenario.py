# General imports
import argparse
import os
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib.cm import get_cmap
import re

import data_loading
import src.models.metrics as metrics
import numpy as np
import sys
import seaborn as sns


# relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import src.models.model_utils as model_utils
import utils.consts as cts
from data_loading import *
from parsing.db_loader import *
import utils.graphics as graph
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc

sns.color_palette("tab10", as_cmap=True)


def create_pat_list(df_pred):
    RBAF_mdclone_info = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info_mdclone.xlsx")  # All patients obtained from rambam
    RBAF_holter_info = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info.xlsx")  # Patients obtained from rambam with holter recordings
    reports_file = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / 'documentation' / 'reports' / 'RBAF_reports.xlsx')  # Patients obtained from rambam with holter reports
    NLP_file = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / 'documentation' / 'reports' / 'Holter_labels_final.xlsx')  # Final labeling per recording according to the reports

    # ALL patients in Rambam
    RBAF_mdclone_info.drop_duplicates(subset=['db_id'], inplace=True)
    print(
        f'{len(RBAF_mdclone_info.db_id.unique())} patients in cardiology at Rambam HCC between January 1st 2011 and October 1st 2021')

    # We are only about patients with Holter recordings
    # We labelled recordings according to the text in the reports, so recordings without reports were discarded
    print('Excluding no holter monitoring or Holter reports:')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(reports_file.holter_id)]
    RBAF_holter_info.loc[
        RBAF_holter_info.holter_id.isin(NLP_file.loc[NLP_file.AFIB.eq(1) | NLP_file.AFL.eq(1), 'Holter_id']), 'AF'] = 1
    RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(NLP_file.loc[NLP_file.AFL.eq(1), 'Holter_id']), 'AFL'] = 1
    RBAF_holter_info.AFL.fillna(0, inplace=True)
    RBAF_holter_info.AF.fillna(0, inplace=True)
    RBAF_holter_info.drop_duplicates('holter_id', inplace=True)
    print_summary(RBAF_holter_info, 'db_id')

    # Patient younger then 18 yrs are not the target population for AF
    print('Excluding patients younger then 18:')
    RBAF_holter_info = RBAF_holter_info.loc[
        RBAF_holter_info.age_at_recording.ge(18) | RBAF_holter_info.age_at_recording.isnull()]
    print_summary(RBAF_holter_info, 'db_id')
    # low sqi patients did not ran through the algorithm
    print('Excluding patients due to bSQI < 0.75:')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(df_pred.id)]
    print_summary(RBAF_holter_info, 'db_id')
    # pacemaker controls AF rhythm, thus can 'fool' the algorithm into false diagnosis
    print('Excluding patients due to pacecmaker:')
    RBAF_holter_info = RBAF_holter_info.loc[RBAF_holter_info.pacemaker_at_holter.ne(1)]
    print_summary(RBAF_holter_info, 'db_id')

    # assuming report older then 2017 maybe followed different guidelines then today
    print('Excluding patients due older then 2017:')
    temp = RBAF_holter_info.loc[RBAF_holter_info.recording_date > pd.Timestamp(2017, 1, 1)].drop_duplicates(
        subset=['db_id', 'recording_date'])
    print_summary(temp, 'db_id')
    return temp, RBAF_mdclone_info, RBAF_holter_info, reports_file, NLP_file

def report_metrics(proba, y, FN, FP, TN, TP, beta=1):
    PPV = len(TP) / (len(TP) + len(FP))
    NPV = len(TN) / (len(TN) + len(FN))
    Se = len(TP) / (len(TP) + len(FN))
    Sp = len(TN) / (len(TN) + len(FP))
    fbeta = (1 + beta ** 2) * PPV * Se / ((beta ** 2) * PPV + Se)
    AUROC = roc_auc_score(y, proba)
    precision_, recall_, thresholds = precision_recall_curve(y, proba)
    AUCPR = auc(recall_, precision_)

    print(f'PPV: {round(PPV, 2)}, NPV:{round(NPV, 2)}, ')
    print(f'Se: {round(Se, 2)}, Sp:{round(Sp, 2)}, F1:{round(fbeta, 2)}, AUROC:{AUROC}, AUCPR: {AUCPR}')

def record_diagnosis(df):
    af_cases = np.array(df["holter_id"][df.apply(
        lambda x: df['diagnosis_merged'].astype(str).str.contains(
            'ATRIAL FIBRILLATION', flags=re.I)).any(axis=1)].values).astype(str)
    afl_cases = np.array(df["holter_id"][df.apply(
        lambda x: df['diagnosis_merged'].astype(str).str.contains(
            'FLUTTER', flags=re.I)).any(axis=1)].values).astype(str)
    df.loc[df.holter_id.isin(af_cases), 'AF'] = 1
    df.loc[df.holter_id.isin(afl_cases), 'AFL?'] = 1
    return df

def _af_pat_lab(afb, thresh=cts.AF_MODERATE_THRESHOLD):
    """ Computes the AF Burden and the global label for a given patient. The AF Burden is computed as the time
    spent on AF divided by the total time of the recording. The different categories of patients are: Non-AF (Time in AF
    does not exceed 30 [sec], Mild AF (Time in AF above 30 [sec] and AFB under 4%), Moderate AF (AFB between 4 and 80%),
    and Severe AF (AFB between 80 and 100%). If the burden of a given pathology for a patient is over 50%, we flage him as a patient
    suffering from another CVD (label cts.PATIENT_LABEL_OTHER_CVD). As a convention, for windows, 0 is the label for NSR, 1 for AF, and above
    2 for other rhythms.
    :param id: The patient afb.
    """
    if afb > 100 * cts.AF_SEVERE_THRESHOLD:  # Assessing the class according to the guidelines
        af_pat_lab = cts.PATIENT_LABEL_AF_SEVERE
    elif afb > 100 * thresh:
        af_pat_lab = cts.PATIENT_LABEL_AF_MODERATE
    else:
        af_pat_lab = cts.PATIENT_LABEL_NON_AF
    return af_pat_lab

def afb_density_dist(pat_df, col='pred_afb', savefig=False, savedir=None, dpi=400):
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax = sns.kdeplot(data=pat_df.loc[pat_df.AFIB_true.eq(0), col],
                     label=f'non-AF, n={np.count_nonzero(pat_df.AFIB_true.eq(0))}', ax=ax)
    ax = sns.kdeplot(data=pat_df.loc[pat_df.AFIB_true.eq(1), col],
                     label=f'AF, n={np.count_nonzero(pat_df.AFIB_true.eq(1))}', ax=ax)
    ax.set_xlabel('Estimated AFB (%)', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=18)

    ax.set_xlim([0, 100])
    ax.legend(fontsize=18)
    fig.tight_layout()
    if savefig:
        plt.savefig(str(savedir / ('AFB density RBAFDB_simulating_scenario_' + col)), dpi=dpi,
                    transparent=True)
    else:
        plt.show()
    return


def afb_hist_plot(pat_df, col='pred_afb', thresh=None, savefig=False, savedir=None, dpi=400, format='png'):
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=(8, 8))
    im_lim = {0: [1400, 1580], 1: [0, 35]}
    bins = np.arange(0, 100.5, 2)
    d = .25
    ax = {}
    gs = GridSpec(2, 2, height_ratios=[1, 7], hspace=0.05)
    ax[0] = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    ax[1] = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2))
    ax[0].set_yticks([1500])
    ax[1].set_yticks(np.arange(0, 35, 10))
    for i in range(2):
        sns.histplot(data=pat_df, x=col, hue='AF',
                     bins=bins,
                     hue_order=[1, 0], ax=ax[i], shrink=0.8)
        ax[i].tick_params(axis="y", which="major", labelsize=18, length=10)
        ax[i].set_ylim(im_lim[i])
        ax[i].set(xlabel=None)
        ax[i].set(ylabel=None)
        ax[i].legend([])
        ax[i].set_xticklabels(())
        if i == 1:
            ax[1].legend([f'non-AF, n={(pat_df["AF"] == 0).sum()}',
                          f'AF, n={(pat_df["AF"] == 1).sum()}'],
                         fontsize=18)
        if thresh is not None:
            ax[i].axvline(thresh, ls='--', c='r', linewidth=2)
    ax[1].set_xticks(np.arange(0, 100.5, 10))
    ax[1].set_xticklabels(list((np.arange(0, 100.5, 10)).astype(int)))
    ax[1].tick_params(axis="both", which="major", labelsize=18, length=10)

    ax[0].spines["bottom"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax[0].plot([0, 1], [0, 0], transform=ax[0].transAxes, **kwargs)
    ax[1].plot([0, 1], [1, 1], transform=ax[1].transAxes, **kwargs)

    fig.supylabel('Count', fontsize=18)
    fig.supxlabel('Estimated AFB (%)', fontsize=18)
    fig.tight_layout()
    if savefig:
        plt.savefig(str(savedir / ('AFB hist RBAFDB_simulating_scenario.' + format)), dpi=dpi,
                    transparent=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model results for AF detection')
    parser.add_argument('--algo', default='ArNet2',
                        help='choose: "XGB" / "1D-CNN" / "1D-CNN+DA" / "ArNet" / "ArNet+DA / "ResNet" / "ArNet2" / "ResNet+DA" / "ArNet2+DA" / "CRNN" / "RNN"')
    parser.add_argument('--add_age_sex', action='store_false',
                        help='create sex and age test dicts')

    args, unk = parser.parse_known_args()
    if unk:
            warnings.warn("Unknown arguments:" + str(unk) + ".")
    algo_py = 'ArNet2'
    feature_extractor_path = cts.path_models['ResNet']
    model_dict = model_utils.load_model(cts.path_models[args.algo], algo=algo_py,
                                        path_feature_extractor=feature_extractor_path)
    X, y, t_s = pickle.load(
        open(cts.REPO_DIR / 'data/splits/model_input' / 'RBAFDB_simulating_scenario_input.pickle', "rb"))
    X_test, y_test, t_s_test = pickle.load(
        open(cts.REPO_DIR / 'data/splits/model_input' / 'RBAFDB_test_input.pickle', "rb"))
    X_temp, y_temp, t_s_temp = np.append(X, X_test, axis=0), np.append(y, y_test, axis=0), np.append(t_s, t_s_test, axis=0)
    RBAFDB_simulating_scenario_info = pd.read_excel(
        cts.REPO_DIR / 'error_analysis' / 'RBAFDB_simulating_scenario_info.xlsx')
    model = model_dict['classifier']
    probas = model.predict_proba(X_temp)[:, 1]
    df = pd.DataFrame(columns=['id', 'proba', 'decision_th', 'pred', 'lab', 'start_time', 'end_time', 'set_name'])
    df['id'] = [i.split('_')[0] for i in X_temp[:, -1]]
    df['proba'] = probas
    df['decision_th'] = model_dict['best_th']
    df['pred'] = (probas > model_dict['best_th'])
    df['start_time'] = t_s_temp[:, 0]
    df['end_time'] = t_s_temp[:, -1]

    pat_list, RBAF_mdclone_info, RBAF_holter_info, reports_file, NLP_file = create_pat_list(df)
    ids_rbdb = np.unique([i.split('_')[0] for i in X_test[:, -1]])
    rbdb_ni_rbdb2 = ids_rbdb[~np.isin(ids_rbdb, pat_list.holter_id)]
    RBAF_holter_info_all = pd.read_excel(
        cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info.xlsx")  # Patients obtained from rambam with holter recordings
    RBAF_holter_info_all['AF'] = 0
    RBAF_holter_info_all['AFL'] = 0
    patient_file = pd.read_excel(cts.REPO_DIR / 'patient_file.xlsx')
    rbdb = patient_file[patient_file.db.eq('RBDB-test')]
    RBAF_holter_info_all.loc[RBAF_holter_info_all.holter_id.isin(rbdb.loc[rbdb.lab.gt(0), 'id']), 'AF'] = 1
    pat_list = pat_list.append(RBAF_holter_info_all.loc[RBAF_holter_info_all.holter_id.isin(rbdb_ni_rbdb2)])
    pat_list.loc[pat_list.holter_id.isin(patient_file.loc[patient_file.lab.ge(1), 'id']), 'AF'] = 1
    pred_afb, pred_af_lab, true_afb = {}, {}, {}
    afb_thresh = 0.04
    print(f'reprorting metrices for threshold: {afb_thresh}')
    for i, pat in enumerate(pat_list.holter_id):
        # print(int(100 * i / len(test_pat)))
        X_pat = X_temp[X_temp[:, -1] == pat + '_RBAFDB']
        y_pat = y_temp[X_temp[:, -1] == pat + '_RBAFDB']
        y_pred_pat = df.loc[df.id.eq(pat), 'proba'].values > model_dict['best_th']
        rr, glob_lab = X_pat[:, :-3], X_pat[0, -2]
        if glob_lab == 4:
            glob_lab = 0
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        pred_afb[pat] = pred_af_burden
        pred_af_lab[pat] = _af_pat_lab(pred_af_burden, thresh=afb_thresh)
    pat_list['pred_afb'] = pat_list['holter_id'].map(pred_afb)
    pat_list['pred_af_lab'] = pat_list['holter_id'].map(pred_af_lab)
    pat_list.loc[pat_list.pred_af_lab.gt(0), 'AFIB_pred'] = 1
    pat_list['Previous AF/AFL'] = pat_list['holter_id'].map(RBAFDB_simulating_scenario_info.set_index('Holter ID')['Previous AF\AFL'])
    pat_list['Died'] = pat_list['holter_id'].map(
        RBAFDB_simulating_scenario_info.set_index('Holter ID')['Died'])
    pat_list['Previous AF/AFL'].fillna(0, inplace=True)
    pat_list.AFIB_pred.fillna(0, inplace=True)

    pat_list.loc[pat_list.age_at_recording.le(60), 'age_group'] = 1
    pat_list.loc[pat_list.age_at_recording.gt(60) & pat_list.age_at_recording.le(75), 'age_group'] = 2
    pat_list.loc[pat_list.age_at_recording.ge(75), 'age_group'] = 3
    pat_list.loc[pat_list.sex.eq("F"), 'sex_group'] = 1
    pat_list.sex_group.fillna(0, inplace=True)

    # TP
    TP_df = pat_list.loc[pat_list.AFIB_pred.eq(1) & pat_list.AF.eq(1)]
    # TN
    TN_df = pat_list.loc[pat_list.AFIB_pred.eq(0) & pat_list.AF.eq(0)]

    # FP
    FP_df = pat_list.loc[pat_list.AFIB_pred.eq(1) & pat_list.AF.eq(0)]
    # for i, j in RBAF_holter_info.iterrows():
    #     if pd.isnull(RBAF_mdclone_info.at[i, 'history of ablation ever-field']):
    #         continue
    #     else:
    #         RBAF_holter_info.at[i, 'Previous ablation'] = 1

    # FN
    FN_df = pat_list.loc[pat_list.AFIB_pred.eq(0) & pat_list.AF.eq(1)]
    temp_FN_df = df.loc[df.id.isin(FN_df.holter_id)]
    temp_FN_df['duration'] = temp_FN_df.end_time - temp_FN_df.start_time
    np.median(temp_FN_df.groupby(['id'])['duration'].agg('sum') / cts.N_MS_IN_S)
    print("FN total AF events duration are of: {:.2f}, ({:.2f}-{:.2f})".format(
        np.median(temp_FN_df.groupby(['id'])['duration'].agg('sum')/cts.N_MS_IN_S),
        np.quantile(temp_FN_df.groupby(['id'])['duration'].agg('sum')/cts.N_MS_IN_S, 0.25),
        np.quantile(temp_FN_df.groupby(['id'])['duration'].agg('sum')/cts.N_MS_IN_S, 0.75)))

    print('reporting for intended case scenario..')
    report_metrics(pat_list.pred_afb, pat_list.AF, FN_df, FP_df, TN_df, TP_df)

    # male vs female
    print(f'reporting for Male group, total of {np.count_nonzero(pat_list.sex_group.eq(0))} patients')
    report_metrics(pat_list.loc[pat_list.sex_group.eq(0), 'pred_afb'], pat_list.loc[pat_list.sex_group.eq(0), 'AF'],
                   FN_df.loc[FN_df.sex_group.eq(0)], FP_df.loc[FP_df.sex_group.eq(0)],
                   TN_df.loc[TN_df.sex_group.eq(0)], TP_df.loc[TP_df.sex_group.eq(0)])
    print(f'reporting for Female group, total of {np.count_nonzero(pat_list.sex_group.eq(1))} patients')
    report_metrics(pat_list.loc[pat_list.sex_group.eq(1), 'pred_afb'], pat_list.loc[pat_list.sex_group.eq(1), 'AF'],
                   FN_df.loc[FN_df.sex_group.eq(1)], FP_df.loc[FP_df.sex_group.eq(1)],
                   TN_df.loc[TN_df.sex_group.eq(1)], TP_df.loc[TP_df.sex_group.eq(1)])

    # age groups
    print(f'reporting for age group <=60, total of {np.count_nonzero(pat_list.age_group.eq(1))} patients')
    report_metrics(pat_list.loc[pat_list.age_group.eq(1), 'pred_afb'], pat_list.loc[pat_list.age_group.eq(1), 'AF'],
                   FN_df.loc[FN_df.age_group.eq(1)], FP_df.loc[FP_df.age_group.eq(1)],
                   TN_df.loc[TN_df.age_group.eq(1)], TP_df.loc[TP_df.age_group.eq(1)])
    print(f'reporting for age group between 60 to 75, total of {np.count_nonzero(pat_list.age_group.eq(2))} patients')
    report_metrics(pat_list.loc[pat_list.age_group.eq(2), 'pred_afb'], pat_list.loc[pat_list.age_group.eq(2), 'AF'],
                   FN_df.loc[FN_df.age_group.eq(2)], FP_df.loc[FP_df.age_group.eq(2)],
                   TN_df.loc[TN_df.age_group.eq(2)], TP_df.loc[TP_df.age_group.eq(2)])
    print(f'reporting for age group >75, total of {np.count_nonzero(pat_list.age_group.eq(3))} patients')
    report_metrics(pat_list.loc[pat_list.age_group.eq(3), 'pred_afb'], pat_list.loc[pat_list.age_group.eq(3), 'AF'],
                   FN_df.loc[FN_df.age_group.eq(3)], FP_df.loc[FP_df.age_group.eq(3)],
                   TN_df.loc[TN_df.age_group.eq(3)], TP_df.loc[TP_df.age_group.eq(3)])

    # cost efficient analysis

    print(f'non AF patients: {np.count_nonzero(pat_list.AF==0)}, '
          f'recordings would have been flagged as AF: {len(FP_df)}, '
          f'recordings with other arrhythmia:'
          f' {np.count_nonzero(NLP_file.loc[NLP_file.Holter_id.isin(FP_df.holter_id), "other"])}')

    print(f'AF patients: {np.count_nonzero(pat_list.AF)}, '
          f'recordings correctly classified: {len(TP_df)}')
    ###################################
    # AF vs other rhythm analysis for FN
    print('Analysing FN windows..')
    FN_file = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(FN_df.holter_id)]
    FN_mdclone = RBAF_mdclone_info.loc[RBAF_mdclone_info.db_id.isin(FN_file.db_id)]
    FN_previous_AF = FN_df.loc[FN_df["Previous AF/AFL"].eq(1)]
    other_cad = FN_mdclone[FN_mdclone[['cad-diagnosis']].notnull().all(1)]
    print(f'number of FN patients: {len(FN_file)}\n'
          f'number of FN which where already diagnosed with AF: '
          f'{np.count_nonzero(FN_df["Previous AF/AFL"])}, {round(100 * (np.count_nonzero(FN_df["Previous AF/AFL"]) / len(FN_file)), 2)}%\n'
          f'number of FN which where diagnosed with AF on mdclone but not written in reports: '
          f'{np.count_nonzero(FN_file["AF"])}, {round(100 * (np.count_nonzero(FN_file["AF"]) / len(FN_file)), 2)}%\n'
          f'number of which with AFL: {np.count_nonzero(FN_file["AFL"])}, {round(100 * (np.count_nonzero(FN_file["AFL"]) / len(FN_file)), 2)}%\n'
          f'number of which with Other cad: {len(other_cad)}, {round(100 * (len(other_cad) / len(FN_file)), 2)}%\n such as {other_cad["cad-diagnosis"].value_counts()}')

    ##################################
    # AF vs other rhythm analysis for FP
    FP_file = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(FP_df.holter_id)]
    FP_mdclone = RBAF_mdclone_info.loc[RBAF_mdclone_info.db_id.isin(FP_file.db_id)]
    FP_previous_AF = FP_df.loc[FP_df["Previous AF/AFL"].eq(1)]
    other_cad = FP_mdclone[FP_mdclone[['cad-diagnosis']].notnull().all(1)]
    print(f'number of FP patients: {len(FP_file)}\n'
          f'number of FP which where already diagnosed with AF: '
          f'{np.count_nonzero(FP_df["Previous AF/AFL"])}, {round(100 * (np.count_nonzero(FP_df["Previous AF/AFL"]) / len(FP_file)), 2)}%\n'
          f'number of FP which where diagnosed with AF on mdclone but not written in reports: '
          f'{np.count_nonzero(FP_file["AF"])}, {round(100 * (np.count_nonzero(FP_file["AF"]) / len(FP_file)), 2)}%\n'
          f'number of which with AFL: {np.count_nonzero(FP_file["AFL"])}, {round(100 * (np.count_nonzero(FP_file["AFL"]) / len(FP_file)), 2)}%\n'
          f'number of which with Other cad: {len(other_cad)}, {round(100 * (len(other_cad) / len(FP_file)), 2)}%\n such as {other_cad["cad-diagnosis"].value_counts()}')

    # plot density distriubtion of afb
    afb_hist_plot(pat_list, col='pred_afb', savefig=True, savedir=cts.REPO_DIR / 'figs' / 'error_analysis',
                  thresh=100 * afb_thresh)
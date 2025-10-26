# General imports
import argparse
import warnings
import sys
import re
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import data.data_loading as data_loading
import utils.consts as cts
import models.metrics as metrics
import models.model_utils as model_utils
from parsing.db_loader import *
from results_viz import plot_by_pathology

def _af_pat_lab(afb):
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
    elif afb > 100 * cts.AF_MODERATE_THRESHOLD:
        af_pat_lab = cts.PATIENT_LABEL_AF_MODERATE
    else:
        af_pat_lab = cts.PATIENT_LABEL_NON_AF
    return af_pat_lab

def load_reannotated(db, pat_list):
    # parse the reannotated patients with the reannotated files, when separating between AF and AFL
    cts.rhythms_dict = {'NSR': 0, 'AFIB': 1, 'AFL': 6, 'NOD': 2, 'AB': 3, 'AT': 4, 'PAT': 5}
    ann_ids = next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1]
    pat_list = pat_list[np.isin(pat_list, ann_ids)]
    for pat in pat_list:
        print(pat)
        db.parse_elem_data(pat=pat, reannotated=True)
        db._win_lab(id=pat, win=db.window_size)
        db._af_win_lab(id=pat, win=db.window_size)
        db._af_pat_lab(id=pat)
    ids = db.return_patient_ids(pat_list=pat_list)
    rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=pat_list, return_binary=False) # y is the true label per window
    if db.name == 'CPSCDB':
        ids = np.array([id.rsplit('_', 1)[0] for id in ids], dtype=ids.dtype)
        AFL_lab = db.rhythms_dict['(AFL']
    else:
        AFL_lab = cts.rhythms_dict['AFL']
    ids = np.array([str(id) + '_' + db.name for id in ids], dtype=ids.dtype)
    X = np.concatenate((rr, ids.reshape(-1, 1)), axis=1)
    return X, y

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

def plot_cosEn(cosEn1, cosEn2, test_dict, savefig=False, savedir=None, dpi=400):
    plt.style.use('seaborn-white')
    ids_AF_win_F = np.argwhere(test_dict['Female_group'][1] == True).reshape(1, -1).flatten()
    ids_AF_win_M = np.argwhere(test_dict['Male_group'][1] == True).reshape(1, -1).flatten()
    ids_nonAF_win_F = np.argwhere(test_dict['Female_group'][0] == True).reshape(1, -1).flatten()
    ids_nonAF_win_M = np.argwhere(test_dict['Male_group'][0] == True).reshape(1, -1).flatten()
    print(f"Female AF CosEn Quantile: {np.median(cosEn1[ids_AF_win_F]):.2f}, ({np.quantile(cosEn1[ids_AF_win_F], 0.25):.2f}-{np.quantile(cosEn1[ids_AF_win_F], 0.75):.2f}) ")
    print(f"Man AF CosEn Quantile: {np.median(cosEn2[ids_AF_win_M]):.2f}, ({np.quantile(cosEn2[ids_AF_win_M], 0.25):.2f}-{np.quantile(cosEn2[ids_AF_win_M], 0.75):.2f}) ")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    sns.kdeplot(data=cosEn1[ids_AF_win_F],
                     label=f'Female', ax=ax[0])
    sns.kdeplot(data=cosEn2[ids_AF_win_M],
                     label=f'Male', ax=ax[0])
    sns.kdeplot(data=cosEn1[ids_nonAF_win_F],
                     label=f'Female', ax=ax[1])
    sns.kdeplot(data=cosEn2[ids_nonAF_win_M],
                     label=f'Male', ax=ax[1])
    for i in range(2):
        ax[i].set_ylabel('', fontsize=18)
        ax[i].tick_params(axis="both", which="major", labelsize=18, length=10)
        ax[i].set_xlabel('cosEn', fontsize=24)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
    ax[1].legend(fontsize=20)
    fig.supylabel('Density', fontsize=24)
    fig.tight_layout()
    if savefig:
        plt.savefig(str(savedir / ('cosEn_Female_Male')), dpi=dpi,
                    transparent=True)
    else:
        plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data input for AF classification')
    parser.add_argument('--model', default='XGB',
                        help='choose: "XGB" / "ArNet" / "ArNet2"')
    parser.add_argument('--add_age_sex', action='store_false',
                        help='create sex and age test dicts')

    args, unk = parser.parse_known_args()

    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    print("Loading Data...")
    data_train, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list)
    _, test_dict_XGB = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list,
                                                       algo='XGB')
    patient_file = pd.read_excel(cts.REPO_DIR / 'patient_file.xlsx')
    df_model_all = pd.read_excel(cts.REPO_DIR / 'output' / 'ArNet2_test_all_pred_error_analysis.csv')
    df_model_all_rlab = pd.read_excel(cts.REPO_DIR / 'error_analysis' / 'test_all_rlab.xlsx')

    ###################################
    # AF vs other rhythm analysis for FN
    print('Analysing FN windows..')
    FN_all = df_model_all_rlab.loc[df_model_all_rlab.pred.eq(False) & df_model_all_rlab.lab.eq(True)]
    FP_all = df_model_all_rlab.loc[df_model_all_rlab.pred.eq(True) & df_model_all_rlab.lab.eq(False)]
    (labels_FN, counts_strat_FN) = np.unique(FN_all.lab_rhythm.values, return_counts=True)
    print(f'out of the FN windows:')
    print("\n".join(
        f"{list(cts.rhythms_dict.keys())[list(cts.rhythms_dict.values()).index(i)]}: {len(FN_all.loc[FN_all.lab_rhythm.eq(i)])}, {np.round(100 * (len(FN_all.loc[FN_all.lab_rhythm.eq(i)]) / len(FN_all)),2)} %"
        for i in labels_FN))
    ###################################

    ###################################
    # other arrhythmias analysis for FP (in short)
    print('Analysing FP windows..')
    (labels_FP, counts_strat_FP) = np.unique(FP_all.lab_rhythm.values, return_counts=True)
    print(f'out of the FP windows:')
    print("\n".join(
        f"{list(cts.rhythms_dict.keys())[list(cts.rhythms_dict.values()).index(i)]}: {len(FP_all.loc[FP_all.lab_rhythm.eq(i)])}, {np.round(100 * (len(FP_all.loc[FP_all.lab_rhythm.eq(i)]) / len(FP_all)),2)} %"
        for i in labels_FP))
    # plot_by_pathology(FP_all, savefig=True, savedir=cts.REPO_DIR / 'figs' / 'error_analysis', dpi=400, format='pdf')
    ###################################

    ###################################
    # other arrhythmias and previous history analysis for FP
    print('Analysing FP patient medical history..')
    ids_FP = FP_all['id'].unique()

    # RBAF
    RBAF_mdclone_info = pd.read_excel(cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info_mdclone.xlsx")
    RBAF_holter_info = pd.read_excel(cts.DATA_DIR / 'rbafdb' / "documentation" / "RBAF_Holter_Info.xlsx")
    # ids_FP_NSR = FP_all.loc[FP_all.lab_rhythm.eq(0), 'id'].unique()
    ids_FP_RBAF = [id_.rsplit('_', 1)[0] for id_ in ids_FP if id_.rsplit('_', 1)[1] == 'RBAFDB']
    RBAF_FP = RBAF_holter_info.loc[RBAF_holter_info.holter_id.isin(ids_FP_RBAF)]
    for i, j in RBAF_FP.iterrows():
        if pd.isnull(RBAF_mdclone_info.at[i, 'history of ablation ever-field']):
            continue
        else:
            RBAF_FP.at[i, 'Previous ablation'] = 1
    RBAF_FP = record_diagnosis(RBAF_FP)

    # JPAF
    JPAF_info = pd.read_excel(cts.DATA_DIR / 'jpafdb' / "List_AF_latest.xlsx")
    JPAF_info["Study ID"] = JPAF_info["Study ID"].astype(str).str.zfill(3)
    ids_FP_JPAF = [id_.rsplit('_', 1)[0] for id_ in ids_FP if id_.rsplit('_', 1)[1] == 'JPAFDB']
    JPAF_FP = JPAF_info.loc[JPAF_info['Study ID'].isin(ids_FP_JPAF)]
    JPAF_FP.rename(columns={'Study ID': 'holter_id', 'ID': 'db_id',
                            'Age': 'age_at_recording', 'Sex': 'sex', 'Date': 'recording_date',
                            'Dx': 'AF', 'Cooments': 'comments'}, inplace=True)
    JPAF_FP['AF'] = JPAF_FP.AF.replace(to_replace=r'^P', value=1, regex=True)
    JPAF_FP['AFL?'] = JPAF_FP['AFL?'].replace({'yes': 1, 'no':0})
    JPAF_FP['Previous ablation'] = JPAF_FP['AFL?'].replace({'yes': 1, 'no':0})

    # UVAF
    UVAF_info = pd.read_excel('/MLAIM/databases/uvfdb/uvfdb_rr/UVA Holter Info.xlsx')
    ids_FP_UVAF = [id_.rsplit('_', 1)[0] for id_ in ids_FP if id_.rsplit('_', 1)[1] == 'UVAFDB']
    ids_FP_UVAF = ["UVA" + id for id in ids_FP_UVAF]
    UVAF_FP = UVAF_info.loc[UVAF_info['Holter ID'].isin(ids_FP_UVAF)]
    UVAF_FP.rename(columns={'Holter ID': 'holter_id', 'Patient ID': 'db_id',
                            'Age at First': 'age_at_recording', 'Gender': 'sex'}, inplace=True)

    # Lets see what useful clinical information can we extract from the FP patients
    df_FP_history = pd.concat([RBAF_FP, JPAF_FP, UVAF_FP], axis=0, ignore_index=True)

    print(f'number of FP patients: {len(ids_FP)}\n'
          f'number of which with previous AF: {len(df_FP_history[df_FP_history.AF.eq(1)])}, {round(100*(len(df_FP_history[df_FP_history.AF.eq(1)]) / len(ids_FP)), 2)}%\n'
          f'number of which with previous AFL: {len(df_FP_history[df_FP_history["AFL?"].eq(1)])}, {round(100*(len(df_FP_history[df_FP_history["AFL?"].eq(1)]) / len(ids_FP)), 2)}%\n'
          f'number of which with previous ablation: {len(df_FP_history[df_FP_history["Previous ablation"].eq(1)])}, {round(100*(len(df_FP_history[df_FP_history["Previous ablation"].eq(1)]) / len(ids_FP)), 2)}%')

    ###################################

    ###################################
    # AFL analysis
    print('Analysing TP AFL windows..')
    TP_all = df_model_all_rlab.loc[df_model_all_rlab.pred.eq(True) & df_model_all_rlab.lab.eq(True)]
    AFIB_only = []
    AFL_only = []
    AFL_AFIB = []
    for pat in TP_all.id.unique():
        wins = df_model_all_rlab.loc[df_model_all_rlab.id.eq(pat)]
        AFL_per = 100 * (len(wins.loc[wins.lab_rhythm.eq(cts.rhythms_dict["AFL"])]) / len(wins))
        AFIB_per = 100 * (len(wins.loc[wins.lab_rhythm.eq(cts.rhythms_dict["AFIB"])]) / len(wins))
        print(pat)
        print(
            f'AFL percentage: {AFL_per}, AFIB percentage: {AFIB_per}')
        if AFL_per > 0 and AFIB_per > 0:
            AFL_AFIB.append(pat)
        elif AFL_per > 0 and AFIB_per ==0:
            AFL_only.append(pat)
        elif AFL_per == 0 and AFIB_per > 0:
            AFIB_only.append(pat)
    print(f'Total AFIB-AFL patients: {len(AFL_AFIB)}, Total AFIB only patients: {len(AFIB_only)}, Total AFL only patients: {len(AFL_only)}')

    FN_AFL = np.count_nonzero(FN_all.lab_rhythm.eq(cts.rhythms_dict['AFL']))
    AFL_wins = np.count_nonzero(df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL']))
    AF_wins = np.count_nonzero(df_model_all_rlab.lab.eq(True))
    print(f'Number of FN AFL windows: {FN_AFL}, percentage out of all AFL windows: {np.round(100*(FN_AFL/AFL_wins), 2)}')


    # analysing E_afb, for AFl patient in specific

    X, y, t_s = test_dict['test_all']
    pred_afb, true_afb, pred_af_lab, true_af_lab, Afl_dict = {}, {}, {}, {}, {}
    for i, pat in enumerate(df_model_all.id.unique()):
        X_pat = X[X[:, -1] == pat]
        y_pat = y[X[:, -1] == pat]
        y_pred_pat = df_model_all.loc[df_model_all.id.eq(pat), 'pred']
        rr = X_pat[:, :-3]
        true_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pat) / np.sum(rr))
        pred_af_burden = 100 * (np.sum(np.sum(rr, axis=1) * y_pred_pat) / np.sum(rr))
        pred_afb[pat] = pred_af_burden
        true_afb[pat] = true_af_burden
        pred_af_lab[pat] = _af_pat_lab(pred_af_burden)
        true_af_lab[pat] = _af_pat_lab(true_af_burden)
        Afl_dict[pat] = len(df_model_all.loc[df_model_all.id.eq(pat) & df_model_all.lab_rhythm.eq(3)]) / len(df_model_all.loc[df_model_all.id.eq(pat)])
    pat_df = pd.DataFrame.from_dict(pred_af_lab, orient='index').reset_index().rename(columns={'index': 'Holter_id', 0: 'pred_af_lab'})
    pat_df['true_af_lab'] = pat_df['Holter_id'].map(true_af_lab)
    pat_df['AFIB_true'] = 1*(pat_df['true_af_lab'] >0)
    pat_df['pred_afb'] = pat_df['Holter_id'].map(pred_afb)
    pat_df['true_afb'] = pat_df['Holter_id'].map(true_afb)
    pat_df['E_AF'] = pat_df['true_afb'] - pat_df['pred_afb']
    pat_df['AFL'] = pat_df['Holter_id'].map(Afl_dict)

    print(f"Patients with AFL%>=10% n={len(pat_df.loc[pat_df.AFL.ge(0.1), 'E_AF'])}, "
          f"{np.median(np.abs(pat_df.loc[pat_df.AFL.ge(0.1), 'E_AF'])):.1f} "
          f"({np.quantile(np.abs(pat_df.loc[pat_df.AFL.ge(0.1), 'E_AF']), 0.25):.1f}-"
          f"{np.quantile(np.abs(pat_df.loc[pat_df.AFL.ge(0.1), 'E_AF']), 0.75):.1f})")

    print(f"Patients with AFL%<10% n={len(pat_df.loc[pat_df.AFL.lt(0.1) & pat_df.AFL.gt(0.), 'E_AF'])}, "
          f"{np.median(np.abs(pat_df.loc[pat_df.AFL.lt(0.1), 'E_AF'])):.1f} "
          f"({np.quantile(np.abs(pat_df.loc[pat_df.AFL.lt(0.1), 'E_AF']), 0.25):.1f}-"
          f"{np.quantile(np.abs(pat_df.loc[pat_df.AFL.lt(0.1), 'E_AF']), 0.75):.1f})")

    ###################################
    # error analysis for sex and age groups

    test_dict_XGB['Female_group'], test_dict_XGB['Male_group'] = data_loading.group_sex(test_dict_XGB['test_all'])
    test_dict_XGB['low_age_group'], test_dict_XGB['mid_age_group'], test_dict_XGB['high_age_group'] = data_loading.group_age(test_dict_XGB['test_all'])
    cosEn_F = test_dict_XGB['Female_group'][0][:, 0]
    cosEn_M = test_dict_XGB['Male_group'][0][:, 0]
    # plot_cosEn(cosEn_F, cosEn_M, test_dict_XGB, savefig=True, savedir=cts.REPO_DIR/ 'figs' / 'error_analysis', dpi=400)
    ids_F = np.unique(test_dict_XGB['Female_group'][0][:, -1])
    ids_M = np.unique(test_dict_XGB['Male_group'][0][:, -1])
    AFL_wins_F = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_F) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL'])])
    AFL_wins_M = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_M) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL'])])
    AF_wins_F = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_F) & df_model_all_rlab.lab.eq(True)])
    AF_wins_M = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_M) & df_model_all_rlab.lab.eq(True)])
    print(f'Number of AFL windows For females: {AFL_wins_F}, percentage out of all AFL windows: {np.round(100*(AFL_wins_F/AFL_wins), 2)},')
    print(f'Prevalence of AFL in females: {np.round(100*(AFL_wins_F/AF_wins_F), 2)}')
    print(f'Number of AFL windows For male: {AFL_wins_M}, percentage out of all AFL windows: {np.round(100*(AFL_wins_M/AFL_wins), 2)}')
    print(f'Prevalence of AFL in male: {np.round(100*(AFL_wins_M/AF_wins_M), 2)}')

    print(f'Number of FN AFL windows in females: {len(FN_all.loc[FN_all.id.isin(ids_F) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFL"])])},'
          f' percentage out of all FN female windows: {np.round(100*(len(FN_all.loc[FN_all.id.isin(ids_F) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFL"])])/len(FN_all)), 2)}')
    print(f'Number of FN AFIB windows in females: {len(FN_all.loc[FN_all.id.isin(ids_F) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFIB"])])},'
          f' percentage out of all FN female windows: {np.round(100*(len(FN_all.loc[FN_all.id.isin(ids_F) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFIB"])])/len(FN_all)), 2)}')

    print(f'Number of FN AFL windows in male: {len(FN_all.loc[FN_all.id.isin(ids_M) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFL"])])},'
          f' percentage out of all FN male windows: {np.round(100*(len(FN_all.loc[FN_all.id.isin(ids_M) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFL"])])/len(FN_all)), 2)}')

    print(f'Number of FN AFIB windows in male: {len(FN_all.loc[FN_all.id.isin(ids_M) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFIB"])])},'
          f' percentage out of all FN male windows: {np.round(100*(len(FN_all.loc[FN_all.id.isin(ids_M) & FN_all.lab_rhythm.eq(cts.rhythms_dict["AFIB"])])/len(FN_all)), 2)}')

    ids_low_age = np.unique(test_dict_XGB['low_age_group'][0][:, -1])
    ids_mid_age = np.unique(test_dict_XGB['mid_age_group'][0][:, -1])
    ids_high_age = np.unique(test_dict_XGB['high_age_group'][0][:, -1])

    AFL_wins_low = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_low_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL'])])
    AF_wins_low = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_low_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFIB'])])
    LAB_wins_low = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_low_age) & df_model_all_rlab.lab.eq(True)])


    AFL_wins_mid = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_mid_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL'])])
    AF_wins_mid = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_mid_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFIB'])])
    LAB_wins_mid = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_mid_age) & df_model_all_rlab.lab.eq(True)])

    AFL_wins_high = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_high_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFL'])])
    AF_wins_high = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_high_age) & df_model_all_rlab.lab_rhythm.eq(cts.rhythms_dict['AFIB'])])
    LAB_wins_high = len(df_model_all_rlab.loc[df_model_all_rlab.id.isin(ids_high_age) & df_model_all_rlab.lab.eq(True)])

    print(f'Number of AFL windows For age group le 60: {AFL_wins_low}, percentage out of all AFL windows: {np.round(100*(AFL_wins_low/AFL_wins), 2)}')
    print(f'Number of AFL windows For age group 60 to 75: {AFL_wins_mid}, percentage out of all AFL windows: {np.round(100*(AFL_wins_mid/AFL_wins), 2)}')
    print(f'Number of AFL windows For age group 75 and above: {AFL_wins_high}, percentage out of all AFL windows: {np.round(100*(AFL_wins_high/AFL_wins), 2)}')

    print(f'#AFL windows For age group le 60: {AFL_wins_low} vs #AF windows {AF_wins_low}, %AFL out of AF windows: {np.round(100*(AFL_wins_low/LAB_wins_low), 2)},%AF out of AF windows: {np.round(100*(AF_wins_low/LAB_wins_low), 2)}')
    print(f'#AFL windows For age group 60 to 75: {AFL_wins_mid} vs #AF windows {AF_wins_mid}, %AFL out of AF windows: {np.round(100*(AFL_wins_mid/LAB_wins_mid), 2)},%AF out of AF windows: {np.round(100*(AF_wins_mid/LAB_wins_mid), 2)}')
    print(f'#AFL windows For age group 75 and above: {AFL_wins_high} vs #AF windows {AF_wins_high}, %AFL out of AF windows: {np.round(100*(AFL_wins_high/LAB_wins_high), 2)},%AF out of AF windows: {np.round(100*(AF_wins_high/LAB_wins_high), 2)}')

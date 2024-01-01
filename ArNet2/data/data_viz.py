# General imports
import os
import pathlib
import sys

import numpy as np
import pickle

import seaborn as sns
from sklearn.model_selection import train_test_split
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

# Relative imports
import utils.consts as cts
import data.data_processing as dp
from parsing.jpaf_parser import JPAFDB_Parser
from parsing.uvafdb_parser import UVAFDB_Parser
from parsing.rbaf_parser import RBAFDB_Parser

def plot_age_distribution(ax, test_df, db_dict, ticks_fontsize, label_fontsize, savefig=False, savedir=None, format='png'):
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plot age distribution
    sns.kdeplot(data=test_df, x="age", hue="db", ax=ax)
    stat_df = test_df.groupby('db').age.agg([np.median, np.std]).reset_index()
    stat_df.replace({'db': db_dict}, inplace=True)
    result = stat_df.apply(lambda row: row["db"] + ': ' + str(row["median"]) + " ± " + str(round(row["std"], 2)),
                           axis=1)
    ax.legend(result, loc=0, fontsize=ticks_fontsize)
    ax.set_xlim([test_df.age.min() - 10, test_df.age.max() + 10])
    # ax.set_title(f'Test sets age distribution')
    ax.set_xlabel('Age', fontsize=label_fontsize)
    ax.set_ylabel('Density', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize, length=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if savefig:
        plt.tight_layout()
        lab = 'test_sets_age_distribution.' + format
        plt.savefig(savedir + lab, dpi=400, transparent=True, format=format)
        return

def plot_sex_distribution(ax, test_df, db_dict, ticks_fontsize, label_fontsize, savefig=False, savedir=None, format='png'):
    # plot sex distribution
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plot sex distribution
    test_df.replace({'db': db_dict}, inplace=True)
    sns.histplot(data=test_df, x='db', hue='sex', ax=ax, shrink=0.8, multiple="dodge", )
    ax.legend(['M', 'F'], loc=4, fontsize=ticks_fontsize)
    ax.set_xlabel('Database', fontsize=label_fontsize)
    ax.set_ylabel('Count', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize, length=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if savefig:
        plt.tight_layout()
        lab = 'test_sets_sex_distribution.' + format
        plt.savefig(savedir + lab, dpi=400, transparent=True, format=format)
    return

def plot_af_distribution(ax, test_df, db_dict, ticks_fontsize, label_fontsize, savefig=False, savedir=None, format='png'):
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plot sex distribution
    af_labels = ['AF', 'Non-AF']
    test_df.replace({'db': db_dict}, inplace=True)
    test_df.loc[test_df.lab.ge(1), 'lab'] = 1
    sns.histplot(data=test_df, x='db', hue='lab', ax=ax, shrink=0.8, multiple="dodge", )
    ax.legend(af_labels, loc=4, fontsize=ticks_fontsize)
    # ax.set_title(f'Test sets AF distribution')
    ax.set_xlabel('Database', fontsize=label_fontsize)
    ax.set_ylabel('Count', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize, length=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if savefig:
        plt.tight_layout()
        lab = 'test_sets_af_distribution.' + format
        plt.savefig(savedir + lab, dpi=400, transparent=True, format=format)

def create_figure(test_df, db_dict, format='png', **kwargs):
    # plt.style.use('seaborn-white')
    ticks_fontsize = kwargs.get('ticks_fontsize', 24)
    label_fontsize = kwargs.get('label_fontsize', 24)
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    plot_af_distribution(ax1, test_df, db_dict, ticks_fontsize, label_fontsize)

    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    plot_sex_distribution(ax2, test_df, db_dict, ticks_fontsize, label_fontsize)

    ax3 = fig.add_subplot(gs[1, :])  # row 1, span all columns
    plot_age_distribution(ax3, test_df, db_dict, ticks_fontsize, label_fontsize)
    if kwargs.get('savefig', False):
        fig.tight_layout()
        lab = 'Distribution for the re-annotated test sets.' + format
        plt.savefig(savedir + lab, dpi=400, transparent=True, format=format)
        return

def error_analysis_examples():
    #FP
    db.plot_ecg('0440', start=16387, end=16397) # proba: 0.64670, rhythm: Non
    db.plot_ecg('111', start=8835.23750, end=8845.82750) # proba:0.90, rhythm: AT
    db.plot_ecg('1623', start=48698, end=48708) # proba: 0.55049

    #FN
    db.plot_ecg('0004', start=60395, end=60403) # proba: 0.01601 AFIB
    db.plot_ecg('V720Gd6e', start=72906, end=72920)  # proba: 0.17167 AFL
    return

if __name__ == '__main__':
    db_dict = {'UVAFDB': 'UVAF', 'JPAFDB': 'SHDB', 'RBAFDB': 'RBDB'}
    test_df = pd.read_excel('/home/shanybiton/repos/Generalization/test_patient_file.xlsx')
    test_df = test_df.loc[test_df.db.isin(db_dict.keys())]
    savedir = '/home/shanybiton/repos/Generalization/figs/data/'
    # plot_age_distribution(test_df, db_dict, savedir=savedir, format='eps')
    # plot_sex_distribution(test_df, db_dict, savedir=savedir, format='eps')
    # plot_af_distribution(test_df, db_dict, savedir=savedir, format='eps')
    create_figure(test_df, db_dict, savefig=True, savedir=savedir, ticks_fontsize=14, label_fontsize=14, format='png')
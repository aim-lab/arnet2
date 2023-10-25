# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:01:02 2019

@author: rta8y
"""
import os
import pandas as pd
import utils.consts as cts
from utils.base_packages import *


def read_rf_file(file):
    with open(file, 'rb') as f:
        byte = f.read(4)
        i = int.from_bytes(byte, byteorder='little')
        n = 6000
        count = 1
        xx = []
        xx.append(i)
        while count < n and byte != b"":
            #        while byte != b"":
            byte = f.read(4)
            i = int.from_bytes(byte, byteorder='little')
            count += 1
            xx.append(i)

        def get_rf_data(xx):

            val_list = []
            for i in xx:
                x = (i & (2 ** 9) - 1)
                j = (i & (2 ** 9)) > 0
                if j:
                    x = x - 2 ** 9
                val_list.append(x)

            return val_list

        x = get_rf_data(xx)
        xy = [b >> 10 for b in xx]
        y = get_rf_data(xy)
        xz = [y >> 10 for y in xy]
        z = get_rf_data(xz)

        rf_df = pd.DataFrame(data={'ecg': x, 'y': y, 'z': z})

        f.close()

    return rf_df


def read_bea_file(bea):
    with open(bea, 'r') as f:
        t = f.readlines()
        x = pd.Series(t).str.split('\n', expand=True)
        x2 = x[0].str.split(' ', expand=True)
        x3 = x2.drop(columns=[1, 3, 4, 5, 6, 7, 8])
        x3[9] = x3[9].str.replace('(', '')
        #        x3['Annotation'] = x4
        #        x3 = x3.drop(columns=[9])
        x3.columns = ['Time', 'Type', 'Annotation']
        f.close()

        return x3


def read_outcome_file(outcomes, patient):
    outcome = pd.read_excel(outcomes)
    patient_outcome = outcome.loc[outcome['Holter ID'] == patient]

    return patient_outcome


os.chdir(cts.DATA_DIR / "uvfdb")

patient = 'UVA0299'
file = patient + '.rf'

rf_df = read_rf_file(file)
rf_df.index = (rf_df.index + 1) * 5

os.chdir('X:\\Bobby\\rfTest')
rf_df.to_csv('Holter32Test.csv')

ecg = rf_df['ecg']

# data = [rf_df]
# rf_df.plot(grid=True)
# fig = rf_df.iplot(kind='scatter', asFigure=True, annotations=bea_df_dict, world_readable=True)
# pyo.plot(fig)



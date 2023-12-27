from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TODO: MESSY!! consider removing completely
lab_intervals_dict={}
lab_count_dict = {}
lab_time_intervals_dict = {}
rr_lab = db.rlab_dict[pat]
rr_lab[np.isnan(rr_lab)] = -159
count_dups = np.array([sum(1 for _ in group) for _, group in groupby(rr_lab)])

unique_lab = np.array([v for i, v in enumerate(rr_lab) if v != rr_lab[i-1]], dtype=int)
lab_intervals_dict[pat] = {}
lab_time_intervals_dict[pat] = {}
labs = np.setdiff1d(np.unique(unique_lab), -159)
len_rec = db.recording_time[pat]
for lab in labs:
    idx = np.where(unique_lab == lab)
    lab_intervals_dict[pat][lab] = count_dups[idx]
    relative_time_in_aryth = (count_dups[idx] / len_rec)
    if (relative_time_in_aryth > thresh).any():
        lab_time_intervals_dict[pat][lab] = relative_time_in_aryth[relative_time_in_aryth > thresh]
lab_count_dict[pat] = {}
for k, v in lab_time_intervals_dict[pat].items():
    lab_count_dict[pat][k] = len(v)
count_df = pd.DataFrame.from_dict(lab_count_dict, orient='index')

count_df_rhythms = count_df[[2, 3, 5, 6, 7, 8, 9]]
count_df_rhythms.rename(columns={2:'VT', 3:'SVT', 5:'Salvo', 6:'Brady',
                                 7:'Trigeminy i', 8:'Trigeminy ii', 9: 'Bigeminy', 149: 'AF'}, errors="raise", inplace=True)
count_df_rhythms['AF'] = np.nan
count_dict = count_df_rhythms.count().to_dict()

fig = plt.figure(figsize=(15, 6), dpi=400)
ax0 = plt.subplot()
ax0.bar(count_dict.keys(), count_dict.values(), alpha=0.85)
fig.savefig('/MLdata/AIMLab/Shany/ErrorAnalysis/rbafdb/count_hist.png')

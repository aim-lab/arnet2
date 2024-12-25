import wfdb
from parsing.jpaf_parser import SHDB_Parser
import numpy as np
import os
import consts as cts
import pandas as pd
import datetime

custom_labels = pd.DataFrame({'symbol': ['N', 'AF', 'AFL', 'AT'],
                              'description': ['Other', 'Atrial Fibrillation', 'Atrial Flutter', 'Atrial Tachycardia']})


def return_time_from_millis(millis):
    hours = datetime.timedelta(seconds=millis).seconds // cts.N_S_IN_HOUR
    minutes = (datetime.timedelta(seconds=millis).seconds // cts.N_SEC_IN_MIN) % cts.N_SEC_IN_MIN
    seconds = datetime.timedelta(seconds=millis).seconds - hours * cts.N_S_IN_HOUR - minutes * cts.N_SEC_IN_MIN
    return hours, minutes, seconds


def prepare_AdditionalData_csv(file_path, dest_path):
    sup_df = pd.read_csv(file_path)

    # Create UID column
    sup_df['UID'] = sup_df['Study ID'].apply(lambda pat: str(pat).zfill(3))
    sup_df['published_Study_ID'] = 'SH' + sup_df['UID']

    # Adjust specific row based on condition
    sup_df.loc[sup_df['Date Holter'].eq('19-Feb-21'), 'UID'] = '064'

    # Clean and handle missing values
    sup_df.replace(['none', 'unknown'], np.nan, inplace=True)
    sup_df.loc[36, 'AF duration (months)'] = np.nan

    # List of date columns to convert
    date_columns = [
        'Date Holter',
        'Date of 1st AF ablation',
        'Date of AF redo ablation',
        'Echo date',
        'PPM date',
    ]

    sup_df = convert_to_datetime(sup_df, columns=date_columns)
    sup_df = convert_to_datetime(sup_df, columns=['Date of first diagnosis of AF/AFL'], format_str='%b-%y')

    # rearrange columns
    sup_df = sup_df[sup_df.columns[-2:].to_list() + sup_df.columns[1:-2].to_list()].rename(
        columns={'published_Study_ID': 'Study ID'})
    sup_df.to_csv(dest_path / 'AdditionalData.csv', index=False)
    return


def prepare_deatil_pat(df):
    dates_col = ['1st AF ablation', '2nd AF ablation', '3rd AF ablation']
    drop_cols = ['ID', 'Date', 'Initial(Fa/Fi)', 'Comments', 'When 1st AF noted', '1st AF ablation', '2nd AF ablation',
                 '3rd AF ablation', 'First or Follow']
    df = df.replace('none', np.NAN)
    df = df.replace('unknown', np.NAN)
    for col in dates_col:
        df[col] = pd.to_datetime(df[col], format='%Y/%m/%d %H:%M:%S.%f')
        df[col] = df[col].dt.year
    detail_pat = df.drop(drop_cols, 1)
    detail_pat.columns = detail_pat.columns.str.replace('?', '')
    detail_pat['Study ID'] = '="' + detail_pat['Study ID'] + '"'
    return detail_pat


def replace_duplicates_with_placeholder(rhythm, rhythm_dict, placeholder=''):
    rhythm_str = np.array([rhythm_dict[i] for i in rhythm])
    seen = ['']
    result = []

    for item in rhythm_str:
        if item in seen:
            result.append(placeholder)
        else:
            seen = [item]
            result.append(item)

    return np.array(result)


# Function to convert dates
def convert_to_datetime(df, columns, format_str='%d-%b-%y'):
    for col in columns:
        df[col] = pd.to_datetime(df[col], format=format_str, errors='coerce')
        df[col] = df[col].dt.strftime('%b-%Y')
    return df


if __name__ == '__main__':
    db = SHDB_Parser(load_on_start=True)  # pat '090' will raise error
    dest_path = cts.BASE_DIR / 'AIMLab' / 'Shany' / 'databases' / 'shdb_physionet' / 'v1.0.1'

    prepare_AdditionalData_csv(file_path=cts.BASE_DIR / 'AIMLab' / 'Shany' / 'databases' / 'shdb_sup/Supplemental_Table_20241209a.csv',
                               dest_path=dest_path)

    additionalData = pd.read_csv(dest_path / 'AdditionalData.csv')
    pat_list = [str(pat).zfill(3) for pat in additionalData['UID']]

    ann_ids = np.intersect1d(np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1]), pat_list)
    db.load_circardian_from_disk(pat_list)

    rhythm_dict = {y: '(' + x for x, y in db.rhythms_dict.items()}
    rhythm_dict[0] = '(N'
    counts_df = pd.DataFrame(columns=rhythm_dict.keys(), index=pat_list)
    ann_type = db.sqi_ref_ann
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print(f'Writing database to {dest_path}..')
    for pat in pat_list:
        print(f'Saving pat: {pat}')
        ecgs = np.concatenate(tuple(
            [db.parse_raw_ecg(pat, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in
             range(1, db.n_leads + 1)]), axis=1)
        ann = db.parse_raw_ecg(pat, lead=db.ref_lead, type=ann_type)[1]
        rrt_dict = np.concatenate(([ann[0] / db.actual_fs], np.cumsum(ann) + ann[0] / db.actual_fs))
        if pat in ann_ids:
            rhythm_ = db.parse_medaim_annotations(pat, ann/db.actual_fs)
            rhythm = replace_duplicates_with_placeholder(rhythm_, rhythm_dict, placeholder='')
            print(f'rhythms available: {np.unique(rhythm)}')
            res = list(zip(*np.unique(rhythm_, return_counts=True)))
            for re in res:
                counts_df.loc[pat, re[0]] = re[-1]
            wfdb.wrann(pat, 'atr', sample=ann, aux_note=list(rhythm), fs=db.actual_fs,
                       write_dir=str(dest_path), label_store=np.array([22]*len(rhythm_)))  # , label_store=np.array([1] * len(ann)))  # rhythms

        wfdb.wrsamp(pat, fs=db.actual_fs, units=['mV', 'mV'], sig_name=['ECG1', 'ECG2'],
                    p_signal=ecgs, fmt=['16', '16'],
                    base_time=datetime.time(*return_time_from_millis(db.circadian_dict[pat]['start_recording'])),
                    base_date=datetime.date(db.circadian_dict[pat]['recording_date'].year, 1, 1),
                    write_dir=str(dest_path))  # ecgs
        wfdb.wrann(pat, 'qrs', ann, aux_note=np.array([' '] * len(ann)), fs=db.actual_fs,
                   write_dir=str(dest_path),
                   symbol=np.array(['N'] * len(ann)))  # , label_store=np.array([1] * len(ann)))  # annotations

    # Count rhythm prevalence
    rhythm_df = db.parse_medaim_reference_rhythm(ann_ids[0])
    for pat in ann_ids[1:]:
        rhythm_df = rhythm_df.append(db.parse_medaim_reference_rhythm(pat))
    print(rhythm_df.groupby('Class').count())

    # Revision 1.0.0
    # detail_pat = db.excel_sheet.loc[db.excel_sheet['Study ID'].isin(pat_list)]
    # detail_pat = prepare_deatil_pat(detail_pat)
    # detail_pat.to_csv(dest_path / 'AdditionalData.csv', index=False)

    # create RECORDS file
    with open(dest_path / 'RECORDS.txt', 'w') as file_handler:
        for item in pat_list:
            file_handler.write("{}\n".format(item))

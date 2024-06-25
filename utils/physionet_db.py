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


def replace_int_to_str(rhythm, rhythms_dict):
    loc = np.nonzero(np.diff(rhythm))[0] if np.nonzero(np.diff(rhythm))[0].size else np.array([0])
    rhythm_idx = rhythm[loc]
    rhythm_str = np.array([rhythms_dict[i] for i in rhythm_idx])
    return rhythm_str, loc


if __name__ == '__main__':
    db = SHDB_Parser(load_on_start=True)
    ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    pat_list = ann_ids[np.isin(ann_ids, db.parsed_patients())]
    db.load_circardian_from_disk(pat_list)

    rhythm_dict = {y: '(' + x for x, y in db.rhythms_dict.items()}
    rhythm_dict[0] = '(N'
    ann_type = db.sqi_ref_ann
    dest_path = cts.BASE_DIR / 'AIMLab' / 'Shany' / 'databases' / 'shdb_physionet'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print(f'Writing database to {dest_path}..')
    for pat in pat_list:
        print(f'Saving pat: {pat}')
        ecgs = np.concatenate(tuple(
            [db.parse_raw_ecg(pat, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in
             range(1, db.n_leads + 1)]), axis=1)
        ann = db.parse_raw_ecg(pat, lead=db.ref_lead, type=ann_type)[1]
        rhythm_ = db.parse_medaim_annotations(pat, np.cumsum(ann) / cts.N_MS_IN_S)
        rhythm, rhythm_sample = replace_int_to_str(rhythm_, rhythm_dict)
        print(f'rhythms available: {np.unique(rhythm)}')

        wfdb.wrsamp(pat, fs=db.actual_fs, units=['mV', 'mV'], sig_name=['ECG1', 'ECG2'],
                    p_signal=ecgs, fmt=['16', '16'],
                    base_time=datetime.time(*return_time_from_millis(db.circadian_dict[pat]['start_recording'])),
                    base_date=datetime.date(db.circadian_dict[pat]['recording_date'].year, 1, 1),
                    write_dir=str(dest_path))  # ecgs
        wfdb.wrann(pat, 'atr', rhythm_sample, aux_note=rhythm, fs=db.actual_fs,
                   write_dir=str(dest_path), symbol=np.array(
                ['+'] * len(rhythm_sample)))  # , label_store=np.array([1] * len(ann)))  # annotations
        wfdb.wrann(pat, 'qrs', ann, aux_note=np.array([' '] * len(ann)), fs=db.actual_fs,
                   write_dir=str(dest_path),
                   symbol=np.array(['N'] * len(ann)))  # , label_store=np.array([1] * len(ann)))  # annotations
    detail_pat = db.excel_sheet.loc[db.excel_sheet['Study ID'].isin(pat_list)]
    detail_pat = prepare_deatil_pat(detail_pat)
    detail_pat.to_csv(dest_path / 'AdditionalData.csv', index=False)

    # create RECORDS file
    with open(dest_path / 'RECORDS.txt', 'w') as file_handler:
        for item in pat_list:
            file_handler.write("{}\n".format(item))

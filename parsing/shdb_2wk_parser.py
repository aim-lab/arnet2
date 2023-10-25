import sys

import numpy as np

sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/CircadianAF')

from base_parser import *

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
import csv_reader as cr
import time
import re
import datetime as dt
import pathlib
import pickle
import glob


class SHDB_2wk_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(SHDB_2wk_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array([])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 125
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.name = "SHDB_2wk"
        self.ecg_format = ".csv"

        self.peak_ann = np.array(['N', 'Q', 'V', 'S'])
        self.peak_ann_dict = {self.peak_ann[i]: i for i in range(len(self.peak_ann))}
        self.sqi_test_ann = 'epltd0'          # The annotation type used to compute and load the SQI variables.
        self.sqi_ref_ann = 'xqrs'
        self.rhythms = np.array(['NSR', 'AFIB', 'AFL'])
        self.rhythms_dict = {'NSR': 0, 'AFIB': 1, 'AFL': 1, 'NOD': 2}
        self.circadian_dict = {}

        """Variables relative to the different paths"""
        cts.DATA_DIR = pathlib.PurePath('/home/shanybiton/repos/CircadianAF/')
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "examples"
        self.generated_anns_path = cts.BASE_DIR / "Shany" / "Annotations" / self.name
        # self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.annotation_types = cts.ANNOTATION_TYPES
        self.main_path = cts.PREPROCESSED_DATA_DIR / self.name

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        # test_pat = self.parsed_patients()
        # self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)
                self.load_circardian_from_disk()
        self.beat_flags = {}
        # self.load_beat_flags()

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the SHDB_2wk) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.ecg_file_name = 'RR'
        self.file_format = ".csv"
        self.csv_dir = "RR_ready"
        # self.beats_shape = {1: 'N', 3:  'N', 4: 'AB', 5: 'I', 6: 'P'}  # The different rhythms present across the dataset. N: NORMAL', AB: 'ABERRANT', I: 'INHIBIT', P: 'PACED'}
        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "List_SHDB_AF2wk.xlsx"
        self.get_META()

        self.searchPar = ['PAF']
        self.searchPer = ['PerAF']
        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        # ids = np.array([dir.split('_')[0].zfill(3) for dir in os.listdir(str(self.raw_ecg_path))])
        ids = [d.split('/')[-1] for d in glob.glob(str(self.raw_ecg_path / '*/*')) if
               os.path.isdir(d)]  # needs to return all the files including days
        return ids

    def _win_lab(self, id, win):
        """ Computes the label of a window. The label is computed based on the most represented label over the window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of beats) along which the raw recording is divided.
        """
        if id not in self.win_lab_dict.keys():
            self.win_lab_dict[id] = {}
        raw_rlab = self.rlab_dict[id]
        rlab = raw_rlab[:(len(raw_rlab) // win) * win].reshape(-1, win)
        counts = np.array([np.sum((rlab == i), axis=1) for i in range(len(cts.rhythms))]).astype(float)
        count_nan = np.sum(np.isnan(rlab), axis=1).astype(float)
        max_lab_count = np.max(counts, axis=0).astype(float)
        self.win_lab_dict[id][win] = np.argmax(counts, axis=0).astype(float)
        self.win_lab_dict[id][win][count_nan > max_lab_count] = np.nan

    def parse_annotation(self, id, type="epltd0", lead=1):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead=1):
        record = self.read_ecg(id).iloc[:, lead].astype(float).values
        re_record = bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs / 2 - 0.5,
                                    signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)
        re_record = dp.resample_by_interpolation(re_record, self.orig_fs, self.actual_fs)
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
        return re_record

    def parse_raw_ecg(self, id, start=0, end=-1, type='epltd0', lead=1):
        record = self.read_ecg(id).iloc[:, lead].astype(float).values
        record = bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs / 2 - 0.5,
                                 signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)

        record = dp.resample_by_interpolation(record, self.orig_fs, self.actual_fs)
        ann = self.parse_annotation(id, type=type, lead=lead)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = int(start * self.actual_fs)
        end_sample = int(end * self.actual_fs)
        record = record[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return record, ann

    def parse_reference_annotation(self, id, combine=True, reannotated=True):
        record = self.read_ecg(id)
        ann = self.read_ann(id, start_time=record.time[0], end_time=record.time.iloc[-1])
        beat = np.array([ann.pos.values])
        tbeats = np.cumsum(ann.pos.values) / cts.N_MS_IN_S
        ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        if reannotated:
            rhythm_df = self.parse_reference_rhythm(id)
            for index, l in rhythm_df.iterrows():
                l1 = np.abs(tbeats - l.Beginning)
                l2 = np.abs(tbeats - l.End)
                begin = np.where(l1 == l1.min())
                end = np.where(l2 == l2.min())
                ltbeats[int(begin[0][0]):int(end[0][0])] = l.Class
        else:
            ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
            # rhythm = np.array([ann.ann.values])
            # rhythm = rhythm[0]
        rhythm = np.array([cts.rhythms_dict[i] for i in ltbeats])
        if combine:
            rhythm[rhythm == cts.rhythms_dict['AFL']] = cts.rhythms_dict['AFIB']
        return (tbeats * self.actual_fs).astype(int), rhythm

    def parse_demographic_features(self, id):
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = np.nan
            self.features_dict[id][win]['Sex'] = np.nan

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        # return self.excel_sheet[self.excel_sheet["Study ID"] == recording_id]["ID"].values[0]
        return np.array([dir.split('_')[0].zfill(3) for dir in os.listdir(str(self.raw_ecg_path))])

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def get_META(self):
        '''
        CSV Columns
        '''
        # load and convert annotation data
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl')
        self.excel_sheet["Study ID"] = self.excel_sheet["Study ID"].astype(str).str.zfill(3)

    def get_dir(self, id):
        return [i for i in glob.glob(str(self.raw_ecg_path / '*/*')) if
                i.split('/')[-1].startswith(id)]

    def read_ecg(self, id):
        id_dir = self.get_dir(str(id))[0]
        example_path = id_dir + '/' + (str(id.rsplit('_', 2)[0]) + self.ecg_format)
        chunks = pd.read_csv(example_path, iterator=True, chunksize=1000000, encoding='unicode_escape',
                             usecols=[0, 1, 2], names=['NA', 'ch1', 'ch2'],
                             header=None, dtype={"NA": 'string', "ch1": 'string', 'ch2': 'string'})
        df2 = pd.concat(chunks, ignore_index=True)

        temp_date = df2.iloc[0]['ch1']
        temp_time = df2.iloc[2]['NA']
        ch2 = df2.iloc[2:]['ch2']
        ch1 = df2.iloc[2:]['ch1']

        temp_date = re.split('\s+', temp_date)
        temp_time = re.split('\s+', temp_time)

        date = dt.datetime.strptime(temp_date[0], '%Y/%m/%d')
        df = pd.DataFrame()
        length = np.size(ch2)
        fs = 1 / self.orig_fs
        [hours, minutes, seconds] = [int(x) for x in temp_time[1].split(':')]
        start = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        end = start.seconds + length * fs
        timestamp = np.arange(start.seconds, end, fs)

        ecg = pd.DataFrame({'time': timestamp, 'data_ch1': ch1, 'data_ch2': ch2, 'date': date})
        ecg.reset_index(drop=True, inplace=True)
        # dt = np.asarray(ecg['data_ch2'].iloc[2:], dtype=float)
        return ecg

    def record_diagnosis(self, patient_id, win):
        afl_cases = np.array(self.excel_sheet["Study ID"][
                                 self.excel_sheet['AFL?'].str.contains('yes', na=False)].values)
        per_af = np.array(self.excel_sheet["Study ID"][
                              self.excel_sheet['Dx'].str.contains('|'.join(self.searchPer), na=False)].values)
        par_af = np.array(self.excel_sheet["Study ID"][
                              self.excel_sheet['Dx'].str.contains('|'.join(self.searchPar), na=False)].values)
        if patient_id in per_af:  # Assessing the class according to the guidelines
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_SEVERE  # persistent AF is equivalent to severe AF
        elif patient_id in par_af:  # Assessing the class according to the guidelines
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_MILD  # paroxysmal AF is equivalent to mild/moderate AF
        elif patient_id in afl_cases:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_OTHER_CVD
        else:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_NON_AF

    def parse_circadian_features(self, id):
        if id not in self.circadian_dict.keys():
            self.circadian_dict[id] = {}
        self.circadian_dict[id]['recording_date'] = self.read_ecg(id).date[0].date()
        self.circadian_dict[id]['start_recording'], self.circadian_dict[id]['end_recording'] = \
            self.read_ecg(id).time.iloc[0], self.read_ecg(id).time.iloc[-1]
        np.save(self.main_path / id / 'circadian_dict.npy', self.__dict__['circadian_dict'][id])

    def load_circardian_from_disk(self, patient_list=None):
        if patient_list is None:
            patient_list = self.parsed_patients()
        for pat in patient_list:
            if os.path.exists(self.main_path / pat / ('circadian_dict.npy')):
                self.__dict__['circadian_dict'][pat] = np.load(self.main_path / pat / ('circadian_dict.npy'),
                                                               allow_pickle=True).item()

    def read_ann(self, id, start_time=None, end_time=None):
        id_dir = self.get_dir(str(id))[0]
        example_path = self.raw_ecg_path / id_dir / self.csv_dir
        RR_df = pd.DataFrame([])
        ann_files = os.listdir(example_path)
        ann_files.sort()
        for f in ann_files:
            chunks = pd.read_csv(example_path / f, iterator=True, chunksize=1000000, encoding='unicode_escape',
                                 usecols=[0, 1, 2], names=['time', 'ann', 'pos'],
                                 header=None, dtype={"NA": 'string', "ann": 'string', 'pos': 'string'})
            df2 = pd.concat(chunks, ignore_index=True)
            if len(df2[df2['pos'].str.contains("RR", na=False)]) > 0:
                df2 = df2.iloc[df2[df2['pos'].str.contains("RR", na=False)].index[0] + 1:]
            RR_df = RR_df.append(df2)
        RR_df.reset_index(inplace=True, drop=True)
        RR_df['pos'] = RR_df['pos'].astype(int)
        # df2 = df2.sort_values(by ='loc', ascending=True, na_position='last')
        ann = RR_df['ann']
        loc = RR_df['pos']
        time_df = RR_df['time']
        real_start = time.strftime('%-H:%M', time.gmtime(start_time))
        time_ann_start = time_df[time_df == real_start].index[0]
        real_end = time.strftime('%-H:%M', time.gmtime(end_time))
        temp_time_df = time_df[time_ann_start + 1:]
        if len(temp_time_df[temp_time_df == real_end]) == 0:
            time_ann_end = time_df.index[-1]
        else:
            if time_df[time_df == real_end].index[-1] < 4000:
                time_ann_end = time_df.index[-1]
            else:
                time_ann_end = time_df[time_df == real_end].index[-1]
        # else:
        #     time_ann_start=0
        #     time_ann_end = len(time_df)
        ann_dict = pd.DataFrame(data={'time': time_df, 'pos': loc.values, 'ann': ann.values})

        return ann_dict.iloc[time_ann_start:time_ann_end + 1]

    def return_data(self, ids, feats_to_use, fillna=True, normalize=False):
        final = tuple()
        # X, y, glob_lab = db.return_features(pat_list=ids, feats_list=feats_to_use,
        #                                                       return_global_label=True)
        ids_rr = db.return_patient_ids(pat_list=ids)
        rr, rrt, _ = db.return_rr(pat_list=ids)
        prec = db.return_preceeding_windows(pat_list=ids)
        data = np.concatenate((rr, prec.reshape(-1, 1), ids_rr.reshape(-1, 1)), axis=1)

        return data, rrt


if __name__ == '__main__':
    windows = [60]
    db = SHDB_2wk_Parser(load_on_start=False)
    ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    # ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    # pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]

    # db.parse_raw_data(patient_list=ids[0])

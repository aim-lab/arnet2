import sys
import mat73

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


class SPAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True, windows_shifted=False):

        super(SPAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(['ns001177', 'ns001117', 'ns004678', 'ns006576'])

        """Helper variables"""
        self.window_size = window_size
        self.windows_shifted = windows_shifted

        """Variables relative to the ECG signals."""
        self.orig_fs = [250, 256]
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = [2, 12]
        self.name = "SPAFDB"
        self.ecg_format = ".mat"

        self.peak_ann = np.array(['N', 'Q', 'V', 'S'])
        self.peak_ann_dict = {self.peak_ann[i]: i for i in range(len(self.peak_ann))}
        self.sqi_test_ann = 'epltd0'          # The annotation type used to compute and load the SQI variables.
        self.sqi_ref_ann = 'xqrs'
        self.rhythms = np.array(['NSR', 'AFIB', 'AFL'])
        self.rhythms_dict = {'NSR': 0, 'AFIB': 1, 'AFL': 1, 'NOD': 2}
        self.circadian_dict = {}

        """Variables relative to the different paths"""
        cts.DATA_DIR = pathlib.PurePath('/MLAIM/databases/')
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "examples"
        self.generated_anns_path = cts.BASE_DIR / "Shany" / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / self.name

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        if os.path.exists(self.main_path):
            parsed_patients = self.parsed_patients()
            test_pat = parsed_patients[0]
            self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
            if load_on_start:
                self.set_window_size(self.window_size)
                self.load_circardian_from_disk()
        self.beat_flags = {}
        # self.load_beat_flags()

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the SPAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.ref_ann_path = cts.DATA_DIR / self.name.lower() / "AF_annotations"
        self.ref_peaks_path = cts.DATA_DIR / self.name.lower() / "peaks"

        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "SPAFBD_description.xlsx"
        self.get_META()
        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        # ids = np.array([dir.split('_')[0].zfill(3) for dir in os.listdir(str(self.raw_ecg_path))])
        ids = np.array([file.split('.')[0] for file in os.listdir(str(self.raw_ecg_path))])  # needs to return all the files including days
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
        file = self.raw_ecg_path / (id + self.ecg_format)
        try:
            loaded = mat73.loadmat(file)
        except Exception:
            loaded = sio.loadmat(file, struct_as_record=True)
        record = loaded['signal'].T[lead]
        orig_fs = int(loaded['fa'].flatten()[0])
        re_record = bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=orig_fs/2 - 0.5,
                                    signal_freq=orig_fs, filter_order=75, notch_freq=50, debug=False)
        re_record = dp.resample_by_interpolation(re_record, orig_fs, self.actual_fs)
        re_record = re_record / 1000
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'])
        return re_record

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type='epltd0', lead=1, correct_peaks=True):
        file = self.raw_ecg_path / (patient_id + self.ecg_format)
        try:
            loaded = mat73.loadmat(file)
        except Exception:
            loaded = sio.loadmat(file, struct_as_record=True)
        record = loaded['signal'].T[lead-1]
        orig_fs = int(loaded['fa'].flatten()[0])
        record = bandpass_filter(data=record, id=patient_id, lead='x', lowcut=0.67, highcut=orig_fs/2 - 0.5,
                                    signal_freq=orig_fs, filter_order=75, notch_freq=50, debug=False)
        record = dp.resample_by_interpolation(record, orig_fs, self.actual_fs)
        record = record / 1000
        ann = self.parse_annotation(patient_id, type=type, lead=lead)
        if correct_peaks:
            # self.create_pool()
            ann = i_o.qrs_adjust_detector(ecg=record, qrs=ann, fs=self.actual_fs, INPUTSIGN=1, n_windows= 2000, pool=self.get_pool())
            # self.destroy_pool()
            # ann = i_o.qrs_adjust(ecg=record, qrs=ann, fs=self.actual_fs, inputsign=1)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = record[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        if self.windows_shifted:
            ann = ann[self.window_size//2:]
        return ecg, ann

    def parse_reference_annotation(self, patient_id, combine=True, reannotated=True):
        beat_file = self.ref_peaks_path / (patient_id + '_peaks' + self.ecg_format)
        rhythm_file = self.ref_ann_path / ('Annotation_' + patient_id[2:] + self.ecg_format)
        try:
            beat_loaded = mat73.loadmat(beat_file)
        except Exception:
            beat_loaded = sio.loadmat(beat_file, struct_as_record=True)
        try:
            rhythm_loaded = mat73.loadmat(rhythm_file)
        except Exception:
            rhythm_loaded = sio.loadmat(rhythm_file, struct_as_record=True)
        beats_orig = beat_loaded['qrs_pos'].flatten()
        orig_fs = int(self.excel_sheet.loc[self.excel_sheet['REC'].eq(patient_id)]['Fs'].values[0])
        beats_orig = ((beats_orig / cts.N_MS_IN_S) * orig_fs).astype(int)
        beats = (beats_orig * (self.actual_fs / orig_fs)).astype(int)
        tbeats = beats / self.actual_fs
        rhythm = rhythm_loaded['AF_annotation'].flatten()
        if combine:
            rhythm[rhythm == cts.rhythms_dict['AFL']] = cts.rhythms_dict['AFIB']
        return (tbeats * self.actual_fs).astype(int), np.insert(rhythm, 0, 0)

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
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl', sheet_name=0)
        self.excel_sheet_AF_analysis = pd.read_excel(self.excel_sheet_path, engine='openpyxl', sheet_name=2)

    def get_dir(self, id):
        return [i for i in glob.glob(str(self.raw_ecg_path / '*/*')) if
                i.split('/')[-1].startswith(id)]

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
    db = SPAFDB_Parser(load_on_start=True)
    ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    ids = np.setdiff1d(ids, db.parsed_patients())
    savedir = pathlib.PurePath('/home/shanybiton/repos/CircadianAF/output') / db.name.lower()

    # beats, rhythms = db.parse_reference_annotation(ids[0], combine=True, reannotated=False)
    # db.generate_annotations(types=db.sqi_ref_ann, pat_list=ids, lead=2)
    # _, ann = db.parse_raw_ecg(ids[0], start=0, end=-1, type='epltd0', lead=1)
    # ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    # pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]

    db.parse_raw_data(patient_list=ids)
    for patient_id in ids:
        beats, rhythm = db.parse_reference_annotation(patient_id, combine=True, reannotated=True)
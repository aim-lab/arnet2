from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class RBAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(RBAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(
            ['M020D42a', 'N919K291', '1021B555', '1021Ccb6', '1220A159', '1021Cd9d', '1319B64e', '1520979c', 'V420Fb4c',
             '7A21B043', 'C92183cc', 'D921E33b', 'G320C992', '4B19C4cc', 'M020D7d1', 'M020D8bd', 'M020Da15', 'M020Dc24', 'N919Gd30',
             'R720B607', 'R720B6bc', 'E020F4e1', '4B19C4cc', '9A21C58a'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 128
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 3
        self.name = "RBAFDB"
        self.ecg_format = ".dat"

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "dataset"
        self.orig_anns_path = None
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / self.name

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        self.window_size = window_size
        test_pat = self.parsed_patients()[0]
        self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "mask_rr")])
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)
                self.load_circardian_from_disk()

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the RBAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.ecg_file_name = 'rawecg'
        self.ecg_files_name = ['rawecg1', 'rawecg2', 'rawecg3']
        self.patient_file_name = 'patient'
        self.control_file_name = 'main'
        self.event_file_name = 'arrhevnt'
        self.beat_time_file_name = 'combtime'
        self.beat_flag_file_name = 'combflag'
        self.dir_format = ".FUL"
        self.file_format = ".dat"
        self.beats_shape = {1: 'N', 3: 'N', 4: 'AB', 5: 'I',
                            6: 'P'}  # The different rhythms present across the dataset. N: NORMAL', AB: 'ABERRANT', I: 'INHIBIT', P: 'PACED'}
        self.file_rhythms = np.array(['DB', 'P', 'VT', 'SVT', 'PREA', 'SALVO', 'BR', 'TRI',
                                      'TRI', 'B', 'T', 'C', 'RoT', 'ISOL', 'PREN', 'COMPP',
                                      'MAHR', 'MIHR', 'ALB', 'N', 'ABER', 'S', 'INH', 'PAT', 'F', 'AFIB',
                                      'AFIB', 'INH', 'CAL', 'AT'])
        self.file_rhythms_dict = {0: 'DB', 1: 'P', 2: 'VT', 3: 'SVT', 4: 'PREA', 5: 'SALVO', 6: 'BR',
                                  7: 'TRI', 8: 'TRI', 9: 'B', 10: 'T', 11: 'C', 12: 'RoT', 13: 'ISOL',
                                  14: 'PREN', 15: 'COMPP', 16: 'MAHR', 17: 'MIHR', 18: 'ALB', 19: '(N',
                                  20: 'ABER', 21: 'S', 22: 'INH', 23: 'PAT', 133: 'F', 147: 'AFIB', 148: 'AFIB',
                                  149: 'INH', 150: 'CAL', 151: 'AT'}
        self.searchPer = ['ACUTE', 'CHRONIC']
        self.searchPar = ['PAROXYSMAL', 'FIBRILLATION']
        self.searchFlutter = ['FLUTTER']
        # TODO: check Hebrew chars in the excel sheet reading
        self.excel_sheet_path_prometheus = cts.DATA_DIR / self.name.lower() / "documentation" / "RBAF_Holter_Info_prometheus.xlsx"
        self.excel_sheet_path_mdclone = cts.DATA_DIR / self.name.lower() / "documentation" / "RBAF_Holter_Info_mdclone.xlsx"
        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "documentation" / "RBAF_Holter_Info.xlsx"
        self.excel_sheet_prometheus = pd.read_excel(self.excel_sheet_path_prometheus, engine='openpyxl')
        self.excel_sheet_mdclone = pd.read_excel(self.excel_sheet_path_mdclone, engine='openpyxl')
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl')
        self.over_18_patients = np.array(self.excel_sheet[self.excel_sheet["age_at_recording"] >= 18]
                                         ["db_id"]).astype('<U32')

        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    # def _af_win_lab(self, id, win):
    #     """ Computes the binary AF label of a window. The label is computed based on the most represented label over the window.
    #     :param id: The patient ID. Assumed to be in the list of IDs present in the database.
    #     :param win: The window size (in number of beats) along which the raw recording is divided.
    #     """
    #     if id not in self.af_win_lab_dict.keys():
    #         self.af_win_lab_dict[id] = {}
    #     self.af_win_lab_dict[id][win] = np.logical_or(
    #         self.win_lab_dict[id][win] == cts.WINDOW_LABEL_AF_RB[0],
    #         self.win_lab_dict[id][win] == cts.WINDOW_LABEL_AF_RB[1])

    # def _af_pat_lab(self, id):
    #     """ Computes the AF Burden and the global label for a given patient. The AF Burden is computed as the time
    #     spent on AF divided by the total time of the recording. The different categories of patients are: Non-AF (Time in AF
    #     does not exceed 30 [sec], Mild AF (Time in AF above 30 [sec] and AFB under 4%), Moderate AF (AFB between 4 and 80%),
    #     and Severe AF (AFB between 80 and 100%). If the burden of a given pathology for a patient is over 50%, we flag him as a patient
    #     suffering from another CVD (label cts.PATIENT_LABEL_OTHER_CVD). As a convention, for windows, 0 is the label for NSR, 1 for AF, and above
    #     2 for other rhythms.
    #     :param id: The patient ID. Assumed to be in the list of IDs present in the database.
    #     """
    #     # Using minimal window size to have the higher granularity
    #     win = min(self.loaded_window_sizes)
    #     raw_rr = self.rr_dict[id][:(len(self.rr_dict[id]) // win) * win].reshape(-1, win)[
    #         self.mask_rr_dict[id][win]].reshape(-1)
    #     raw_rlab = self.rlab_dict[id][:(len(self.rlab_dict[id]) // win) * win].reshape(-1, win)[
    #         self.mask_rr_dict[id][win]].reshape(-1)
    #     if np.all(np.isnan(raw_rlab)):  # Case where the labels are not available (like in SHHS)
    #         self.af_burden_dict[id] = np.nan
    #         self.af_pat_lab_dict[id] = np.nan
    #         self.other_cvd_burden_dict[id] = np.nan
    #         self.missing_af_label = np.append(self.missing_af_label, id)
    #     else:
    #         time_in_af = raw_rr[np.logical_or(
    #             raw_rlab == cts.WINDOW_LABEL_AF_RB[0],
    #             raw_rlab == cts.WINDOW_LABEL_AF_RB[1])].sum()  # Deriving time in AF.
    #         self.af_burden_dict[id] = time_in_af / self.recording_time[id]  # Computing AF Burden.
    #
    #         np.logical_or(np.isnan(raw_rlab), raw_rlab == cts.WINDOW_LABEL_NON_AF_RB)
    #
    #         self.other_cvd_burden_dict[id] = np.sum(~np.logical_or(np.isnan(raw_rlab),
    #                                                                raw_rlab == cts.WINDOW_LABEL_NON_AF_RB)) / \
    #                                          self.recording_time[id]  # Computing Other CVD Burden.
    #         if self.af_burden_dict[id] > cts.AF_SEVERE_THRESHOLD:  # Assessing the class according to the guidelines
    #             self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_SEVERE
    #         elif self.af_burden_dict[id] > cts.AF_MODERATE_THRESHOLD:
    #             self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_MODERATE
    #         elif time_in_af > cts.AF_MILD_THRESHOLD:
    #             self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_MILD
    #         elif self.other_cvd_burden_dict[id] > 0.5:
    #             self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_OTHER_CVD
    #         else:
    #             self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_NON_AF

    def parse_available_ids(self):
        return np.array([file.split('.')[0] for file in os.listdir(str(self.raw_ecg_path))])

    def parse_reference_annotation(self, id, combine=True, export_to_physiozoo=False, reannotated=False):
        orig_beats = dr.combtime_file_reader(self.raw_ecg_path / (id + '.FUL') / (
                    self.beat_time_file_name + '.dat'))  # , self.recording_time_stamp[id]['start_recording'])
        beats = np.round(orig_beats * (self.actual_fs / self.orig_fs)).astype(int) # with respect to time 0
        tbeats = beats/ self.actual_fs
        ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        if reannotated:
            rhythm_df = self.parse_reference_rhythm(id)
            for index, l in rhythm_df.iterrows():
                l1 = np.abs(tbeats - l.Beginning)
                l2 = np.abs(tbeats - l.End)
                begin = np.where(l1 == l1.min())
                end = np.where(l2 == l2.min())
                ltbeats[int(begin[0][0]):int(end[0][0])] = l.Class
            rhythm = np.array([cts.rhythms_dict[i] for i in ltbeats])
            if combine:
                rhythm[rhythm == cts.rhythms_dict['AFL']] = cts.rhythms_dict['AFIB']
        else:
            arrhevent_data = dr.read_arrhevnt_file(
                self.raw_ecg_path / (id + self.dir_format) / (self.event_file_name + self.file_format))
            arrhevent_data_sorted = arrhevent_data.sort_values(by=['beginning'])
            start_rhythm = [(x / 128) for x in arrhevent_data_sorted['beginning']]
            end_rhythm = [(x / 128) for x in arrhevent_data_sorted['end']]
            event_data = np.array(arrhevent_data['class']).astype(int)
            rhythm_time_intervals = np.append(np.array(start_rhythm).reshape(-1, 1), np.array(end_rhythm).reshape(-1, 1), 1)
            rhythm_intervals = np.round(rhythm_time_intervals * self.actual_fs).astype(int)
            rhythms = np.concatenate((rhythm_intervals, np.array(event_data.astype(int)).reshape(-1, 1)), axis=1)
            rhythm = np.full(len(beats), np.nan)
            for i, r in enumerate(rhythms):
                rhythm[np.logical_and(
                    beats >= rhythms[i][0],
                    beats <= rhythms[i][1]
                )] = rhythms[i][2]
        if export_to_physiozoo:
            return beats, rhythm, arrhevent_data_sorted, event_data
        return beats, rhythm

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
        record = dr.read_ecg_file(
            self.raw_ecg_path / (id + '.FUL') / (self.ecg_file_name + str(lead) + '.dat'))
        re_record = dp.resample_by_interpolation(record, self.orig_fs, cts.EPLTD_FS)
        wfdb.wrsamp(id, fs=cts.EPLTD_FS, units=['mV'],
                    sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
        return record

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type='epltd0', lead=1):
        record = dr.read_ecg_file(self.raw_ecg_path / (patient_id + '.FUL') / (self.ecg_files_name[lead-1] + '.dat'))
        record = bandpass_filter(data=record, id=patient_id, lead='x', lowcut=0.67, highcut=self.orig_fs/2 - 0.5,
                                    signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)

        record = dp.resample_by_interpolation(record, self.orig_fs, self.actual_fs)
        ann = self.parse_annotation(patient_id, type=type, lead=lead)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = int(start * self.actual_fs)
        end_sample = int(end * self.actual_fs)
        record = record[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return record, ann

    def parse_demographic_features(self, patient_id):
        db_id = self.parse_patient_id(patient_id)
        for win in self.loaded_window_sizes:
            sex = self.excel_sheet.loc[self.excel_sheet['db_id'] == db_id, 'sex'].values
            age = self.excel_sheet.loc[self.excel_sheet['db_id'] == db_id, 'age_at_recording'].values
            if len(age)==0:
                self.features_dict[patient_id][win]['Age'] = np.nan
            else:
                self.features_dict[patient_id][win]['Age'] = age[0]
            if len(sex)==0:
                self.features_dict[patient_id][win]['Sex'] = np.nan
            else:
                self.features_dict[patient_id][win]['Sex'] = sex[0]


    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        return self.excel_sheet[self.excel_sheet["holter_id"].astype(str) == recording_id]["db_id"].values[0]
    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def record_diagnosis(self, patient_id, win):
        af_cases = np.array(self.excel_sheet["holter_id"][self.excel_sheet.apply(
            lambda x: self.excel_sheet['diagnosis_merged'].astype(str).str.contains(
                'ATRIAL FIBRILLATION', flags=re.I)).any(axis=1)].values).astype(str)
        per_af = np.array(self.excel_sheet["holter_id"][
                     self.excel_sheet['diagnosis_merged'].str.contains('|'.join(self.searchPer), na=False)].values).astype(str)
        par_af = np.setdiff1d(af_cases, per_af)
        if patient_id in per_af:  # Assessing the class according to the guidelines
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_AF_SEVERE #persistent AF is equivalent to severe AF
        elif patient_id in par_af:  # Assessing the class according to the guidelines
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_AF_MILD #paroxysmal AF is equivalent to mild/moderate AF
        else:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_NON_AF


    def load_beat_flags(self, wins=None, pat_list=None):
        """ This function loads the number of ectopic beats per window for the UVAF dataset.
            :param wins: The windows for which the number of ectopics should be loaded.
        """
        print("Loading beat flags")
        if pat_list is None:
            pat_list = db.parsed_patients()
        if wins is None:
            wins = self.window_sizes
        for pat in pat_list:
            self.beat_flags[pat] = {}
            for win in wins:
                beat_flags = dr.combflag_file_reader(
                    self.raw_ecg_path / (pat + '.FUL') / (self.beat_flag_file_name + '.dat'))
                beat_flags[beat_flags == 3] = 1
                self.beat_flags[pat][win] = beat_flags[:(len(beat_flags) // win) * win].reshape(-1, win)

    def parse_circadian_features(self, patient_id):
        if patient_id not in self.circadian_dict.keys():
            self.circadian_dict[patient_id] = {}
        self.circadian_dict[patient_id]['recording_date'], self.circadian_dict[patient_id]['analysis_date'], \
        self.circadian_dict[patient_id]['start_recording'], self.circadian_dict[patient_id]['end_recording'] = \
            dr.read_timestamp(self.raw_ecg_path / (patient_id + '.FUL') / (self.control_file_name + '.dat'))
        np.save(self.main_path / patient_id / 'circadian_dict.npy', self.__dict__['circadian_dict'][patient_id])

    def load_circardian_from_disk(self, pat_list=None):
        self.__dict__['circadian_dict'] = {}
        if pat_list is None:
            pat_list = self.parsed_ecgs
        for pat in pat_list:
            if os.path.exists(self.main_path / pat / ('circadian_dict.npy')):
                self.__dict__['circadian_dict'][pat] = np.load(self.main_path / pat / ('circadian_dict.npy'),
                                                               allow_pickle=True).item()

    def load_af_prob(self, rbaf_features, rbaf_pred_proba, rbaf_pred, pat_list=None, exclude_low_sqi_win=True,
                     win_thresh=cts.SQI_WINDOW_THRESHOLD):
        if pat_list is None:
            correct_pat_list = self.non_corrupted_ecg_patients()
        else:
            correct_pat_list = np.array([elem for elem in pat_list])
            if len(correct_pat_list) == 0:
                return np.array([[]]), np.array([])  # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][
                                             self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}
        X_rrt = np.concatenate(tuple(
            self.rrt_dict[elem][:(len(self.rrt_dict[elem]) // self.window_size) * self.window_size].reshape(-1,
                                                                                                            self.window_size)[
                masks[elem]] for elem in correct_pat_list), axis=0)
        num_duplicates = {pat: np.sum(masks[pat]) for pat in masks.keys()}

        rec_time = np.concatenate(tuple(
            self.circadian_dict[elem]['start_recording'] * np.ones(num_duplicates[elem], dtype=object) for elem in
            correct_pat_list), axis=0)
        df = pd.DataFrame([])
        df['patient_id'] = rbaf_features[:, 61]
        df['window'] = rbaf_pred
        df['prob'] = rbaf_pred_proba
        df['start'] = X_rrt[:, 0] + rec_time
        df['end'] = X_rrt[:, -1] + rec_time

        return df

    def get_num_leads(self, patient_id):
        files_id = np.array([f.split('.')[0] for f in os.listdir(self.raw_ecg_path / (patient_id + '.FUL'))])
        return len(files_id[np.isin(files_id, self.ecg_files_name)])

    #
    # def export_mat(self, id, df_af, n_lead, ann_type='epltd0'):
    #     point = 5 * db.actual_fs  # taking 5 sec before and after each segment
    #
    #     # ecg, annot = db.parse_raw_ecg(id)
    #     for i, r in df_af.iterrows():
    #         ecg = []
    #         annot = []
    #         for i in range(1, n_lead+1):
    #             record = dr.read_ecg_file(
    #                 self.raw_ecg_path / (id + '.FUL') / (self.ecg_file_name + str(i) + '.dat'))
    #             re_record = dp.resample_by_interpolation(record, self.orig_fs, cts.EPLTD_FS)
    #             re_record = re_record[int(r.start_time * self.actual_fs) - point:int(r.end_time * self.actual_fs) + point]
    #             wfdb.wrsamp(id, fs=cts.EPLTD_FS, units=['mV'],
    #                         sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
    #             detector = getattr(i_o,
    #                                ann_type + '_detector')  # Calling the correct wrapper in the feature comp module.
    #             detector(id)  # Running the wrapper
    #             shutil.move(id + '.' + ann_type, self.generated_anns_path / 'wins' / ann_type / (
    #                     id + '.' + ann_type))
    #             ann = wfdb.rdann(str(self.generated_anns_path / 'wins' / ann_type / id), ann_type).sample
    #             ecg.append(re_record)
    #             annot.append(ann)
    #         data = np.array(ecg)
    #         rqrs = np.array(annot)
    #         mdic = {"data": data, "rqrs": rqrs, "fs": db.actual_fs, "start_": r.start_time, "end_": r.end_time,
    #                 "recording_hour": db.circadian_dict[id]['start_recording']}
    #         savemat(
    #             "/home/shanybiton/repos/Lund/wins/" + str(id) + "_af_win_start_" + str(r.start_time) + "_end_" + str(
    #                 r.end_time) + ".mat", mdic)


if __name__ == '__main__':
    import pathlib

    windows = [60]
    db = RBAFDB_Parser(load_on_start=True)
    # ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    # pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]
    # db.parse_raw_data(patient_list=pat_list)

    # pat_list  = []# add here the holter ids you need to extract examples for
    # for pat in pat_list:
    #     directory = pathlib.PurePath("/MLdata/AIMLab/medAIM/RBAFDB/") / str(pat) # where to save the examples
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     db.export_to_physiozoo(pat, directory=directory, export_rhythms=False, force=True, n_leads=db.get_num_leads(pat))
    # ids = db.parse_available_ids()
    # ids = np.setdiff1d(ids, db.missing_ecg)
    # for patient_id in ids:
    #     if patient_id not in db.features_dict.keys():
    #         db.features_dict[patient_id] = {}
    #         db.features_dict[patient_id][db.loaded_window_sizes[0]] = {}
    #     db.parse_demographic_features(patient_id)

    # db.print_summary()
    # id = 'A720D410'
    # _, _, db.excel_id_mapping.loc[db.excel_id_mapping.Holter_ID.eq(id), 'Age'], db.excel_id_mapping.loc[db.excel_id_mapping.Holter_ID.eq(id), 'Sex'] = dr.read_patient_file(
    #     db.raw_ecg_path / (str(id) + '.FUL') / (db.patient_file_name + '.txt'))
    # ids = db.parse_available_ids()
    # ids = ids[~np.isin(ids, db.missing_ecg)]
    # for id in ids:
    #     db.parse_circadian_features(id)
    # ids_2 = []
    # # ids = db.parsed_patients()
    # for id in ids:
    #     if db.get_num_leads(id) == 2:
    #         ids_2.append(id)
    # ids_3 = ids[~np.isin(ids, ids_2)]
    # db.generate_annotations(pat_list=ids, lead=1)
    # _test_pat = np.load(cts.REPO_DIR / "data" / "splits" / (db.name + "_test_pat.npy"))
    # ids = db.parsed_patients()
    # ids = ids[np.where(ids=='O5209cbd')[0][0]:]
    # print(ids)

    # ids = np.setdiff1d(db.parse_available_ids(), db.corrupted_ecg)
    # ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    # db.generate_annotations(pat_list=ids, force=True, lead=3)
    # db.annotation_types = ['epltd0', 'gqrs', 'wqrs', 'jqrs', 'xqrs']
    # db.loaded_window_sizes = np.append(db.loaded_window_sizes, windows[0])
    # ids = db.parse_available_ids()
    # parse_id = ids[~np.isin(ids, db.parsed_patients())]
    # db.load_from_disk(pat_list=ids)
    # df_clusters = pd.read_csv("/home/shanybiton/AIMLabProjects/rbafdb-project/AF_patient_clusters.csv")
    # ids_AF = df_clusters.loc[df_clusters.cluster.eq(3), "patient"].values
    # for pat in ids_AF:
    #     directory = pathlib.PurePath("/MLdata/AIMLab/medAIM/RBAFDB/") / str(pat)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     db.export_to_physiozoo(pat, directory=directory, export_rhythms=False, force=True, n_leads=3)
    # db.generate_annotations(pat_list=db.parsed_patients(), force=True, lead=1)
    # db.generate_annotations(pat_list=db.parsed_patients(), force=True, lead=2)
    # db.generate_annotations(pat_list=db.parsed_patients(), force=True, lead=3)
    # ids = db.return_patient_ids(pat_list=db.all_patients())
    # rr, rrt, _, win_start, win_end = db.return_rr(pat_list=db.all_patients())
    # prec = db.return_preceeding_windows(pat_list=db.all_patients())
    # data = np.concatenate((rr, prec.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    # # # read Tom's probabilities
    # output = np.load("/MLdata/AIMLab/Tom/rbaf_new_pred_binary.npy")
    # proba = np.load("/MLdata/AIMLab/Tom/rbaf_new_pred_proba.npy")
    #
    # data_proba = np.concatenate((data, output.reshape(-1, 1)), axis=1)
    # df = pd.DataFrame(columns=['id', 'proba'])
    # df['id'] = data_proba[:, 61]
    # df['proba'] = proba
    # df['AF'] = data_proba[:, 62]
    # df['start_time'] = win_start
    # df['end_time'] = win_end
    # correct_pat_list = db.all_patients()
    # masks = {pat: np.logical_and(db.mask_rr_dict[pat][db.window_size],
    #                              db.signal_quality_dict[pat][db.window_size][
    #                                  db.sqi_test_ann] >= cts.SQI_WINDOW_THRESHOLD) for pat
    #          in correct_pat_list}
    #
    # # time = np.concatenate(
    # #     tuple(db.circadian_dict[elem]['start_recording'] * np.ones(np.sum(masks[elem])) for elem in correct_pat_list), axis=0)
    # df['start_time'] = df['start_time'] #+ time
    # df['end_time'] = df['end_time'] #+ time
    # af_grouping = df.groupby(['id']).agg({'proba': 'sum'}).astype(float).reset_index()
    # af_df = df.loc[df.AF.eq(True)]
    # af_event = pd.DataFrame(columns=af_df.columns)
    # j = 0
    # for id in af_df.id.unique():
    #     temp_df = af_df.loc[af_df.id.eq(id)].reset_index(drop=True)
    #     af_event = af_event.append(temp_df.iloc[0])
    #     for i in range(len(temp_df) - 1):
    #         en = af_event.end_time.values[j]
    #         st = temp_df.start_time.values[i + 1]
    #         if np.isclose(st,en):
    #             af_event.at[j, 'end_time'] = temp_df.end_time.values[i + 1]
    #         else:
    #             j += 1
    #             af_event = af_event.append(temp_df.iloc[i + 1]).reset_index(drop=True)
    # af_event['duration'] = af_event['end_time'] - af_event['start_time']
    # import utils.dat_reader as dr
    #
    # point = 5 * db.actual_fs  # taking 5 sec before and after each segment
    # ann_type = 'epltd0'
    # for i, r in temp_df.iloc[:3].iterrows():
    #     ecg = []
    #     annot = []
    #     for i in range(1, 3):
    #         record = db.read_ecg(r.id).iloc[:, i].astype(float).values
    #         re_record = dp.resample_by_interpolation(record, db.orig_fs, db.actual_fs)
    #         re_record = re_record[int(r.start_time * db.actual_fs) - point:int(r.end_time * db.actual_fs) + point]
    #         wfdb.wrsamp(id, fs=db.actual_fs, units=['mV'],
    #                     sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
    #         detector = getattr(i_o,
    #                            ann_type + '_detector')  # Calling the correct wrapper in the feature comp module.
    #         detector(id)  # Running the wrapper
    #         shutil.move(id + '.' + ann_type, db.generated_anns_path / 'wins' / ann_type / (
    #                 id + '.' + ann_type))
    #         ann = wfdb.rdann(str(db.generated_anns_path / 'wins' / ann_type / id), ann_type).sample
    #         timeline = np.arange(0, len(re_record) / db.actual_fs, 1 / db.actual_fs)
    #         plt.plot(timeline, ecg, label='Signal')
    #         rr = np.diff(ann) / db.actual_fs
    #         plt.plot(timeline[ann][:-1], rr, label='RR Interval')

    # for id in af_df.id.unique()[-5:]:
    #     temp_df = af_df.loc[af_df.id.eq(id)].reset_index(drop=True)
    #     if len(temp_df)==0:
    #         continue
    #     db.plot_AF_win(temp_df, 0, -1, figname=str('AF_events_' + str(id) + '.png'))
    # db.plot_win_len(af_df, savefig=False)

    # directory = pathlib.PurePath('/MLdata/AIMLab/medAIM/RBAFDB')
    # for id in af_grouping.loc[af_grouping.proba.ge(10) & af_grouping.proba.lt(300)]['id']:
    #     db.load_patient_from_disk(id)
    #     dir = directory / id
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     if pathlib.Path(db.raw_ecg_path / (id + '.FUL') / (db.ecg_files_name[2] + '.dat')).is_file():
    #         db.export_to_physiozoo(id, n_leads=3, directory=dir)
    #     else:
    #         db.export_to_physiozoo(id, n_leads=2, directory=dir)
    # np.save('/home/shanybiton/repos/Generalization/data/rbaf.npy', data, allow_pickle=True)
    # np.save('/home/shanybiton/repos/Generalization/data/rbaf_rrt.npy', rrt, allow_pickle=True)
    # prec_train = db.return_preceeding_windows(pat_list=ids)
    # db.parse_raw_data(patient_list=ids)
    # for id in ids:
    #     db.export_to_physiozoo(id, export_rhythms=False, force=True, n_leads=3)
    # db.generate_annotations(pat_list=ids[600:], types=['wqrs', 'gqrs'])
    # db.parse_raw_data(patient_list=ids[600:])
    # for id_ in ids:
    #    db.parse_circadian_features(patient_id=id_)
    #    db.load_patient_from_disk(pat=id_)
    '''
    db.circadian_dict[id_] = {}
    db.circadian_dict[id_]['start_recording'] = {}
    db.circadian_dict[id_]['end_recording'] = {}
    db.features_dict[id_] = {}
    db.features_dict[id_][windows[0]] = {}
    db.parse_demographic_features(id_)
    db.parse_raw_data(patient_list = [id_])
    '''

from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class SHDB_2wk_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(SHDB_2wk_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
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
        self.ref_lead = 1
        self.name = "SHDB_2wk"
        self.ecg_format = ".csv"

        """ General variables for signal processing/Filtering """
        self.sqi_test_ann = 'epltd0'  # The annotation type used to compute and load the SQI variables.
        self.sqi_ref_ann = 'xqrs'

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "examples"
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
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

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only to SHDB_2wk) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.ecg_file_name = 'RR'
        self.file_format = ".csv"
        self.csv_dir = "RR_ready"
        self.searchPar = ['PAF']
        self.searchPer = ['PerAF']
        self.peak_ann = np.array(['N', 'Q', 'V', 'S'])
        self.peak_ann_dict = {self.peak_ann[i]: i for i in range(len(self.peak_ann))}
        self.circadian_dict = {}
        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "List_SHDB_AF2wk.xlsx"
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl')
        self.excel_sheet["Study ID"] = self.excel_sheet["Study ID"].astype(str).str.zfill(3)

        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        ids = [d.split('/')[-1] for d in glob.glob(str(self.raw_ecg_path / '*/*')) if
               os.path.isdir(d)]  # needs to return all the files including days
        return ids

    def parse_reference_annotation(self, id, combine=True, reannotated=True):
        record = self.read_ecg(id)
        ann = self.read_ann(id, start=record.time[0], end=record.time.iloc[-1])
        tbeats = np.cumsum(ann.pos.values) / cts.N_MS_IN_S
        # if reannotated:
        #     rhythm_df = self.parse_reference_rhythm(id)
        #     for index, l in rhythm_df.iterrows():
        #         l1 = np.abs(tbeats - l.Beginning)
        #         l2 = np.abs(tbeats - l.End)
        #         begin = np.where(l1 == l1.min())
        #         end = np.where(l2 == l2.min())
        #         ltbeats[int(begin[0][0]):int(end[0][0])] = l.Class
        # else:
        #     ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        #     # rhythm = np.array([ann.ann.values])
        #     # rhythm = rhythm[0]
        ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        rhythm = np.array([self.rhythms_dict[i] for i in ltbeats])
        return (tbeats * self.actual_fs).astype(int), rhythm

    def parse_annotation(self, id, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        # check if peaks file exist. Sometimes, the detector fails to work
        dest_path = str(self.generated_anns_path / type / str(lead) / id)
        if os.path.exists(dest_path + '.' + type):
            return wfdb.rdann(dest_path, type).sample
        return np.array([])

    def record_to_wfdb(self, id, lead, filter_signal=True):
        record = self.parse_raw_ecg(id, lead=lead, read_ann=False, filter_signal=filter_signal)
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'], )
        return record

    def parse_raw_ecg(self, id, lead, start=0, end=-1, type='epltd0', filter_signal=True, read_ann=True, ):
        record = self.read_ecg(id).iloc[:, lead].astype(float).values
        if filter_signal:
            record = dp.bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs / 2 - 0.5,
                                        signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)

        record = dp.resample_by_interpolation(record, self.orig_fs, self.actual_fs)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = int(start * self.actual_fs)
        end_sample = int(end * self.actual_fs)
        record = record[start_sample:end_sample]
        if read_ann:
            ann = self.parse_annotation(id, type=type, lead=lead)
            ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
            ann -= start_sample
            return record, ann
        else:
            return record

    def parse_demographic_features(self, id):
        #  Demographic features are not available for this database
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = np.nan
            self.features_dict[id][win]['Sex'] = np.nan

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        return np.array([dir.split('_')[0].zfill(3) for dir in os.listdir(str(self.raw_ecg_path))])

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def get_dir(self, id):
        """ This function returns the full path starting with id"""
        return [i for i in glob.glob(str(self.raw_ecg_path / '*/*')) if
                i.split('/')[-1].startswith(id)]

    def read_ecg(self, id):
        """ This function reads and returns the ecg signal belonging to id.
        The ecg is stored in a csv file with two columns: ch1 and ch2 storing two ecg channels."""
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
        length = np.size(ch2)
        fs = 1 / self.orig_fs
        [hours, minutes, seconds] = [int(x) for x in temp_time[1].split(':')]
        start = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        end = start.seconds + length * fs
        timestamp = np.arange(start.seconds, end, fs)

        ecg = pd.DataFrame({'time': timestamp, 'data_ch1': ch1, 'data_ch2': ch2, 'date': date})
        ecg.reset_index(drop=True, inplace=True)
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
        """ This functions creates a dict which holds two keys: recording_date and start_recording.
        recording_date: the date of start of recording.
        start_recording: the relative time of the day for when the recording started."""
        if id not in self.circadian_dict.keys():
            self.circadian_dict[id] = {}
        self.circadian_dict[id]['recording_date'] = self.read_ecg(id).date[0].date()
        self.circadian_dict[id]['start_recording'], self.circadian_dict[id]['end_recording'] = \
            self.read_ecg(id).time.iloc[0], self.read_ecg(id).time.iloc[-1]
        np.save(self.main_path / id / 'circadian_dict.npy', self.__dict__['circadian_dict'][id])

    def load_circardian_from_disk(self, patient_list=None):
        """
        This functions loads circadian_dict that was created by parse_circadian_features.
        """
        if patient_list is None:
            patient_list = self.parsed_patients()
        for pat in patient_list:
            if os.path.exists(self.main_path / pat / ('circadian_dict.npy')):
                self.__dict__['circadian_dict'][pat] = np.load(self.main_path / pat / ('circadian_dict.npy'),
                                                               allow_pickle=True).item()

    def read_ann(self, id, start=None, end=None):
        """
        This functions read the R-peaks .csv file per id.
        Then it returns for a given id the reference annotation
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param start: The beginning of the ECG.
        :param end: The end of the ECG.
        :returns peaks: A numpy array listing the indices of the peaks in the raw ECG.
        :returns rhythms: A numpy array listing the rhythms corresponding to the peaks in the raw ECG.
        """
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
        real_start = time.strftime('%-H:%M', time.gmtime(start))
        time_ann_start = time_df[time_df == real_start].index[0]
        real_end = time.strftime('%-H:%M', time.gmtime(end))
        temp_time_df = time_df[time_ann_start + 1:]
        if len(temp_time_df[temp_time_df == real_end]) == 0:
            time_ann_end = time_df.index[-1]
        else:
            if time_df[time_df == real_end].index[-1] < 4000:
                time_ann_end = time_df.index[-1]
            else:
                time_ann_end = time_df[time_df == real_end].index[-1]
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
    db.get_dir(ids[0])
    # ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    # pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]

    # db.parse_raw_data(patient_list=ids[0])

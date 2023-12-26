from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class SPAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(SPAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(['ns001177', 'ns001117', 'ns004678', 'ns006576'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = [250, 256]
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = [2, 12]
        self.name = "SPAFDB"
        self.ecg_format = ".mat"

        """ General variables for signal processing/Filtering """
        self.sqi_test_ann = 'epltd0'          # The annotation type used to compute and load the SQI variables.
        self.sqi_ref_ann = 'xqrs'

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "examples"
        self.orig_anns_path = cts.DATA_DIR / self.name.lower() / "peaks"
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
        self.beat_flags = {}
        # self.load_beat_flags()

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the SPAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.circadian_dict = {}
        self.peak_ann = np.array(['N', 'Q', 'V', 'S'])
        self.peak_ann_dict = {self.peak_ann[i]: i for i in range(len(self.peak_ann))}
        self.ref_ann_path = cts.DATA_DIR / self.name.lower() / "AF_annotations"
        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "SPAFBD_description.xlsx"
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl', sheet_name=0)
        self.excel_sheet_AF_analysis = pd.read_excel(self.excel_sheet_path, engine='openpyxl', sheet_name=2)

        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        ids = np.array([file.split('.')[0] for file in os.listdir(str(self.raw_ecg_path))])  # needs to return all the files including days
        return ids

    def parse_annotation(self, id, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        # check if peaks file exist. Sometimes, the detector fails to work
        dest_path = str(self.generated_anns_path / type / str(lead) / id)
        if os.path.exists(dest_path + '.' + type):
            return wfdb.rdann(dest_path, type).sample
        return np.array([])

    def record_to_wfdb(self, id, lead, filter_signal=True, correct_peaks=True):
        record = self.parse_raw_ecg(id, lead=lead, read_ann=False, filter_signal=filter_signal,
                                    correct_peaks=correct_peaks)
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'])
        return record

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type='epltd0', filter_signal=True, read_ann=True, correct_peaks=True):
        file = self.raw_ecg_path / (patient_id + self.ecg_format)
        try:
            loaded = mat73.loadmat(file)
        except Exception:
            loaded = sio.loadmat(file, struct_as_record=True)
        ecg = loaded['signal'].T[lead-1]
        orig_fs = int(loaded['fa'].flatten()[0])
        if filter_signal:
            ecg = dp.bandpass_filter(data=ecg, id=patient_id, lead='x', lowcut=0.67, highcut=orig_fs/2 - 0.5,
                                        signal_freq=orig_fs, filter_order=75, notch_freq=50, debug=False)
        ecg = dp.resample_by_interpolation(ecg, orig_fs, self.actual_fs)
        ecg = ecg / 1000  # scale amplitude
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        if read_ann:
            ann = self.parse_annotation(patient_id, type=type, lead=lead)
            if correct_peaks:
                ann = i_o.qrs_adjust_detector(ecg=ecg, qrs=ann, fs=self.actual_fs, INPUTSIGN=1, n_windows= 2000, pool=self.get_pool())
            ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
            ann -= start_sample
            return ecg, ann
        else:
            return ecg

    def parse_reference_annotation(self, patient_id, combine=True):
        beat_file = self.orig_anns_path / (patient_id + '_peaks' + self.ecg_format)
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

    # TODO: imrove function. the "end_recording" is weird as in contains AN hours without any reference to number of
    #  passed from start of recording. consider just use inddate instead of end_recording or drop this variable
    #  completely.
    def parse_circadian_features(self, id):
        """ This functions creates a dict which holds two keys: recording_date and start_recording.
        recording_date: the date of start of recording.
        start_recording: the relative time of the day for when the recording started."""
        if id not in self.circadian_dict.keys():
            self.circadian_dict[id] = {}
        self.circadian_dict[id]['recording_date'] = np.nan
        self.circadian_dict[id]['start_recording'] = self.excel_sheet_AF_analysis.loc[
            self.excel_sheet_AF_analysis["REC"].eq(id), 'Holter start time'].values[0]
        self.circadian_dict[id]['end_recording'] = (dt.datetime.combine(
            dt.date(1,1,1), self.circadian_dict[id]['start_recording']) +
                                                    dt.timedelta(seconds=self.excel_sheet_AF_analysis.loc[
                                                        self.excel_sheet_AF_analysis["REC"].eq(id), 'Unnamed: 3']
                                                                 .values[0])).time()
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

if __name__ == '__main__':
    windows = [60]
    db = SPAFDB_Parser(load_on_start=False)
    # ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    # ids = np.setdiff1d(ids, db.parsed_patients())
    # savedir = pathlib.PurePath('/home/shanybiton/repos/CircadianAF/output') / db.name.lower()

    # beats, rhythms = db.parse_reference_annotation(ids[0], combine=True, reannotated=False)
    # db.generate_annotations(types=db.sqi_ref_ann, pat_list=ids, lead=2)
    # _, ann = db.parse_raw_ecg(ids[0], start=0, end=-1, type='epltd0', lead=1)
    # ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    # pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]

    # db.parse_raw_data(patient_list=ids)
    # for patient_id in ids:
    #     beats, rhythm = db.parse_reference_annotation(patient_id, combine=True, reannotated=True)
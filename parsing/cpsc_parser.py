import sys
#relative paths
sys.path.append('/home/shanybiton/repos/Generalization')
sys.path.append('/home/shanybiton/repos/Generalization/utils')
sys.path.append('/home/shanybiton/repos/Generalization/parsing')
sys.path.append('/home/shanybiton/repos/Generalization/preprocessing')

from base_parser import *

warnings.filterwarnings('ignore')


class CPSCDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(CPSCDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Missing records"""
        self.missing_ecg = np.array(['data_51_7', 'data_51_6', 'data_51_1', 'data_51_2', 'data_51_4',
       'data_51_5', 'data_51_3', 'data_51_8', 'data_50_11', 'data_51_9',
       'data_50_12'])

        """ Helper variables"""
        self.window_size = window_size

        """ General variables for signal processing/Filtering """
        self.min_annotation_len = 60      # Below this number of peaks, the recording is removed.

        """Variables relative to the ECG signals."""
        self.orig_fs = cts.EPLTD_FS
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.name = "CPSCDB"
        self.ecg_format = "wfdb"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J', 'MISSB',
                                 'PSE', 'MB', 'M'])
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "cpsc2021" / "1.0.0"
        self.orig_anns_path = cts.DATA_DIR / self.name.lower() / "cpsc2021" / "1.0.0"
        self.generated_anns_path = cts.BASE_DIR / "Shany" / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.BASE_DIR / "Shany" / "PreprocessedDatabases" / self.name
        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        self.window_sizes = np.array([60], dtype=int) # Window sizes available in the dataset
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the CPSCDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.excel_sheet_path_I = cts.DATA_DIR / self.name.lower() / "cpsc2021" / "1.0.0" / "PatientInfo_training_I.csv"
        self.excel_sheet_path_II = cts.DATA_DIR / self.name.lower() / "cpsc2021" / "1.0.0" / "PatientInfo_training_II.csv"
        self.get_META()
        self.over_18_patients = np.array(self.excel_sheet[self.excel_sheet["Age"] >= 18]["Patient"].astype(str))

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = np.array([x.split('/')[-1][:-1] for x in f.readlines()])
        return records

    def parse_reference_annotation(self, id, reannotated=False):
        ann = wfdb.rdann(str(self.raw_ecg_path / self.get_recording_dir(id)), 'atr')
        rhythm = i_o.pad_rhythm(np.array(ann.aux_note), missing=['', 'None', '\x01 Aux'])
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann.sample, rhythm

    def parse_annotation(self, id, type="epltd0", lead=1):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        dest_path = str(self.generated_anns_path / type / str(lead) / id)
        if os.path.exists(dest_path + '.' + type):
            return wfdb.rdann(dest_path, type).sample
        return np.array([])

    def record_to_wfdb(self, id, lead=1):
        record = wfdb.rdrecord(str(self.raw_ecg_path / self.get_recording_dir(id)))
        ecg = record.p_signal[:, lead-1]
        ecg_resampled = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_resampled.reshape(-1, 1), fmt=['16'])

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type='epltd0', lead=1):
        record = wfdb.rdrecord(str(self.raw_ecg_path / self.get_recording_dir(patient_id)))
        ecg = record.p_signal[:, lead-1]
        ecg = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        ann = self.parse_annotation(patient_id, type=type)
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return ecg, ann

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        age = float(self.excel_sheet.loc[self.excel_sheet["Patient"] == self.parse_patient_id(id), "Age"])
        sex = float(
            self.excel_sheet.loc[self.excel_sheet["Patient"] == self.parse_patient_id(id)]["Sex"] == 'F')  # True: Female, False: Male
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = age
            self.features_dict[id][win]['Sex'] = sex

    def parse_patient_id(self, recording_id):
        return recording_id.split('_')[1]

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
        self.excel_sheet_I = pd.read_csv(self.excel_sheet_path_I)
        self.excel_sheet_II = pd.read_csv(self.excel_sheet_path_II)
        self.excel_sheet = pd.concat([self.excel_sheet_I, self.excel_sheet_II])
        self.excel_sheet.replace({'female': 'F', 'male': 'M', 'unknown': np.nan}, inplace=True)
        self.excel_sheet.Patient = self.excel_sheet.Patient.astype(str)

    def get_recording_dir(self, patient_id):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = f.read().splitlines()
        for line in records:
            if patient_id in line:
                return line
        raise IOError("The requested id does not exist.")
        return

    def record_diagnosis(self, patient_id, win):
        _, fields = wfdb.rdsamp(str(self.raw_ecg_path / self.get_recording_dir(patient_id)))
        sample_descrip = fields['comments']
        if 'non atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_NON_AF
        elif 'persistent atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_SEVERE  # persistent AF is equivalent to severe AF
        elif 'paroxysmal atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_AF_MILD #paroxysmal AF is equivalent to mild/moderate AF
        return

if __name__ == "__main__":
    db = CPSCDB_Parser(window_size=60, load_on_start=True)
    ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    db.parse_raw_data(patient_list=ids, window_sizes=[60])
    # for id in db.parse_available_ids():
    #     db.parse_demographic_features('data_24_26')
    #     db.save_patient_to_disk(id)
    db.save_to_disk()
    # unique_ids = np.unique([id.rsplit('_', 1)[0] for id in db.parse_available_ids()])

    # db.generate_annotations(lead=1, force=True)
    # db.generate_annotations(pat_list = db.parse_available_ids()[~np.isin(db.parse_available_ids(), db.missing_ecg)], lead=2)
    # for pat in db.parse_available_ids():
    #     db.features_dict[pat] = {}
    #     db.features_dict[pat][60] = {}
    #     db.record_diagnosis(pat, win=60)
    # db.parse_raw_data(window_sizes= db.window_sizes, patient_list = db.parse_available_ids()[~np.isin(db.parse_available_ids(), db.missing_ecg)])
    # db.parse_raw_data(patient_list = ['data_58_9'])
    # db.save_to_disk()
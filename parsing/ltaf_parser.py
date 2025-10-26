from base_parser import *


class LTAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(LTAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes --------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array([])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 128
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.ref_lead = 1
        self.name = "LTAFDB"
        self.ecg_format = ".wfdb"

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / "af_long_term" / "afltdb"
        self.orig_anns_path = cts.DATA_DIR / "af_long_term" / "afltdb"
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / self.name[:4]

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        if os.path.exists(self.main_path):
            parsed_patients = self.parsed_patients()
            test_pat = parsed_patients[0]
            self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
            if load_on_start:
                self.set_window_size(self.window_size)

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only to LTAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.circadian_dict = {}
        self.over_18_patients = self.parse_available_ids()

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = np.array([x[:-1] for x in f.readlines()])
        return records

    def parse_reference_annotation(self, id):
        ann = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')
        if id == 64:  # ID 64 notes did not present the label AFIB at the beginning.
            ann.aux_note[0] = '(AFIB'
        rhythm = i_o.pad_rhythm(np.array(ann.aux_note), missing=['', '\x01 Aux'])
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann.sample, rhythm

    def parse_annotation(self, id, lead, type='epltd0'):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead):
        record = self.parse_raw_ecg(id, lead=lead, read_ann=False)
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'])

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type='epltd0', read_ann=True, ):
        record = wfdb.rdrecord(str(self.raw_ecg_path / patient_id))
        ecg = record.p_signal[:, lead-1]
        ecg = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        if read_ann:
            ann = self.parse_annotation(patient_id, type=type)
            ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
            ann -= start_sample
            return ecg, ann
        else:
            return ecg

    def parse_ahi(self, id):
        self.ahi_dict[id] = None  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = None  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        pass  # This data is not available for this dataset.

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def parse_circadian_features(self, id):
        """ This functions creates a dict which holds two keys: recording_date and start_recording.
        recording_date: the date of start of recording."""
        _, fields = wfdb.rdsamp(str(self.raw_ecg_path / id))
        if id not in self.circadian_dict.keys():
            self.circadian_dict[id] = {}
        self.circadian_dict[id]['recording_date'] = fields['base_date']
        np.save(self.main_path / id / 'circadian_dict.npy', self.__dict__['circadian_dict'][id])

    def record_diagnosis(self, patient_id, win):
        """
        This function records the AF diagnosis extracted from holter free text OR tabular diagnosis (diagnosis_merged).
        The different classes are paroxysmal AF (AF severe) and persistent AF (AF mild)
        """
        _, fields = wfdb.rdsamp(str(self.raw_ecg_path / patient_id))
        sample_descrip = fields['comments']
        if 'non atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_NON_AF
        elif 'persistent atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_SEVERE  # persistent AF is equivalent to severe AF
        elif 'paroxysmal atrial fibrillation' in sample_descrip:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_AF_MILD #paroxysmal AF is equivalent to mild/moderate AF
        return


if __name__ == '__main__':
    db = LTAFDB_Parser(load_on_start=True)
    # db.generate_annotations(types=cts.ANNOTATION_TYPES, pat_list=db.parse_available_ids(), lead=1)
    # db.generate_annotations(types=cts.ANNOTATION_TYPES, pat_list=db.parse_available_ids(), lead=2)
    # for pat in db.parse_available_ids():
    #     db.features_dict[pat] = {}
    #     db.features_dict[pat][60] = {}
    #     db.parse_circadian_features(pat)
    # db.parse_raw_data()
    # for pat in db.signal_quality_dict.keys():
    #     for win in cts.BASE_WINDOWS:
    #         curr_sig_qual = copy.deepcopy(db.signal_quality_dict[pat][win])
    #         del db.signal_quality_dict[pat][win]
    #         db.signal_quality_dict[pat][win] = {}
    #         db.signal_quality_dict[pat][win]['xqrs'] = curr_sig_qual
    #
    # print("Adding PIP")
    # db.add_feature('PIP')
    # print("Adding PSS")
    # db.add_feature('PSS')
    # print("Adding PAS")
    # db.add_feature('PAS')
    # print("Adding IALS")
    # db.add_feature('IALS')
    #
    # for pat in db.features_dict.keys():
    #     for win in cts.BASE_WINDOWS:
    #         np.save(db.main_path / pat / 'signal_quality' / str(win), db.signal_quality_dict[pat][win])
    #         np.save(db.main_path / pat / 'features' / str(win), db.features_dict[pat][win])
    # #db.parse_raw_data(total_run=True)

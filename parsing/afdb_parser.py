from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class AFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(AFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(['00735', '03665'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 250
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.ref_lead = 1
        self.name = "AFDB"
        self.ecg_format = ".wfdb"

        """Variables relative to the different paths."""
        self.raw_ecg_path = cts.DATA_DIR / "afdb"
        self.orig_anns_path = cts.DATA_DIR / "afdb"
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

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only to AFDB) ---------------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.over_18_patients = self.parse_available_ids()
    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """

    def parse_available_ids(self):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = np.array([x[:-1] for x in f.readlines()])
        return records

    def parse_reference_annotation(self, id):
        ann = self.parse_annotation(id, type='manual')
        ann_rhythm = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')
        rhythm_samp = ann_rhythm.sample
        rhythm_names = ann_rhythm.aux_note
        rhythm = np.ndarray(shape=len(ann), dtype=object)
        for j in range(len(rhythm_samp)):
            if j < len(rhythm_samp) - 1:
                rhythm[np.where(np.logical_and(ann > rhythm_samp[j], ann <= rhythm_samp[j + 1]))[0]] = rhythm_names[j]
            else:
                rhythm[np.where(ann > rhythm_samp[j])[0]] = rhythm_names[j]
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann, rhythm

    def parse_annotation(self, id, lead, type='epltd0'):
        if type == 'manual':
            return wfdb.rdann(str(self.raw_ecg_path / id), 'qrs').sample
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        else:
            return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead):
        record = self.parse_raw_ecg(id, lead=lead, read_ann=False)
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'])
        return record

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type="epltd0", read_ann=True, ):
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
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        pass

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def _af_pat_clinical_lab(self, patient_id, win):
        raw_rr = self.rr_dict[patient_id][:(len(self.rr_dict[patient_id]) // win) * win].reshape(-1, win)[
            self.mask_rr_dict[patient_id][win]].reshape(-1)
        raw_rlab = self.rlab_dict[patient_id][:(len(self.rlab_dict[patient_id]) // win) * win].reshape(-1, win)[
            self.mask_rr_dict[patient_id][win]].reshape(-1)
        time_in_af = raw_rr[raw_rlab == cts.WINDOW_LABEL_AF].sum()  # Deriving time in AF.
        if self.af_burden_dict[patient_id] > cts.AF_PERSISTENT_THRESHOLD:
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_SEVERE  # persistent AF is equivalent to severe AF
        elif self.af_burden_dict[patient_id] > cts.AF_MODERATE_THRESHOLD or time_in_af > cts.AF_MILD_THRESHOLD:
            self.features_dict[patient_id][win][
                'diagnosis'] = cts.PATIENT_LABEL_AF_MILD  # paroxysmal AF is equivalent to mild/moderate AF
        elif self.other_cvd_burden_dict[patient_id] > 0.5:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_OTHER_CVD
        else:
            self.features_dict[patient_id][win]['diagnosis'] = cts.PATIENT_LABEL_NON_AF

if __name__ == '__main__':
    db = AFDB_Parser(load_on_start=True)
    # db.generate_annotations(lead=2, force=True)
    # db.generate_annotations(lead=1, force=True)
    # for pat in db.parse_available_ids():
    #     db.parse_reference_annotation(pat)
    #     db.features_dict[pat] = {}
    #     db.features_dict[pat][60] = {}
    #     db.record_diagnosis(pat, win=60)
    # db.export_to_physiozoo('04015', export_rhythms=True, n_leads=db.n_leads)
    # db.parse_raw_data()
    # db.save_to_disk()
    # a = 5

from base_parser import *


class PTBXL_Parser(BaseParser):

    def __init__(self):

        super(PTBXL_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes --------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Variables relative to the ECG signals."""
        self.orig_fs = 500
        self.actual_fs = 400
        self.n_leads = 12
        self.ref_lead = 6
        self.name = "PTBXL"
        self.ecg_format = ".wfdb"
        self.lead_index = ('I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
        self.lead_dict = {self.lead_index[i]: i for i in range(len(self.lead_index))}

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / "PTB-XL" / "1.0.1"
        self.orig_anns_path = None
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
        self.main_path = cts.PREPROCESSED_DATA_DIR / "PTB-XL"
        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the PTBXL) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.spreadsheet_format = ".csv"
        self.spreadsheet_name = "ptbxl_database"
        self.scp_name = "scp_statements"
        self.path_to_spreadsheet = self.DATA_DIR / (self.spreadsheet_name + self.spreadsheet_format)
        self.path_to_scp = self.DATA_DIR / (self.scp_name + self.spreadsheet_format)
        self.res = "hr"
        self.get_META()
        self.traces_ids = self.META.ecg_id.values
        self.x = self.META["filename_" + self.res]
        self.col_pat = "patient_id"
        self.col_exam = "ecg_id"

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        # EXAM IDS
        return self.META.patient_id.values

    def parse_annotation(self, id, type="epltd0", lead=6):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        if np.isin(self.corrupted_ecg, id).any():
            print('The record is flat')
            ann = []
        else:
            ecg_len = len(self.record_to_wfdb(id, lead=lead))
            filename = str(id) + "_lead_" + str(lead)
            ann = wfdb.rdann(str(self.generated_anns_path / type / filename), type).sample
            ann = ann[ann > ecg_len] - ecg_len
        return ann

    def record_to_wfdb(self, id, lead):
        loc = np.where(self.traces_ids == int(id))[0][0]
        rel_path = self.get_ecg_path(int(id))
        record = wfdb.rdrecord(str(self.ECG_DATA_DIR) + "/" + rel_path)
        record = self.x[loc].reshape(-1, self.x.shape[-1])
        # ecg = record.p_signal[:, lead]
        ecg = record[:, lead]
        # re_ecg = dp.resample_by_interpolation(ecg, self.orig_fs, cts.EPLTD_FS)
        signal_epltd = np.concatenate((ecg, ecg))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=signal_epltd.reshape(-1, 1), fmt=['16'])
        return ecg

    def parse_raw_ecg(self, exam_id, start=0, end=-1, type='epltd0', lead=0):
        ecg = self.record_to_wfdb(exam_id, lead=lead)
        ann = self.parse_annotation(exam_id, type=type, lead=lead)

        return ecg, ann

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def reorganize_data(self, data, lead_index, orig_index, actual_fs, orig_fs):
        ecg = np.zeros(shape=(4096, 12))
        for i_lead, lead in enumerate(lead_index):
            ecg_lead = data[:, orig_index[lead]]
            ecg_lead = signal.resample(ecg_lead, int(len(ecg_lead) * actual_fs / orig_fs))
            # ecg_lead = bandpass_filter(data=ecg_lead, id=i_lead, lead=lead, lowcut=0.67, highcut=100, signal_freq=400,
            #                            filter_order=75, notch_freq=50, debug=False)
            padding = 4096 - len(ecg_lead)
            offset = int(round(padding / 2))
            ecg[offset:len(ecg_lead) + offset, i_lead] = ecg_lead
            # ecg[:, i_lead] = ecg_lead
        return ecg

    def get_ecg_path(self, id):
        col = "filename_" + self.res
        path = self.META.loc[self.META.ecg_id == id, col].values[0]
        return path

    def get_trace(self, id):
        rel_path = self.get_ecg_path(int(id))
        record = wfdb.rdrecord(str(self.ECG_DATA_DIR) + "/" + rel_path)
        return record.p_signal

    def get_META(self):
        '''
        CSV Columns
        '''
        # load and convert annotation data
        self.META = pd.read_csv(self.path_to_spreadsheet)
        self.META.scp_codes = self.META.scp_codes.apply(lambda x: ast.literal_eval(x))

        # load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(self.path_to_scp, index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        self.META['diagnostic_superclass'] = self.META.scp_codes.apply(self.aggregate_diagnostic)

    def aggregate_diagnostic(self, y_dict):
        tmp = []
        for key in y_dict.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def get_extended_dataframe(self):
        # Sort by date
        self.META['date_exam'] = pd.to_datetime(self.META['recording_date']).dt.date
        self.META.sort_values('date_exam', inplace=True, ascending=False)
        # temp = self.META[~self.META['scp_codes'].str.contains('AFIB')]
        id_duplicate = self.META.loc[self.META.duplicated(subset='patient_id'), "patient_id"].unique()
        # df = self.META.loc[self.META.patient_id.isin(id_duplicate)]
        df = self.META.copy()
        df["AF"] = df.scp_codes.apply(lambda x: 1 if "AFIB" in x.keys() else 0)
        n_exams = len(df)
        # Get some fields
        ids = np.array(df['ecg_id'])
        patient_ids = np.array(df['patient_id'].astype(int))
        date = pd.to_datetime(df['recording_date']).values
        condition = np.array(df['AF'], dtype=bool)
        # Get number of patients
        patients = np.unique(patient_ids)
        n_patients = len(patients)
        # Get converters
        hash_exams = dict(zip(ids, range(n_exams)))
        hash_patients = dict(zip(patients, range(n_patients)))
        # Get information about next exam (notice the exams are ordered in descending order)
        next_exam_id = -1 * np.ones(n_exams, dtype=int)
        count_exams = np.zeros(n_patients, dtype=int)
        date_last_exam = np.zeros(n_patients, dtype='datetime64[ns]')
        first_exam_patient = -1 * np.ones(n_patients, dtype=int)
        patients_with_condition = np.zeros(n_patients, dtype=bool)
        for n in range(n_exams):
            n_patient = hash_patients[patient_ids[n]]
            next_exam_id[n] = first_exam_patient[n_patient]
            patients_with_condition[n_patient] = patients_with_condition[n_patient] or condition[n]
            if first_exam_patient[n_patient] > 0:
                n_next = hash_exams[first_exam_patient[n_patient]]
            else:
                date_last_exam[n_patient] = date[n]
            first_exam_patient[n_patient] = ids[n]
            count_exams[n_patient] += 1
        # First appearance
        count_appearances_exam = np.zeros(n_exams, dtype=int)
        count_appearances = np.zeros(n_patients, dtype=int)
        crescent_counter = np.zeros(n_patients, dtype=int)
        count_exams_first_appearance = np.zeros(n_patients, dtype=int)
        date_first_appearance = np.zeros(n_patients, dtype='datetime64[ns]')
        for n in range(n_exams)[::-1]:
            n_patient = hash_patients[patient_ids[n]]
            crescent_counter[n_patient] += 1
            if condition[n]:
                count_appearances[n_patient] += 1
            count_appearances_exam[n] = count_appearances[n_patient]
            if condition[n] and count_exams_first_appearance[n_patient] == 0:
                count_exams_first_appearance[n_patient] = crescent_counter[n_patient]
                date_first_appearance[n_patient] = date[n]
        time_to_last_exam = np.zeros(n_exams, dtype='timedelta64[ns]')
        for n in range(n_exams):
            n_patient = hash_patients[patient_ids[n]]
            time_to_last_exam[n] = date_last_exam[n_patient] - date[n]
        # Convert to weeks
        time_to_last_exam = np.array(time_to_last_exam, dtype=int) / (1e9 * 60 * 60 * 24 * 7)
        time_to_first_appearance = np.zeros(n_exams, dtype='timedelta64[ns]')
        for n in range(n_exams):
            if count_appearances_exam[n] < 1:
                n_patient = hash_patients[patient_ids[n]]
                time_to_first_appearance[n] = date_first_appearance[n_patient] - date[n]
        # Convert to weeks
        time_to_first_appearance = np.array(time_to_first_appearance, dtype=int) / (1e9 * 60 * 60 * 24 * 7)
        df['time_to_first_appearance'] = time_to_first_appearance
        df['time_to_last_exam'] = time_to_last_exam

        # Create dataframe containing information about patient
        return df

    def pad(self, id):
        loc = np.where(self.traces_ids == int(id))  # find the location of the exam
        record = self.x[loc].reshape(-1, self.x.shape[-1])  # shape: (4096,12)
        ecg_raw = record[:, 6]  # taking lead V1 {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}
        # signal_epltd = np.concatenate((re_ecg, re_ecg))
        return np.where(ecg_raw != 0)

    def _generate_ann_per_lead(self, ann_type, id, files):
        id = str(id)
        leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ann_files = [i for i in files if i.startswith(id)]
        if len(ann_files) == 0:
            for lead in leads:
                self.generate_annotations(types=[ann_type], pat_list=[id], force=True, lead=lead)
        ref_files = np.array([id + "_lead_" + str(lead) + '.' + ann_type for lead in leads] + [id + '.' + ann_type])
        create_files = ref_files[~np.isin(ref_files, ann_files)]
        result = [re.search('lead_(.*).' + ann_type, f) for f in create_files]
        l = [int(g.group(1)) for g in result]
        for lead in l:
            self.generate_annotations(types=[ann_type], pat_list=[id], force=True, lead=lead)
        return

    def ann_to_dict(self, ids=[], ann_type=[]):
        leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        lead_dict = {}
        for id_ in ids:
            id_ = str(id_)
            print("creating 12-lead annotation dict for id", id_)
            lead_dict[id_] = {}
            for ann in ann_type:
                lead_dict[id_][ann] = {}
                for l in leads:
                    lead_dict[id_][ann][str(l)] = self.parse_annotation(id_, type=ann, lead=l)
            self.save_peaks_to_disk(id_, lead_dict[id_])
        return lead_dict

    def save_peaks_to_disk(self, pat, peaks_dict):
        """ This function saves one patient to the disk.
        :param pat: The patient ID to be saved. """
        filename = str(pat) + '_peaks_dict'
        np.save(self.generated_dict_path / filename, peaks_dict)


if __name__ == '__main__':
    db = PTBXL_Parser()
    # afib_pred_data = db.get_extended_dataframe()
    # class_df = get_class(afib_pred_data, C1_thresh=4.43, C2_thresh=260.714, col_pat="patient_id", col_exam="ecg_id")
    # exam_info = split_C(afib_pred_data, class_df, C2_thresh=260.714, C1_thresh=4.43, col_pat="patient_id", col_exam="ecg_id")
    exams = pd.read_csv("/MLdata/AIMLab/ShanySheina/dataset/PTBXL/exams_info.csv")
    classes = exams.loc[exams.C.ge(0)]
    ids = classes["ecg_id"].values
    x_C = np.empty(shape=(len(classes), 4096, db.get_trace(ids[0]).shape[1]))
    for i, id in enumerate(ids):
        temp = db.get_trace(id)
        x_C[i] = db.reorganize_data(temp, cts.lead_index, db.lead_dict, db.actual_fs, db.orig_fs)
    db.x = x_C
    db.traces_ids = ids
    leads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    C = 2
    db.generated_dict_path = pathlib.PurePath('/MLdata/AIMLab/Shany/Annotations/PTBXL/C' + str(C))
    db.generated_anns_path = pathlib.PurePath('/MLdata/AIMLab/Shany/Annotations/PTBXL')
    ids_C = classes.loc[classes.C.eq(C), "ecg_id"].astype(int)
    for lead in leads:
        db.generate_annotations(types=cts.ANNOTATION_TYPES, pat_list=ids_C.values[3000:], force=True, lead=lead)
    db.ann_to_dict(ids=ids_C.values[3000:], ann_type=cts.ANNOTATION_TYPES)

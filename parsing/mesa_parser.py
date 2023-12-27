from base_parser import *


# TODO: write this class to fit all other parsers + add reference annotations
class MESA_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(MESA_Parser, self).__init__()

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
        self.orig_fs = 256
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 1
        self.ref_lead = 1
        self.name = "MESA"
        self.ecg_format = ".edf"

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / 'mesa' / 'polysomnography' / 'edfs'
        self.orig_anns_path = None
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

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only to MESA) ----------------- #
        # ------------------------------------------------------------------------------- #
        """

        self.afib_lab_file_path = cts.DATA_DIR / 'mesa' / 'datasets' / 'mesa-sleep-dataset-0.3.0.csv'
        self.afib_tab = pd.read_csv(self.afib_lab_file_path)
        self.curr_edf = None
        self.af_inc_lab_dict = {}
        self.af_prev_lab_dict = {}
        self.afib_lab_dict = {}

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """

    def parse_available_ids(self):
        return np.array([file[11:15] for file in os.listdir(str(self.raw_ecg_path))])

    def parse_annotation(self, id, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead):
        record = self.parse_raw_ecg(id, lead=lead, read_ann=False)
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'])
        return record

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type='epltd0', read_ann=True, ):
        edf = pyedflib.EdfReader(str(self.raw_ecg_path / ('mesa-sleep-' + patient_id + self.ecg_format)))
        self.curr_edf = edf
        ecg_idx = np.where(np.array(edf.getSignalLabels()) == 'EKG')[0][lead-1]
        ecg_raw = edf.readSignal(ecg_idx)
        Fs = np.round(edf.samplefrequency(ecg_idx))
        ecg = signal.resample(ecg_raw, int(len(ecg_raw) * cts.EPLTD_FS / Fs))
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        edf._close()
        del edf
        if read_ann:
            ann = self.parse_annotation(patient_id, type=type)
            ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
            ann -= start_sample
            return ecg, ann
        else:
            return ecg

    # rlab is not relevant here. Same for af burden. Will have rlab all ones and zeros, and af burden 0 or 100%. They
    # should not be considered.
    # TODO: replace this function with parse_reference_annotation
    def parse_elem_data(self, pat):
        dicts_to_fill = [self.start_windows_dict, self.end_windows_dict, self.av_prec_windows_dict, self.n_excluded_windows_dict, self.mask_rr_dict]
        for dic in dicts_to_fill:
            if pat not in dic.keys():
                dic[pat] = {}
        ann = self.parse_annotation(pat)
        keyword = 'unuhrou5'
        label = int((float(self.afib_tab.loc[self.afib_tab['mesaid'] == int(pat), keyword]) == 2))

        self.rr_dict[pat] = np.diff(ann) / self.actual_fs
        self.rrt_dict[pat] = np.concatenate(([ann[0] / self.actual_fs], np.cumsum(self.rr_dict[pat]) + ann[0] / self.actual_fs))
        self.rlab_dict[pat] = label * np.ones_like(self.rr_dict[pat])
        self.recording_time[pat] = self.rrt_dict[pat][-1]
        self.excluded_portions_dict[pat] = np.array([[0, self.rrt_dict[pat][0]]])

        for win in self.window_sizes:
            rr = self.rr_dict[pat]
            rr_reshaped = rr[:(len(rr) // win) * win].reshape(-1, win)
            n_windows, _ = rr_reshaped.shape
            rrt = self.rrt_dict[pat]
            start_windows = rrt[:-1][:(len(rr) // win) * win].reshape(-1, win)[:, 0]
            end_windows = rrt[1:][:(len(rr) // win) * win].reshape(-1, win)[:, 0]

            self.mask_rr_dict[pat][win] = np.ones(n_windows, dtype=bool)
            self.n_excluded_windows_dict[pat][win] = 0
            self.start_windows_dict[pat][win] = start_windows
            self.end_windows_dict[pat][win] = end_windows
            self.av_prec_windows_dict[pat][win] = dp.cumsum_reset(self.mask_rr_dict[pat][win])

    def parse_ahi(self, id):
        self.ahi_dict[id] = float(self.afib_tab.loc[self.afib_tab['mesaid'] == int(id), 'ahi_a0h3a'])

    def parse_odi(self, id):
        i_spo2 = np.where(np.array(self.curr_edf.getSignalLabels()) == 'SpO2')[0][0]
        n_spo2 = self.curr_edf.getNSamples()[i_spo2]
        sf_spo2 = self.curr_edf.samplefrequency(i_spo2)
        data_spo2 = self.curr_edf.readSignal(i_spo2)
        data_spo2[np.where(data_spo2 == 0)] = np.nan
        data_resamp = fc.sc_resamp(data_spo2, sf_spo2)
        data_resamp_med = fc.sc_median(data_resamp, medfilt_lg=9)
        det_desat, table_desat_aa, table_desat_bb, table_desat_cc = fc.sc_desaturations(data_resamp_med)
        total_recording_time = ((n_spo2 / sf_spo2) / cts.N_SEC_IN_MIN) / cts.N_MIN_IN_HOUR  # In hours
        self.odi_dict[id] = len(table_desat_aa) / total_recording_time  # Definition of ODI: Number of desaturations per hour

    def parse_demographic_features(self, id):
        # Age (age_s1), Hypertension (HTNDerv_s + visit), metabolic syndrome (??), BMI (bmi_s+visit), gender (gender,
        # 0 female, 1 male), necksize (neck20, only in shhs1)
        age = float(self.afib_tab[self.afib_tab["mesaid"] == int(id)]["sleepage5c"])
        gender = float(self.afib_tab[self.afib_tab["mesaid"] == int(id)]["gender1"])

        dict_dem = {"Age": age, "Gender": gender}
        for win in self.window_sizes:
            for dem_name, dem_val in dict_dem.items():
                self.features_dict[id][win][dem_name] = dem_val


if __name__ == '__main__':
    db = MESA_Parser(load_on_start=True)
    db.parse_raw_ecg(db.parsed_patients()[0])
    # pat_list = db.parse_available_ids()
    # db.parse_raw_data(patient_list=pat_list)
    # db.save_to_disk()
    # print('exporting input variables')
    # ids = db.return_patient_ids(pat_list=pat_list, exclude_low_sqi_win=False)
    # rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=pat_list, return_binary=True, exclude_low_sqi_win=False)
    # prec = db.return_preceeding_windows(pat_list=pat_list, exclude_low_sqi_win=False)
    # feats, y, glob_lab = db.return_features(pat_list=pat_list, feats_list=np.append(cts.SELECTED_FEATURES, 'sqi'),
    #                                         return_global_label=True, exclude_low_sqi_win=False)
    # sqi_feat = feats[:,-1]
    # # Process data
    # feats, mean_feats = dp.fillna(feats)
    # # Concatenate the features
    # X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    # final = tuple()
    # timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    # final = final + (X, y, timestamp, sqi_feat)
    # print('saving input variables')
    # with open(cts.REPO_DIR / 'inputs' / 'RDS_input.pickle', 'wb') as f:
    #     pickle.dump(final, f)
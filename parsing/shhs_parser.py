from base_parser import *


# TODO: write this class to fit all other parsers + add reference annotations
class SHHS_Parser(BaseParser):

    def __init__(self, window_size=60, visit=1, afib_lab_type='afib', load_on_start=True):
        """Note: In SHHS and MESA, the window labels and the rlab dictionnary are not relevant. They are based on the global
        label of the patient (binary label for AF) which can be found among the variables. The afib_lab_type argument refers to
        the type of AF label used: AFIB, AFIB incident or AFIB prevalent. AFIB has been used all along.
        """
        super(SHHS_Parser, self).__init__()
        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Missing records"""
        self.missing_ecg = np.array(['200146', '200246', '200279', '201115', '201669', '201821', '202248', '202308',
                                     '202317', '203169', '204217'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 125  # Warning ! Some files present a different sample frequency !
        self.actual_fs = cts.EPLTD_FS
        self.name = "SHHS" + str(visit)
        self.ecg_format = ".edf"
        self.visit = visit

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / 'shhs' / 'polysomnography' / 'edfs' / ('shhs' + str(self.visit))
        self.orig_anns_path = None
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        # self.filename = "SHHS" + str(visit) + ".pkl"
        self.main_path = cts.PREPROCESSED_DATA_DIR / ("SHHS" + str(self.visit))
        self.window_size = window_size

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
        # ---------------- Local variables (relevant only to SHHS) ----------------- #
        # ------------------------------------------------------------------------------- #
        """

        self.visit = visit
        self.missing_cvd_summary = np.array(['204709', '204806'])
        self.afib_lab_type = afib_lab_type
        self.afib_lab_file_path = cts.DATA_DIR / 'shhs' / 'datasets' / ('shhs' + str(self.visit) + '-dataset-0.14.0.csv')
        self.afib_inc_prev_lab_file_path = cts.DATA_DIR / 'shhs' / 'datasets' / 'shhs-cvd-summary-dataset-0.14.0.csv'
        self.new_af_label_sheetname = "Labels_SHHS_1_RoiV2.csv"     # Reannotated files by Roi Efraim, Rambam Cardiologist.
        if self.visit == 1:
            self.afib_tab = pd.read_csv(self.afib_lab_file_path)
        else:
            self.afib_tab = pd.read_csv(self.afib_lab_file_path, encoding="ISO-8859-1")
        self.afib_inc_prev_tab = pd.read_csv(self.afib_inc_prev_lab_file_path)
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
        return np.array([file[6:12] for file in os.listdir(str(self.raw_ecg_path))])

    def parse_annotation(self, pat, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id):
        file = self.raw_ecg_path / ("shhs" + str(self.visit) + "-" + id + ".edf")
        edf = pyedflib.EdfReader(str(self.raw_ecg_path / file))
        ecg_idx = np.where(np.array(edf.getSignalLabels()) == 'ECG')[0][0]
        ecg_raw = edf.readSignal(ecg_idx)
        fs = edf.getSampleFrequencies()[ecg_idx]
        ecg_resampled = signal.resample(ecg_raw, int(len(ecg_raw) * self.actual_fs / fs))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_resampled.reshape(-1, 1), fmt=['16'])
        self.curr_edf = edf
        return ecg_resampled

    def parse_raw_ecg(self, lead, patient_id, start=0, end=-1, type='epltd0', read_ann=True, ):
        edf = pyedflib.EdfReader(str(self.raw_ecg_path / ('shhs' + str(self.visit) + '-' + patient_id + self.ecg_format)))
        self.curr_edf = edf
        ecg_idx = np.where(np.array(edf.getSignalLabels()) == 'ECG')[0][0]
        ecg_raw = edf.readSignal(ecg_idx)
        Fs = np.round(edf.samplefrequency(ecg_idx))
        ecg = signal.resample(ecg_raw, int(len(ecg_raw) * cts.EPLTD_FS / Fs))
        ann = self.parse_annotation(patient_id, type=type)
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        edf.close()
        return ecg, ann

    # rlab is not relevant here. Same for af burden. Will have rlab all ones and zeros, and af burden 0 or 100%. They should not be considered.
    def parse_elem_data(self, pat):

        dicts_to_fill = [self.start_windows_dict, self.end_windows_dict, self.av_prec_windows_dict, self.n_excluded_windows_dict, self.mask_rr_dict]
        for dic in dicts_to_fill:
            if pat not in dic.keys():
                dic[pat] = {}

        ann = self.parse_annotation(pat)
        keyword = 'afib'
        if self.visit == 1:
            keyword = keyword.upper()
        if self.afib_lab_type == 'afib':
            label = float(self.afib_tab.loc[self.afib_tab['nsrrid'] == int(pat), keyword])
        elif self.afib_lab_type == 'afibprev':
            label = self.afib_inc_prev_tab.loc[self.afib_tab['nsrrid'] == int(pat), keyword]
        elif self.afib_lab_type == 'afibinc':
            label = self.afib_tab.loc[self.afib_tab['nsrrid'] == int(pat), keyword]

        self.rr_dict[pat] = np.diff(ann) / self.actual_fs
        self.rrt_dict[pat] = np.concatenate(([ann[0] / self.actual_fs], np.cumsum(self.rr_dict[pat]) + ann[0] / self.actual_fs))
        self.rlab_dict[pat] = label * np.ones_like(self.rr_dict[pat])
        self.excluded_portions_dict[pat] = np.array([[0, self.rrt_dict[pat][0]]])
        self.recording_time[pat] = self.rrt_dict[pat][-1]

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
        self.ahi_dict[id] = float(self.afib_tab.loc[self.afib_tab['nsrrid'] == int(id), 'ahi_a0h3a'])

    def parse_odi(self, id):
        i_spo2 = np.where(np.array(self.curr_edf.getSignalLabels()) == 'SaO2')[0][0]
        n_spo2 = self.curr_edf.getNSamples()[i_spo2]
        sf_spo2 = self.curr_edf.samplefrequency(i_spo2)
        data_spo2 = self.curr_edf.readSignal(i_spo2)
        data_spo2[np.where(data_spo2 == 0)] = np.nan
        data_resamp = fc.sc_resamp(data_spo2, sf_spo2)
        data_resamp_med = fc.sc_median(data_resamp, medfilt_lg=9)
        det_desat, table_desat_aa, table_desat_bb, table_desat_cc = fc.sc_desaturations(data_resamp_med)
        total_recording_time = ((n_spo2 / sf_spo2) / cts.N_SEC_IN_MIN) / cts.N_MIN_IN_HOUR  # In hours
        self.odi_dict[id] = len(table_desat_aa) / total_recording_time  # Definition of ODI: Number of desaturations per hour
    # ------------------------------------------------------------------------------------ #

    def reannotate_af_label(self):
        """ This function sets new labels to the recordings re-annotated by the peer cardiologist."""
        new_labels_sheet = pd.read_csv(cts.ERROR_ANALYSIS_DIR / "Reannotation-SHHS" / self.new_af_label_sheetname)
        for i, pat in enumerate(new_labels_sheet['Patient']):
            if str(pat) in self.rr_dict.keys():
                is_af = np.logical_and(new_labels_sheet['Label'][i] >= cts.PATIENT_LABEL_AF_MILD, new_labels_sheet['Label'][i] <= cts.PATIENT_LABEL_AF_SEVERE)
                self.rlab_dict[str(pat)] = is_af * np.ones_like(self.rlab_dict[str(pat)])
                for win in self.window_sizes:
                    self._win_lab(str(pat), win)
                    self._af_win_lab(str(pat), win)
                self.af_pat_lab_dict[str(pat)] = new_labels_sheet['Label'][i]
                self.af_burden_dict[str(pat)] = float(is_af) # Not really significant, as we don't truly have the af_burden. This is just for consistency.

    def label_hist(self):
        labels_af = np.array([self.af_pat_lab_dict[elem] for elem in self.all_patients()])
        labels_af = labels_af[~np.isnan(labels_af)]
        labels_af = np.array([np.sum(labels_af == i) for i in [cts.PATIENT_LABEL_NON_AF, cts.PATIENT_LABEL_AF_SEVERE]]) # Patients have AFB of 0 or 100%, i.e. they're considered or as non-AF or as Severe.
        fig, axes = graph.create_figure(tight_layout=False)
        axes[0][0].bar(np.array([1, 2]), labels_af, tick_label=['$NSR$', '$AF$'], width=0.2)
        axes[0][0].text(1.0, 2900, '$p=' + str(labels_af[0]) + '$', ha='center', va='bottom', fontsize=24)
        axes[0][0].text(2.0, 71, '$p=' + str(labels_af[1]) + '$', ha='center', va='bottom', fontsize=24)
        graph.complete_figure(fig, axes, main_title="Label histogram SHHS", savefig=True, put_legend=[[False]],
                              x_titles=[['Rhythm types']], y_titles=[["Count"]])
        plt.show()

    def parse_demographic_features(self, id):
        # Age (age_s1), Hypertension (HTNDerv_s + visit), metabolic syndrome (??), BMI (bmi_s+visit), gender (gender, 1 male, 2 female), necksize (neck20, only in shhs1)
        age = float(self.afib_tab[self.afib_tab["nsrrid"] == int(id)]["age_s" + str(self.visit)])
        hypertension = float(self.afib_tab[self.afib_tab["nsrrid"] == int(id)]["HTNDerv_s" + str(self.visit)])
        bmi = float(self.afib_tab[self.afib_tab["nsrrid"] == int(id)]["bmi_s" + str(self.visit)])
        gender = float(self.afib_tab[self.afib_tab["nsrrid"] == int(id)]["gender"])
        if self.visit == 1:
            necksize = float(self.afib_tab[self.afib_tab["nsrrid"] == int(id)]["NECK20"])
        else:
            path = cts.DATA_DIR / 'shhs' / 'datasets' / 'shhs1-dataset-0.14.0.csv'
            tab = pd.read_csv(path)
            necksize = float(tab[tab["nsrrid"] == int(id)]["NECK20"])
        dict_dem = {"Age": age, "Hypertension": hypertension, "BMI": bmi, "Gender": gender, "necksize": necksize}
        for win in self.window_sizes:
            for dem_name, dem_val in dict_dem.items():
                self.features_dict[id][win][dem_name] = dem_val


if __name__ == '__main__':

    db = SHHS_Parser(visit=1, load_on_start=False)
    db.load_from_disk()
    db.create_pool()
    for pat in db.parsed_patients():
        db._sqi(pat, 60, 'xqrs')
        np.save(db.main_path / pat / 'signal_quality' / str(60), db.signal_quality_dict[pat][60])
        np.save(db.main_path / pat / 'features' / str(60), db.features_dict[pat][60])

    #
    # for pat in to_load:
    #     for win in cts.BASE_WINDOWS:
    #         np.save(db.main_path / pat / 'signal_quality' / str(win), db.signal_quality_dict[pat][win])
    #         np.save(db.main_path / pat / 'features' / str(win), db.features_dict[pat][win])

from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class UVAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True, load_ectopics=True, windows_shifted=False):

        super(UVAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes --------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Missing records"""
        self.missing_ecg = np.array(['0280', '0308'])

        """ Helper variables"""
        self.window_size = window_size
        self.windows_shifted = windows_shifted

        """Variables relative to the ECG signals."""
        self.orig_fs = cts.EPLTD_FS
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 3
        self.ref_lead = 1
        self.name = "UVAFDB"
        self.ecg_format = "rf"

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR
        self.orig_anns_path = None
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / (self.name + ("_shifted" if windows_shifted else ""))

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        self.window_size = window_size
        test_pat = self.parsed_patients()[0]
        self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "mask_rr")])
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)
        self.is_ectopic = {}
        self.n_ectopics = {}

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only to UVAFDB) -------------------- #
        # ------------------------------------------------------------------------------- #
        """
        # Like the SQI step, the rate of missing annotations below which the patient is excluded.
        self.beats_bea = np.array(['NORMAL', 'PVC', 'APC', 'AESC', 'VESC', 'PACE', 'PFUS',
                                   # The different rhythms present across the dataset.
                                   'UNKNOWN', 'UNCLASS', 'SUBTYPE', 'RHYTHM', 'AUX', 'SUB', 'ARFCT',
                                   'VFON', 'FLWAV', 'VFOFF', 'RONT',
                                   'FUSION'])
        self.bea_path = self.raw_ecg_path / "uvfdb_rr" / "BEA"
        self.excel_sheet_path = self.raw_ecg_path / "uvfdb_rr" / "UVA Holter Info.xlsx"
        self.excel_sheet = pd.read_excel(self.excel_sheet_path)

        # Quick and dirty
        if load_ectopics:
            self.load_ectopics([self.window_size])

        self.excel_sheet['age_at_recording'] = (self.excel_sheet["Age (days)"] + self.excel_sheet[
            "Days from First"]) // 365
        self.over_18_patients = np.array(self.excel_sheet[self.excel_sheet["age_at_recording"] >= 18]
                                         ["Holter ID"].apply(lambda x: x[3:])).astype('<U32')

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        return np.array([file[3:7] for file in os.listdir(str(self.raw_ecg_path)) if file != "uvfdb_rr"])

    def parse_reference_annotation(self, id, combine=True): #, reannotated=False):
        _, _, _, beats, _, _, rhythm, _ = self.readbea(self.bea_path / ("UVA" + id + '.bea'))
        beats = ((beats / cts.N_MS_IN_S) * self.actual_fs).astype(int)
        # tbeats = beats / self.actual_fs
        # ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        # if reannotated:
        #     rhythm_df = self.parse_reference_rhythm(id)
        #     for index, l in rhythm_df.iterrows():
        #         l1 = np.abs(tbeats - l.Beginning)
        #         l2 = np.abs(tbeats - l.End)
        #         begin = np.where(l1 == l1.min())
        #         end = np.where(l2 == l2.min())
        #         ltbeats[int(begin[0][0]):int(end[0][0])] = l.Class
        #     rhythm = np.array([cts.rhythms_dict[i] for i in ltbeats])
        #     if combine:
        #         rhythm[rhythm == cts.rhythms_dict['AFL']] = cts.rhythms_dict['AFIB']
        # else:
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return beats, rhythm

    def parse_annotation(self, id, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead):
        file = self.raw_ecg_path / ("UVA" + id + ".rf")
        record = self._read_rf(file, lead=lead - 1)
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=record.reshape(-1, 1), fmt=['16'])
        return record

    def parse_raw_ecg(self, id, lead, start=0, end=-1, type='epltd0'):
        ecg = self._read_rf(self.raw_ecg_path / ('UVA' + id + '.rf'), lead=lead - 1)
        ann = self.parse_annotation(id, type=type, lead=lead)
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        if self.windows_shifted:
            ann = ann[self.window_size // 2:]
        return ecg, ann

    def parse_demographic_features(self, id):
        age = float(self.excel_sheet[self.excel_sheet["Holter ID"] == "UVA" + id]["age_at_recording"])
        sex = float(
            self.excel_sheet[self.excel_sheet["Holter ID"] == "UVA" + id]["Gender"] == 'F')  # True: Female, False: Male
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = age
            self.features_dict[id][win]['Sex'] = sex

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        return self.excel_sheet[self.excel_sheet["Holter ID"] == "UVA" + recording_id]["Patient ID"]

    # TODO: create clinical_lab dict
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

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def load_ectopics(self, wins=None):
        """ This function loads the number of ectopic beats per window for the UVAF dataset.
            :param wins: The windows for which the number of ectopics should be loaded.
        """
        print("Loading ectopics")
        if wins is None:
            wins = self.window_sizes
        for pat in self.parsed_patients():
            self.n_ectopics[pat] = {}
            for win in wins:
                self.n_ectopics[pat][win] = np.load(self.main_path / pat / "n_ectopics" / (str(win) + ".npy"))

    # TODO: move to a new script
    def generate_beats_hist(self, figsize=(15, 15), remove_N=True):
        """ This function generates a bar plot with the number of beats for the most represented rhythms in the dataset."""
        max_y = 0.5 * 1e8
        jump = 2
        beats_threshold = 1e6
        self.extract_reann_pat()
        beats_table_reann, beats_table_not_reann = np.zeros((len(self.reann_pat), len(self.rhythms)),
                                                            dtype=int), np.zeros(
            (len(self.not_reann_pat), len(self.rhythms)), dtype=int)
        pat_table_reann, pat_table_not_reann = np.zeros((len(self.reann_pat), len(self.rhythms)), dtype=int), np.zeros(
            (len(self.not_reann_pat), len(self.rhythms)), dtype=int)

        for i, pat in enumerate(self.reann_pat):
            rhythms, counts = np.unique(self.rlab_dict[pat], return_counts=True)
            for j, rhy in enumerate(rhythms):
                beats_table_reann[i, int(rhy)] = counts[j]
                pat_table_reann[i, int(rhy)] = 1

        for i, pat in enumerate(self.not_reann_pat):
            rhythms, counts = np.unique(self.rlab_dict[pat], return_counts=True)
            for j, rhy in enumerate(rhythms):
                beats_table_not_reann[i, int(rhy)] = counts[j]
                pat_table_not_reann[i, int(rhy)] = 1

        beats_per_rhythm_reann = np.sum(beats_table_reann, axis=0)
        beats_per_rhythm_not_reann = np.sum(beats_table_not_reann, axis=0)

        beats_table = np.concatenate((beats_table_reann, beats_table_not_reann), axis=0)
        pat_table = np.concatenate((pat_table_reann, pat_table_not_reann), axis=0)
        patients_per_rhythm = np.sum(pat_table, axis=0)
        beats_per_rhythm = np.sum(beats_table, axis=0)

        idx_sort = np.argsort(beats_per_rhythm)[::-1]
        beats_per_rhythm = beats_per_rhythm[idx_sort]
        rhythms = self.rhythms[idx_sort]
        beats_per_rhythm_reann = beats_per_rhythm_reann[idx_sort]
        beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[idx_sort]
        patients_per_rhythm = patients_per_rhythm[idx_sort]

        mask = beats_per_rhythm.copy() > beats_threshold
        beats_per_rhythm = beats_per_rhythm[mask]
        patients_per_rhythm = patients_per_rhythm[mask]

        beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[mask]
        beats_per_rhythm_reann = beats_per_rhythm_reann[mask]

        rhythms = rhythms[mask]
        beats_per_rhythm_disp = ['{:.1e}'.format(float(x)) for x in beats_per_rhythm]

        # Scaling first element ((N)
        beats_per_rhythm_reann[0] = beats_per_rhythm_reann[0] * max_y / beats_per_rhythm[0]
        beats_per_rhythm_not_reann[0] = beats_per_rhythm_not_reann[0] * max_y / beats_per_rhythm[0]

        if remove_N:
            beats_per_rhythm_reann = beats_per_rhythm_reann[1:]  # Removing the (N class
            beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[1:]
            rhythms = rhythms[1:]
            beats_per_rhythm = beats_per_rhythm[1:]
            beats_per_rhythm_disp = beats_per_rhythm_disp[1:]
            patients_per_rhythm = patients_per_rhythm[1:]
            max_y = beats_per_rhythm_reann[0] + beats_per_rhythm_not_reann[0] + 2000000
        # Plotting Beats repartition
        fig, axes = graph.create_figure(figsize=figsize)
        p1 = axes[0][0].bar(np.arange(0, 2 * len(rhythms), jump), beats_per_rhythm_not_reann,
                            tick_label=[r[1:] for r in rhythms],
                            label='No manual correction')
        p2 = axes[0][0].bar(np.arange(0, 2 * len(rhythms), jump), beats_per_rhythm_reann,
                            tick_label=[r[1:] for r in rhythms],
                            bottom=beats_per_rhythm_not_reann, label='Manually corrected')
        for i, rhy in enumerate(rhythms):
            if 0 < beats_per_rhythm[i] < max_y:
                plt.text(jump * i - 0.5 + 0.1, beats_per_rhythm[i] + 1000000,
                         'n=' + beats_per_rhythm_disp[i] + ',\np=' + str(patients_per_rhythm[i])
                         , fontsize=24)
            elif beats_per_rhythm[i] > max_y:
                plt.text(jump * (i + 1) - 1, 0.9 * max_y,
                         'n=' + beats_per_rhythm_disp[i] + ',\np=' + str(patients_per_rhythm[i])
                         , fontsize=24)
        graph.complete_figure(fig, axes, savefig=True, x_titles=[['Rhythm types']], y_titles=[['Count']],
                              y_lim=[[[0, max_y]]], main_title='UVAFDB_beats_distribution')

    # TODO: move to a new script
    def generate_events_hist(self):
        """ This function generates a histogram of events lengths per patient label category across the dataset."""
        AF_events_lengths = [[], [], [], []]
        for pat in self.all_patients():
            is_af = (self.rlab_dict[pat] == cts.WINDOW_LABEL_AF).reshape(-1)
            is_af = np.array([0, *is_af.tolist(), 0])
            starts = np.where(np.diff(is_af) == 1)[0] + 1
            ends = np.where(np.diff(is_af) == -1)[0]
            label = self.af_pat_lab_dict[pat]
            if label == cts.PATIENT_LABEL_OTHER_CVD:
                label = cts.PATIENT_LABEL_NON_AF  # Merging Other CVD with Non-AF
            AF_events_lengths[label].extend((ends - starts).tolist())
        fig, axes = graph.create_figure(subplots=(1, 2))
        all_data = np.concatenate(tuple(AF_events_lengths))
        print("Number of events: " + str(len(all_data)))
        print("Number of events smaller than 60 seconds: " + str(np.sum(all_data < 60)))
        max_val = np.round(np.percentile(all_data, 98))
        max_val += (500 - max_val % 500)
        bins_first = np.arange(0, 60, 10)
        bins_second = np.arange(100, max_val, 500)
        bins = np.concatenate((bins_first, bins_second))

        for i, lab in enumerate([cts.PATIENT_LABELS[1], cts.PATIENT_LABELS[0]]):
            data = AF_events_lengths[lab]
            hist, bin_edges = np.histogram(data, bins)
            axes[0][0].bar(range(0, len(hist) * 2, 2), hist, width=2 * (1 - i / len(cts.PATIENT_LABELS)),
                           color=cts.COLORS[lab],
                           label=cts.GLOBAL_LAB_TITLES[lab])

        for i, lab in enumerate(cts.PATIENT_LABELS[2:4]):
            data = AF_events_lengths[lab]
            hist, bin_edges = np.histogram(data, bins)
            axes[0][1].bar(range(0, len(hist) * 2, 2), hist, width=2 * (1 - i / len(cts.GLOBAL_LAB_TITLES)),
                           color=cts.COLORS[lab],
                           label=cts.GLOBAL_LAB_TITLES[lab])

        graph.complete_figure(fig, axes, xticks_fontsize=12,
                              x_ticks=[[[2 * (0.5 + i) for i, j in enumerate(hist)]] * 2],
                              x_ticks_labels=[[['%d' % (bins[i + 1]) for i, j in enumerate(hist)]] * 2],
                              x_titles=[['Events lengths (in number of beats)'] * 2], xlabel_fontsize=20,
                              y_titles=[['Count', '']], savefig=True, main_title='UVAFDB_Events_lengths')

    # TODO: move to a new script
    def generate_af_burden_hist(self, figsize=(15, 10)):
        """ This function generates a histogram of the AF burden per patient label category across the dataset."""
        AF_Burdens = [[], [], [], []]
        for pat in self.af_burden_dict.keys():
            lab = self.af_pat_lab_dict[pat]
            if lab == cts.PATIENT_LABEL_OTHER_CVD:
                lab = cts.PATIENT_LABEL_NON_AF  # Merging Other CVD with Non-AF
            AF_Burdens[lab].append(self.af_burden_dict[pat])

        fig, axes = graph.create_figure(figsize=figsize)
        labels = np.array([cts.PATIENT_LABEL_AF_MILD, cts.PATIENT_LABEL_AF_MODERATE, cts.PATIENT_LABEL_AF_SEVERE])
        bins = np.append([0, cts.AF_MODERATE_THRESHOLD * 100], np.arange(10, 101, 5))
        rwidths = np.array([1, 1, 1, 1])[::-1]
        n_points = np.sum([len(x) for x in AF_Burdens[1:]])
        for i, lab in enumerate(labels[::-1]):
            data = AF_Burdens[lab]
            weights = np.ones_like(data) / n_points
            axes[0][0].hist(np.array(data) * 100,
                            bins=bins, label=cts.GLOBAL_LAB_TITLES[lab],
                            color=cts.COLORS[lab], rwidth=rwidths[i], weights=weights)
        graph.complete_figure(fig, axes, x_titles=[['AF Burden (%)']], y_titles=[['Proportion of AF Patients']],
                              xlim=[[[0, 1]]],
                              savefig=True, main_title='UVAFDB_AF_Burden_hist', x_lim=[[[0, 100]]])

    # TODO: move to a new script
    def generate_features_hist(self, feats_names=cts.SELECTED_FEATURES):
        """ This function generates a histogram of features per patient label category across the dataset.
        :param feats_names: The list of features to include in the histograms subplot."""
        n_feats_per_plot = 8
        n_rows = 4
        n_cols = 2
        n_bins = 40
        X, y = self.return_features(feats_list=feats_names, return_binary=False)
        y[y >= cts.WINDOW_LABEL_OTHER] = cts.WINDOW_LABEL_OTHER
        n_plots = (X.shape[1] // n_feats_per_plot) + 1
        for k in range(n_plots):
            fig, axes = graph.create_figure(subplots=(n_rows, n_cols), figsize=(30, 15))

            start, end = k * n_feats_per_plot, min((k + 1) * n_feats_per_plot, len(feats_names))
            for n, idx in enumerate(np.arange(start, end, 1)):
                i, j = n // n_cols, n % n_cols
                lim1 = 0.9 * np.percentile(X[:, idx], 1)
                lim2 = 1.1 * np.percentile(X[:, idx], 98)
                bins = np.linspace(lim1, lim2, n_bins)
                for lab in cts.WINDOW_LABELS[:-1]:  # One label in the plot
                    weights = np.ones_like(X[y == lab, idx]) / float(
                        len(X[y == lab, idx]))
                    axes[i][j].hist(X[y == lab, idx],
                                    bins=bins, rwidth=1 - lab / (2 * len(cts.WINDOW_LAB_TITLES)),
                                    label=cts.WINDOW_LAB_TITLES[lab], color=cts.COLORS[lab],
                                    weights=weights)

            put_legend = np.zeros((n_rows, n_cols), dtype=bool)
            put_legend[0][1] = True
            fig.subplots_adjust(left=0.05)
            fig.subplots_adjust(bottom=0.05)
            if end - start == n_feats_per_plot:
                x_titles = np.array(feats_names[start:end]).reshape(n_rows, n_cols)
            else:
                x_titles = '' * np.ones((n_rows, n_cols), dtype=object)
                i, j = 0, 0
                for n, idx in enumerate(np.arange(start, end, 1)):
                    x_titles[i, j] = feats_names[idx]
                    j = (j + 1) % n_cols
                    i = (n + 1) // n_rows
            graph.complete_figure(fig, axes, x_titles=x_titles,
                                  y_titles=[['Density', '']] * n_rows, legend_fontsize=18, ylabel_fontsize=18,
                                  xlabel_fontsize=16,
                                  main_title='UVAFDB_features_' + str(self.window_size) + '_beats_plot_number_' + str(
                                      k),
                                  savefig=True, put_legend=put_legend, xticks_fontsize=16, yticks_fontsize=16)

    def extract_reann_pat(self):
        """ This function looks into the Excel report present in the directory of the .BEA files to report which
        patients have been reannotated. """
        # First extracting reannotated patients from UVA Info file
        res = pd.read_excel(self.excel_sheet_path)
        self.reann_pat = np.setdiff1d(np.array(res[res['Comments'].notnull()]['Holter ID'].apply(lambda x: x[-4:])),
                                      self.missing_ecg)
        self.not_reann_pat = np.setdiff1d(np.array(res[res['Comments'].isnull()]['Holter ID'].apply(lambda x: x[-4:])),
                                          self.missing_ecg)
        self.reann_pat = np.intersect1d(self.reann_pat, self.parsed_ecgs)
        self.not_reann_pat = np.intersect1d(self.not_reann_pat, self.parsed_ecgs)

    def _read_rf(self, file, lead):
        """ This function reads the raw ECG files, which are given in an encoded (.rf) format.
        :param file: The path to the .rf file.
        :param lead: ECG lead."""
        n_chans = 3
        n_bits_per_chan = 10
        f = open(self.raw_ecg_path / file, "rb")
        A = np.fromfile(f, dtype=np.uint32)
        masks_abs_val = [0x1ff, 0x7fc00, 0x1ff00000]
        masks_sign = [0x200, 0x80000, 0x20000000]
        ecgs = np.zeros((len(A), n_chans))
        ecgs_sign = np.zeros((len(A), n_chans))
        for i in range(n_chans):
            ecgs[:, i] = np.bitwise_and(A, masks_abs_val[i]) >> n_bits_per_chan * i
            ecgs_sign[:, i] = np.bitwise_and(A, masks_sign[i]) >> (n_bits_per_chan * (i + 1) - 1)
            ecgs[ecgs_sign[:, i] == 1, i] -= 2 ** (n_bits_per_chan - 1)
        Vptp = 5.0
        ecgs *= (Vptp / 2 ** n_bits_per_chan)  # Conversion from A/D value to [mV]
        return ecgs[:, lead - 1]

    def bea_spliter(self, line):
        """ Helper function the read properly the different lines in the .bea files which contain the annotations from the Holter.
        :param line: The input line belonging to the .bea file.
        :returns res: The line with an extension for the rhythm in case it was missing."""
        res = line.split()
        if len(res) == 2:
            res.append(None)
        return res

    # Need to return (beat, rhythm) to have the actual annotation. The rest can be derived from there.
    def readbea(self, beafile, remove_special=False):
        """ Reads a bea file and returns the different important elements.
        :param beafile: The path to the .bea file.
        :param remove_special: If True, removes some of the beats with a given label according to Lake&Moorman's convention. If False, leaves the beats as-is.
        :returns rr: The RR intervals.
        :returns rlab: The labels corresponding to the RR intervals.
        :returns rrt: The timestamp corresponding to each RR interval.
        """
        try:
            with open(beafile, 'r') as file:
                lines = np.array(file.readlines())
                splited = np.array(list(map(self.bea_spliter, lines)))
                splited[:, 0] = splited[:, 0].astype(int)
                beat = splited[:, 0]
                label = splited[:, 1]
                rhythm = splited[:, 2]
                not_none = np.where(rhythm != None)[0]
                not_none = np.append(not_none, len(rhythm))
                diffs = np.diff(not_none)
                sing_rhy = rhythm[not_none[:-1]]
                rhythm[not_none[0]:] = np.repeat(sing_rhy, diffs)
                rhythm[0:not_none[0]] = sing_rhy[0]
                rhythm_lab = np.array(list(map(lambda x: self.rhythms_dict[x], rhythm))).reshape(-1, 1)
        except IOError:
            sys.exit(1)

        bnum = [0, 2, 3, 4, 5, 6, 7, 8, 8, 12, 10, 11, 12, 13, 20, 21, 22, 9, 9]
        # bnum = [0, 2, 3, 4, 5, 6, 7, 8, 8, 9,  10, 11, 12, 13, 20, 21, 22, 9, 9] # Correct version. Currently using the same version as Lake and Moorman
        bnum_dict = {self.beats_bea[i]: bnum[i] for i in range(len(bnum))}
        lab = np.array(list(map(lambda x: bnum_dict[x], label))).reshape(-1, 1)

        if remove_special:
            mask_good = (lab < 10).reshape(-1)
            if np.sum(mask_good) >= 2:
                good_beats = beat[mask_good]
                rr = np.diff(good_beats)
                rrt = good_beats[0] + np.cumsum(rr)
                rlab = rhythm_lab[mask_good][1:]  # [1:] to select the second extremity of the RR interval for the beat
        else:
            rr = np.diff(beat)
            rrt = beat[0] + np.cumsum(rr)
            rlab = rhythm_lab[1:]

        mask = (lab == 11).reshape(-1)  # Label AUX
        rtime = beat[mask]

        # For lab, converting to binary label only: True for Ectopic (APC, PVC), False otherwise
        lab = lab.reshape(-1)
        lab = np.logical_or(lab == 1, lab == 2)
        return rr, rrt, rlab, beat, lab, label, rhythm, rtime


if __name__ == "__main__":
    # db = UVAFDB_Parser(window_size=60, load_ectopics=False, load_on_start=True, windows_shifted=False)
    db = UVAFDB_Parser(window_size=60, load_ectopics=False)
    # train_pat_ = np.load(cts.REPO_DIR / "data" / "splits" / "ids" / (db.name + "_train_pat.npy"), allow_pickle=True)
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve

    # val_pat_ = np.load(cts.REPO_DIR / "data" / "splits" / "ids" / (db.name + "_val_pat.npy"), allow_pickle=True)
    # pat = np.append(train_pat_, val_pat_)
    # df_model_2 = pd.DataFrame(columns ={'id', 'proba'})
    # df1 = pd.DataFrame.from_dict(db.features_dict[p][60]['AFEv'])
    # df1.rename(columns={0:'proba'}, inplace=True)
    # df1['id'] = p
    # df_model_2 = df_model_2.append(df1, ignore_index=True)
    # df_model_2.to_csv(cts.REPO_DIR / 'output' / 'Lorenz_train_pred.csv', index=False)
    # print("Accuracy: " + str(round(accuracy, 2)))
    # print("F" + str(beta) + "-Score: " + str(round(fbeta, 2)))
    # print("Sensitivity: " + str(round(sensitivity, 2)))
    # print("Specificity: " + str(round(specificity, 2)))
    # print("PPV: " + str(round(PPV, 2)))
    # print("NPV: " + str(round(NPV, 2)))
    # UVAF_db.extract_reann_pat()
    # reann_pat = UVAF_db.reann_pat
    # not_reann_pat = UVAF_db.not_reann_pat
    #
    # # Exclude low sqi and under 18 patients
    # baseline = db.non_corrupted_ecg_patients()
    # set_ids = np.load(cts.REPO_DIR / 'data/splits/ids/UVAFDB_test_pat.npy', allow_pickle=True)
    #
    # db.generate_annotations(pat_list=set_ids, lead=1, types=['jqrs'])

    # baseline = np.intersect1d(baseline, db.high_sqi_patients())
    # baseline = np.intersect1d(baseline, UVAF_db.over_18_patients)
    # for i, id in enumerate(baseline):
    #     UVAF_db.parse_demographic_features(id)
    #
    # # Divide patients between reannotated and not
    # reann_pat = np.intersect1d(reann_pat, baseline)
    # not_reann_pat = np.intersect1d(not_reann_pat, baseline)
    # n_pat = len(reann_pat) + len(not_reann_pat)
    #
    # # Split reannotated patients with stratification
    # test_part = 0.2
    # # reann_pat_labels = list(UVAF_db.af_pat_lab_dict[elem] for elem in reann_pat)
    # reann_pat_afb = list(UVAF_db.af_burden_dict[elem] for elem in reann_pat)
    # strat_bins_afb = [-1, 0, 0.04, 0.8, 1]
    # strat_afb = np.digitize(reann_pat_afb, strat_bins_afb, right=True)  # right = True -> 0 is isolated as one
    # reann_pat_age = [UVAF_db.features_dict[id][60]['Age'] for id in reann_pat]
    # strat_bins_age = [0, 30, 50, 70, 100]
    # strat_age = np.digitize(reann_pat_age, strat_bins_age, right=True)
    # strat_gender = [UVAF_db.features_dict[id][60]['Gender'] for id in reann_pat]
    # for i, bin_afb in enumerate(strat_bins_afb):
    #     for j, bin_age in enumerate(strat_bins_age):
    #         print(bin_afb)
    #         print(bin_age)
    #         print(np.sum(np.logical_and(strat_afb == i + 1, strat_age == j + 1)))
    #         print()
    # train_pat_reann, test_pat = train_test_split(reann_pat, test_size=int(test_part * n_pat),
    #                                              stratify=np.vstack((strat_afb, strat_age, strat_gender)).T, random_state=cts.SEED)
    # import matplotlib.pyplot as plt
    # plt.hist([UVAF_db.features_dict[id][60]['Age'] for id in baseline], label='baseline_data')
    # plt.hist([UVAF_db.features_dict[id][60]['Age'] for id in train_pat_reann], label='train_pat_reann')
    # plt.hist([UVAF_db.features_dict[id][60]['Age'] for id in test_pat], label='test_pat')
    # plt.title('Age')
    # plt.legend()
    # plt.show()
    # plt.hist([UVAF_db.features_dict[id][60]['Gender'] for id in baseline], label='baseline_data')
    # plt.hist([UVAF_db.features_dict[id][60]['Gender'] for id in train_pat_reann], label='train_pat_reann')
    # plt.hist([UVAF_db.features_dict[id][60]['Gender'] for id in test_pat], label='test_pat')
    # plt.title('Gender')
    # plt.legend()
    # plt.show()
    # # afb_list = list(db.af_burden_dict.values())
    # # bins = [-1] + list(np.linspace(start=0, stop=1, num=20))  # So that first bin contains only non_AF patients 30 s / 24 h = 0.0003
    # # y_binned = np.digitize(afb_list, bins, right=True)
    # # print(afb_list)
    # # print(list(y_binned))

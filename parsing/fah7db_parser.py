from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)

class FAH7DB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True, ):

        super(FAH7DB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes --------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Missing records"""
        self.missing_ecg = np.array(['ecg_PatAF09'])

        """ Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 128
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.ref_lead = 1
        self.name = "FAH7DB"
        self.ecg_format = ".mat"

        """Variables relative to the different paths"""
        # TODO: move database to MLAIM/databases (?)
        self.raw_ecg_path = cts.BASE_DIR / "AIMLab" / "Shany" / 'databases' / "basal"
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
        # ---------------- Local variables (relevant only to FAH7DB) --------------- #
        # ------------------------------------------------------------------------------- #
        """

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        return np.array([dir for dir in os.listdir(str(self.raw_ecg_path))])

    def parse_reference_annotation(self, id, combine=True, reannotated=False):
        # No rhythms available
        _, ann = self.parse_raw_ecg(id, type=self.sqi_ref_ann)
        ltbeats = np.array(['NSR' for i in ann]).astype(object)
        rhythm = np.array([cts.rhythms_dict[i] for i in ltbeats])
        return ann, rhythm

    def parse_annotation(self, id, type="epltd0", lead=1):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead=1):
        file = self.raw_ecg_path / id / (id + "_" + str(lead) + self.ecg_format)
        loaded = sio.loadmat(file, struct_as_record=True)
        record = loaded['ecg'].flatten().flatten()
        re_record = bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs/2 - 0.5,
                                    signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)
        re_record = dp.resample_by_interpolation(re_record, self.orig_fs, self.actual_fs)
        re_record = re_record / 1000
        re_record = re_record[self.start_ann[id]:]
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'])
        return re_record

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type='epltd0', correct_peaks=True, filter_signal=True, read_ann=True, ):
        file = self.raw_ecg_path / patient_id / (patient_id + "_" + str(lead) + self.ecg_format)
        loaded = sio.loadmat(file, struct_as_record=True)
        record = loaded['ecg'].flatten().flatten()
        if filter_signal:
            record = dp.bandpass_filter(data=record, id=patient_id, lead='x', lowcut=0.67, highcut=self.orig_fs/2 - 0.5,
                                        signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)
        record = dp.resample_by_interpolation(record, self.orig_fs, self.actual_fs)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        record = record[start_sample:end_sample]
        if read_ann:
            ann = self.parse_annotation(patient_id, type=type, lead=lead)
            if correct_peaks:
                ann = i_o.qrs_adjust_detector(ecg=record, qrs=ann, fs=self.actual_fs, INPUTSIGN=1, n_windows= 2000)
            ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
            ann -= start_sample
            return record, ann
        else:
            return record

    def parse_demographic_features(self, id):
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = np.nan
            self.features_dict[id][win]['Sex'] = np.nan

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        return recording_id
    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """


if __name__ == "__main__":
    # db = UVAFDB_Parser(window_size=60, load_ectopics=False, load_on_start=True, windows_shifted=False)
    db = FAH7DB_Parser(load_on_start=False, window_size=60, load_ectopics=False)
    savedir = pathlib.PurePath('/home/shanybiton/repos/CircadianAF/output') / db.name.lower()
    st = [22200, 26000, 13600, 21600, 16700, 12100, 30500, 280000, 21300, 2000, 19300, 18250, 15000, 18750, 55000, 19200,
          18000, 105000, 22300]
    ids = db.parse_available_ids()
    ids_ = ids[~np.isin(db.parsed_patients(), db.missing_ecg)]
    db.start_ann = {ids_[i]: st[i] for i in range(len(ids_))}
    # db.generate_annotations(types=db.sqi_ref_ann, pat_list=ids_, lead=2)
    # db.parse_raw_data(patient_list=ids_)
    # db.save_to_disk()
    # for id in ids:
    #     rep_ids = db.return_patient_ids(pat_list=[id], exclude_low_sqi_win=False)
    #     rr, rrt, y, win_start, win_end, win_lab = db.return_rr(pat_list=[id], return_binary=True, exclude_low_sqi_win=False)
    #     prec = db.return_preceeding_windows(pat_list=[id], exclude_low_sqi_win=False)
    #     feats, y, glob_lab = db.return_features(pat_list=[id], feats_list=np.append(cts.SELECTED_FEATURES, 'sqi'),
    #                                             return_global_label=True, exclude_low_sqi_win=False)
    #     sqi_feat = feats[:,-1]
    #     medHR_feat = feats[:, 7]
    #     # Process data
    #     feats, mean_feats = dp.fillna(feats)
    #     # Concatenate the features
    #     X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), rep_ids.reshape(-1, 1)), axis=1)
    #     final = tuple()
    #     timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    #     final = final + (X, y, timestamp, sqi_feat, medHR_feat)
    #     print('saving input variables')
    #     with open(savedir / (id + '_input.pickle'), 'wb') as f:
    #         pickle.dump(final, f)
    #
    # times = pd.read_excel('/MLAIM/AIMLab/Shany/databases/fah7db/times.xlsx')
    # times['recording length'] = 0
    # for id in ids:
    #     pat_id = id.split('_')[1]
    #     times.loc[times['Unnamed: 0'].eq(pat_id), 'recording length'] = db.recording_time[id]
    # times.to_excel('/MLAIM/AIMLab/Shany/databases/fah7db/times_2.xlsx', index=False)

    # directory = pathlib.PurePath('/home/shanybiton/repos/CircadianAF/error_analysis/')
    # ecg_header = ['---\n',
    #               'Mammal:            human\n',
    #               'Fs:                ' + str(cts.EPLTD_FS) + '\n',
    #               'Integration_level: electrocardiogram\n',
    #               '\n'
    #               'Channels:\n']
    # channels = [['\n'
    #              '    - type:   electrography\n',
    #              '      name:   data' + str(i) + '\n',
    #              '      unit:   mV\n',
    #              '      enable: yes\n'] for i in range(1, db.n_leads + 1)]
    # end_header = ['\n',
    #               '---\n',
    #               '\n']
    # [ecg_header.extend(channels[i]) for i in range(db.n_leads)]
    # ecg_header.extend(end_header)
    #
    # peaks_header = ['---\n',
    #                 'Mammal:            human\n',
    #                 'Fs:                ' + str(db.actual_fs) + '\n',
    #                 'Integration_level: electrocardiogram\n',
    #                 '\n'
    #                 'Channels:\n',
    #                 '\n'
    #                 '    - type:   peak\n',
    #                 '      name:   interval\n',
    #                 '      unit:   index\n',
    #                 '      enable: yes\n',
    #                 '\n',
    #                 '---\n',
    #                 '\n'
    #                 ]
    #
    # sig_qual_header = ['---\n',
    #                    'type: quality annotation\n',
    #                    'source file: ',  # To be completed by filename
    #                    '\n',
    #                    '---\n',
    #                    '\n',
    #                    'Beginning\tEnd\t\tClass\n']
    #
    # rhythms_header = copy.deepcopy(sig_qual_header)
    # rhythms_header[1] = 'type: rhythms annotation\n'
    # for pat_id in ids_:
    #     pat = pat_id.split('_')[1]
    #     if not os.path.exists(directory/pat):
    #         os.makedirs(directory/pat)
    #     _, ann = db.parse_raw_ecg(pat_id, start=0, end=-1, type='epltd0', lead=1)
    #     ecgs = np.concatenate(tuple([db.parse_raw_ecg(pat_id, start=0, end=-1, type='epltd0', lead=lead,
    #                                                   correct_peaks=False)[0].reshape(-1, 1) for
    #                                  lead in range(1, db.n_leads+1)]), axis=1)
    #
    #     df = pd.read_csv(savedir / ('circAF_df_' + pat + '.csv'))
    #     start_event = df.loc[df.pred.eq(True) & df.sqi.ge(cts.SQI_WINDOW_THRESHOLD), 'start_time']
    #     end_event = df.loc[df.pred.eq(True) & df.sqi.ge(cts.SQI_WINDOW_THRESHOLD), 'end_time']
    #     final_rhythms_str = np.repeat(['AFIB'], len(start_event))
    #
    #     final_rhythms = final_rhythms_str
    #     for i in range(1, 7):
    #         rhythms_full_path = directory / pat / (
    #                 pat + '_rhythms_day_' + str(i) + '_n_leads_' + str(db.n_leads) + '.txt')
    #         start = int((i - 1) * cts.N_HOURS_IN_DAY * cts.N_S_IN_HOUR)
    #         end = int((i) * cts.N_HOURS_IN_DAY * cts.N_S_IN_HOUR)
    #         mask_int_rhythm = np.logical_and(start_event < end, end_event > start)
    #         start_events, end_events = start_event[mask_int_rhythm] - start, end_event[mask_int_rhythm] - start
    #         end_events[end_events > (end - start)] = end - start
    #         final_rhythms_str = final_rhythms[mask_int_rhythm]
    #         with open(rhythms_full_path, 'w+') as rhythms_file:
    #             rhythms_file.writelines(rhythms_header)
    #
    #             rhythms_file.write('\n'.join(
    #                 ['%.5f\t%.5f\t%s' % (start_events.values[i], end_events.values[i], final_rhythms_str[i]) for i in
    #                  range(len(start_events))]))
    #
    #         ecg_full_path = directory / pat / (
    #                 pat + '_ecg_day_' + str(i) + '_n_leads_' + str(db.n_leads) + '.txt')
    #         start = int((i - 1) * cts.N_HOURS_IN_DAY * cts.N_S_IN_HOUR)
    #         end = int((i) * cts.N_HOURS_IN_DAY * cts.N_S_IN_HOUR)
    #         ecg_day_i = ecgs_day_i = np.vstack(tuple([ecgs[start * db.actual_fs:end * db.actual_fs,lead] for
    #                                  lead in range(db.n_leads)])).transpose()
    #         join_func = lambda x: ' '.join(['%.2f' % i for i in x])
    #         ecg_str = np.apply_along_axis(join_func, 1, ecg_day_i)
    #         with open(ecg_full_path, 'w+') as ecg_file:
    #             ecg_file.writelines(ecg_header)
    #             ecg_file.write('\n'.join(ecg_str))
    #
    #         peaks_full_path = directory / pat / (
    #                 pat + '_peaks_day_' + str(i) + '_n_leads_' + str(db.n_leads) + '.txt')
    #         mask_int_ann = np.logical_and(ann < end, ann > start)
    #         ann_day_i = ann[mask_int_ann] - start
    #         with open(peaks_full_path, 'w+') as peaks_file:
    #             peaks_file.writelines(peaks_header)
    #             peaks_file.write('\n'.join(['%d' % i for i in ann_day_i]))
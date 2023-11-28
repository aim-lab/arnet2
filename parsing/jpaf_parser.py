from base_parser import *

warnings.filterwarnings('ignore')
random.seed(cts.SEED)


class JPAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(JPAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(['090'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 125
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 2
        self.ref_lead = 1
        self.name = "JPAFDB"
        self.ecg_format = ".csv"

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / self.name.lower() / "examples"
        self.orig_anns_path = None
        self.generated_anns_path = cts.GEN_ANN_DIR / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / self.name

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        test_pat = self.parsed_patients()
        self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)
                self.load_circardian_from_disk()

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the JPAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """
        self.ecg_file_name = 'RR'
        self.file_format = ".csv"
        self.csv_dir = "RRData"
        self.searchPar = ['PAF']
        self.searchPer = ['PerAF']
        self.peak_ann = np.array(['N', 'Q', 'V', 'S'])
        self.peak_ann_dict = {self.peak_ann[i]: i for i in range(len(self.peak_ann))}
        self.circadian_dict = {}
        self.excel_sheet_path = cts.DATA_DIR / self.name.lower() / "List_AF_latest.xlsx"
        self.get_META()
        self.over_18_patients = np.array(self.excel_sheet[self.excel_sheet["Age"] >= 18]
                                         ["Study ID"]).astype('<U32')

        """
        # ------------------------------------------------------------------------- #
        # ----- Parsing functions: have to be overridden by the child classes ----- #
        # ------------------------------------------------------------------------- #
        """
        """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        return np.array([dir.split('_')[0].zfill(3) for dir in os.listdir(str(self.raw_ecg_path))])

    def parse_annotation(self, id, lead, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def record_to_wfdb(self, id, lead):
        record = self.read_ecg(id).iloc[:, lead].astype(float).values
        re_record = dp.bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs / 2 - 0.5,
                                       signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)
        re_record = dp.resample_by_interpolation(re_record, self.orig_fs, self.actual_fs)
        wfdb.wrsamp(id, fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
        return re_record

    def parse_raw_ecg(self, id, lead, start=0, end=-1, type='epltd0'):
        record = self.read_ecg(id).iloc[:, lead].astype(float).values
        record = dp.bandpass_filter(data=record, id=id, lead='x', lowcut=0.67, highcut=self.orig_fs / 2 - 0.5,
                                    signal_freq=self.orig_fs, filter_order=75, notch_freq=50, debug=False)

        record = dp.resample_by_interpolation(record, self.orig_fs, self.actual_fs)
        ann = self.parse_annotation(id, type=type, lead=lead)
        ann = i_o.qrs_adjust(ecg=record, qrs=ann, fs=self.actual_fs, inputsign=1)
        if end == -1:
            end = int(len(record) / self.actual_fs)
        start_sample = int(start * self.actual_fs)
        end_sample = int(end * self.actual_fs)
        record = record[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return record, ann

    def parse_reference_annotation(self, id, combine=True):  # , reannotated=True):
        record = self.read_ecg(id)
        ann = self.read_ann(id, start_time=record.time[0], end_time=record.time.iloc[-1])
        beat = np.array([ann.pos.values])
        tbeats = np.cumsum(ann.pos.values) / cts.N_MS_IN_S
        # ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        # if reannotated:
        #     rhythm_df = self.parse_reference_rhythm(id)
        #     for index, l in rhythm_df.iterrows():
        #         l1 = np.abs(tbeats - l.Beginning)
        #         l2 = np.abs(tbeats - l.End)
        #         begin = np.where(l1 == l1.min())
        #         end = np.where(l2 == l2.min())
        #         ltbeats[int(begin[0][0]):int(end[0][0])] = l.Class
        # else:
        ltbeats = np.array(['NSR' for i in tbeats]).astype(object)
        rhythm = np.array([self.rhythms_dict[i] for i in ltbeats])
        if combine:
            rhythm[rhythm == self.rhythms_dict['AFL']] = self.rhythms_dict['AFIB']
        return (tbeats * self.actual_fs).astype(int), rhythm

    def parse_demographic_features(self, id):
        age = float(self.excel_sheet.loc[self.excel_sheet["Study ID"] == id, 'Age'])
        sex = float(
            self.excel_sheet.loc[self.excel_sheet["Study ID"] == id, "Sex"] == 'F')  # True (1): Female, False (0): Male
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = age
            self.features_dict[id][win]['Sex'] = sex

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_patient_id(self, recording_id):
        return self.excel_sheet[self.excel_sheet["Study ID"] == recording_id]["ID"].values[0]

    def _af_pat_clinical_lab(self, patient_id, win):
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
        self.excel_sheet = pd.read_excel(self.excel_sheet_path, engine='openpyxl')
        self.excel_sheet["Study ID"] = self.excel_sheet["Study ID"].astype(str).str.zfill(3)

    def get_dir(self, id):
        return [i for i in os.listdir(self.raw_ecg_path) if i.startswith(id)]

    def read_ecg(self, id):
        id_dir = self.get_dir(str(id))[0]
        example_path = self.raw_ecg_path / id_dir / (str(id) + self.ecg_format)
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

    def parse_circadian_features(self, id):
        if id not in self.circadian_dict.keys():
            self.circadian_dict[id] = {}
        self.circadian_dict[id]['recording_date'] = self.read_ecg(id).date[0].date()
        self.circadian_dict[id]['start_recording'], self.circadian_dict[id]['end_recording'] = \
            self.read_ecg(id).time.iloc[0], self.read_ecg(id).time.iloc[-1]
        np.save(self.main_path / id / 'circadian_dict.npy', self.__dict__['circadian_dict'][id])

    def load_circardian_from_disk(self, patient_list=None):
        if patient_list is None:
            patient_list = self.parsed_patients()
        for pat in patient_list:
            if os.path.exists(self.main_path / pat / ('circadian_dict.npy')):
                self.__dict__['circadian_dict'][pat] = np.load(self.main_path / pat / ('circadian_dict.npy'),
                                                               allow_pickle=True).item()

    def read_ann(self, id, start_time=None, end_time=None):
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
        real_start = time.strftime('%-H:%M', time.gmtime(start_time))
        time_ann_start = time_df[time_df == real_start].index[0]
        real_end = time.strftime('%-H:%M', time.gmtime(end_time))
        temp_time_df = time_df[time_ann_start + 1:]
        if len(temp_time_df[temp_time_df == real_end]) == 0:
            time_ann_end = time_df.index[-1]
        else:
            if time_df[time_df == real_end].index[-1] < 4000:
                time_ann_end = time_df.index[-1]
            else:
                time_ann_end = time_df[time_df == real_end].index[-1]
        # else:
        #     time_ann_start=0
        #     time_ann_end = len(time_df)
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
    db = JPAFDB_Parser(load_on_start=True)
    # ids = np.setdiff1d(db.parse_available_ids(), db.missing_ecg)
    ann_ids = np.array(next(os.walk(cts.REANNOTATION_DIR / (db.name + '-annotated')))[1])
    pat_list = ann_ids[~np.isin(ann_ids, db.parsed_patients())]
    db.plot_ecg(patient_id='127', start=20654 - 42, end=20654 + 16, savefig=True)
    # db.parse_raw_data(patient_list=pat_list)

    # for id in db.parse_available_ids():
    #     db.parse_demographic_features(id)
    #     db.save_patient_to_disk(id)
    # db.save_to_disk()
    # pat='006'
    # db.parse_elem_data(pat=pat, reannotated=True)
    # db._win_lab(id=pat, win=60)
    # db._af_win_lab(id=pat, win=60)
    # db._af_pat_lab(id=pat)
    # _test_pat = np.load(cts.REPO_DIR / "data" / "splits" / (db.name + "_test_pat.npy"))
    # db.print_summary()
    # db.annotation_types = ['epltd0', 'gqrs', 'wqrs', 'jqrs', 'xqrs']
    # db.loaded_window_sizes = np.append(db.loaded_window_sizes, windows[0])
    # all_ids = db.parse_available_ids()
    # all_ids = np.sort(all_ids)
    # all_ids = next(os.walk('/MLdata/AIMLab/Shany/medAIM/JPAFDB-annotated'))[1]
    # all_ids = next(os.walk('/MLdata/AIMLab/Shany/PreprocessedDatabases/JPAFDB'))[2]

    #
    # db.parse_raw_data(patient_list=['130', '131', '132', '133'])
    # ids = np.array(np.arange(134, 144), dtype='str')
    # for pat in ids:
    #     print(pat)
    #     ecg, ann = db.parse_raw_ecg(pat)
    #     db.recording_time[pat] = len(ecg) / db.actual_fs
    # directory = pathlib.PurePath("/MLAIM/AIMLab/Shany/medAIM/GS") / db.name / str(pat)
    # directory2 = pathlib.PurePath("/home/shanybiton/repos/Generalization/temp/") / str(pat)
    #
    # if not os.path.exists(directory2):
    #     os.makedirs(directory2)
    #
    # db.export_to_physiozoo(pat, directory=directory2, export_rhythms=False, force=True, n_leads=db.n_leads)

    # db.generate_annotations(pat_list=['130', '131', '132', '133'], force=False, lead=2)
    # ids = np.setdiff1d(db.parse_available_ids(), db.parsed_patients())
    # ids = []
    # for id in db.parsed_patients():
    #     if 'Age' not in db.features_dict[id][db.window_size].keys():
    #         ids.append(id)
    #
    # db.parse_raw_data(patient_list=ids)

    # # db.parse_reference_annotation(all_ids[0])
    # ids = db.return_patient_ids(pat_list=all_ids)
    # _, y, glob_lab = db.return_features(pat_list=all_ids, feats_list=cts.IMPLEMENTED_FEATURES,
    #                                                    return_global_label=True)
    # rr, rrt, _, win_start, win_end = db.return_rr(pat_list=all_ids)
    # prec = db.return_preceeding_windows(pat_list=all_ids)
    # data = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    # final = tuple()
    # timestamp = np.concatenate((win_start.reshape(-1, 1), win_end.reshape(-1, 1)), axis=1)
    # final = final + (data, y, timestamp)
    #
    # df = pd.DataFrame(columns=['id', 'proba'])
    # df['id'] = ids
    # df['proba'] = y
    # df['start_time'] = win_start
    # df['end_time'] = win_end
    # af_df = df.loc[df.proba.eq(True)]
    # af_event = pd.DataFrame(columns=af_df.columns)
    # j = 0
    # for id in af_df.id.unique():
    #     temp_df = af_df.loc[af_df.id.eq(id)].reset_index(drop=True)
    #     af_event = af_event.append(temp_df.iloc[0])
    #     for i in range(len(temp_df) - 1):
    #         en = af_event.end_time.values[j]
    #         st = temp_df.start_time.values[i + 1]
    #         if np.isclose(st, en):
    #             af_event.at[j, 'end_time'] = temp_df.end_time.values[i + 1]
    #         else:
    #             j += 1
    #             af_event = af_event.append(temp_df.iloc[i + 1]).reset_index(drop=True)
    # af_event['duration'] = af_event['end_time'] - af_event['start_time']
    # ann_type = 'epltd0'
    # df = pd.DataFrame([])
    # for id in all_ids:
    #     df = df.append(db.parse_reference_rhythm(id))
    # df.rename(columns={'End': 'end_time', 'Beginning': 'start_time'}, inplace=True)
    # start_bin = (df.end_time - df.start_time).min()
    # end_bin = np.quantile((df.end_time - df.start_time), 0.85)
    # db.plot_win_len(df, start_bin=start_bin, end_bin=end_bin, step=(end_bin-start_bin)//20, figsize=(8, 8), savefig=True, figname='event_length.png')

    # point = 5 * db.actual_fs  # taking 5 sec before and after each segment
    # for i, r in temp_df.iloc[:3].iterrows():
    #     ecg = []
    #     annot = []
    #     for i in range(1, 3):
    #         ecg, annot = db.parse_raw_ecg(r.id, r.start_time, r.end_time, type='epltd0')
    # record = db.read_ecg(r.id).iloc[:, i].astype(float).values
    # re_record = dp.resample_by_interpolation(record, db.orig_fs, db.actual_fs)
    # re_record = re_record[int(r.start_time * db.actual_fs) - point:int(r.end_time * db.actual_fs) + point]
    # re_record = bandpass_filter(data=re_record, id=af_df.id.unique()[0], lead='x', lowcut=0.67, highcut=90,
    #                            signal_freq=db.actual_fs, filter_order=75, notch_freq=50, debug=False)
    # wfdb.wrsamp(id, fs=db.actual_fs, units=['mV'],
    #             sig_name=['V5'], p_signal=re_record.reshape(-1, 1), fmt=['16'], )
    # detector = getattr(i_o,
    #                    ann_type + '_detector')  # Calling the correct wrapper in the feature comp module.
    # detector(id)  # Running the wrapper
    # shutil.move(id + '.' + ann_type, db.generated_anns_path / 'wins' / ann_type / (
    #         id + '.' + ann_type))
    # ann = wfdb.rdann(str(db.generated_anns_path / 'wins' / ann_type / id), ann_type).sample
    # cann = i_o.qrs_adjust(ecg=ecg,qrs=annot,fs=db.actual_fs,inputsign=1, debug=1)
    #
    # timeline = np.arange(0, len(ecg) / db.actual_fs, 1 / db.actual_fs)
    # plt.plot(timeline, ecg, label='Signal', zorder=0)
    # rr = np.diff(annot) / db.actual_fs
    # plt.scatter(timeline[cann], ecg[cann], label='RR Interval', c='k', zorder=1)
    # plt.xlim(10,30)
    # plt.show()
    # plt.close()
    #
    # # db.generate_annotations(pat_list=[temp_df.id.unique()[0]], force=True)
    # for i, r in temp_df.iloc[:3].iterrows():
    #     db.plot_ecg(patient_id=r.id, start=r.start_time, end=r.end_time, savefig=False)
    #
    # for i, r in temp_df.iloc[::20].iterrows():
    #     db.plot_ecg(patient_id=r.id, start=r.start_time, end=r.end_time, correct_peaks=True, savefig=True)
    # for id in af_df.id.unique()[-5:]:
    #     db.parse_circadian_features(id)
    #     temp_df =af_df.loc[af_df.id.eq(id)].reset_index(drop=True)
    #     db.export_mat(id, temp_df, n_lead=2, ann_type='epltd0')
    #     db.plot_AF_win(temp_df, 0, -1, figname=str('AF_events_' + str(id) + '.png'))

    # ids = db.parse_available_ids()
    # db.parse_raw_data(patient_list=ids[54:])
    # db.load_from_disk(pat_list=ids)
    # feats_to_use = np.append(cts.SELECTED_FEATURES, 'sqi')
    # data = db.return_data(ids, feats_to_use, normalize=True)
    # with open('/home/shanybiton/repos/Generalization/jpaf_input.pickle', 'wb') as f:
    #     pickle.dump(final, f)

    # db.generate_annotations(types=cts.ANNOTATION_TYPES, pat_list=[ids[0]], force=True)
    # for pat in ids[-6:]:
    #     ecg = db.read_ecg(pat).iloc[:, 1].astype(float).values
    #     db.recording_time[pat] = len(ecg) / db.actual_fs
    #     directory = pathlib.PurePath("/MLdata/AIMLab/medAIM/JPAFDB/") / str(pat)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     db.export_to_physiozoo(pat, directory=directory, export_rhythms=False, force=True, n_leads=2)

    # db.parse_raw_data(patient_list=ids)
    # db.load_from_disk(pat_list=ids)
    # X, y = db.return_features(pat_list=ids)
    # prec_train = db.return_preceeding_windows(pat_list=ids)

    # db.generate_annotations(pat_list=ids[600:], types=['wqrs', 'gqrs'])
    # db.parse_raw_data(patient_list=ids[600:])
    # for id_ in ids:
    #    db.parse_circadian_features(patient_id=id_)
    #    db.load_patient_from_disk(pat=id_)
    '''
    db.circadian_dict[id_] = {}
    db.circadian_dict[id_]['start_recording'] = {}
    db.circadian_dict[id_]['end_recording'] = {}
    db.features_dict[id_] = {}
    db.features_dict[id_][windows[0]] = {}
    db.parse_demographic_features(id_)
    db.parse_raw_data(patient_list = [id_])
    '''

from utils.base_packages import *
import utils.consts as cts
import utils.graphics as graph
import utils.feature_comp as fc
import utils.data_processing as dp
from preprocessing.Feature_extractor import bandpass_filter
import utils.in_out as i_o
import scipy.interpolate as interp
from scipy.io import savemat


# TODO: Restructure parse_elem_data function. (Maybe create a function per parser to return the reference annotations). This has been done for UVAF, need to implement the parse_ref_ann for all other databases


class BaseParser:
    """Class defining a base parser. All the parsers relative to the different datasets will inherit this class.
    This class offers the possibility to load the dataset onto the system, with different window sizes, features, sqi.
    It further provide tools to display elementary statistics on the datasets, such as the time in AF, the total recording time,
    display of the ECG signals for a given segment, and more.
    Each child dataset has its own init function, in which the following the paths and the information relative to the
    specific dataset will be set, to ensure the dataset will be loaded properly."""

    def __init__(self):
        """The purpose of the __init__ function for the BaseParser class
        is to define the different dictionaries that will be found among the different parsers.
        Note: Depending on the dataset, some of those dictionnaries may be empty. Data extraction is taken care of
        apart in each dataset with the elementary parsing functions whose signature can be found below. """

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overridden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- # """

        # Missing records
        self.missing_ecg = np.array([])  # List of the patients who's ECG is missing (patient is listed in the database but file is absent)

        # Helper variables
        self.curr_ecg = None
        self.curr_id = None

        """ Variables relative to the ECG signals. """
        self.orig_fs = None                                 # Sampling frequency of the original files
        self.actual_fs = None                               # Sampling frequency of the resampled files (resampling is necessary to use the EPLTD C Code)
        self.n_leads = 1                                    # Number of ECG leads in the database
        self.ref_lead = 1                                   # The lead according to which all the elementary dictionnaries are computed
        self.name = None                                    # Name of the Dataset
        self.ecg_format = None                              # The format of the ECG files. Should be "wfdb", "edf", "rf"
        self.rhythms = np.array(['NSR', 'AFIB', 'AB', 'AFL', 'B', 'BII', 'IVR', 'NOD',
                            'P', 'PREX', 'SBR', 'SVTA', 'T', 'VFL', 'VT', 'J',
                            'PAT', 'AT', 'VTS', 'AIVRS', 'IVRS', 'AIVR'])
                                                            # The rhythms defined in the dataset
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}
                                                            # Mapping between rhythms and unique ids

        """ Variables relative to the different paths. """
        self.raw_ecg_path = None                            # Path of the raw ECG files
        self.orig_anns_path = None                          # Path of the raw annotations, if exist (otherwise None)
        self.generated_anns_path = None                     # Path of the generated annotations
        self.annotation_types = None                        # Generated annotation types
        self.main_path = None                               # Main path for the processed database
        self.window_size = None                             # Current window size used in the dataset
        self.window_sizes = np.array([60], dtype=int)       # Window sizes available in the dataset
        self.loaded_window_sizes = np.array([], dtype=int)  # Window sizes currently loaded in the object

        """
        # ------------------------------------------------------------------------------- #
        # ------------------------- Overridden variables end here ------------------------ #
        # ------------------------------------------------------------------------------- #
        """

        """Raw data. Used further to derive metrics and features."""

        self.rr_dict = {}                   # Key: Patient_ID, Value: RR intervals divided by windows
        self.rrt_dict = {}                  # Key: Patient_ID, Value: Relative time of RR intervals divided by windows
        self.rlab_dict = {}                 # Key: Patient_ID, Value: Rhythm label per beat
        self.start_windows_dict = {}        # Key: Patient ID, Value: Windows starts (time elapsed from recording start)
        self.end_windows_dict = {}          # Key: Patient ID, Value: Windows ends (time elapsed from recording start)
        self.av_prec_windows_dict = {}      # Key: Patient ID, Value: Successive preceding windows available
        self.recording_time = {}            # Key: Patient ID, Value: Recording time in hours

        """Dictionaries obtained after computation through the methods."""
        self.signal_quality_dict = {}       # Key: Patient_ID, Value: SQI for each window
        self.features_dict = {}             # Key: Patient_ID, Value: Computed features for each window
        self.af_burden_dict = {}            # Key: Patient_ID, Value: AF Burden
        self.other_cvd_burden_dict = {}     # Key: Patient_ID, Value: Burden of CVDs different than AF

        """ Labels dicts."""
        self.win_lab_dict = {}              # Key: Patient_ID, Value: Rhythm label for each window
        self.af_win_lab_dict = {}           # Key: Patient_ID, Value: Binary label for AF for each window
        self.af_pat_lab_dict = {}           # Key: Patient_ID, Value: Type of AF based on AF burden (NSR, AF_Pa, AF_Pe, O)
        self.ahi_dict = {}                  # Key: Patient_ID, Value: Apnea Hypopnea Index
        self.odi_dict = {}                  # Key: Patient_ID, Value: Oxygen Desaturation Index

        """ Patients and exclusions. """
        self.corrupted_ecg = np.array([])           # List of the patients who's ECG is corrupted (file is present but can't extract a minimal number of annotations (1000))
        self.missing_af_label = np.array([])        # List of patients without AF label
        self.low_sqi = np.array([])                 # List of patients which need to be excluded if the signal quality condition is demanded
        self.parsed_ecgs = np.array([])             # List of the patients already parsed
        self.excluded_portions_dict = {}            # Key: Patient ID, Value: List of the segments excluded ([start_time, end_time])
        self.n_excluded_windows_dict = {}           # Key: Patient ID, Value: Number of Excluded windows in preprocessing
        self.mask_rr_dict = {}                      # Key: Patient ID, Value: Mask to exclude windows obtained after preprocessing


        """ General variables for signal processing/Filtering """
        self.min_annotation_len = 1000      # Below this number of peaks, the recording is removed.
        self.agw = 0.05                     # Agreement window for SQI computation
        self.pool = None                    # Multiprocessing pool to generate the features and compute the sqi for all the windows.
        self.sqi_test_ann = 'xqrs'          # The annotation type used to compute and load the SQI variables.
        self.sqi_ref_ann = 'epltd0'

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """

    def parse_available_ids(self):
        """The function returns all the patient IDs available in the database.
        :returns arr: numpy array listing all the patients."""
        raise NotImplementedError("Needs to be called by a child class.")

    def parse_reference_annotation(self, id):
        """ This function returns for a given patient the reference annotation, if available.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :returns peaks: A numpy array listing the indices of the peaks in the raw ECG.
        :returns rhythms: A numpy array listing the rhythms corresponding to the peaks in the raw ECG."""
        raise NotImplementedError("Needs to be called by a child class.")

    def parse_annotation(self, id, type="epltd0", lead=1):
        """ Returns, if exists, for a given ID, the peak annotation.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param type: Annotation type. Can be epltd0, xqrs, gqrs.
        :param lead: ECG lead. Can be one of the leads presented in the database.
        :returns ann: A numpy array listing the indices of the peaks in the raw ECG."""
        raise NotImplementedError("Needs to be called by a child class.")

    def record_to_wfdb(self, id):
        """ This functions converts a raw ECG signal to the wfdb format under the code directory.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        raise NotImplementedError("Needs to be called by a child class.")

    def parse_raw_ecg(self, patient_id, lead, start=0, end=-1, type='epltd0'):
        """Returns the raw ECG and the corresponding annotation for a given patient. The signal is resampled at 200 [Hz]
        to generate the epltd annotation."
        :param patient_id: The ID of the patient.
        :param lead: The ECG lead.
        :param start: The beginning of the ECG.
        :param end: The end of the ECG.
        :param type: The annotation type.
        :returns raw_ecg: the ECG recording.
        :returns ann: the peaks annotation indices.
        """
        raise NotImplementedError("Needs to be called by a child class.")

    def parse_ahi(self, id):
        """Computes/Extracts from raw data the AHI for each patient.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        raise NotImplementedError("Needs to be called by a child class.")

    def parse_odi(self, id):
        """Computes/Extracts from raw data the ODI for each patient.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        raise NotImplementedError("Needs to be called by a child class.")
    def parse_patient_id(self, recording_id):
        """Computes/Extracts from raw data the ID for each patient.
        :param id: The recording ID. Assumed to be in the list of IDs present in the database.
        """
    def parse_demographic_features(self, id):
        """Computes/Extracts from raw data the demographic features for each patient.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        raise NotImplementedError("Needs to be called by a child class.")

    # ------------------------------------------------------------------------- #
    # ------------------------ Computational functions ------------------------ #
    # ------------------------------------------------------------------------- #

    def annot_available(self, id, type='epltd0', lead=1):
        """ Returns for a given ID a boolean value indicating if the annotation has been generated and is available.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param type: Annotation type. Can be epltd0, xqrs, gqrs.
        """
        return os.path.exists(self.generated_anns_path / type / str(lead) / (id + '.' + type))

    def parse_elem_data(self, pat):
        """ This function is responsible for extracting the basic raw data for the given id.
        It fills the following elementary dictionaries: rr_dict (RR intervals), rlab_dict (label for each RR interval),
        rrt (timestamp of each RR interval), mask RR (which windows we can rely on based on the presence of proper annotations),
        start_windows, end_windows (respectively the timestamps of the beginning and the end of the windows), n_excluded_windows
        (number of windows excluded because their annotations were not reliable), av_prec_windows (for each window,
        the number of consecutive windows preceding it after the exclusions.
        :param pat: The patient ID. Assumed to be in the list of IDs present in the database.
        """

        dicts_to_fill = [self.start_windows_dict, self.end_windows_dict, self.av_prec_windows_dict,
                         self.n_excluded_windows_dict, self.mask_rr_dict]
        for dic in dicts_to_fill:
            if pat not in dic.keys():
                dic[pat] = {}
        sec_int = 0  # Security interval for RR exclusion (number of beats to exclude around a problematic RR

        # Extracting data based on Automatic annotation (default EPLTD)
        ecg, ann = self.parse_raw_ecg(pat, type=self.sqi_ref_ann, lead=self.ref_lead)
        self.recording_time[pat] = len(ecg) / self.actual_fs
        rr = np.diff(ann) / self.actual_fs
        start_rr, end_rr = ann[:-1] / self.actual_fs, ann[1:] / self.actual_fs
        interbeats = np.append(np.insert((start_rr + end_rr) / 2, 0, max(0, start_rr[0] - 1)), end_rr[-1] + 1.0)

        # Extracting the reference annotation
        ref_ann, ref_rhythm = self.parse_reference_annotation(pat)
        ref_rr = np.diff(ref_ann) / self.actual_fs
        start_ref_rr, end_ref_rr = ref_ann[:-1] / self.actual_fs, ref_ann[1:] / self.actual_fs
        ref_rlab = ref_rhythm[1:]   # To have the same dimension as ref_rr

        # First filtering - Removing RR intervals at which equal zero
        mask = ref_rr > cts.RR_OUTLIER_THRESHOLD_INF
        ref_rr, start_ref_rr, end_ref_rr, ref_rlab = ref_rr[mask], start_ref_rr[mask], end_ref_rr[mask], ref_rlab[mask]
        ref_interbeats = np.append(np.insert((start_ref_rr + end_ref_rr) / 2, 0, max(0, start_ref_rr[0] - 1)), end_ref_rr[-1] + 1.0)

        # Second filtering - Based on missing reference annotations
        mask_sup = ref_rr <= cts.RR_OUTLIER_THRESHOLD_SUP
        irreg_rr_indices = np.where(~mask_sup)[0]
        self.excluded_portions_dict[pat] = np.array([[ref_interbeats[x - sec_int], ref_interbeats[x + sec_int + 2]] for x in irreg_rr_indices])
        if len(self.excluded_portions_dict[pat]) > 0:
            self.excluded_portions_dict[pat] = np.concatenate(([[0, start_ref_rr[0]]], self.excluded_portions_dict[pat]), axis=0)  # Beggining of recordings without any label are considered excluded
        else:
            self.excluded_portions_dict[pat] = np.array([[0, start_ref_rr[0]]])

        # Building interpolator for the label
        ref_rr, start_ref_rr, end_ref_rr, ref_rlab = ref_rr[mask_sup], start_ref_rr[mask_sup], end_ref_rr[mask_sup], ref_rlab[mask_sup]
        f = interp.interp1d(end_ref_rr, ref_rlab, kind='nearest', fill_value="extrapolate")

        # Saving raw data
        self.rlab_dict[pat] = f(start_rr)
        self.rr_dict[pat] = rr
        self.rrt_dict[pat] = start_rr

        # Constituting final filters for windows
        for win in self.window_sizes:
            start_win = interbeats[:-2][:(len(rr) // win) * win].reshape(-1, win)[:, 0]  # [:, 0] to select the beginning of the window
            end_win = interbeats[2:][:(len(rr) // win) * win].reshape(-1, win)[:, -1]   # [:, -1] to select the end of the window
            mask_start = np.logical_or.reduce(tuple([np.logical_and(start_win > x[0], start_win <= x[1]) for x in self.excluded_portions_dict[pat]]))   # The window begins in an excluded portion.
            mask_end = np.logical_or.reduce(tuple([np.logical_and(end_win > x[0], end_win <= x[1]) for x in self.excluded_portions_dict[pat]]))         # The window ends in an excluded portion.
            mask_between = np.logical_or.reduce(tuple([np.logical_and(start_win <= x[0], end_win > x[1]) for x in self.excluded_portions_dict[pat]]))   # The window contains an excluded portion.
            final_mask = np.logical_not(np.logical_or.reduce((mask_start, mask_end, mask_between)))     # We don't want any of the three
            self.start_windows_dict[pat][win] = start_win
            self.end_windows_dict[pat][win] = end_win
            self.mask_rr_dict[pat][win] = final_mask
            # Depend on the mask applied on the windows
            self.n_excluded_windows_dict[pat][win] = (~final_mask).sum()
            self.av_prec_windows_dict[pat][win] = dp.cumsum_reset(final_mask)

    def parse_reference_rhythm(self, id):
        dest_path = cts.REANNOTATION_DIR / (self.name + '-annotated')
        if not os.path.exists(dest_path):
            print('No re-annotated recordings for' + self.name)
        else:
            directory = dest_path / id
            f = next(os.walk(directory))[2]
            rhythm_f = [s for s in f if "rhythm" in s]
            df = pd.read_csv(directory / rhythm_f[0], sep="\s+|;|,", error_bad_lines=False, skiprows=6, engine='python')
            return df


    def generate_annotations(self, types=None, pat_list=None, force=False, lead=1):
        """ This function generates the peak annotations for all the patients in the database.
        :param types: Tuple containing the names of the annotations to be generated.
        :param pat_list: List of the patients for whom the annotations should be generated. If None, generates on all the patients.
        :param force: If False, the annotations are not computed if already existing. If true, computes the annotations anyway.
        """
        if pat_list is None:
            pat_list = self.parse_available_ids()

        if types is None:
            types = self.annotation_types
        else:
            types = np.append(types, self.sqi_ref_ann)
        # Generate paths
        for ann_type in types:
            dest_path = str(self.generated_anns_path / ann_type / str(lead))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
        # Generate annotations
        self.create_pool()  # Creating a multiprocessing pool to work concurrently on a large number of windows.
        for i, id in enumerate(pat_list):
            available_anns = np.array([self.annot_available(id, ann_type, lead) for ann_type in types])
            if np.any(np.logical_not(available_anns)) or force:
                self.record_to_wfdb(id, lead=lead)
                for j, ann_type in enumerate(types):
                    try:
                        if not available_anns[j] or force:
                            print("Generating " + str(ann_type) + " annotation for patient ID " + str(id))
                            detector = getattr(i_o, ann_type + '_detector')     # Calling the correct wrapper in the feature comp module.
                            detector(id, pool=self.get_pool())                  # Running the wrapper
                            shutil.move('/home/shanybiton/repos/Generalization/parsing/' + id + '.' + ann_type, self.generated_anns_path / ann_type / str(lead) / (
                                    id + '.' + ann_type))                       # The wrapper provides the annotation file in the local directory. We here migrate it to the anns directory.
                    except:
                        continue
                # Cleaning local directory
                if os.path.exists(id + '.hea'):
                    os.remove(id + '.hea')
                if os.path.exists(id + '.dat'):
                    os.remove(id + '.dat')
                for j, ann_type in enumerate(self.annotation_types):
                    if os.path.exists(id + '.' + ann_type):
                        os.remove(id + '.' + ann_type)

        self.destroy_pool()

    def generate_wrqrs_annotations(self, lead, pat_list=None, force=False, tol=0.05):
        """ This function performs a correction on the wqrs annotation. The gqrs annotation locates the Q onset rather than the
            R-Peak. To correct the behaviour, this function generates the rqrs annotation, which looks for the local maximum
            absolute value over the ECG with a given tolerance window.
            :param lead: ECG lead.
            :param pat_list: List of the patients for whom the annotations should be generated. If None, generates on all the patients.
            :param force: If False, the annotations are not computed if already existing. If true, computes the annotations anyway.
            :param tol: The tolerance window on which a local maximum should be searched.
        """
        dest_path = str(self.generated_anns_path / 'wrqrs')
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if pat_list is None:
            pat_list = self.parse_available_ids()
        for i, id in enumerate(pat_list):
            ann_available = self.annot_available(id, 'wrqrs', lead)
            if not ann_available or force:
                print(f"Generating rqrs annotation for patient ID {str(id)} for lead {str(lead)}")
                ecg, ann = self.parse_raw_ecg(id, type='wqrs', lead=lead)
                idx_start = np.array([max(ann[i] - int(self.actual_fs * tol), 0) for i in range(len(ann))])
                idx_end = np.array([min(ann[i] + int(self.actual_fs * tol), len(ecg) - 1) for i in range(len(ann))])
                rqrs_ann = np.array([np.argmax(np.abs(ecg[idx_start[i]:idx_end[i]])) + idx_start[i] for i in range(len(ann))])
                wfdb.wrann(id, 'wrqrs', rqrs_ann, symbol=['q'] * len(rqrs_ann))
                shutil.move(id + '.wrqrs', self.generated_anns_path / 'wrqrs' / str(lead) / (id + '.wrqrs'))

    def generate_rqrs_annotations(self, lead, pat_list=None, force=False, tol=0.05):
        """ This function performs a correction on the gqrs annotation. The gqrs annotation locates the Q onset rather than the
            R-Peak. To correct the behaviour, this function generates the rqrs annotation, which looks for the local maximum
            absolute value over the ECG with a given tolerance window.
            :param lead: ECG lead.
            :param pat_list: List of the patients for whom the annotations should be generated. If None, generates on all the patients.
            :param force: If False, the annotations are not computed if already existing. If true, computes the annotations anyway.
            :param tol: The tolerance window on which a local maximum should be searched.
        """
        if pat_list is None:
            pat_list = self.parse_available_ids()
        for i, id in enumerate(pat_list):
            ann_available = self.annot_available(id, 'rqrs', lead)
            if not ann_available or force:
                print(f"Generating rqrs annotation for patient ID {str(id)} for lead {str(lead)}")
                ecg, ann = self.parse_raw_ecg(id, type='gqrs', lead=lead)
                idx_start = np.array([max(ann[i] - int(self.actual_fs * tol), 0) for i in range(len(ann))])
                idx_end = np.array([min(ann[i] + int(self.actual_fs * tol), len(ecg) - 1) for i in range(len(ann))])
                rqrs_ann = np.array([np.argmax(np.abs(ecg[idx_start[i]:idx_end[i]])) + idx_start[i] for i in range(len(ann))])
                wfdb.wrann(id, 'rqrs', rqrs_ann, symbol=['q'] * len(rqrs_ann))
                shutil.move(id + '.rqrs', self.generated_anns_path / 'rqrs' / str(lead) / (id + '.rqrs'))

    def parse_raw_data(self, window_sizes=cts.BASE_WINDOWS, gen_ann=False, feats=cts.IMPLEMENTED_FEATURES, patient_list=None, test_anns=None):
        """ This function is responsible of performing all the necessary computations for the dataset, among which:
        all the features according to the input, the sqi, the labels per window, the ahi, the odi, and the demographic features if available.
        This function has usually a long running time (at least for the big databases).
        :param window_sizes: The different window sizes along which the windows are derived.
        :param get_ann: If True, calls the function gen_ann in Force mode, and runs all the annotations.
        :param feats: List of the features to compute. The function 'comp_" + feature name must be implemented in the utils.feature_comp module.
        :param total_run: if True, runs the function on all the IDs, otherwise runs on the non-parsed IDs only.
        :param test_anns: All the test annotations upon which SQI needs to run. If None, uses all the available annotations.
        """
        self.window_sizes = window_sizes
        self.generate_annotations(pat_list=patient_list, force=gen_ann, types=test_anns)
        # self.parsed_ecgs = self.parsed_patients()
        if patient_list is None:
            patient_list = self.parse_available_ids()

        self.create_pool()
        for i, id in enumerate(patient_list):
            print("Parsing patient number " + id)
            ecg, ann = self.parse_raw_ecg(id, lead=self.ref_lead, type=self.sqi_ref_ann)
            self.curr_ecg, self.curr_id = ecg, id
            if len(ann) > self.min_annotation_len:
                self.parse_elem_data(id)        # First deriving rr, rlab, rrt dictionnaries and other basic elements.
                for win in window_sizes:                # Running on all the required windows
                    self.window_sizes = np.unique(np.append(self.window_sizes, win))
                    self.loaded_window_sizes = np.append(self.loaded_window_sizes, win)
                    self._features(id, win, feats=feats)    # Computing all the features
                    if test_anns is None:
                        test_anns = np.setdiff1d(self.annotation_types, self.sqi_ref_ann)
                    for ann_type in test_anns:  # Computing SQI
                        self._sqi(id, win, test_ann=ann_type)
                    self._win_lab(id, win)                  # Generating label for each window
                    self._af_win_lab(id, win)               # Generating binary AF label for each window.
                self.parse_demographic_features(id)         # Parsing demographics
                self._af_pat_lab(id)                        # Generating patient label among the different categories based on AF burden
                self.parse_ahi(id)                          # Computing AHI
                self.parse_odi(id)                          # Computing ODI
                self.save_patient_to_disk(id)               # Saving
            else:
                print("Corrupted ECG.")
                self.corrupted_ecg = np.append(self.corrupted_ecg, id)  # If the recording presents less than 1000 peaks, it is considered as corrupted and is not considered.
            #except:
            #    print('Error')
            #    continue
        self.destroy_pool()

    def _features(self, id, win, feats=cts.IMPLEMENTED_FEATURES):
        """ Generates the features given the patient ID and the window size.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of beats) along which the raw recording is divided.
        :param feats: List of the features to compute. The function 'comp_" + feature name must be implemented in the utils.feature_comp module.
        """
        raw_rr = self.rr_dict[id]
        rr = raw_rr[:(len(raw_rr) // win) * win].reshape(-1, win)
        if id not in self.features_dict.keys():
            self.features_dict[id] = {}
        self.features_dict[id][win] = {}
        for i, feat in enumerate(feats):
            if feat not in self.get_available_features(id, win):
                func = getattr(fc, 'comp_' + feat)
                self.features_dict[id][win][feat] = np.array(self.pool.starmap(func, zip(rr,)))

    def _sqi(self, id, win, lead=1, test_ann='xqrs'):
        """ Computes the Signal Quality Index (SQI) of each window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of beats) along which the raw recording is divided.
        :param lead: The ECG lead for which the annotations were computed.
        :param test_ann: The test annotation to use for the computation of the bsqi function (in utils.feature_comp module)"""

        if id not in self.signal_quality_dict.keys():
            self.signal_quality_dict[id] = {}
        if win not in self.signal_quality_dict[id].keys():
            self.signal_quality_dict[id][win] = {}
        raw_rrt = self.rrt_dict[id]
        rrt = raw_rrt[:(len(raw_rrt) // win) * win].reshape(-1, win)
        if len(self.rrt_dict[id]) % win == 0:               # Adding timestamp of the end of last RR.
            additional_rrt = self.rrt_dict[id][-1] + 1.0
        else:
            additional_rrt = self.rrt_dict[id][(len(raw_rrt) // win) * win]
        rrt = np.concatenate((rrt, np.append(rrt[1:, 0], additional_rrt).reshape(-1, 1)), axis=1)
        refqrs = (rrt * self.actual_fs).astype(int)
        ecg_win_starts = self.start_windows_dict[id][win] * self.actual_fs
        ecg_win_ends = self.end_windows_dict[id][win] * self.actual_fs
        testqrs = self.parse_annotation(id, type=test_ann, lead=lead)
        testqrs = [testqrs[np.where(
            np.logical_and(testqrs > ecg_win_starts[i], testqrs <= ecg_win_ends[i]))] for i in
                   range(len(ecg_win_starts))]
        # self.create_pool()
        self.signal_quality_dict[id][win][test_ann] = np.array(self.pool.starmap(fc.bsqi,
                                                             zip(refqrs, testqrs,
                                                                 self.agw * np.ones(
                                                                     len(testqrs)),
                                                                 self.actual_fs * np.ones(
                                                                     len(testqrs)))))

    def _win_lab(self, id, win):
        """ Computes the label of a window. The label is computed based on the most represented label over the window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of beats) along which the raw recording is divided.
        """
        if id not in self.win_lab_dict.keys():
            self.win_lab_dict[id] = {}
        raw_rlab = self.rlab_dict[id]
        rlab = raw_rlab[:(len(raw_rlab) // win) * win].reshape(-1, win)
        counts = np.array([np.sum((rlab == i), axis=1) for i in range(len(self.rhythms))]).astype(float)
        count_nan = np.sum(np.isnan(rlab), axis=1).astype(float)
        max_lab_count = np.max(counts, axis=0).astype(float)
        self.win_lab_dict[id][win] = np.argmax(counts, axis=0).astype(float)
        self.win_lab_dict[id][win][count_nan > max_lab_count] = np.nan


    def _af_win_lab(self, id, win):
        """ Computes the binary AF label of a window. The label is computed based on the most represented label over the window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of beats) along which the raw recording is divided.
        """
        if id not in self.af_win_lab_dict.keys():
            self.af_win_lab_dict[id] = {}
        self.af_win_lab_dict[id][win] = self.win_lab_dict[id][win] == cts.WINDOW_LABEL_AF

    def _af_pat_lab(self, id):
        """ Computes the AF Burden and the global label for a given patient. The AF Burden is computed as the time
        spent on AF divided by the total time of the recording. The different categories of patients are: Non-AF (Time in AF
        does not exceed 30 [sec], Mild AF (Time in AF above 30 [sec] and AFB under 4%), Moderate AF (AFB between 4 and 80%),
        and Severe AF (AFB between 80 and 100%). If the burden of a given pathology for a patient is over 50%, we flage him as a patient
        suffering from another CVD (label cts.PATIENT_LABEL_OTHER_CVD). As a convention, for windows, 0 is the label for NSR, 1 for AF, and above
        2 for other rhythms.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        # Using minimal window size to have the higher granularity
        win = min(self.loaded_window_sizes)
        raw_rr = self.rr_dict[id][:(len(self.rr_dict[id]) // win) * win].reshape(-1, win)[self.mask_rr_dict[id][win]].reshape(-1)
        raw_rlab = self.rlab_dict[id][:(len(self.rlab_dict[id]) // win) * win].reshape(-1, win)[self.mask_rr_dict[id][win]].reshape(-1)
        if np.all(np.isnan(raw_rlab)):  # Case where the labels are not available (like in SHHS)
            self.af_burden_dict[id] = np.nan
            self.af_pat_lab_dict[id] = np.nan
            self.other_cvd_burden_dict[id] = np.nan
            self.missing_af_label = np.append(self.missing_af_label, id)
        else:
            time_in_af = raw_rr[raw_rlab == cts.WINDOW_LABEL_AF].sum()                                          # Deriving time in AF.
            self.af_burden_dict[id] = time_in_af / self.recording_time[id]                                      # Computing AF Burden.
            self.other_cvd_burden_dict[id] = np.sum(raw_rlab > cts.WINDOW_LABEL_AF) / self.recording_time[id]   # Computing Other CVD Burden.
            if self.af_burden_dict[id] > cts.AF_SEVERE_THRESHOLD:                                               # Assessing the class according to the guidelines
                self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_SEVERE
            elif self.af_burden_dict[id] > cts.AF_MODERATE_THRESHOLD:
                self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_MODERATE
            elif time_in_af > cts.AF_MILD_THRESHOLD:
                self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_AF_MILD
            elif self.other_cvd_burden_dict[id] > 0.5:
                self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_OTHER_CVD
            else:
                self.af_pat_lab_dict[id] = cts.PATIENT_LABEL_NON_AF

    # Assumption: The function name is "comp_feature_name" and is implemented in the feature_comp module.
    def add_feature(self, feature_name, wins=cts.BASE_WINDOWS):
        """ Computes a new feature in the dataset for the specified window sizes. The function 'comp_" + feature name must be implemented in the utils.feature_comp module.
        :param feature_name: The name of the feature to compute.
        :param wins: List of window sizes for which the feature should be computed."""
        existing_features = self.get_available_features()
        if np.any(existing_features == feature_name):
            print("Feature " + feature_name + " already exists in the database")
        else:
            self.create_pool()
            func = getattr(fc, 'comp_' + feature_name)
            for i, id in enumerate(self.non_corrupted_ecg_patients()):
                if i % 100 == 0:
                    print(id)
                for win in wins:
                    if win not in self.loaded_window_sizes:     # In case this window has not been loaded, we load the features dictionnary and add the feature.
                        self.features_dict[id][win] = np.load(self.main_path / id / "features" / (str(win) + '.npy'), allow_pickle=True).item()
                    raw_rr = self.rr_dict[id]
                    rr = raw_rr[:(len(raw_rr) // win) * win].reshape(-1, win)
                    self.features_dict[id][win][feature_name] = np.array(self.pool.starmap(func, zip(rr,)))
                    np.save(self.main_path / id / "features" / str(win), self.features_dict[id][win])
            self.destroy_pool()

    def add_window(self, win):
        """ Generates the data for a whole new window size. If the dataset has already been parsed with regard to
        this window size, the function returns immediately.
        :param win: Integer specifying the window size."""

        if win in self.window_sizes:
            pass
        self.window_sizes = np.append(self.window_sizes, win)
        self.loaded_window_sizes = np.append(self.loaded_window_sizes, win)
        # By default, computing for all available ECGs
        self.create_pool()
        for id in self.non_corrupted_ecg_patients():
            self._features(id, win)
            self._sqi(id, win)
            self._win_lab(id, win)
            self._af_win_lab(id, win)
            self.save_patient_to_disk(id)
        self.destroy_pool()

    def recompute_sqi(self, win_thresh=cts.SQI_WINDOW_THRESHOLD, file_thresh=cts.SQI_FILE_THRESHOLD):
        """ This function fills the low_sqi variable which retains all the files presenting a low SQI.
        The criterion is as follows: if more than X% of the windows have a SQI lower than y, the patient is considered
        as one having a low sqi.
        :param win_thresh: The threshold on SQI to qualify a window of being of bad quality.
        :param file_thresh: The percentage of corrupted windows to consider a file as bad quality."""

        self.low_sqi = np.array([])
        for id in self.non_corrupted_ecg_patients():
            total_mask = np.logical_and(self.mask_rr_dict[id][self.window_size], self.signal_quality_dict[id][self.window_size][self.sqi_test_ann] >= win_thresh)
            self.av_prec_windows_dict[id][self.window_size] = dp.cumsum_reset(total_mask)
            if self.has_low_sqi(id, win_thresh, file_thresh):
                self.low_sqi = np.append(self.low_sqi, id)

    def recompute_ann_exclusions(self, ann_exclusion_rate=cts.MISSING_ANN_THRESHOLD):
        """ This function fills the corrupted_ecg variable which retains all the files presenting a high percentage of missing annotations.
        The criterion is as follows: if more than X% of the windows are missing annotations, the patient is considered
        as corrupted. Also takes in consideration patients presenting annotations with less that 1000 R-Peaks (may happen in SHHS).
        :param ann_exclusion_rate: The percentage of corrupted windows to consider a file as corrupted."""
        if os.path.exists(self.main_path / 'corrupted_ecgs.npy'):
            flat_ecg = np.load(self.main_path / 'corrupted_ecgs.npy')
        else:
            flat_ecg = np.array([])
        percentage_ann = np.array([self.mask_rr_dict[pat][self.window_size].sum() / self.mask_rr_dict[pat][self.window_size].shape[0] for pat in self.all_patients()])
        self.corrupted_ecg = np.append(self.corrupted_ecg, self.all_patients()[percentage_ann < ann_exclusion_rate])
        self.corrupted_ecg = np.unique(np.concatenate((self.corrupted_ecg, flat_ecg)))

    # ------------------------------------------------------------------------- #
    # ------------------------- Getters and setters --------------------------- #
    # ------------------------------------------------------------------------- #

    def set_sqi_test_ann(self, new_test_ann):
        """ This function defines a new annotation type upon which the sqi is returned.
        :param new_test_ann: The new annotation type to use."""
        assert new_test_ann in self.annotation_types
        self.sqi_test_ann = new_test_ann
        self.recompute_sqi()

    def set_window_size(self, win_size):
        """ This function modifies the working window size of the parser. If the window hasn't been loaded yet, the function
        loads it beforehand.
        :param win_size: The new window size."""
        assert np.any(self.window_sizes == win_size)
        if win_size not in self.loaded_window_sizes:
            print("Window size " + str(win_size) + " exists but has not been loaded. Loading this window size...")
            self.load_from_disk(wins=[win_size])
            print("Loading completed.")
        self.window_size = win_size

    def has_low_sqi(self, id, win_thresh=cts.SQI_WINDOW_THRESHOLD, file_thresh=cts.SQI_FILE_THRESHOLD):
        """ This function determines if the whole recording of a patient is of bad quality.
        The criterion is as follows: if more than X% of the windows have a SQI lower than y, the patient is considered
        as one having a low sqi.
        :param win_thresh: The threshold on SQI to qualify a window of being of bad quality.
        :param file_thresh: The percentage of corrupted windows to consider a file as bad quality.
        :param test_ann: The annotation type used to compute the SQI."""
        sig_qual = self.signal_quality_dict[id][self.window_size][self.sqi_test_ann]
        return np.sum(sig_qual < win_thresh) >= file_thresh * len(sig_qual)

    def total_time(self, pat_list=None):
        """ This function returns the total recording time of the database.
        :param pat_list: The list of patients for which the sum should be cumulated. If None, returns the time collected for all the patients.
        :returns total: The total recording time (in Hours)."""
        total = 0.0
        if pat_list is None:
            pat_list = self.parsed_patients()
        for pat in pat_list:
            total += self.recording_time[pat] / cts.N_S_IN_HOUR
        return total

    def total_time_in_af(self, pat_list=None):
        """ This function returns the total time in AF in the database.
        :param pat_list: The list of patients for which the sum should be cumulated. If None, returns the time collected for all the patients.
        :returns total: The total AF time (in Hours)."""

        total = 0.0
        if pat_list is None:
            pat_list = self.parsed_patients()
        for pat in pat_list:
            total += np.sum(self.rr_dict[pat][self.rlab_dict[pat] == cts.WINDOW_LABEL_AF]) / cts.N_S_IN_HOUR
        return total

    def get_raw_ecg_path(self):
        """Returns the path of the raw data.
        :returns path: The path in which the raw recordings are located."""
        return self.raw_ecg_path

    def get_available_preeceding_windows(self, pat_list=None, exclude_low_sqi_win=True, win_thresh=cts.SQI_WINDOW_THRESHOLD):
        """ Returns for each window and for each patient the number of consecutive windows preceding it, i.e.
        the number of preceding windows which were not excluded by the different criteria.
        :param exclude_low_sqi_win: If true, considers the low SQI windows to be removed.
        :param win_thresh: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param pat_list. The list of patients to be considered. If None, returns the result for all the patients.
        :returns dict: A dictionary containing for each patient an array of the preceding windows."""

        if pat_list is None:
            pat_list = self.parsed_patients()

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size] >= win_thresh) for pat in
                     pat_list} #[self.sqi_test_ann]
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in pat_list}

        res_dict = {pat: dp.cumsum_reset(masks[pat]) for pat in masks.keys()}
        return res_dict


    def all_patients(self):
        """ Returns all the patients parsed and loaded in the system.
        :returns list: List of the patients parsed in the system."""
        return np.array(list(self.rr_dict.keys()))

    # Excludes the corrupted ecg (annotation length under 1000)
    def non_corrupted_ecg_patients(self):
        """ Returns all the patients parsed and loaded in the system, whose recording is not considered as corrupted.
        :returns list: List of the non-corrupted patient recordings parsed in the system."""
        return np.setdiff1d(self.all_patients(), self.corrupted_ecg)

    # Excludes the patients without af label (global af label not available)
    def af_label_available_patients(self):
        """ Returns all the patients parsed and loaded in the system, whose AF label is available.
        :returns list: List of the non-corrupted patient recordings parsed in the system."""
        return np.setdiff1d(self.all_patients(), self.missing_af_label)

    # Excludes the patients with low sqi (with the condition defined in has_low_sqi function)
    def high_sqi_patients(self):
        """ Returns all the patients parsed and loaded in the system, whose SQI is high.
        :returns list: List of the recordings with high SQI parsed in the system."""
        return np.setdiff1d(self.all_patients(), self.low_sqi)

    def parsed_patients(self):
        """ Returns the patients already parsed and saved in the preprocessed path.
        :returns list: List of the patients parsed."""
        return np.array([x for x in os.listdir(self.main_path) if os.path.isdir(self.main_path / x)])

    def get_available_features(self, pat=None, win=None):
        """ Returns the features computed during the parsing operation.
        :param pat: The patient ID to verify. If None, selects a random ID.
        :param win: The window size to verify. If None, selects the first window in the list.
        :returns list: The list of the feature names available."""
        if pat is None:
            pat = self.non_corrupted_ecg_patients()[0]
        if win is None:
            win = self.loaded_window_sizes[0]
        return np.array(list(self.features_dict[pat][win].keys()))

    def return_features(self, pat_list=None, feats_list=None, return_binary=True, return_global_label=False, exclude_low_sqi_win=True, win_thresh=cts.SQI_WINDOW_THRESHOLD):
        """Concatenates all the features contained in the whole dataset and returns them in the form X, y.
        :param pat_list: The list of patients for whom the features should be returned.
        :param feats_list: The list of features to return. The columns in the output will correspond to the features given as input.
        :param return_binary: If true, returns the binary AF label, otherwise the general rhythm label (af_win_lab vs. win_lab)
        :param return_global_label: If true, returns as well the global label (patient label) for each patient.
        :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
        :return X: Feature matrix (n_samples, n_features).
        :return y: AF label per window. (n_samples, )"""

        if pat_list is None:
            correct_pat_list = self.non_corrupted_ecg_patients()
        else:
            correct_pat_list = np.array([elem for elem in pat_list])
            if len(correct_pat_list) == 0:
                return np.array([[]]), np.array([])     # Returning empty arrays in case of empty lists.

        if feats_list is None:
            feats_list = self.get_available_features()

        test_pat = self.non_corrupted_ecg_patients()[0]
        n_windows = {elem: len(self.win_lab_dict[elem][self.window_size][self.mask_rr_dict[elem][self.window_size]]) for elem in correct_pat_list}
        tuple_X = ()

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}

        for feat in feats_list:
            if feat == 'sqi':
                to_add = np.concatenate(tuple(
                    self.signal_quality_dict[elem][self.window_size][self.sqi_test_ann][masks[elem]].reshape(-1, 1) for elem in
                    correct_pat_list), axis=0)
            elif type(self.features_dict[test_pat][self.window_size][feat]) == float:
                to_add = np.concatenate(tuple(
                    self.features_dict[elem][self.window_size][feat] * np.ones((np.sum(masks[elem]), 1)) for elem in
                    correct_pat_list), axis=0)
            else:
                to_add = np.concatenate(
                    tuple(self.features_dict[elem][self.window_size][feat][masks[elem][:len(masks[elem])]].reshape(-1, 1) for elem in correct_pat_list),
                    axis=0)
            tuple_X = tuple_X + (to_add,)

        X = np.concatenate(tuple_X, axis=1)

        if return_binary:
            y = np.concatenate(tuple(self.af_win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
        else:
            y = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)

        if return_global_label:
            global_lab = np.concatenate(tuple(self.af_pat_lab_dict[elem] * np.ones(np.sum(masks[elem])) for elem in correct_pat_list), axis=0)
            return X, y, global_lab
        else:
            return X, y

    def return_rr(self, pat_list=None, return_binary=True, return_global_label=False, exclude_low_sqi_win=True, win_thresh=cts.SQI_WINDOW_THRESHOLD):
        """Concatenates all the RR windows contained in the whole dataset and returns them in the form X, y.
        :param pat_list: The list of patients for whom the features should be returned.
        :param return_binary: If true, returns the binary AF label, otherwise the general rhythm label (af_win_lab vs. win_lab)
        :param return_global_label: If true, returns as well the global label (patient label) for each patient.
        :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
        :return X: RR intervals matrix (n_windows, window_size).
        :return y: AF label per window. (n_samples, )"""

        if pat_list is None:
            correct_pat_list = self.non_corrupted_ecg_patients()
        else:
            correct_pat_list = np.array([elem for elem in pat_list])
            if len(correct_pat_list) == 0:
                return np.array([[]]), np.array([])     # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}

        X = np.concatenate(tuple(self.rr_dict[elem][:(len(self.rr_dict[elem]) // self.window_size) * self.window_size].reshape(-1, self.window_size)[masks[elem]] for elem in correct_pat_list), axis=0)
        X_rrt = np.concatenate(tuple(
            self.rrt_dict[elem][:(len(self.rr_dict[elem]) // self.window_size) * self.window_size].reshape(-1, self.window_size)[masks[elem]] for elem in correct_pat_list), axis=0)
        win_start = np.concatenate(tuple(
            self.start_windows_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
        win_end = np.concatenate(tuple(
            self.end_windows_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
        win_lab = np.concatenate(tuple(
            self.rlab_dict[elem][:(len(self.rlab_dict[elem]) // self.window_size) * self.window_size].reshape(-1,
                                                                                                            self.window_size)[
                masks[elem]] for elem in correct_pat_list), axis=0)
        if return_binary:
            y = np.concatenate(tuple(self.af_win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
        else:
            y = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)

        if return_global_label:
            global_lab = np.concatenate(tuple(self.af_pat_lab_dict[elem] * np.ones(np.sum(masks[elem])) for elem in correct_pat_list), axis=0)
            return X, X_rrt, y, global_lab, win_start, win_end
        else:
            return X, X_rrt, y, win_start, win_end, win_lab

    def return_patient_ids(self, pat_list=None, exclude_low_sqi_win=True, win_thresh=cts.SQI_WINDOW_THRESHOLD):
        """Concatenates all the patient IDs contained in the whole dataset and returns them in the form of a single vector of the size of the number of windows.
        :param pat_list: The list of patients for whom the features should be returned.
        :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
        :return y: The patient IDs. (n_samples, )"""

        if pat_list is None:
            correct_list = self.non_corrupted_ecg_patients()
        else:
            correct_list = np.array([elem for elem in pat_list])
            if len(correct_list) == 0:
                return np.array([])     # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][self.sqi_test_ann] >= win_thresh) for pat in correct_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_list}
        num_duplicates = {pat: np.sum(masks[pat]) for pat in masks.keys()}

        y = np.concatenate(tuple(elem * np.ones(num_duplicates[elem], dtype=object) for elem in correct_list), axis=0)

        return y

    def return_preceeding_windows(self, pat_list=None, exclude_low_sqi_win=True, win_thresh=cts.SQI_WINDOW_THRESHOLD):
        """Concatenates all the available preceding windows for the patient IDs contained in the whole dataset and returns them
        in the form of a single vector of the size of the number of windows. This function is helpful to leverage the temporality between windows
        in the DL models.
        :param pat_list: The list of patients for whom the features should be returned.
        :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
        :return y: The number of preceeding windows. (n_samples, )"""

        if pat_list is None:
            correct_list = self.non_corrupted_ecg_patients()
        else:
            correct_list = np.array([elem for elem in pat_list])
            if len(correct_list) == 0:
                return np.array([])     # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][self.sqi_test_ann] >= win_thresh) for pat in correct_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_list}

        y = np.concatenate(tuple(self.av_prec_windows_dict[elem][self.window_size][masks[elem]] for elem in correct_list), axis=0)

        return y
    # ------------------------------------------------------------------------- #
    # ---------------------- Import/Export functions -------------------------- #
    # ------------------------------------------------------------------------- #

    def plot_ecg(self, patient_id, lead, disp_peaks=True, start=0, end=-1, ann_type='epltd0', savefig=False, add_peak=None,
                 correct_peaks=False, format='png'):
        """
        Plots the ECG raw signal with annotation peaks and the RR intervals.
        :param patient_id: The ID of the patient.
        :param lead: ECG lead
        :param disp_peaks: Display or not the R Peaks
        :param start: The beginning of the ECG.
        :param end: The end of the ECG.
        :param ann_type: The type of annotation (can be "epltd", "xqrs", "gqrs")
        :param savefig: Boolean value to indicate if the figure should be saved under cts.SNAPSHOTS_DIR or not.
        """
        ecg, annot = self.parse_raw_ecg(patient_id, lead, start, end, type=ann_type)
        fig, axes = graph.create_figure(subplots=(2, 1), sharex=True)
        timeline = np.arange(0, len(ecg) / self.actual_fs, 1 / self.actual_fs)
        axes[0][0].plot(timeline[:len(ecg)], ecg, label='Signal')
        if disp_peaks:
            axes[0][0].scatter(timeline[annot], ecg[annot], marker='x', color='r', label=ann_type)
            if correct_peaks:
                cannot = i_o.qrs_adjust(ecg=ecg, qrs=annot, fs=self.actual_fs, inputsign=1)
                axes[0][0].scatter(timeline[cannot], ecg[cannot], marker='x', color='orange', label=('c-' + ann_type))
            if add_peak is not None:
                _, annot2 = self.parse_raw_ecg(patient_id, lead, start, end, type=add_peak)
                axes[0][0].scatter(timeline[annot2], ecg[annot2], marker='x', color='purple', label=add_peak)

        rr = np.diff(annot) / self.actual_fs
        axes[1][0].plot(timeline[annot][:-1], rr, label='RR Interval')
        if correct_peaks:
            crr = np.diff(cannot) / self.actual_fs
            axes[1][0].plot(timeline[cannot][:-1], crr, color='orange', label='c-RR Interval')
        graph.complete_figure(fig, axes, x_titles=[[''], ['Time (s)']], y_titles=[['ECG (mV)'], ['RR interval (s)']],
                              legend_fontsize=20, savefig=savefig, format=format, put_legend=False * np.ones(axes.shape, dtype=bool),
                              main_title='ECG_Plot_' + str(self.name) + '_start=' + str(start) + '_end=' + str(end) + "_disp_peaks=" + str(disp_peaks))
        plt.show()

    def export_to_physiozoo(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False, ann_type='epltd0', export_rhythms=False, n_leads=1):
        """ This function exports a given recording to the physiozoo format for further investigation.
        The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
        The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
        :param pat: The patient ID to export.
        :param directory: The directory under which the files will be saved.
        :param start: The beginning timestamp for the ECG portion to be saved.
        :param end: The ending timestamp for the ECG portion to be saved.
        :param force: If True, generate the files anyway, otherwise verifies if the file has already been exported under the directory
        :param export_rhythms: If True, exports a file with the different rhythms over the recording."""

        ecg_header = ['---\n',
                      'Mammal:            human\n',
                      'Fs:                ' + str(cts.EPLTD_FS) + '\n',
                      'Integration_level: electrocardiogram\n',
                      '\n'
                      'Channels:\n']
        channels = [['\n'
                      '    - type:   electrography\n',
                      '      name:   data' + str(i) + '\n',
                      '      unit:   mV\n',
                      '      enable: yes\n'] for i in range(1, n_leads+1)]
        end_header = ['\n',
                      '---\n',
                      '\n']
        [ecg_header.extend(channels[i]) for i in range(n_leads)]
        ecg_header.extend(end_header)

        peaks_header =   ['---\n',
                          'Mammal:            human\n',
                          'Fs:                ' + str(self.actual_fs) + '\n',
                          'Integration_level: electrocardiogram\n',
                          '\n'
                          'Channels:\n',
                          '\n'
                          '    - type:   peak\n',
                          '      name:   interval\n',
                          '      unit:   index\n',
                          '      enable: yes\n',
                          '\n',
                          '---\n',
                          '\n'
                          ]

        sig_qual_header = ['---\n',
                           'type: quality annotation\n',
                           'source file: ',  # To be completed by filename
                           '\n',
                           '---\n',
                           '\n',
                           'Beginning\tEnd\t\tClass\n']

        rhythms_header = copy.deepcopy(sig_qual_header)
        rhythms_header[1] = 'type: rhythms annotation\n'

        if not os.path.exists(directory):
            os.makedirs(directory)



        ecgs = np.concatenate(tuple([self.parse_raw_ecg(pat, start=start, end=end, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in range(1, n_leads+1)]), axis=1)
        ann = self.parse_raw_ecg(pat, lead=self.ref_lead, start=start, end=end, type=ann_type)[1]
        if end == -1:
            end = self.recording_time[pat]

        ecg_full_path = directory / (pat + '_ecg_start_' + str(start) + '_end_' + str(end) + '_n_leads_' + str(n_leads) + '.txt')
        peaks_full_path = directory / (pat + '_peaks_start_' + str(start) + '_end_' + str(end) + '_' + str(ann_type) + '.txt')
        rhythms_full_path = directory / (pat + '_rhythms_start_' + str(start) + '_end_' + str(end) + '.txt')
        # Raw ECG
        join_func = lambda x: ' '.join(['%.2f' % i for i in x])
        ecg_str = np.apply_along_axis(join_func, 1, ecgs)
        if not os.path.exists(ecg_full_path) or force:
            with open(ecg_full_path, 'w+') as ecg_file:
                ecg_file.writelines(ecg_header)
                ecg_file.write('\n'.join(ecg_str))

        # Peaks
        if not os.path.exists(peaks_full_path) or force:
            with open(peaks_full_path, 'w+') as peaks_file:
                peaks_file.writelines(peaks_header)
                peaks_file.write('\n'.join(['%d' % i for i in ann]))

        # Rhythms
        if export_rhythms:
            if not os.path.exists(rhythms_full_path) or force:
                with open(rhythms_full_path, 'w+') as rhythms_file:
                    rhythms_header[2] = 'source file: ' + pat + '_rhythms.txt\n'
                    rhythms_file.writelines(rhythms_header)
                    rhythms = np.array([int(i) for i in self.win_lab_dict[pat][self.window_size]])
                    periods = np.concatenate(([0], np.where(np.diff(rhythms.astype(int)))[0] + 1, [len(rhythms) - 1]))
                    start_idx, end_idx = periods[:-1], periods[1:]
                    final_rhythms = rhythms[start_idx]
                    mask_rhythms = final_rhythms > 0    # We do not keep NSR as rhythm
                    raw_rrt = self.rrt_dict[pat]
                    rrt = raw_rrt[:(len(raw_rrt) // self.window_size) * self.window_size].reshape(-1, self.window_size)
                    start_events, end_events = rrt[start_idx, 0], rrt[end_idx, 0]
                    mask_int_events = np.logical_and(start_events < end, end_events > start)
                    mask_rhythms = np.logical_and(mask_rhythms, mask_int_events)
                    start_events, end_events = start_events[mask_rhythms] - start, end_events[mask_rhythms] - start
                    end_events[end_events > (end - start)] = end - start
                    final_rhythms = final_rhythms[mask_rhythms]
                    final_rhythms_str = np.array([self.rhythms[i][1:] for i in final_rhythms])
                    rhythms_file.write('\n'.join(['%.5f\t%.5f\t%s' % (start_events[i], end_events[i], final_rhythms_str[i]) for i in range(len(start_events))]))

    def report_low_sqi(self):
        if len(self.low_sqi) > 0:
            """Export in an excel table a summary of the excluded patients because of bad quality."""
            low_sqi = np.array([[int(i), self.signal_quality_dict[i][self.window_size][self.sqi_test_ann].mean()] for i in self.low_sqi])
            sorted_idx = np.argsort(low_sqi[:, 1])
            low_sqi = low_sqi[sorted_idx]
            np.savetxt(cts.ERROR_ANALYSIS_DIR / (self.name + '_low_sqi_' + str(self.window_size) + '_beats_' + str(self.sqi_test_ann) + '.csv'), low_sqi, fmt="%d,%.2f", header='PatientID,SQI,Manual Review,Comments', comments='')
        else:
            print("All the patients satisfy the SQI criteria.")

    def print_summary(self, pat_list=None):
        """ Print a summary of the characteristics of the database.
        :param pat_list: The list for which the data should be generated. If None, selects all the patients in the database."""
        if pat_list is None:
            pat_list = self.parsed_patients()

        print("Information about " + str(self.window_size) + "-beats windows dataset:")
        print("Total time: " + str(self.total_time(pat_list)))
        print("Total time in AF: " + str(self.total_time_in_af(pat_list)))
        print("Mean time of the recordings: " + str(np.mean([self.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_list])))
        print("Std time of the recordings: " + str(np.std([self.recording_time[pat] / cts.N_S_IN_HOUR for pat in pat_list])))
        all_labs = np.array([self.af_pat_lab_dict[pat] for pat in pat_list])
        mask_non_af_pat = np.logical_or(all_labs == cts.PATIENT_LABEL_NON_AF, all_labs == cts.PATIENT_LABEL_OTHER_CVD)
        mask_mild_pat = all_labs == cts.PATIENT_LABEL_AF_MILD
        mask_moderate_pat = all_labs == cts.PATIENT_LABEL_AF_MODERATE
        mask_severe_pat = all_labs == cts.PATIENT_LABEL_AF_SEVERE
        print("Total number of patients: " + str(len(pat_list)))
        print("Total number of Non-AF patients: " + str(np.sum(mask_non_af_pat)))
        print("Total number of Mild (AFIB patients: " + str(np.sum(mask_mild_pat)))
        print("Total number of Moderate (AFIB patients: " + str(np.sum(mask_moderate_pat)))
        print("Total number of Severe (AFIB patients: " + str(np.sum(mask_severe_pat)))
        all_wins = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size] for elem in pat_list), axis=0)
        try:
            non_af_wins = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size] for elem in pat_list[mask_non_af_pat]), axis=0)
        except ValueError:
            non_af_wins = np.array([])
        try:
            mild_af_wins = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size] for elem in pat_list[mask_mild_pat]), axis=0)
        except ValueError:
            mild_af_wins = np.array([])

        try:
            moderate_af_wins = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size] for elem in pat_list[mask_moderate_pat]), axis=0)
        except ValueError:
            moderate_af_wins = np.array([])

        try:
            severe_af_wins = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size] for elem in pat_list[mask_severe_pat]), axis=0)
        except ValueError:
            severe_af_wins = np.array([])

        print("Total number of valid windows: " + str(len(all_wins)))
        print("Total number of windows belonging to Non-AF patients: " + str(len(non_af_wins)))
        print("Total number of windows belonging to Mild (AFIB patients: " + str(len(mild_af_wins)))
        print("Total number of windows belonging to Moderate (AFIB patients: " + str(len(moderate_af_wins)))
        print("Total number of windows belonging to Severe (AFIB patients: " + str(len(severe_af_wins)))

        """ Summary of the exclusions"""
        excluded_pat_for_anns = np.intersect1d(self.corrupted_ecg, pat_list)
        excluded_sqi = np.intersect1d(self.low_sqi, pat_list)
        excluded_sqi_not_ann = np.setdiff1d(excluded_sqi, excluded_pat_for_anns)
        all_exclusions = np.concatenate((excluded_pat_for_anns, excluded_sqi_not_ann))
        patients_after_exclusions = np.setdiff1d(pat_list, all_exclusions)


        _, _, y_include_low_sqi, _, _, _ = self.return_rr(pat_list=patients_after_exclusions, exclude_low_sqi_win=False)
        _, _, y_without_low_sqi, _, _, _ = self.return_rr(pat_list=patients_after_exclusions, exclude_low_sqi_win=True)
        print("Excluded patients because of annotations: " + str(len(excluded_pat_for_anns)))
        print("Additional excluded patients because of SQI: " + str(len(excluded_sqi_not_ann)))
        print("Remaining windows after annotations exclusions: " + str(len(y_include_low_sqi)))
        print("Remaining windows after annotations and SQI exclusions: " + str(len(y_without_low_sqi)))

    def __getitem__(self, item):
        """ Returns all the information relative to a given patient in the database.
        :param item: The patient ID.
        :returns pat_dict: Dictionnary including all the relevant information regarding the patient."""
        pat_list = self.parse_available_ids()
        if item not in pat_list:
            raise IndexError('Patient ID does not exist in the database/has not been loaded.')
        pat_dict = {
                        'rr': self.rr_dict[item],
                        'rrt': self.rrt_dict[item],
                        'rlab': self.rlab_dict[item],
                        'sqi': self.signal_quality_dict[item][self.window_size][self.sqi_test_ann],
                        'features': self.features_dict[item][self.window_size],
                        'af_burden': self.af_burden_dict[item],
                        'other_cvd_burden': self.other_cvd_burden_dict[item],
                        'win_lab': self.win_lab_dict[item][self.window_size],
                        'af_win_lab': self.af_win_lab_dict[item][self.window_size],
                        'af_pat_lab': self.af_pat_lab_dict[item],
                        'ahi': self.ahi_dict[item],
                        'odi': self.odi_dict[item]
        }
        return pat_dict


    def create_pool(self):
        """ Creates a multiprocessing pool to generate the features and other computations."""
        if ('pool' in self.__dict__.keys() and self.pool is None) or 'pool' not in self.__dict__.keys():
            print("Create Pool with " + str(cts.N_PROCESSES) + " processes.")
            self.pool = multiprocessing.Pool(cts.N_PROCESSES)

    def get_pool(self, create=True):
        """ Returns the multiprocessing pool of the dataset.
        :param create: If true, creates a pool in case it has not been created yet."""
        if 'pool' in self.__dict__.keys() and self.pool is not None:
            return self.pool
        elif create:
            self.create_pool()
            return self.pool
        else:
            return None

    def destroy_pool(self):
        """ Destroys and deletes the pool, if there is such one."""
        if 'pool' in self.__dict__.keys():
            if self.pool is not None:
                self.pool.close()
                del self.pool

    def save_to_disk(self, parallel=True, n_threads=2):
        """ This function saves under the preprocessing data path all the results.
        :param parallel: If True, a Thread pool is created to divide the workload.
        :param n_threads: If parallel=True, the number of threads to be used to build the pool. """
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        if parallel:
            pool = multiprocessing.pool.ThreadPool(processes=n_threads)
            pool.map(self.save_patient_to_disk, self.non_corrupted_ecg_patients())
            pool.close()
            del pool
        else:
            for pat in self.non_corrupted_ecg_patients():
                self.save_patient_to_disk(pat)
        np.save(self.main_path / "corrupted_ecgs.npy", self.corrupted_ecg)

    def load_from_disk(self, parallel=True, n_threads=2, pat_list=None, wins=None):
        """ This function loads from the disk partly or completely the available preprocessed data.
        :param parallel: If True, a Thread pool is created to divide the workload.
        :param n_threads: If parallel=True, the number of threads to be used to build the pool.
        :param pat_list: The list of patients to load. If None, loads all the parsed patients.
        :param wins: The window sizes to load from the disk."""
        if os.path.exists(self.main_path):
            self.parsed_ecgs = np.array([x for x in os.listdir(self.main_path) if os.path.isdir(self.main_path / x)])
            if pat_list is None:
                pat_list = self.parsed_ecgs
            if wins is None:
                wins = self.window_sizes
                self.loaded_window_sizes = self.window_sizes
            else:
                self.loaded_window_sizes = np.append(self.loaded_window_sizes, wins)
            if parallel:
                pool = multiprocessing.pool.ThreadPool(processes=n_threads)
                pool.starmap(self.load_patient_from_disk, zip(pat_list, [wins for _ in range(len(pat_list))]))
                pool.close()
                del pool
            else:
                for pat in pat_list:
                    self.load_patient_from_disk(pat, wins)
            if os.path.exists(self.main_path / "missing_af_label.npy"):
                self.missing_af_label = np.load(self.main_path / "missing_af_label.npy")
            if os.path.exists(self.main_path / "corrupted_ecgs.npy"):
                self.corrupted_ecg = np.load(self.main_path / "corrupted_ecgs.npy")

            self.recompute_ann_exclusions()
            self.recompute_sqi()
            for id in self.non_corrupted_ecg_patients():
                if np.isnan(self.af_pat_lab_dict[id]):
                    self.missing_af_label = np.append(self.missing_af_label, id)


    def save_patient_to_disk(self, pat):
        """ This function saves one patient to the disk.
        :param pat: The patient ID to be saved. """
        win_indep_subfolders = ['rr', 'rlab', 'rrt', 'excluded_portions']
        win_dep_subfolders = ['features', 'signal_quality', 'af_win_lab', 'win_lab', 'start_windows', 'end_windows', 'av_prec_windows', 'mask_rr']
        if not os.path.exists(self.main_path / pat):
            os.makedirs(self.main_path / pat)
        for dir in win_indep_subfolders:
            np.save(self.main_path / pat / dir, self.__dict__[dir + '_dict'][pat])
        for dir in win_dep_subfolders:
            if not os.path.exists(self.main_path / pat / dir):
                os.makedirs(self.main_path / pat / dir)
            for win in self.__dict__[dir + '_dict'][pat].keys():
                if not os.path.exists(self.main_path / pat / dir):
                    os.makedirs(self.main_path / pat / dir)
                np.save(self.main_path / pat / dir / str(win), self.__dict__[dir + '_dict'][pat][win])
        np.save(self.main_path / pat / 'pat_values', np.array([self.af_burden_dict[pat],
                                                                    self.other_cvd_burden_dict[pat],
                                                                    self.af_pat_lab_dict[pat],
                                                                    self.ahi_dict[pat],
                                                                    self.odi_dict[pat],
                                                                    self.recording_time[pat],
                                                                    self.n_excluded_windows_dict[pat]]))

    def load_patient_from_disk(self, pat, wins=None):
        """ This function loads one patient to the disk.
        :param pat: The patient ID to be saved.
        :param wins: The windows to load.
        """
        win_indep_subfolders = ['rr', 'rlab', 'rrt', 'excluded_portions']
        win_dep_subfolders = ['features', 'signal_quality', 'af_win_lab', 'win_lab', 'start_windows', 'end_windows', 'av_prec_windows', 'mask_rr', 'n_excluded_windows']
        if wins is None:
            wins = self.window_sizes
        for dir in win_indep_subfolders:
            if os.path.exists(self.main_path / pat / (dir + '.npy')):
                self.__dict__[dir + '_dict'][pat] = np.load(self.main_path / pat / (dir + '.npy'), allow_pickle=True)
        for dir in win_dep_subfolders:
            if os.path.exists(self.main_path / pat / dir):
                self.__dict__[dir + '_dict'][pat] = {}
                for win in wins:
                    res = np.load(self.main_path / pat / dir / (str(win) + '.npy'), allow_pickle=True)
                    if dir == 'features' or dir == 'signal_quality':
                        self.__dict__[dir + '_dict'][pat][win] = res.item()
                    else:
                        self.__dict__[dir + '_dict'][pat][win] = res

        vals = np.load(self.main_path / pat / 'pat_values.npy', allow_pickle=True)
        self.af_burden_dict[pat] = vals[0]
        self.other_cvd_burden_dict[pat] = vals[1]
        self.af_pat_lab_dict[pat] = vals[2]
        self.ahi_dict[pat] = vals[3]
        self.odi_dict[pat] = vals[4]
        self.recording_time[pat] = vals[5]
        self.n_excluded_windows_dict[pat] = vals[6]

    def plot_AF_win(self, af_df, start, end, figname='AF_events.png'):
        self.parse_circadian_features(af_df.id.unique()[0])
        time = self.circadian_dict[af_df.id.unique()[0]]['start_recording']
        point = 5 * self.actual_fs  # taking 5 sec before and after each segment
        ecg, annot = self.parse_raw_ecg(af_df.id.unique()[0])
        if end == -1:
            end = len(ecg)
        ecg_lead = bandpass_filter(data=ecg, id=af_df.id.unique()[0], lead='x', lowcut=0.67, highcut=90,
                                   signal_freq=self.actual_fs,
                                   filter_order=75, notch_freq=50, debug=False)
        ecg_mask = ecg_lead.copy()
        for i, r in af_df.iterrows():
            ecg_mask[int((r.start_time - start) * self.actual_fs) - point:int((r.end_time - start) * self.actual_fs) + point] = 0
        af_ecg = ecg_lead - ecg_mask
        seg_af_ecg = af_ecg[start:end]
        plt.style.use('seaborn-white')
        fig, ax = plt.subplots(figsize=(14, 3), ncols=1)
        ax.plot(seg_af_ecg, linewidth=0.7, zorder=0)
        for i, r in af_df.iterrows():
            if int(r.start_time) > start:
                ax.fill_between((int((r.start_time - start) * self.actual_fs), int((r.end_time - start) * self.actual_fs)),
                                -2,
                                2, facecolor='orange', alpha=0.4,
                                zorder=1)
        x_time = np.arange(start, end, (end - start) / 10)
        str_time = pd.to_datetime(pd.Series(x_time / self.actual_fs) + time, unit='s', utc=True).dt.strftime('%H:%M:%S')
        ax.set(xticks=(x_time - start))
        ax.set(yticks=[-1, 0, 1])
        ax.set_xticklabels(str_time, fontsize=14)
        ax.set_xlabel('time [%H:%M:%S]', fontsize=16)
        ax.set_ylabel('Amplitude [mV]', fontsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_xlim([start, end - 10000])
        ax.set_ylim([-2, 2])
        fig.tight_layout()
        plt.savefig(cts.SNAPSHOTS_DIR / 'Lund' / 'figs' / self.name / figname, dpi=400, transparent=True)
        plt.close()

    def export_mat(self, df_af, n_lead, ann_type='epltd0'):
        for i, r in df_af.iterrows():
            ecg = []
            annot = []
            for j in range(1, n_lead+1):
                ecg_j, ann_j = self.parse_raw_ecg(r.id, j, r.start_time, r.end_time, type=self.sqi_ref_ann)
                cann_j = i_o.qrs_adjust(ecg=ecg_j,qrs=ann_j,fs=self.actual_fs,inputsign=1, debug=0)
                annot.append(cann_j)
                ecg.append(ecg_j)
            data = np.array(ecg)
            rqrs = np.array(annot)
            if self.name=='UVAFDB':
                mdic = {"data": data, "rqrs": rqrs, "fs": self.actual_fs, "start_": r.start_time, "end_": r.end_time}
            else:
                mdic = {"data": data, "rqrs": rqrs, "fs": self.actual_fs, "start_": r.start_time, "end_": r.end_time,
                        "recording_hour": self.circadian_dict[r.id]['start_recording']}
            savemat(
                "/home/shanybiton/repos/Lund/wins/" + self.name + "/" + str(r.id) + "_af_win_start_" + str(r.start_time) + "_end_" + str(
                    r.end_time) + ".mat", mdic)

    def plot_win_len(self, df, start_bin=20, end_bin=120, step=10, figsize=(8, 8), savefig=False,
                     figname='window_length.png'):
        periods = df.end_time - df.start_time
        n_bins = np.arange(start_bin, end_bin, step)
        counts, bin_edges = np.histogram(periods, bins=n_bins)
        _range = np.ceil(bin_edges).astype(int).astype(str)

        center = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.style.use('seaborn-white')

        fig, ax = plt.subplots(figsize=figsize, ncols=1)
        ax.hist((df.end_time - df.start_time), bins=n_bins,
                alpha=1, color="#307da6", rwidth=0.85)
        plt.axvline((df.end_time - df.start_time).mean(), color='k', linestyle='dashed', linewidth=1,
                    label='Mean window length: {:.2f} sec'.format(np.mean(df.end_time - df.start_time)))
        # tick_lab = ['15k', '30k', '45k', '60k']
        # ax.set(yticks=[15000, 30000, 45000, 60000])
        ax.set(xticks=center)
        # ax.set_xticklabels(list(np.arange(2, n_bins - 1)), fontsize=22)
        # ax.set_yticklabels(tick_lab, fontsize=22)
        ylabels = ['{:,.2f}'.format(x) + 'K' for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.xaxis.set_tick_params(labelsize=16, labelrotation=45)
        ax.set_xlabel('Window length (sec)', fontsize=24)
        ax.set_ylabel('Counts', fontsize=24)
        ax.legend()

        fig.tight_layout()
        s = 0

        for p in ax.patches:
            s += p.get_height()

        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2.,
                    p.get_height(),
                    '{:.1%}'.format(p.get_height() / np.sum(counts)),
                    fontsize=10,
                    color='red',
                    ha='center',
                    va='bottom')
        if savefig:
            fig.savefig(cts.SNAPSHOTS_DIR / self.name[:4] / figname, dpi=400, transparent=True)
        plt.close()
        print('Number of events excluded from histogram is: {}, meaning {:.1%} of events are not shown'.format(
            len(periods.loc[periods > end_bin]), len(periods.loc[periods > end_bin]) / len(df)))
        return

    def export_model_output(self, pat, af_df, directory=cts.ERROR_ANALYSIS_DIR, force=False, algo='ArNet2'):
        rhythms_header = ['---\n',
                           'type: quality annotation\n',
                           'source file: ',  # To be completed by filename
                           '\n',
                           '---\n',
                           '\n',
                           'Beginning\tEnd\t\tClass\n']
        rhythms_header[1] = 'type: rhythms annotation\n'

        rhythms_full_path = directory / f'{pat}_rhythms_output_{algo}.txt'
        start_events = np.array(af_df.start_time.values)
        end_events = np.array(af_df.end_time.values)
        final_rhythms_str = np.array(['AFIB']* len(end_events))
        if not os.path.exists(rhythms_full_path) or force:
                with open(rhythms_full_path, 'w+') as rhythms_file:
                    rhythms_header[2] = 'source file: ' + pat + '_rhythms.txt\n'
                    rhythms_file.writelines(rhythms_header)
                    rhythms_file.write('\n'.join(['%.5f\t%.5f\t%s' % (start_events[i], end_events[i], final_rhythms_str[i]) for i in range(len(start_events))]))
        return


if __name__ == '__main__':
    from base_packages import *
    import consts as cts

else:
    from base_packages import *
    import consts as cts

import wfdb.processing as processing
import multiprocessing
from itertools import chain

def pad_rhythm(rhythm, missing=None):
    """ Helper function which recieves the changes in the cardiac rhythm labels and pads the whole vector.
        Example:
            in = ['AFIB', '', '', '', '', '', 'N', '', '', '', 'SBR', '']
            out = ['AFIB', 'AFIB', 'AFIB', 'AFIB', 'AFIB', 'AFIB', 'N', 'N', 'N', 'N', 'SBR', 'SBR']
        This function in used to parse the '.bea' files summarizing the beats detected in the UVAF database.
    :param rhythm:      The input vector representing the changes in the cardiac rhythm (list of strings or labels).
    :param missing:     The different strings or labels (list) which characterize a missing rhythm. (If None, considering only '' as a missing rhythm)
    :returns rhythm:    The padded vector of rhythms.
    """
    cond = np.ones(len(rhythm), dtype=bool)
    if missing != None:
        for char in missing:
            cond = np.logical_and(cond, rhythm != char)
    else:
        cond = rhythm != missing
    not_none = np.where(cond)[0]
    if len(not_none) == 0:
        rhythm = np.array(['(N']*len(rhythm))
    else:
        not_none = np.append(not_none, len(rhythm))
        diffs = np.diff(not_none)
        sing_rhy = rhythm[not_none[:-1]]
        rhythm[not_none[0]:] = np.repeat(sing_rhy, diffs)
        rhythm[0:not_none[0]] = sing_rhy[0]
    return rhythm


def str_to_float(string):
    """ This function converts a string to a float number, if possible. Otherwise, return np.nan.
    :param string: The input string representing a floating point number.
    :return value: The corresponding float value.
    """
    try:
        return float(string)
    except ValueError:
        return np.nan


def read_csv_with_headers(path, mask_pat=None):

    """ This function reads a '.csv' file containing headers.
    :param path:        The path of the '.csv' input file.
    :param mask_pat:    Mask to select only some part of the patients (boolean numpy array).
    :return column:     Dictionnary containing as key the headers and as values the different corresponding columns.
    """

    f = open(path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)
    column = {}
    for h in headers:
        column[h] = []

    converters = [str_to_float] * (len(headers))

    for row in reader:
        for h, v, conv in zip(headers, row, converters):
            column[h].append(conv(v))

    keys = list(column.keys())
    for k in keys:
        if mask_pat is None:
            column[k] = np.array(column[k])
        else:
            column[k] = np.array(column[k])[mask_pat]
        column[k.lower()] = column.pop(k)
    return column


# Calls C code for epltd dtection. Assumes the ecg file with .dat format is present in the code directory (parsing project).
# Assumes ID is received as a string with the correct format.
def epltd0_detector(id, prog_dir=cts.EPLTD_PROG_DIR.as_posix(), ecg_dir=cts.PARSING_PROJECT_DIR.as_posix(), n_windows=1000, pool=None):

    """ This function generates a wfdb annotation based on the 'epltd' detector containing the peaks of an input wfdb ECG recording.
    The function calls the C code for epltd detection previously compiled. By default, the function assumes the ECG recording is present in the
    directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
    :param id:          The ID of the patient.
    :param prog_dir:    The directory of the EPLTD C program to be ran.
    :param ecg_dir:     The directory of the wfdb ECG input recording.
    :param n_windows:   Unused. Received as parameter to match the other detectors structure.
    :param pool:        Unused. Received as parameter to match the other detectors structure.
    :returns:           None

    """
    command = ';'.join([ 'EPLTD_PROG_DIR=' + prog_dir,
                'ECG_DIR=' + ecg_dir,
                'cd $ECG_DIR',
                'command=\"$EPLTD_PROG_DIR -r ' + str(id) + '\"',
                'eval $command'])
    if os.name == 'nt':
        command = 'wsl ' + command
    os.system(command)


def jqrs(ecg, fs, thr, rp, debug):
    '''The function is an Implementation of an energy based qrs detector [1]_. The algorithm is an
    adaptation of the popular Pan & Tompkins algorithm [2]_. The function assumes
    the input ecg is already pre-filtered i.e. bandpass filtered and that the
    power-line interference was removed. Of note, NaN should be represented by the
    value -32768 in the ecg (WFDB standard).
    .. [1] Behar, Joachim, Alistair Johnson, Gari D. Clifford, and Julien Oster.
        "A comparison of single channel fetal ECG extraction methods." Annals of
        biomedical engineering 42, no. 6 (2014): 1340-1353.
    .. [2] Pan, Jiapu, and Willis J. Tompkins. "A real-time QRS detection algorithm."
        IEEE Trans. Biomed. Eng 32.3 (1985): 230-236.
    :param ecg: vector of ecg signal amplitude (mV)
    :param fs: sampling frequency (Hz)
    :param thr: threshold (nu)
    :param rp: refractory period (sec)
    :param debug: plot results (boolean)
    :return: qrs_pos: position of the qrs (sample)
    '''
    INT_NB_COEFF = int(np.round(7 * fs / 256)) # length is 30 for fs=256Hz
    dffecg = np.diff(ecg) # differenciate (one datapoint shorter)
    sqrecg = np.square(dffecg) # square ecg
    intecg = signal.lfilter(np.ones(INT_NB_COEFF, dtype=int),
                            1, sqrecg) # integrate
    mdfint = intecg
    delay = math.ceil(INT_NB_COEFF / 2)
    mdfint = np.roll(mdfint, -delay) # remove filter delay for scanning back through ecg
    # thresholding
    mdfint_temp = mdfint
    mdfint_temp_ = np.delete(mdfint_temp, np.where(ecg==-32768)) # exclude the NaN (encoded in WFDB format)
    xs = np.sort(mdfint_temp)
    ind_xs = int(np.round(98 / 100 * len(xs)))
    en_thres = xs[ind_xs]
    poss_reg = mdfint > thr * en_thres
    tm = np.arange(start=1/fs, stop=(len(ecg)+1)/fs, step=1/fs).reshape(1, -1)
    # search back
    SEARCH_BACK = 1
    if SEARCH_BACK:
        indAboveThreshold = np.where(poss_reg)[0] # indices of samples above threshold
        RRv = np.diff(tm[0, indAboveThreshold]) # compute RRv
        medRRv = np.median(RRv[RRv > 0.01])
        indMissedBeat = np.where(RRv > 1.5 * medRRv)[0] # missed a peak?
        # find interval onto which a beat might have been missed
        indStart = indAboveThreshold[indMissedBeat]
        indEnd = indAboveThreshold[indMissedBeat + 1]
        for i in range(0, len(indStart)):
            # look for a peak on this interval by lowering the energy threshold
            poss_reg[indStart[i]: indEnd[i]] = mdfint[indStart[i]: indEnd[i]] > (0.25 * thr * en_thres)
    # find indices into boudaries of each segment
    left = np.where(np.diff(np.pad(1 * poss_reg, (1, 0), 'constant')) == 1)[0] # remember to zero pad at start
    right = np.where(np.diff(np.pad(1 * poss_reg, (0, 1), 'constant')) == -1)[0] # remember to zero pad at end
    nb_s = len(left < 30 * fs)
    loc = np.zeros([1, nb_s], dtype=int)
    for j in range(0, nb_s):
        loc[0, j] = np.argmax(np.abs(ecg[left[j]:right[j] + 1]))
        loc[0, j] = int(loc[0, j] + left[j])
    sign = np.median(ecg[loc])
    # loop through all possibilities
    compt = 0
    NB_PEAKS = len(left)
    maxval = np.zeros([NB_PEAKS])
    maxloc = np.zeros([NB_PEAKS], dtype=int)
    for j in range(0, NB_PEAKS):
        if sign > 0:
            # if sign is positive then look for positive peaks
            maxval[compt] = np.max(ecg[left[j]:right[j] + 1])
            maxloc[compt] = np.argmax(ecg[left[j]:right[j] + 1])
        else:
            # if sign is negative then look for negative peaks
            maxval[compt] = np.min(ecg[left[j]:right[j] + 1])
            maxloc[compt] = np.argmin(ecg[left[j]:right[j] + 1])
        maxloc[compt] = maxloc[compt] + left[j]
        # refractory period - has proved to improve results
        if compt > 0:
            if (maxloc[compt] - maxloc[compt - 1] < fs * rp) & (np.abs(maxval[compt]) < np.abs(maxval[compt - 1])):
                maxval = np.delete(maxval, compt)
                maxloc = np.delete(maxloc, compt)
            elif (maxloc[compt] - maxloc[compt - 1] < fs * rp) & (np.abs(maxval[compt]) >= np.abs(maxval[compt - 1])):
                maxval = np.delete(maxval, compt - 1)
                maxloc = np.delete(maxloc, compt - 1)
            else:
                compt = compt + 1
        else:
            # if first peak then increment
            compt = compt + 1
    qrs_pos = maxloc # datapoints QRS positions
    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.plot(tm[0, :], np.append(mdfint, 0))
        ax1.plot(tm[0, left], mdfint[left], 'og')
        ax1.plot(tm[0, right], mdfint[right], 'om')
        ax1.title.set_text('raw ECG (blue) and zero-pahse FIR filtered ECG (red)')
        ax2.plot(tm[0, :], ecg)
        ax2.plot(tm[0, qrs_pos], ecg[qrs_pos], '+r')
        ax2.title.set_text('Integrated ecg with scan boundaries over scaled ECG')
        ax3.plot(tm[0, qrs_pos[0:len(qrs_pos) - 1]], np.diff(qrs_pos) / fs, '+r')
        ax3.title.set_text('ECG with R-peaks (black) and S-points (green) over ECG')
        fig.show()
    return qrs_pos


def wqrs_detector(id, prog_dir=cts.WQRS_PROG_DIR.as_posix(), ecg_dir=cts.PARSING_PROJECT_DIR.as_posix(), pool=None):
    """ This function generates a wfdb annotation based on the 'wqrs' detector containing the peaks of an input wfdb ECG recording.
    The function calls the C code for wqrs detection previously compiled. By default, the function assumes the ECG recording is present in the
    directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
    :param id:          The ID of the patient.
    :param prog_dir:    The directory of the WQRS C program to be ran.
    :param ecg_dir:     The directory of the wfdb ECG input recording.
    :param n_windows:   Unused. Received as parameter to match the other detectors structure.
    :param pool:        Unused. Received as parameter to match the other detectors structure.
    :returns:           None
    """
    command = ';'.join([ 'WQRS_PROG_DIR=' + prog_dir,
                'ECG_DIR=' + ecg_dir,
                'cd $ECG_DIR',
                'command=\"$WQRS_PROG_DIR -r ' + str(id) + '\"',
                'eval $command'])
    if os.name == 'nt':
        command = 'wsl ' + command
    os.system(command)


def xqrs_detector(id, ecg_dir=cts.PARSING_PROJECT_DIR, n_windows=2000, pool=None):

    """ This function generates a wfdb annotation based on the 'xqrs' detector containing the peaks of an input wfdb ECG recording.
    The function calls the C code for epltd detection previously compiled. By default, the function assumes the ECG recording is present in the
    directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
    :param id:          The ID of the patient.
    :param prog_dir:    The directory of the EPLTD C program to be ran.
    :param ecg_dir:     The directory of the wfdb ECG input recording.
    :param n_windows:   The number of windows to divide the ECG into. Each window is duplicated to ensure all the peaks are detected.
    :param pool:        Pool of multiprocessors to be used to divide the workload on several cores. Each core deals with a window at a time.
    :returns:           None

    """

    record = wfdb.rdrecord(str(ecg_dir / id))
    fs = record.fs
    ecg = record.p_signal[:, 0]
    pool_to_close = False
    if n_windows > 1:
        if pool is None:
            pool = multiprocessing.Pool(cts.N_PROCESSES)
            pool_to_close = True
        borders = np.round(np.linspace(0, len(ecg), n_windows + 1)).astype(int)
        ecg_wins = [ecg[borders[i]:borders[i+1]] for i in range(len(borders) - 1)]
        lengths = np.array([len(e) for e in ecg_wins])
        ecg_wins = [np.tile(e, 2) for e in ecg_wins]
        fss = fs * np.ones(n_windows)
        sampfrom = np.zeros(n_windows, dtype=int)
        sampto = 'end' * np.ones(n_windows, dtype=object)
        conf = np.array([None] * n_windows)
        learn = True * np.ones(n_windows, dtype=bool)
        verbose = False * np.ones(n_windows, dtype=bool)
        res = pool.starmap(processing.xqrs_detect, zip(ecg_wins, fss, sampfrom, sampto, conf, learn, verbose))
        res = [res[i][res[i] > lengths[i]] - lengths[i] for i in range(len(res))]
        ann = np.concatenate(tuple([res[i] + borders[i] for i in range(n_windows)])).astype(int)
        if pool_to_close:
            pool.close()
    else:
        ann = processing.xqrs_detect(ecg, fs, verbose=True)
    print('done')
    wfdb.wrann(id, 'xqrs', ann, symbol=['q'] * len(ann), write_dir='/home/shanybiton/repos/Generalization/parsing/')


def gqrs_detector(id, ecg_dir=cts.PARSING_PROJECT_DIR, n_windows=2000, pool=None):

    """ This function generates a wfdb annotation based on the 'gqrs' detector containing the peaks of an input wfdb ECG recording.
    The function calls the C code for epltd detection previously compiled. By default, the function assumes the ECG recording is present in the
    directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
    :param id:          The ID of the patient.
    :param prog_dir:    The directory of the EPLTD C program to be ran.
    :param ecg_dir:     The directory of the wfdb ECG input recording.
    :param n_windows:   The number of windows to divide the ECG into. Each window is duplicated to ensure all the peaks are detected.
    :param pool:        Pool of multiprocessors to be used to divide the workload on several cores. Each core deals with a window at a time.
    :returns:           None

    """

    record = wfdb.rdrecord(str(ecg_dir / id))
    fs = record.fs
    ecg = record.p_signal[:, 0]
    pool_to_close = False
    if n_windows > 1:
        if pool is None:
            pool = multiprocessing.Pool(cts.N_PROCESSES)
            pool_to_close = True
        borders = np.round(np.linspace(0, len(ecg), n_windows + 1)).astype(int)
        ecg_wins = [ecg[borders[i]:borders[i + 1]] for i in range(len(borders) - 1)]
        lengths = np.array([len(e) for e in ecg_wins])
        ecg_wins = [np.tile(e, 2) for e in ecg_wins]
        fss = fs * np.ones(n_windows)
        res = pool.starmap(processing.gqrs_detect, zip(ecg_wins, fss))
        res = [res[i][res[i] > lengths[i]] - lengths[i] for i in range(len(res))]
        ann = np.concatenate(tuple([res[i] + borders[i] for i in range(n_windows)])).astype(int)
        if pool_to_close:
            pool.close()
    else:
        ann = processing.gqrs_detect(ecg, fs)
    wfdb.wrann(id, 'gqrs', ann, symbol=['q'] * len(ann))


def jqrs_detector(id, ecg_dir=cts.PARSING_PROJECT_DIR, n_windows=1, pool=None, fs=200, THRES=cts.SQI_WINDOW_THRESHOLD, REF_PERIOD=0.52, DEB=False):
    """
    This function generates a wfdb annotation based on the 'jqrs' detector containing the peaks of an input wfdb ECG recording.
        The function calls the function for jqrs detection impleneted above. By default, the function assumes the ECG recording is present in the
        directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
        By default, the function assumes the ECG recording is present in the directory under the name "id.dat, id.hea",
        and further assumes the signal has been sampled with a frequency of 200 [Hz].
        :param id:          The ID of the patient.
        :param ecg_dir:     The directory of the wfdb ECG input recording.
        :param n_windows:   The number of windows to divide the ECG into. Each window is duplicated to ensure all the peaks are detected.
        :param pool:        Pool of multiprocessors to be used to divide the workload on several cores. Each core deals with a window at a time.
        :returns:           None
    """
    record = wfdb.rdrecord(str(ecg_dir / id))
    ecg = record.p_signal[:, 0]
    pool_to_close = False
    if n_windows > 1:
        if pool is None:
            pool = multiprocessing.Pool(cts.N_PROCESSES)
            pool_to_close = True
        borders = np.round(np.linspace(0, len(ecg), n_windows + 1)).astype(int)
        ecg_wins = [ecg[borders[i]:borders[i + 1]] for i in range(len(borders) - 1)]
        lengths = np.array([len(e) for e in ecg_wins])
        ecg_wins = [np.tile(e, 2) for e in ecg_wins]
        fss = fs * np.ones(n_windows)
        thr = THRES * np.ones(n_windows)
        rp = REF_PERIOD * np.ones(n_windows)
        debug = DEB * np.ones(n_windows)
        res = pool.starmap(jqrs, zip(ecg_wins, fss, thr, rp, debug))
        res = [res[i][res[i] > lengths[i]] - lengths[i] for i in range(len(res))]
        ann = np.concatenate(tuple([res[i] + borders[i] for i in range(n_windows)])).astype(int)
        if pool_to_close:
            pool.close()
    else:
        ann = jqrs(ecg, fs, THRES, REF_PERIOD, DEB)
    wfdb.wrann(id, 'jqrs', ann, symbol=['q'] * len(ann))

def qrs_adjust(ecg,qrs,fs,inputsign,tol=0.05,debug=0):
    """
    This function is used to adjust the qrs location by looking for a local min or max around the unput qrs points.
    Adjusted from mhrv tooblox for matlab.
        :param ecg:         array of ecg signal amplitude (mV)
        :param qrs:         peak position (sample number)
        :param fs:          sampling frequency (Hz)
        :param inputsign:   sign of the peak to look for (-1 for negative and 1 for positive)
        :param tol:         tolerance window (sec)
        :param debug:       (boolean)
        :return: cqrs:      adjusted (corrected) qrs positions (sample number)
    """
    # general
    cqrs = np.zeros(qrs.shape).reshape(-1, 1)  # corrected qrs array
    NB_qrs = len(qrs)
    WINDOW = np.ceil(fs*tol) # allow a maximum of tol in sec shift
    NB_SAMPLES = len(ecg)

    # local refinement to correct for sampling freq resolution
    for i in range(NB_qrs):
        if qrs[i] > WINDOW and qrs[i]+WINDOW < NB_SAMPLES:
            if inputsign >0:
                indm = np.argmax(ecg[int(qrs[i]-WINDOW):int(qrs[i]+WINDOW)])
            else:
                indm = np.argmin(ecg[int(qrs[i]-WINDOW):int(qrs[i]+WINDOW)])
            cqrs[i] = qrs[i] + indm - WINDOW
        elif qrs[i] < WINDOW:
            # managing left border
            if inputsign > 0:
                indm = np.argmax(ecg[1:int(qrs[i]+WINDOW)])
            else:
                indm = np.min(ecg[1:int(qrs[i]+WINDOW)])
            cqrs[i] = indm
        elif qrs[i] + WINDOW > NB_SAMPLES:
            # managing right border
            if inputsign > 0:
                indm = np.argmax(ecg[int(qrs[i]-WINDOW):])
            else:
                indm = np.argmin(ecg[int(qrs[i]-WINDOW):])
            cqrs[i] = qrs[i]+indm-WINDOW
        else:
            cqrs[i]=qrs[i]
        cqrs = cqrs.reshape(1,-1).astype(int).flatten()
    if debug:
        plt.plot(ecg)
        plt.scatter(qrs, ecg[qrs], marker='x', color='r', label='qrs')
        plt.scatter(cqrs, ecg[cqrs], marker='x', color='purple', label='cqrs')
        plt.show()
        plt.close()
    return cqrs

def qrs_adjust_detector(ecg, qrs, fs, INPUTSIGN, TOL=0.05, DEB=0, n_windows=1, pool=None):
    """
    This function generates a wfdb annotation based on the 'jqrs' detector containing the peaks of an input wfdb ECG recording.
        The function calls the function for jqrs detection impleneted above. By default, the function assumes the ECG recording is present in the
        directory under the name "id.dat, id.hea", and further assumes the signal has been sampled with a frequency of 200 [Hz].
        By default, the function assumes the ECG recording is present in the directory under the name "id.dat, id.hea",
        and further assumes the signal has been sampled with a frequency of 200 [Hz].
        :param id:          The ID of the patient.
        :param ecg_dir:     The directory of the wfdb ECG input recording.
        :param n_windows:   The number of windows to divide the ECG into. Each window is duplicated to ensure all the peaks are detected.
        :param pool:        Pool of multiprocessors to be used to divide the workload on several cores. Each core deals with a window at a time.
        :returns:           None
    """
    pool_to_close = False
    if n_windows > 1:
        if pool is None:
            pool = multiprocessing.Pool(cts.N_PROCESSES)
            pool_to_close = True
        borders = np.round(np.linspace(0, len(ecg), n_windows + 1)).astype(int)
        borders = borders[np.where(borders <= qrs[-1])]
        ecg_wins = [ecg[borders[i]:borders[i + 1]] for i in range(len(borders) - 1)]
        qrs_wins = [qrs[np.where(np.logical_and(qrs >= borders[i], qrs < borders[i + 1]))]-borders[i] for i in range(len(borders) - 1)]
        lengths = np.array([len(e) for e in ecg_wins])
        n_windows = len(lengths)
        ecg_wins = [np.tile(e, 2) for e in ecg_wins]
        qrs_wins = [np.tile(e, 2) for e in qrs_wins]
        fss = fs * np.ones(n_windows)
        debug = DEB * np.ones(n_windows)
        tol = TOL * np.ones(n_windows)
        inputsign = INPUTSIGN * np.ones(n_windows)
        res = pool.starmap(qrs_adjust, zip(ecg_wins, qrs_wins, fss, inputsign, tol, debug))
        # res = [res[i][res[i] > lengths[i]] - lengths[i] for i in range(len(res))]
        combined = list(itertools.chain.from_iterable([res[i] + borders[i] for i in range(n_windows)]))
        ann = np.array(combined).astype(int)
        if pool_to_close:
            pool.close()
    else:
        ann = qrs_adjust(ecg, qrs, fs, INPUTSIGN, tol=TOL, debug=DEB)
    return ann
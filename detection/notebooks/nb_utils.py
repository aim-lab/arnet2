import os
import numpy as np
import pandas as pd
import wfdb
import requests
import pickle
import subprocess


def load_ecg_record(record_name, db_path, sampling_rate, lead):
    record = wfdb.rdrecord(f'{db_path}/{record_name}')
    if record.fs != sampling_rate:
        print(f'Warning: {record_name} has fs={record.fs}, expected {sampling_rate}')
        return None
    signal = record.p_signal[:, lead-1]
    return signal


def create_ecg_windows(signal, window_size_samples):
    n_windows = len(signal) // window_size_samples
    windows = []
    for i in range(n_windows):
        start = i * window_size_samples
        end = (i + 1) * window_size_samples
        window = signal[start:end]
        windows.append(window)
    return np.array(windows)


def download_file(url, filename):
    """
    Downloads a file from a specified URL and saves it to the given filename.

    :param url: URL to the file to be downloaded.
    :param filename: Path where the downloaded file will be saved.
    """
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)


def url_exists(url):
    """Return True if the URL exists (status code 200)."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False



def pad_rhythm(rhythm, missing=None):
    """
    Helper function which receives the changes in the cardiac rhythm labels and pads the whole vector.
        Example:
            in = ['AFIB', '', '', '', '', '', 'N', '', '', '', 'SBR', '']
            out = ['AFIB', 'AFIB', 'AFIB', 'AFIB', 'AFIB', 'AFIB', 'N', 'N', 'N', 'N', 'SBR', 'SBR']
        This function in used to parse the '.bea' files summarizing the beats detected in the UVAF database.
    :param rhythm: The input vector representing the changes in the cardiac rhythm (list of strings or labels).
    :param missing: The different strings or labels (list) which characterize a missing rhythm. (If None, considering only '' as a missing rhythm)
    :returns rhythm: The padded vector of rhythms.
    """
    cond = np.ones(len(rhythm), dtype=bool)
    if missing != None:
        for char in missing:
            cond = np.logical_and(cond, rhythm != char)
    else:
        cond = rhythm != missing
    not_none = np.where(cond)[0]
    if len(not_none) == 0:
        rhythm = np.array(['(N'] * len(rhythm))
    else:
        not_none = np.append(not_none, len(rhythm))
        diffs = np.diff(not_none)
        sing_rhy = rhythm[not_none[:-1]]
        rhythm[not_none[0]:] = np.repeat(sing_rhy, diffs)
        rhythm[0:not_none[0]] = sing_rhy[0]
    return rhythm


def calc_rr_window_labels(rhythm_sequence, window_size_beats, threshold=0.5):
    """
    Calculate per-window AF labels for RR-interval data.

    Original simple approach - direct beat-level processing.

    Parameters:
    -----------
    rhythm_sequence : list
        Beat-level rhythm labels: ['(AFIB', '(N', '(SBR', etc.]
    window_size_beats : int
        Number of beats per window (e.g., 60)
    threshold : float
        Fraction of beats that must be AF to label window as AF (default 0.5)

    Returns:
    --------
    np.array : Binary labels per window (1=AF, 0=non-AF)
    """
    rhythms = pad_rhythm(np.array(rhythm_sequence), missing=['', 'None'])
    rlab = rhythms[:(len(rhythms) // window_size_beats) * window_size_beats].reshape(-1, window_size_beats)
    af_counts = np.sum((rlab == '(AFIB'), axis=1)
    return (af_counts >= window_size_beats * threshold).astype(int)


def calc_ecg_window_labels(rhythm_sequence, beat_timestamps, window_start_times, window_end_times, sampling_rate=200):
    """
    Calculate per-window AF labels for ECG data using time-based windows.

    Based on Noam AF_DETECT base_wave_parser.py _win_lab_by_start_end_time method.
    Correctly handles beat-level annotations for time-based ECG windows.

    Parameters:
    -----------
    rhythm_sequence : list
        Beat-level rhythm labels: ['(AFIB', '(N', '(SBR', etc.]
    beat_timestamps : array-like
        Timestamps of each beat in seconds
    window_start_times : array-like
        Start times of windows in seconds
    window_end_times : array-like
        End times of windows in seconds
    sampling_rate : int
        ECG sampling rate in Hz (default 200)

    Returns:
    --------
    np.array : Binary labels per window (1=AF, 0=non-AF)
    """

    # Convert inputs to numpy arrays
    rhythms = pad_rhythm(np.array(rhythm_sequence), missing=['', 'None'])
    timestamps = np.array(beat_timestamps) * sampling_rate  # Convert to sample indices
    start_times = np.array(window_start_times) * sampling_rate
    end_times = np.array(window_end_times) * sampling_rate

    # Find beats within each time window
    window_labels = []
    for i in range(len(start_times)):
        # Find beats in this time window
        beats_in_window = (timestamps > start_times[i]) & (timestamps <= end_times[i])
        rhythms_in_window = rhythms[beats_in_window]

        if len(rhythms_in_window) == 0:
            # No beats in window - label as non-AF (0)
            window_labels.append(0)
            continue

        # Count AF beats in this window (following Noam AF_DETECT approach)
        af_count = np.sum(rhythms_in_window == '(AFIB')
        total_beats = len(rhythms_in_window)

        # Label as AF if majority of beats are AF
        window_labels.append(1 if af_count > total_beats / 2 else 0)

    return np.array(window_labels)


def calc_window_labels(rhythm_sequence, window_size, data_type='rr', beat_timestamps=None, window_starts=None, window_ends=None, threshold=0.5):
    """
    Unified function for calculating window labels.

    Parameters:
    -----------
    rhythm_sequence : list
        Beat-level rhythm labels: ['(AFIB', '(N', '(SBR', etc.]
    window_size : int or float
        Window size: beats for RR, seconds for ECG
    data_type : str
        'rr' for RR data, 'ecg' for ECG data
    beat_timestamps : array-like, optional
        Beat timestamps in seconds (required for ECG)
    window_starts : array-like, optional
        Window start times in seconds (required for ECG)
    window_ends : array-like, optional
        Window end times in seconds (required for ECG)
    threshold : float
        AF threshold (default 0.5)

    Returns:
    --------
    np.array : Binary labels per window (1=AF, 0=non-AF)
    """

    if data_type == 'rr':
        return calc_rr_window_labels(rhythm_sequence, window_size, threshold)

    elif data_type == 'ecg':
        if beat_timestamps is None or window_starts is None or window_ends is None:
            raise ValueError("beat_timestamps, window_starts, and window_ends required for ECG data")
        return calc_ecg_window_labels(rhythm_sequence, beat_timestamps, window_starts, window_ends)

    else:
        raise ValueError("data_type must be 'rr' or 'ecg'")


def create_window_times(recording_duration_sec, window_duration_sec=30.0):
    """
    Create window start and end times for ECG analysis.

    Parameters:
    -----------
    recording_duration_sec : float
        Total duration of ECG recording in seconds
    window_duration_sec : float
        Duration of each analysis window (default 30.0 for ArNetECG)

    Returns:
    --------
    start_times : np.array
        Window start times in seconds: [0, 30, 60, 90, ...]
    end_times : np.array
        Window end times in seconds: [30, 60, 90, 120, ...]
    """
    n_windows = int(recording_duration_sec // window_duration_sec)
    if n_windows == 0:
        n_windows = 1

    start_times = np.arange(n_windows) * window_duration_sec
    end_times = (np.arange(n_windows) + 1) * window_duration_sec

    # Ensure last window doesn't exceed recording duration
    end_times = np.clip(end_times, 0, recording_duration_sec)

    return start_times, end_times


# Backward compatibility - assumes 1 beat per second for RR data
def calc_y(rhythm_sequence, window_size_beats):
    """Legacy function for RR-interval data"""
    # For RR data, assume 1 beat per second (rough approximation)
    beat_timestamps = np.arange(len(rhythm_sequence))  # One beat per second
    return calc_window_labels(rhythm_sequence, window_size_beats, beat_timestamps=beat_timestamps, threshold=0.5)

def calc_preceding_windows(annotations, fs, window_size):
    """
    Calculate preceding windows based on R-peak annotations.

    :param annotations: R-peak annotations.
    :param fs: Sampling frequency.
    :param window_size: Size of the window in beats.
    :returns: Masked windows to exclude certain portions of the data.
    """
    start_rr, end_rr = annotations[:-1] / fs, annotations[1:] / fs
    rr_data = np.diff(annotations) / fs
    interbeats = np.append(np.insert((start_rr + end_rr) / 2, 0, max(0, start_rr[0] - 1)), end_rr[-1] + 1.0)
    excluded_portions_dict = np.array([[0, start_rr[0]]])

    # Split the data into windows and apply the mask
    start_win = interbeats[:-2][:(len(rr_data) // window_size) * window_size].reshape(-1, window_size)[:,
                0]  # [:, 0] to select the beginning of the window
    end_win = interbeats[2:][:(len(rr_data) // window_size) * window_size].reshape(-1, window_size)[:,
              -1]  # [:, -1] to select the end of the window
    mask_start = np.logical_or.reduce(tuple([np.logical_and(start_win > x[0], start_win <= x[1]) for x in excluded_portions_dict]))  # The window begins in an excluded portion.
    mask_end = np.logical_or.reduce(tuple([np.logical_and(end_win > x[0], end_win <= x[1]) for x in
                                           excluded_portions_dict]))  # The window ends in an excluded portion.
    mask_between = np.logical_or.reduce(tuple([np.logical_and(start_win <= x[0], end_win > x[1]) for x in excluded_portions_dict]))  # The window contains an excluded portion.
    final_mask = np.logical_not(
        np.logical_or.reduce((mask_start, mask_end, mask_between)))

    return np.cumsum(final_mask)  # Return cumulative sum mask


import numpy as np

def compute_prec_succ_from_window_starts(start_times: np.ndarray, expected_step: float | None = None, tol: float = 1e-3):
    """
    start_times: shape [n_windows], each window start timestamp (seconds).
    expected_step: expected delta between consecutive window starts. If None, uses median diff.
    tol: allowed deviation (seconds) to still consider windows consecutive.

    Returns:
      prec[i] = consecutive count ending at i (including i)
      succ[i] = consecutive count starting at i (including i)
    """
    start_times = np.asarray(start_times, dtype=np.float64)
    n = len(start_times)
    if n == 0:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    diffs = np.diff(start_times)
    if expected_step is None:
        expected_step = np.median(diffs) if len(diffs) else 0.0

    # break between i-1 and i if delta not ~ expected_step
    is_break = np.zeros(n, dtype=bool)
    if n >= 2 and expected_step > 0:
        is_break[1:] = np.abs(diffs - expected_step) > tol
    else:
        is_break[1:] = False  # nothing to infer

    prec = np.zeros(n, dtype=np.int32)
    run = 0
    for i in range(n):
        if i == 0 or is_break[i]:
            run = 1
        else:
            run += 1
        prec[i] = run

    succ = np.zeros(n, dtype=np.int32)
    run = 0
    for i in range(n - 1, -1, -1):
        if i == n - 1 or is_break[i + 1]:
            run = 1
        else:
            run += 1
        succ[i] = run

    return prec, succ

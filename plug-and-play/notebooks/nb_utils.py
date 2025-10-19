import os
import numpy as np
import pandas as pd
import wfdb
import requests
import pickle
import subprocess


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


def calc_y(rhythm, window_size):
    """
    Generates window labels based on the rhythm sequence.

    :param rhythm: The sequence of rhythm labels (AF or non-AF).
    :param window_size: The size of each window in beats.
    :returns: Binary array with labels (1 for AF, 0 for non-AF).
    """
    rhythms = pad_rhythm(np.array(rhythm), missing=['', 'None'])
    rlab = rhythms[:(len(rhythms) // window_size) * window_size].reshape(-1, window_size)
    counts = np.sum((rlab == '(AFIB'), axis=1)
    return counts >= window_size // 2  # If half or more are AF, label as AF


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

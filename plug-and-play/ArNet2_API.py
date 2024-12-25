# Relative imports
import pickle
import os
import numpy as np
import pathlib
import pandas as pd

from ArNet2.src.models.ArNet2 import ArNet2

def load_model(path, algo, path_feature_extractor=None):
    with open(path, 'rb') as file:
        model_dict = pickle.load(file)
        if algo != 'XGB':
            hypercomb = model_dict["hyperparameters"]
            if algo == "ArNet2":
                model = ArNet2(**hypercomb, path_feature_extractor=path_feature_extractor)
            else:
                model = ArNet2(**hypercomb)
            model.set_state_dict(model_dict['classifier'])
            model_dict['classifier'] = model
    return model_dict


def run_model(beat2beat_signal, path):
    algo_py = 'ArNet2'
    feature_extractor_path = path['resnet']
    model_dict = load_model(path=path['arnet2'], algo=algo_py, path_feature_extractor=feature_extractor_path)
    model = model_dict['classifier']
    probas = model.predict_proba(beat2beat_signal)[:, 1]
    return probas, probas > model_dict['best_th']


def export_to_physiozoo(directory, start_events, end_events, rhythms, unique_name):
    """
    This function exports a given recording to the physiozoo format for further investigation.
    The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
    The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
    :param directory: The directory under which the files will be saved.
    :param rhythms: bool array with True for AF and False for non-AF.
    :param start_events: time array of the start time of all windows.
    :param end_events: time array of the end time of all windows.
    """
    rhythms_header = ['---\n',
                      'type: rhythms annotation\n',
                      'source file: ',  # To be completed by filename
                      '\n',
                      '---\n',
                      '\n',
                      'Beginning\tEnd\t\tClass\n']
    diff = 10  # events with less then 10 seconds difference count as same event
    if not os.path.exists(directory):
        os.makedirs(directory)
    rhythms_full_path = pathlib.PurePath(directory) / (str(unique_name) + '.txt')
    # Rhythms
    with open(rhythms_full_path, 'w+') as rhythms_file:
        rhythms_header[2] = 'source file: rhythms.txt\n'
        rhythms_file.writelines(rhythms_header)
        final_rhythms = np.array(rhythms.astype(int))
        mask_rhythms = final_rhythms > 0  # We do not keep NSR as rhythm
        start_events, end_events = start_events[mask_rhythms], end_events[mask_rhythms]
        final_rhythms = final_rhythms[mask_rhythms]
        events = pd.DataFrame({'start_events': start_events, 'end_events': end_events, 'rhythms': final_rhythms})
        events = (events.groupby((events.start_events - events.end_events.shift() > diff).cumsum()).agg(
            {'start_events': 'min', 'end_events': 'max', 'rhythms': 'first'})[
            ['start_events', 'end_events', 'rhythms']])
        final_rhythms_str = np.array(['AFIB' for i in np.array(events.rhythms)])
        rhythms_file.write('\n'.join(
            ['%.5f\t%.5f\t%s' % (events.start_events.to_list()[i], events.end_events.to_list()[i], final_rhythms_str[i])
             for i
             in
             range(len(final_rhythms_str))]))


def process_input(raw_rr, raw_ts):
    win = 60
    rr = raw_rr[:(len(raw_rr) // win) * win].reshape(-1, win)
    ts = raw_ts[:(len(raw_ts) // win) * win].reshape(-1, win)
    start_win = ts[:, 0]
    end_win = ts[:, -1]
    glob_lab = np.ones(len(rr))
    prec_windows = np.arange(len(rr))
    ids = np.ones(len(rr))
    X = np.concatenate((rr, prec_windows.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    return X, start_win, end_win

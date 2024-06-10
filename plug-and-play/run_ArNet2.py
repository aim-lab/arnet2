import numpy as np
import argparse
from warnings import warn
import yaml

# Relative imports
import pandas as pd
from ArNet2_API import run_model


def read_file(input_file):
    try:
        df = pd.read_excel(input_file)
        filetype = 'EXCEL'
    except Exception:
        df = pd.read_csv(input_file)
        filetype = 'CSV'
    return df


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


def run(input_file, config):
    df_input = read_file(input_file)
    raw_rr = df_input[df_input.columns[0]].to_numpy(dtype='float64')
    raw_ts = df_input[df_input.columns[1]].to_numpy(dtype='float64')
    X, start_win, end_win = process_input(raw_rr, raw_ts)
    probas, y_pred = run_model(X, config['path'])
    df_pred = pd.DataFrame(columns=['rr_id', 'start_time', 'end_time', 'proba', 'pred'])
    df_pred['rr_id'] = X[:, 1]
    df_pred['start_time'] = start_win
    df_pred['end_time'] = end_win
    df_pred['proba'] = probas
    df_pred['pred'] = y_pred
    return df_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True,
                                     description='Use ArNEt2 in a plug-and-play manner.')
    parser.add_argument("--config", type=str, default="config.yml",
                        help="model hyperparameters")
    parser.add_argument('--device', default='cuda:1', help='Device')
    parser.add_argument('--path_to_database', type=str,
                        help='path to folder containing tnmg database')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    print(args)

    # Get config
    config_path = "./config/" + args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    input_file = './df_test.csv'
    df_pred = run(input_file, config)
    df_pred.to_csv('./df_test_proba.csv')
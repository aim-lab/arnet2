import numpy as np
import argparse
from warnings import warn
import yaml
import os
import pandas as pd
from ArNet2_API import run_model


def read_file(input_file):
    """
    Reads input file (CSV or Excel). Returns the data and its type.
    """
    try:
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = pd.read_excel(input_file)
            filetype = 'EXCEL'
        else:
            df = pd.read_csv(input_file)
            filetype = 'CSV'
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_file}' not found.")
    except Exception as e:
        raise ValueError(f"Failed to read the file: {e}")

    return df


def process_input(raw_rr, raw_ts, win=60):
    """
    Processes the raw RR intervals and timestamps into windows.
    """
    # Reshape raw RR and timestamp arrays into windows of 'win' size
    n_windows = len(raw_rr) // win
    rr = raw_rr[:n_windows * win].reshape(-1, win)
    ts = raw_ts[:n_windows * win].reshape(-1, win)
    start_win = ts[:, 0]
    end_win = ts[:, -1]

    # Create placeholders for the global labels and other features
    glob_lab = np.ones(len(rr))
    prec_windows = np.arange(len(rr))
    ids = np.ones(len(rr))

    # Concatenate the features into a single array
    X = np.concatenate((rr, prec_windows.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    return X, start_win, end_win


def run(input_file, config):
    """
    Run the ArNet2 model on the input data and return predictions.
    """
    # Read and preprocess the input file
    df_input = read_file(input_file)
    raw_rr = df_input[df_input.columns[0]].to_numpy(dtype='float64')
    raw_ts = df_input[df_input.columns[1]].to_numpy(dtype='float64')
    X, start_win, end_win = process_input(raw_rr, raw_ts)

    # Run the model and get predictions
    probas, y_pred = run_model(X, config['path'])

    # Create the output DataFrame
    df_pred = pd.DataFrame(columns=['rr_id', 'start_time', 'end_time', 'proba', 'pred'])
    df_pred['rr_id'] = X[:, 1]
    df_pred['start_time'] = start_win
    df_pred['end_time'] = end_win
    df_pred['proba'] = probas
    df_pred['pred'] = y_pred
    return df_pred


def save_output(df_pred, save_path, output_name):
    """
    Save the output DataFrame to the specified path.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check file extension and save accordingly
    output_file = os.path.join(save_path, f"{output_name}.csv")
    df_pred.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run ArNet2 for Atrial Fibrillation detection.')
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the model's hyperparameters YAML file.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input data file (.csv, .xlsx)')
    parser.add_argument('--save_path', type=str, default='./', help='Directory to save the output file')
    parser.add_argument('--output_name', type=str, default='af_detection_results', help='Unique name for the output file')

    args = parser.parse_args()
    # Load the configuration file
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run the model and save predictions
    df_pred = run(args.input_file, config)
    save_output(df_pred, args.save_path, args.output_name)

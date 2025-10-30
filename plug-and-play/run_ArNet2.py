import numpy as np
import argparse
import os
import pandas as pd
import pickle
import tensorflow as tf


def parse_args():
    """
    Parse command line arguments to specify the operation mode and input/output files.

    Returns:
        args: A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run ArNet2 for AF detection and prediction.')

    # Input and Output arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file (CSV/Excel format)')
    # Config file argument (optional)
    parser.add_argument('--saved_model', type=str, default='./exported_model', help='Path to the .bp saved model')
    parser.add_argument(
        '--inference_mode',
        type=str,
        choices=['full', 'window'],
        default='full',
        help='Select which trained model to use: full: for full ArNet2 temporal sequence modeling (default option) or window: for running part 1 only.')
    parser.add_argument('--save_output_path', type=str, default='./results/predictions', help='Path to save the prediction results')
    parser.add_argument('--output_name', type=str, default='predictions', help='Name for the output file')

    return parser.parse_args()


def validate_rr_csv(filepath):
    """
    Validate an RR-interval CSV file for prediction.

    Conditions:
    1. Must have exactly 3 columns.
    2. First column ('rr_data'):
       - Must not contain nulls.
       - Must be numeric.
       - All values <= 100 (since >100s RR interval is unrealistic).
    3. Second column ('rr_time'):
       - Must contain float or numeric values.
    4. Third column ('patient_id'):
       - Can be string or numeric.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        bool: True if the file passes validation, False otherwise.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # --- Check column count ---
    if df.shape[1] != 3:
        print(f"Invalid number of columns ({df.shape[1]}). Expected 3 columns.")
        return False

    # Extract column names
    col1, col2, col3 = df.columns[:3]

    # --- Check 1st column: rr_data ---
    rr_data = df[col1]
    if rr_data.isnull().any():
        print("Missing values in RR data column.")
        return False

    if not np.issubdtype(rr_data.dtype, np.number):
        # Try to convert
        try:
            rr_data = rr_data.astype(float)
        except Exception:
            print("RR data column is not numeric.")
            return False

    if (rr_data > 100).any():
        print("RR data contains values > 100 seconds (unrealistic).")
        return False

    # --- Check 2nd column: rr_time ---
    rr_time = df[col2]
    try:
        rr_time.astype(float)
    except Exception:
        print("RR time column contains non-numeric values.")
        return False

    # --- Check 3rd column: patient_id ---
    patient_id = df[col3]
    if not (patient_id.apply(lambda x: isinstance(x, (str, int, float)) or pd.isna(x))).all():
        print("patient_id column must contain string or numeric values.")
        return False

    print(f"File '{filepath}' passed validation.")
    return True


def load_data(input_file):
    """
    Load the input data file (CSV/Excel/Pickle) for either training or prediction.

    Args:
        input_file (str): Path to the input data file.

    Returns:
        data: Loaded data (either CSV, Excel, or Pickle).
    """
    try:
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            data = pd.read_excel(input_file)
            filetype = 'EXCEL'
        elif input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
            filetype = 'CSV'
        elif input_file.endswith('.pickle'):
            data = pickle.load(open(input_file, "rb"))
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_file}' not found.")
    except Exception as e:
        raise ValueError(f"Failed to read the file: {e}")

    return data


def prepare_data_for_prediction(raw_rr, raw_ts, patient_id, win=60):
    """
    Prepare the raw RR intervals and timestamps into windows for prediction.

    Args:
        raw_rr (np.ndarray): Raw RR intervals.
        raw_ts (np.ndarray): Raw timestamps corresponding to the RR intervals.
        patient_id (np.ndarray): Recording id from which the raw_rr and raw_ts are derived corresponding to the RR intervals.
        win (int, optional): Size of the window to split the data into. Defaults to 60.

    Returns:
        tuple: The processed data for prediction (X, start_win, end_win).
    """
    # Reshape raw RR and timestamp arrays into windows of 'win' size
    n_windows = len(raw_rr) // win
    rr = raw_rr[:n_windows * win].reshape(-1, win)
    ts = raw_ts[:n_windows * win].reshape(-1, win)
    ids = np.repeat(patient_id, len(rr))

    start_win = ts[:, 0]
    end_win = ts[:, -1]

    # Create placeholders for the global labels and other features
    glob_lab = np.ones(len(rr))
    prec_windows = np.arange(len(rr), dtype=int)

    # Concatenate the features into a single array
    X = np.concatenate((rr, prec_windows.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    return X, start_win, end_win


def process_data_for_all_ids(data, win=60):
    """
    Process the data for all unique patient_ids and prepare it for prediction.

    Args:
        data (pd.DataFrame): The dataframe containing 'rr', 'time', and 'patient_id' columns.
        win (int, optional): Size of the window to split the data into. Defaults to 60.

    Returns:
        X_full: An array of all windowed data.
        start_win_dict: Dictionary of windowed start indices with patient_id as keys.
        end_win_dict: Dictionary of windowed end indices with patient_id as keys.
    """
    # Initialize list to store the results for all patient_ids
    X_parts = []
    start_win_dict, end_win_dict = {}, {}

    # Iterate over each unique patient_id
    for patient_id in data[data.columns[2]].unique():
        # Extract the subset of data belonging to the current patient_id
        subset = data[data.iloc[:, 2] == patient_id]

        # raw arrays
        raw_rr = subset.iloc[:, 0].to_numpy(dtype='float64')
        raw_ts = subset.iloc[:, 1].to_numpy(dtype='float64')

        # if not enough samples to make at least one window, skip
        if len(raw_rr) < win:
            continue

        # Call the prepare_data_for_prediction function for the current patient_id subset
        X, start_win, end_win = prepare_data_for_prediction(raw_rr, raw_ts, patient_id.astype(str), win)

        # Store the results (X, start_win, end_win) for each patient_id
        if X.size:
            X_parts.append(X)
            start_win_dict[patient_id] = start_win
            end_win_dict[patient_id] = end_win
    X_full = np.vstack(X_parts) if X_parts else np.empty((0, win + 3), dtype=object)

    return X_full, start_win_dict, end_win_dict


def create_prediction_df(X, probas, y_pred, start_win_dict, end_win_dict):
    """
    Create a DataFrame for the predictions.

    Args:
        X (np.ndarray): Input features.
        probas (np.ndarray): Predicted probabilities.
        y_pred (np.ndarray): Predicted labels.
        start_win_dict (dict): Dict mapping patient_id -> array of start times.
        end_win_dict (dict): Dict mapping patient_id -> array of end times.

    Returns:
        pandas.DataFrame: DataFrame containing the predictions.
    """
    dfs = []

    # Ensure patient_ids are strings for consistency
    patient_ids = np.array(X[:, -1], dtype=str)

    for patient_id in np.unique(patient_ids):
        # mask for this patient_id
        mask = patient_ids == patient_id

        # skip if no start/end window info
        key_dtype = type(next(iter(start_win_dict)))
        if patient_id.astype(key_dtype) not in start_win_dict.keys() or patient_id.astype(key_dtype) not in end_win_dict.keys():
            continue

        start_win = start_win_dict[patient_id.astype(key_dtype)]
        end_win = end_win_dict[patient_id.astype(key_dtype)]

        # ensure lengths match
        n_windows = min(len(start_win), mask.sum())
        if n_windows == 0:
            continue

        df_pred = pd.DataFrame({
            'patient_id': [patient_id] * n_windows,
            'prec_window': X[mask, -3],
            'start_time': start_win[:n_windows],
            'end_time': end_win[:n_windows],
            'proba': probas[mask][:n_windows],
            'pred': y_pred[mask][:n_windows],
        })
        dfs.append(df_pred)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def save_output(df_pred, save_path, output_name):
    """
    Save the output DataFrame to the specified path.

    Args:
        df_pred (pandas.DataFrame): The predictions DataFrame.
        save_path (str): Directory path to save the output.
        output_name (str): The name of the output file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f"{output_name}.csv")
    df_pred.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """
    Main function to run either training or prediction based on the command line arguments.
    """
    # Parse command line arguments
    args = parse_args()

    # Load the data file
    data = load_data(args.input_file)

    print("Predicting...")

    validate_rr_csv(args.input_file)

    # Prepare data for prediction

    X, start_win, end_win = process_data_for_all_ids(data)

    loaded = tf.saved_model.load(args.saved_model)
    infer = loaded.signatures["predict_fixed"]
    out = infer(
        x=X[:, :-3].astype('float32'),
        prec_windows=X[:, -3].astype('int32'),
        glob_lab=X[:, -2].astype('float32'),
        ids=X[:, -1],
    )
    probas = out["probs"]
    y_pred = out['pred']

    # Create prediction DataFrame and save it
    prediction_df = create_prediction_df(X, probas, y_pred, start_win, end_win)
    save_output(prediction_df, args.save_output_path, args.output_name)


if __name__ == '__main__':
    main()

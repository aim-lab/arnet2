"""
AF detection runner: run ArNet2 (RR-based) or ArNetECG (ECG-based) and save predictions.

Usage:
  python run_detection.py --model_type ArNet2 --input_file ... --saved_model ...
  python run_detection.py --model_type ArNetECG --input_file ... --saved_model ...
"""
import numpy as np
import argparse
import os
import pandas as pd
import pickle
import tensorflow as tf


# Supported detection algorithms
MODEL_TYPES = ("ArNet2", "ArNetECG")


def parse_args():
    """
    Parse command line arguments for AF detection.

    Returns:
        args: Namespace with input_file, saved_model, model_type, etc.
    """
    parser = argparse.ArgumentParser(
        description="Run AF detection (ArNet2 or ArNetECG) and save predictions."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to input data (CSV/Excel/NPY/Pickle)")
    parser.add_argument("--saved_model", type=str, default="./exported_model", help="Path to SavedModel directory")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODEL_TYPES),
        default="ArNet2",
        help="Detection algorithm: ArNet2 (RR intervals) or ArNetECG (raw ECG windows).",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["full", "window"],
        default="full",
        help="ArNet2 only: full (temporal sequence) or window (per-window classification).",
    )
    parser.add_argument("--save_output_path", type=str, default="./results/predictions", help="Directory for output CSV")
    parser.add_argument("--output_name", type=str, default="predictions", help="Output filename (without .csv)")
    return parser.parse_args()


def validate_rr_csv(filepath):
    """
    Validate an RR-interval CSV for prediction.

    Expects 3 columns: rr_data, rr_time, patient_id.
    rr_data must be numeric, no nulls, all values <= 100.

    Returns:
        bool: True if valid.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    if df.shape[1] != 3:
        print(f"Invalid number of columns ({df.shape[1]}). Expected 3 columns.")
        return False

    col1, col2, col3 = df.columns[:3]
    rr_data = df[col1]
    if rr_data.isnull().any():
        print("Missing values in RR data column.")
        return False
    if not np.issubdtype(rr_data.dtype, np.number):
        try:
            rr_data = rr_data.astype(float)
        except Exception:
            print("RR data column is not numeric.")
            return False
    if (rr_data > 100).any():
        print("RR data contains values > 100 seconds (unrealistic).")
        return False

    rr_time = df[col2]
    try:
        rr_time.astype(float)
    except Exception:
        print("RR time column contains non-numeric values.")
        return False

    print(f"File '{filepath}' passed validation.")
    return True


def load_data(input_file):
    """
    Load input from CSV, Excel, Pickle, or NPY.

    Returns:
        data: DataFrame or ndarray depending on format.
    """
    try:
        if input_file.endswith(".xlsx") or input_file.endswith(".xls"):
            return pd.read_excel(input_file)
        if input_file.endswith(".csv"):
            return pd.read_csv(input_file)
        if input_file.endswith(".pickle") or input_file.endswith(".pkl"):
            with open(input_file, "rb") as f:
                return pickle.load(f)
        if input_file.endswith(".npy"):
            return np.load(input_file, allow_pickle=True)
        raise ValueError(f"Unsupported file format: {input_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{input_file}' not found.")
    except Exception as e:
        raise ValueError(f"Failed to read the file: {e}")


def prepare_data_for_prediction(raw_rr, raw_ts, patient_id, win=60):
    """
    Prepare raw RR intervals and timestamps into windows for prediction.

    Returns:
        (X, start_win, end_win): windowed features and time ranges.
    """
    n_windows = len(raw_rr) // win
    rr = raw_rr[: n_windows * win].reshape(-1, win)
    ts = raw_ts[: n_windows * win].reshape(-1, win)
    ids = np.repeat(patient_id, len(rr))
    start_win = ts[:, 0]
    end_win = ts[:, -1]
    glob_lab = np.ones(len(rr))
    prec_windows = np.arange(len(rr), dtype=int)
    X = np.concatenate((rr, prec_windows.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)
    return X, start_win, end_win


def process_data_for_all_ids(data, win=60):
    """
    Process data for all patient_ids into windowed arrays for prediction.

    Args:
        data: DataFrame with columns [rr, time, patient_id] (by position).
        win: Window size in beats.

    Returns:
        (X_full, start_win_dict, end_win_dict).
    """
    X_parts = []
    start_win_dict, end_win_dict = {}, {}

    for patient_id in data[data.columns[2]].unique():
        subset = data[data.iloc[:, 2] == patient_id]
        raw_rr = subset.iloc[:, 0].to_numpy(dtype="float64")
        raw_ts = subset.iloc[:, 1].to_numpy(dtype="float64")
        if len(raw_rr) < win:
            continue
        X, start_win, end_win = prepare_data_for_prediction(raw_rr, raw_ts, patient_id.astype(str), win)
        if X.size:
            X_parts.append(X)
            start_win_dict[patient_id] = start_win
            end_win_dict[patient_id] = end_win

    X_full = np.vstack(X_parts) if X_parts else np.empty((0, win + 3), dtype=object)
    return X_full, start_win_dict, end_win_dict


def create_prediction_df(X, probas, y_pred, start_win_dict, end_win_dict):
    """Build predictions DataFrame for RR-based model (with start/end times)."""
    dfs = []
    patient_ids = np.array(X[:, -1], dtype=str)

    for patient_id in np.unique(patient_ids):
        mask = patient_ids == patient_id
        key_dtype = type(next(iter(start_win_dict)))
        if patient_id.astype(key_dtype) not in start_win_dict or patient_id.astype(key_dtype) not in end_win_dict:
            continue
        start_win = start_win_dict[patient_id.astype(key_dtype)]
        end_win = end_win_dict[patient_id.astype(key_dtype)]
        n_windows = min(len(start_win), mask.sum())
        if n_windows == 0:
            continue
        df_pred = pd.DataFrame({
            "patient_id": [patient_id] * n_windows,
            "prec_window": X[mask, -3],
            "start_time": start_win[:n_windows],
            "end_time": end_win[:n_windows],
            "proba": probas[mask][:n_windows],
            "pred": y_pred[mask][:n_windows],
        })
        dfs.append(df_pred)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def create_ecg_prediction_df(X, probas, y_pred):
    """Build predictions DataFrame for ECG-based model (no start/end times)."""
    dfs = []
    patient_ids = np.array(X[:, -1], dtype=str)
    for patient_id in np.unique(patient_ids):
        mask = patient_ids == patient_id
        n_windows = mask.sum()
        if n_windows == 0:
            continue
        df_pred = pd.DataFrame({
            "patient_id": [patient_id] * n_windows,
            "prec_window": X[mask, 6000],
            "proba": probas[mask][:n_windows],
            "pred": y_pred[mask][:n_windows],
        })
        dfs.append(df_pred)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def save_output(df_pred, save_path, output_name):
    """Write predictions DataFrame to CSV under save_path."""
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"{output_name}.csv")
    df_pred.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def _run_arnet2(data, args):
    """Run ArNet2 (RR-based) detection and return prediction DataFrame."""
    validate_rr_csv(args.input_file)
    X, start_win, end_win = process_data_for_all_ids(data)
    loaded = tf.saved_model.load(args.saved_model)
    infer = loaded.signatures["predict_fixed"]
    out = infer(
        x=X[:, :-3].astype("float32"),
        prec_windows=X[:, -3].astype("int32"),
        glob_lab=X[:, -2].astype("float32"),
        ids=X[:, -1],
    )
    return create_prediction_df(X, out["probs"], out["pred"], start_win, end_win)


def _prepare_ecg_X(data):
    """Convert loaded ECG data to array and validate shape (6000 ECG + 4 metadata cols)."""
    X = data.values if isinstance(data, pd.DataFrame) else data
    if X.dtype == object:
        print("Warning: Input data has object dtype. Converting to float32...")
        X_float = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        for i in range(X.shape[1]):
            try:
                X_float[:, i] = X[:, i].astype(np.float32)
            except (ValueError, TypeError):
                if i == X.shape[1] - 1:
                    unique_ids = np.unique(X[:, i])
                    id_map = {id_val: float(idx) for idx, id_val in enumerate(unique_ids)}
                    X_float[:, i] = np.array([id_map[val] for val in X[:, i]], dtype=np.float32)
                else:
                    raise ValueError(f"Cannot convert column {i} to float32")
        X = X_float

    window_size = 6000
    expected_cols = window_size + 4
    if X.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns (ECG: {window_size} + 4 metadata), got {X.shape[1]}"
        )
    return X


def _run_arnet_ecg(data, args):
    """Run ArNetECG (ECG-based) detection and return prediction DataFrame."""
    X = _prepare_ecg_X(data)
    loaded = tf.saved_model.load(args.saved_model)
    infer = loaded.signatures["predict_fixed"]
    ids_col = X[:, 6003]
    if ids_col.dtype.kind in "fi":
        ids_col = ids_col.astype("int").astype(str)
    out = infer(
        x=X[:, :6000].astype("float32"),
        prec_windows=X[:, 6000].astype("int32"),
        succ_windows=X[:, 6001].astype("int32"),
        glob_lab=X[:, 6002].astype("float32"),
        ids=ids_col,
    )
    return create_ecg_prediction_df(X, out["probs"], out["pred"])


def main():
    """Load data, run selected detection algorithm, save predictions."""
    args = parse_args()
    data = load_data(args.input_file)
    print(f"Using model: {args.model_type}")
    print("Predicting...")

    if args.model_type == "ArNet2":
        prediction_df = _run_arnet2(data, args)
    elif args.model_type == "ArNetECG":
        prediction_df = _run_arnet_ecg(data, args)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    save_output(prediction_df, args.save_output_path, args.output_name)


if __name__ == "__main__":
    main()

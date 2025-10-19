import numpy as np
import argparse
import yaml
import os
import pandas as pd
import pickle
import datetime

import ArNet2.src.models.model_utils as model_utils
from ArNet2.src.models.core import ArNet2
import utils.consts as cts
import utils.metrics as metrics


def parse_args():
    """
    Parse command line arguments to specify the operation mode and input/output files.

    Returns:
        args: A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run ArNet2 for AF detection and prediction.')

    # Input and Output arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input data file (CSV/Excel format)')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help='Mode to run: "train" for training the model, "predict" for generating predictions')

    # Config file argument (optional)
    parser.add_argument('--config', type=str, default='./config/config.yml', help='Path to the configuration file')
    parser.add_argument('--save_model_path', type=str, default='./model', help='Path to save the trained model')
    parser.add_argument('--save_output_path', type=str, default='./results', help='Path to save the prediction results')
    parser.add_argument('--output_name', type=str, default='predictions', help='Name for the output file')
    parser.add_argument('--model_name', type=str, default='ArNet2', help='Model name to save and load')

    return parser.parse_args()


def load_config(config_path):
    """
    Load the configuration file from the specified path.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        config (dict): Loaded configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # Load the YAML configuration

    return config


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

def save_model(final_dict, model, path, algo):
    """
    Save a trained model to a specified path.

    Args:
        final_dict (dict): Model dictionary containing the hyperparameters and evaluation metrics.
        model (dict): The trained ArNet2 model.
        path (str): Path to save the model.
        algo (str): Algorithm name to save the model (e.g., "ArNet2").

    Returns:
        None
    """
    # Check if the directory exists, create it otherwise
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the model in a subdirectory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_file = f"{path}/{algo}_{timestamp}.pkl"
    print(f"Trained model saved to {model_file}")
    # Save model
    with open(model_file, 'wb') as file:
        final_dict['classifier'] = model.get_state_dict()
        pickle.dump(final_dict, file)


def load_model(path, algo, path_feature_extractor=None):
    """
    Load a trained model from a specified path.

    Args:
        path (str): Path to the saved model.
        algo (str): Algorithm name to load the model (e.g., "ArNet2").
        path_feature_extractor (str, optional): Path to the feature extractor model.

    Returns:
        model_dict: Loaded model dictionary containing the classifier and hyperparameters.
    """
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


def define_decision_threshold(probas, y):
    """
    Define the decision threshold for classification based on the F-beta score.

    Args:
        probas (np.ndarray): Predicted probabilities for each sample.
        y (np.ndarray): True labels for each sample.

    Returns:
        float: The optimal decision threshold.
    """
    best_th = metrics.maximize_f_beta(probas, y)
    return best_th


def update_model_dict(X, y, probas, decision_th, model_dict, set_name='train'):
    """
    Update the model dictionary with evaluation metrics.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        probas (np.ndarray): Predicted probabilities.
        decision_th (float): Decision threshold.
        model_dict (dict): Dictionary holding the model and hyperparameters.
        set_name (str, optional): The dataset name ('train' or 'test'). Defaults to 'train'.

    Returns:
        model_dict: Updated model dictionary with evaluation metrics.
    """
    metrics_dict, best_th_dict, mean_abs_afb_error_dict = {}, {}, {}

    best_th_dict[set_name] = decision_th
    metrics_dict[set_name] = metrics.model_metrics(probas, y, probas > decision_th, print_metrics=True)
    mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y, probas > decision_th)

    # Update model_dict with the evaluation metrics
    model_dict[f'metrics_{set_name}'] = dict(zip(cts.METRICS, metrics_dict[set_name]))
    model_dict[f'metrics_{set_name}']['mean_abs_afb_error'] = mean_abs_afb_error_dict[set_name]

    return model_dict


def train_model(X_train, y_train, config, n_epochs=5):
    """
    Train the ArNet2 model on the given training data.

    Args:
        X_train (np.ndarray): Input features for training.
        y_train (np.ndarray): True labels for training.
        config (dict): Configuration dictionary with model parameters.
        n_epochs (int, optional): Number of epochs to train the model. Defaults to 5.

    Returns:
        model: The trained model.
        model_dict: The model dictionary containing the classifier and hyperparameters.
    """
    # Initialize the model
    algo_py = 'ArNet2'
    feature_extractor_path = config['path']['resnet']

    # Initialize ArNet2 model
    model = model_utils.class_funcs[algo_py](**cts.hypercomb[algo_py], path_feature_extractor=feature_extractor_path)

    # Train the model
    model.fit(X_train, y_train, n_epochs=n_epochs)

    # Save model and hyperparameters
    model_dict = {'hyperparameters': cts.hypercomb[algo_py]}

    return model, model_dict


def predict_with_model(model, X_test):
    """
    Generate predictions using the trained ArNet2 model.

    Args:
        model: The trained ArNet2 model.
        X_test (np.ndarray): Input features for testing.

    Returns:
        np.ndarray: Predicted probabilities for each test sample.
    """
    probas = model.predict_proba(X_test)[:, 1]
    return probas


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

    # Load the configuration file
    config = load_config(args.config)

    # Load the data file
    data = load_data(args.input_file)

    if args.mode == 'train':
        print("Training model...")

        # Extract training data
        X_train, y_train, _ = data

        # Train the model
        model, model_dict = train_model(X_train, y_train, config)
        probas = predict_with_model(model, X_train)

        # Define decision threshold
        decision_th = define_decision_threshold(probas, y_train)

        # Update model dictionary with metrics
        model_dict = update_model_dict(X_train, y_train, probas, decision_th, model_dict, set_name='train')

        save_model(model_dict, model, args.save_model_path, 'ArNet2')

    elif args.mode == 'predict':
        print("Predicting...")

        # Prepare data for prediction

        X, start_win, end_win = process_data_for_all_ids(data)

        # Load the trained model
        model_dict = load_model(path=config['path']['arnet2'], algo='ArNet2',
                                path_feature_extractor=config['path']['resnet'])
        model = model_dict['classifier']

        # Predict
        probas = predict_with_model(model, X)
        y_pred = probas > model_dict['best_th']

        # Create prediction DataFrame and save it
        prediction_df = create_prediction_df(X, probas, y_pred, start_win, end_win)
        save_output(prediction_df, args.save_output_path, args.output_name)


if __name__ == '__main__':
    main()

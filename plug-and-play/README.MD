# Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes

This README outlines the steps to seproduce the results for the ArNet2 AF detection pipeline.

ArNet2 is a deep learning algorithm developed to detect AF events from windows/long-term beat-to-beat interval data. AF is the most common arrhythmia, linked to a significantly higher risk of stroke and increased mortality. Early and accurate detection of AF is crucial for timely intervention, risk assessment, and personalized treatment strategies.

For more details and to cite this work, please refer to our paper [1]

## Model Overview
ArNet2 is a deep learning algorithm designed to detect AF from long-term beat-to-beat (RR interval) data. For the purpose of AF detection in long-term cardiac monitoring and AF burden (AFB) estimation, ArNet2 focuses on generalization performance to ensure robustness across different patient groups and conditions. This approach builds on our previous work developing the original ArNet model [2].

## Architecture
ArNet2 algorithm work in two parts:

### Part 1: Residual Embedding Layer (Figure part a)

- **Input**: A 60-beat window of RR intervals.

- **Embedding**: The first part of the model consists of residual blocks adopted from He et al., which create an embedding by learning a representation of the ECG window.

- **Feature Extraction**: The embedded features are then passed to the first fully connected layer for extraction.

The output of the first part is a binary classification (AF or non-AF) for each input window.

### Part 2: Recurrent Neural Network (RNN) (Figure part c)

- **Sequence Modeling**: The second part of the model consists of a sequence encoder made of four identical GRU (Gated Recurrent Units) layers. These layers account for time dependencies and select relevant past information to accurately predict AF.

- **Input**: The embedded features of the selected window for classification, concatenated with the embedded features of the 9 preceding windows from Part 1.

- **AF Severity Classification**: Each GRU is trained for different AF severity levels: non-AF, AFmild, AFmod, and AFsev [1]. , and the relevant GRU layer is selected based on the patient's AFB.

- **Output**: A probability label of AF for each window, with the classification made based on the selected GRU.

During training, the true AFB is used for selection. However, during inference, the estimated AFB from the first step is used to guide the selection of the relevant GRU layer.

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-023-00791-1/MediaObjects/41746_2023_791_Fig2_HTML.png?as=webp" alt="ArNet2" width="400"/>
</p>

*Figure: Overview of the ArNet2 model architecture for AF detection.*

## Reproducibility and Usage

This section covers how to reproduce the training and prediction steps.

### 1. Running prediction

To run the trained ArNet2 model for prediction, run the following command:
```
python3 run_ArNet2.py --mode predict --input_file /path/to/your/input_file.csv [--mode_type full|window --output_name predictions --config /path/to/config_file.yml --save_output_path /path/to/results/dir]
```
Where:

- `--mode`: Run the code with pretrained model defined in config to generate predictions.
- `--input_file`: Path to the **input data file** (CSV/Excel format).
- `mode_type`: Choose whether to use the full architecture or just window classification:
  - `full`: Uses both Part 1 & 2 for full temporal sequence modeling (default option).
  - `window`: Uses only Part 1, running classification per 60-beat window independently (suitable for short signals).
- `--output_name` *(optional)*: **output predictions** filename under which the predictions will be saved.
- `--save_output_path` *(optional)*: Path where the **output predictions** will be saved as a CSV file.
- `--config` *(optional)*: Path to the **config file**. If not provided, it will default to `./config/config.yml`.

Example usage:

```angular2html
python3 run_ArNet2.py --mode predict --input_file ./data/df_test.csv --output_name predictions
```

This will:
- `--mode`: Run the code to predict.
- Take input data from `./data/df_test.csv`.
- Save the output predictions to `./results/df_out.csv`.
- Use the custom configuration file `./config/custom_config.yml`.

### 2. Model Usage Options: Window Classification vs. Full Architecture

ArNet2 provides flexibility in how it can be used based on the length of your ECG recordings. By setting the `--mode_type` argument, you can choose between two modes:

#### 2.1 Window Classification (Part 1 only):

- **Use Case**: Ideal for short ECG recordings where each recording is represented by a 60-beat window. This approach uses only Part 1 of the model to perform a binary classification (AF vs. non-AF) on each window independently.

- **How It Works**:

  **Input**: A 60-beat window of RR intervals.

  **Output**: A classification (AF or non-AF) for each 60-beat window.

- **Minimal Signal Length**: A minimum of 60 beats (approximately 48 seconds) of signal is required for Part 1 to run. If shorter signals are available, they may be padded or truncated accordingly.

#### 2.2 Full Architecture (Parts 1 & 2):

- **Use Case**: Suitable for longer ECG recordings or continuous data where temporal dependencies across windows can be used to improve detection. This method utilizes Part 2, which captures time-related patterns.

- **How It Works**:

  **Input**: Multiple, preceding 60-beat windows, where each window is classified first by Part 1, and then the temporal context of the windows is modeled by the Part 2.

  **Output**: A classification (AF or non-AF) for each 60-beat window.

- **Minimal Signal Length**: The signal must be at least 10 preceding 60-beat windows, or 480.3 seconds (8 minutes) long, for Part 2 to effectively leverage time dependencies. For shorter signals, empty arrays will be initialized to replace the missing preceding windows.

### 3. Preparing Data for Training and Prediction

#### 3.1 Prediction Data 

For prediction, you only need a CSV file with the RR intervals (beats), their relative time
and third column with a string or numeric identifier for the recording or patient. See example `./data/df_test.csv`
The algorithm will then split the data into 60-beat windows, and use this to make predictions.

#### 3.2 Training Data

For training, the input data should already be split into 60-beat windows, with the following additional columns:

| rr\_data (60 beats) | global\_label | patient\_id  | prec\_window | win\_start | win\_end |
|---------------------|---------------|--------------|--------------|------------|----------|
| \[0.8, 0.7, ...]    | 1             | 101          | 3            | 0.0        | 0.6      |
| \[0.7, 0.9, ...]    | 2             | 102          | 9            | 0.6        | 1.2      |
| \[0.9, 0.6, ...]    | 2             | 102          | 2            | 1.2        | 2.2      |
| ...                 | ...           | ...          | ...          | ...        | ...      |

Where:

- `rr_data`: The 60 RR intervals (beats) in the window.
- `global_label`: The label assigned to the patient based on their AF burden (e.g., `NON_AF`, `AF_MILD`, `AF_MODERATE`, `AF_SEVERE`).
- `patient_id`: The patient ID to which the window belongs. This is important for leveraging the temporal relationships between windows.
- `win_start`: The start time of the window.
- `win_end`: The end time of the window.

See example `./data/training_data.pickle`

### 4. Run training

To train the ArNet2 model, execute the following command:
```
python3 run_ArNet2.py --mode train --input_file /path/to/your/input_file.csv [--model_name ArNet2 --config /path/to/config_file.yml --save_model_path /path/to/save/model]
```
Where:

- `--mode`: Run the code to train a model.
- `--input_file`: Path to the **input data file** (pickle format).
- `--model_name`*(optional)*: The filename under which the **trained model** will be saved.
- `--save_model_path` *(optional)*: Path where the **trained model** will be saved as a pkl file.
- `--config` *(optional)*: Path to the **config file**. If not provided, it will default to `./config/config.yml`.

Example usage:

```angular2html
python3 run_ArNet2.py --mode train --input_file ./data/training_data.pickle --model_name ArNet2
```

This will:

- Train a model using an input data from `./data/training_data.pickle`.
- And save the trained model to `./model/ArNet2_{datetime-string}.pkl`.

## Example Notebooks

Example notebooks for training and prediction can be found under the notebooks directory:

- [Training Notebook](./notebooks/train_example.ipynb)
- [Prediction Notebook](./notebooks/predict_example.ipynb)

These notebooks demonstrate how to prepare the data, train the model, and run predictions using the trained model.

## Citation
```
[1] Biton, Shany, et al. 
"Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes." 
NPJ Digital Medicine 6.1 (2023): 44.

[2] Chocron, Armand, et al. 
"Remote atrial fibrillation burden estimation using deep recurrent neural network." 
IEEE Transactions on Biomedical Engineering 68.8 (2020): 2447-2455.
```


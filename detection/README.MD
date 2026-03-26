# Detection pre-trained models
This README outlines the steps to reproduce the results for the AF detection pipeline, which includes two algorithms: **ArNet2** (RR-interval based) and **ArNetECG** (raw-ECG based).

## Table of contents

- [ArNet2: Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes](#arnet2-generalizable-and-robust-deep-learning-algorithm-for-atrial-fibrillation-diagnosis-across-geography-ages-and-sexes)
  - [Model Overview](#model-overview)
  - [Architecture](#architecture)
- [ArNetECG: Deep Learning for AF Detection from the Raw Electrocardiogram](#arnetecg-deep-learning-for-af-detection-from-the-raw-electrocardiogram)
  - [Model Overview](#model-overview-1)
  - [Architecture](#architecture-1)
- [Reproducibility and Usage](#reproducibility-and-usage)
  - [1. Running prediction](#1-running-prediction)
  - [2. Model Usage Options: Window Classification vs. Full Architecture](#2-model-usage-options-window-classification-vs-full-architecture)
  - [3. Preparing Data for Training and Prediction](#3-preparing-data-for-training-and-prediction)
- [Example Notebooks](#example-notebooks)
- [Citation](#citation)

## ArNet2: Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes

ArNet2 is a deep learning algorithm developed to detect AF events from windows/long-term beat-to-beat interval data. AF is the most common arrhythmia, linked to a significantly higher risk of stroke and increased mortality. Early and accurate detection of AF is crucial for timely intervention, risk assessment, and personalized treatment strategies.

<p align="center">
  <img src="af vs nsr.png" alt="AF vs Normal Sinus Rhythm" width="400"/>
</p>

*Figure: Comparison of AF and Normal Sinus Rhythm (NSR) patterns in RR interval data.*

For more details and to cite this work, please refer to our paper [1].

### Model Overview
ArNet2 is a deep learning algorithm designed to detect AF from long-term beat-to-beat (RR interval) data. For the purpose of AF detection in long-term cardiac monitoring and AF burden (AFB) estimation, ArNet2 focuses on generalization performance to ensure robustness across different patient groups and conditions. This approach builds on our previous work developing the original ArNet model [2].

#### Performance and Validation
ArNet2 was developed on a large retrospective dataset of 2,147 patients totaling 51,386 hours obtained from continuous ECG recordings. The model's generalization performance was evaluated on manually annotated test sets from four centers (USA, Israel, Japan, and China) totaling 402 patients, and further validated on a retrospective dataset of 1,825 consecutive Holter recordings from Israel. For the task of window classification, ArNet2 achieved an F1 score of 0.95, outperforming benchmark state-of-the-art models including ArNet (F1 = 0.93), XGBoost (F1 = 0.92), and the AFEv rule-based approach (F1 = 0.90). The model demonstrated robust generalization across geography, ages, and sexes.

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-023-00791-1/MediaObjects/41746_2023_791_Fig3_HTML.png?as=webp" alt="ArNet2 Performance" width="1000"/>
</p>

*Figure: ArNet2 performance comparison across different models and datasets.*

### Architecture
ArNet2 algorithm work in two parts:

#### Part 1: Residual Embedding Layer (Figure part a)

- **Input**: A 60-beat window of RR intervals.

- **Embedding**: The first part of the model consists of residual blocks adopted from He et al., which create an embedding by learning a representation of the ECG window.

- **Feature Extraction**: The embedded features are then passed to the first fully connected layer for extraction.

The output of the first part is a binary classification (AF or non-AF) for each input window.

#### Part 2: Recurrent Neural Network (RNN) (Figure part c)

- **Sequence Modeling**: The second part of the model consists of a sequence encoder made of four identical GRU (Gated Recurrent Units) layers. These layers account for time dependencies and select relevant past information to accurately predict AF.

- **Input**: The embedded features of the selected window for classification, concatenated with the embedded features of the 9 preceding windows from Part 1.

- **AF Severity Classification**: Each GRU is trained for different AF severity levels: non-AF, AFmild, AFmod, and AFsev [1]. , and the relevant GRU layer is selected based on the patient's AFB.

- **Output**: A binary label classification of AF or non-AF for each window, with the classification made based on the selected GRU.

During training, the true AFB is used for selection. However, during inference, the estimated AFB from the first step is used to guide the selection of the relevant GRU layer.

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-023-00791-1/MediaObjects/41746_2023_791_Fig2_HTML.png?as=webp" alt="ArNet2" width="400"/>
</p>

*Figure: Overview of the ArNet2 model architecture for AF detection.*

## ArNetECG: Deep Learning for AF Detection from the Raw Electrocardiogram

ArNetECG detects atrial fibrillation (and atrial flutter) directly from **raw single-lead ECG** rather than derived RR intervals. By processing the raw waveform, the model leverages both **morphological** information (e.g. f-waves) and **rhythm** information, yielding better detection of AF and atrial flutter than rhythm-only (RR-based) models. This approach is extended and rigorously validated in RawECGNet [3].

For more details and to cite this work, please refer to [3].

### Model Overview

ArNetECG is a deep learning algorithm designed to detect AF from **raw ECG signals** in long-term cardiac monitoring. Unlike RR-interval models, it takes the continuous waveform as input, enabling use of both rhythm and morphology. The model focuses on generalization across recording conditions, lead positions, and populations. RawECGNet [3] builds on this approach with extensive benchmarking across geography, ethnicity, lead position, age, and sex.

#### Performance and Validation

Validated on external test sets with over 5,000 hours of continuously annotated data from two independent datasets. **RBDB**: F1 0.91–0.94 across leads; **SHDB**: F1 0.93. Outperformed ArNet2 (F1 0.89–0.91 on RBDB, 0.91 on SHDB). The study includes quantitative error analysis of false positives and negatives and demonstrates robust generalization across geography, lead position, age, and sex.

### Architecture

- **Input**: Raw single-lead ECG at **200 Hz**. The model operates on **30-second windows** (6000 samples per window).

- **Representation**: The network processes the raw waveform to extract both rhythm and morphological features (including f-waves), providing a richer signal than beat-to-beat intervals alone.

- **Output**: Binary classification (AF/non-AF) per 30 s window; the same pipeline supports detection of atrial flutter.

**Running ArNetECG:** Use `run_detection.py` with `--model_type ArNetECG`. Input must be preprocessed into 30 s windows: shape `(n_windows, 6004)` with columns `[ECG 0..5999, prec_windows, succ_windows, global_label, ids]`. See the [ArNetECG prediction notebook](./notebooks/predict_example_ecg.ipynb) for data preparation and usage.

## Reproducibility and Usage

This section covers how to reproduce the training and prediction steps.

### 1. Running prediction

To run the trained ArNet2 model for prediction, run the following command:
```
python3 run_detection.py --model_type ArNet2 --input_file /path/to/your/input_file.csv [--inference_mode full|window --output_name predictions --save_output_path /path/to/results/dir]
```
Where:

- `--input_file`: Path to the **input data file** (CSV/Excel format).
- `inference_mode`: Choose whether to use the full architecture or just window classification:
  - `full`: Uses both Part 1 & 2 for full temporal sequence modeling (default option).
  - `window`: Uses only Part 1, running classification per 60-beat window independently (suitable for short signals).
- `--saved_model` *(optional)*: Path to the **.pb model file** (.bp format).
- `--output_name` *(optional)*: **output predictions** filename under which the predictions will be saved.
- `--save_output_path` *(optional)*: Path where the **output predictions** will be saved as a CSV file.

Example usage:

```angular2html
python3 run_detection.py --model_type ArNet2 --saved_model ./exported_model --input_file ./data/df_test.csv --output_name predictions
```

This will:
- Take input data from `./data/df_test.csv`.
- Load the pre-trained model from `./exported_model`
- Save the output predictions to `./results/predictions/predictions.csv`.

### 2. Model Usage Options: Window Classification vs. Full Architecture

ArNet2 provides flexibility in how it can be used based on the length of your ECG recordings. By setting the `--inference_mode` argument, you can choose between two modes:

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

## Example Notebooks

Example notebooks for prediction can be found under the notebooks directory:

- [ArNet2 (RR) prediction](./notebooks/predict_example.ipynb) — prepare RR data and run ArNet2.
- [ArNetECG (raw ECG) prediction](./notebooks/predict_example_ecg.ipynb) — prepare 30 s ECG windows and run ArNetECG.

## Citation
```
[1] Biton, Shany, et al. 
"Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes." 
NPJ Digital Medicine 6.1 (2023): 44.

[2] Chocron, Armand, et al. 
"Remote atrial fibrillation burden estimation using deep recurrent neural network." 
IEEE Transactions on Biomedical Engineering 68.8 (2020): 2447-2455.

[3] Ben-Moshe, Noam, et al.
"RawECGNet: Deep Learning Generalization for Atrial Fibrillation Detection From the Raw ECG."
IEEE Journal of Biomedical and Health Informatics 28.9 (2024): 5180-5188.
```


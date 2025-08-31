# ArNet2: Atrial Fibrillation Detection from Long-Term Beat-to-Beat Intervals

This README provides the necessary steps to set up the environment, install dependencies, and reproduce the results for the ArNet2 AF detection pipeline.

ArNet2 is an algorithm designed for detecting Atrial Fibrillation (AF) events based on long-term beat-to-beat intervals.

## 1. Installation
Follow these steps to set up your environment and install dependencies.

### 1.1 Create Conda Environment from `environment.yml`

1. Navigate to the project directory where the environment.yml file is located.

2. Create the Conda environment:
```
conda env create -f environment.yml
```
3. Activate the environment:
```
conda activate arne2
```

### 1.2 Install Additional Dependencies Using Poetry
1. Install Poetry (if not already installed):
```
pip install poetry
```
2. Navigate to the project directory where the `pyproject.toml` file is located.
3. Install dependencies:
```
poetry install
```

### 1.3 Verify TensorFlow and CUDA Compatibility

Ensure that your system meets the following requirements for TensorFlow 2.12 with GPU support:

Python Version: 3.8, 3.9, or 3.10

- TensorFlow Version: 2.12.*
- CUDA Version: 12.*

To verify the installation:
```angular2html
import tensorflow as tf
print(tf.__version__)  # Should print 2.12.0
print(tf.config.list_physical_devices('GPU'))  # Should list available GPUs
```

## 2. Setting `PYTHONPATH`

To ensure that Python can access `ArNet2` directory during runtime, you need to set the `PYTHONPATH` environment variable.

Run the following command in your terminal to set the `PYTHONPATH` variable with the necessary directory:

```angular2html
export PYTHONPATH="/root/folder/of/repo/ArNet2/src/models:$PYTHONPATH"
```

Note: Replace `/root/folder/of/repo` with the actual path to the repo directory

## 3. Reproducibility and Usage

This section covers how to reproduce the training and prediction steps.

### 3.1 Running prediction

To run the trained ArNet2 model for prediction, run the following command:
```
python3 run_ArNet2.py --input_file /path/to/your/input_file.csv --output_name /path/to/save/output_file.csv [--config /path/to/config_file.yml]
```
Where:

- `--input_file`: Path to the **input data file** (CSV/Excel format).
- `--output_name`: Path where the **output predictions** will be saved as a CSV file.
- `--config` *(optional)*: Path to the **config file**. If not provided, it will default to `./config/config.yml`.

Example usage:

```angular2html
python3 run_ArNet2.py --input_file ./data/df_test.csv --output_name ./results/df_out.csv
```

This will:

- Take input data from `./data/df_test.csv`.
- Save the output predictions to `./results/df_out.csv`.
- Use the custom configuration file `./config/custom_config.yml`.
- 
- ### 3.2 Preparing Data for Training and Prediction

#### 3.2.1 Prediction Data 

For prediction, you only need a CSV file with the RR intervals (beats) and their relative time. See example `./data/df_test.csv`
The algorithm will then split the data into 60-beat windows, and use this to make predictions.

#### 3.2.2 Training Data

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

For ease of use, we have provided a Jupyter notebook that demonstrates how to prepare the data for training.
You can find the notebook here:

- Data Preparation Example Notebook

By running this notebook, you'll generate a `.pickle` file containing the pre-processed data, which is ready for training with the ArNet2 model.

### 3.3 Run training

To train the ArNet2 model, execute the following command:
```angular2html

```



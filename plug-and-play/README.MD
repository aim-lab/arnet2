# ArNet2: Atrial Fibrillation Detection from Long-Term Beat-to-Beat Intervals

ArNet2 is an algorithm designed for detecting Atrial Fibrillation (AF) events based on long-term beat-to-beat intervals.

## Requirements
- **Python 3.8 (64-bit)**
- **CUDA 10.1** (make sure to install this version before proceeding)
- **cuDNN 7.6.5** (compatible with CUDA 10.1)
- **TensorFlow 2.4.1 (GPU)**

Required libraries and dependencies can be installed using the provided `env.yml` file:

```
conda env create -f env.yml
```

## Setup

1. **Activate the environment:**
After installing the dependencies, activate the conda environment with:
```
conda activate tf_gpu
```

2. **Run ArNet2 on your dataset:**
Use the following command to execute the ArNet2 algorithm:
```
python3 run_ArNet2.py --input_file <input_file> --save_path <save_path> --output_name <output_name>
```
**Parameters:**
- `--input_file <input_file>`: Path to the input data file (supports .mat, .csv, or .xlsx formats). The file should contain:
  - `rr_data`: The RR intervals (beat-to-beat data).
  - `rr_time`: The time of the R-peaks in seconds.
- `--save_path <save_path>`: The directory where the output file will be saved.
- `-output_name <output_name>`: A unique name for the output file, which will be used to save the results.

**Example:**
To run ArNet2 with a sample input file `df_test.csv`, save the output in the `./results/` directory, and name the output file `af_detection_results`, use:
```
python3 run_ArNet2.py --input_file df_test.csv --save_path ./results/ --output_name af_detection_results
```

This command processes the `df_test.csv` file, runs the AF event detection, and saves the results as `af_detection_results` in the specified `./results/` folder.
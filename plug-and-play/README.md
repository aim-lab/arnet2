# Run ArNet2 on external dataset

ArNet2 is an Atrial Fibrillation (AF) events detection algorithm from long term beat-to-beat intervals.

## Requirements
- 64-bit Python 3.8.
- CUDA=10.0 must be installed first.
- Installation of the required library dependencies with
```
conda env create -f env.yml
```

## Usage

Activate environment
```
conda activate tf_gpu
```

run main code
```
python3 run_ArNet2.py input_file save_path ouput_name
```
Where
- `input_file`: location of a .mat/.csv/.excel file with the following fields:
  - `rr_data`: rr intervals
  - `rr_time`: time of the R-peaks in seconds
- `save_path`: location to save the output file
- `ouput_name`: A unique name for the output file.

Example file: `df_test.csv`.

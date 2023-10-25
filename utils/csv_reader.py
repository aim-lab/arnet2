import pandas as pd
# from plot_ecg import plot_ecg
import os
import time
import pathlib
import utils.consts as cts



if __name__ == '__main__':

    # ann_dict = read_ann('/MLdata/AIMLab/databases/jpafdb/010_20201203/010_RRData/RR_00ready.csv', ecg.time[0])
    csv_dir = "RRData"
    main_dir = pathlib.PurePath(cts.DATA_DIR / 'examples')
    d = [x for x in os.listdir(main_dir)]
    for ecg_example in d[42:]:
        print("parsing example " + ecg_example)
        # ann_dict = pd.DataFrame()
        # read first example ecg + annotations
        example_path = main_dir / ecg_example
        ecg = read_ecg(example_path / (ecg_example.split("_")[0] + ".csv"))
        len_csv = len([name for name in os.listdir(example_path / csv_dir)])
        ann_dict = read_ann(example_path / csv_dir, start_time=ecg.time[0], end_time=ecg.time.iloc[-1])
        # ann_files = os.listdir(example_path / csv_dir)
        # ann_files.sort()

        # for i, x in enumerate(ann_files):
        #     if i == 0:
        #         ann_dict_temp = read_ann(example_path / csv_dir / x, start_time=ecg.time[0])
        #     elif i+1 ==len_csv:
        #         ann_dict_temp = read_ann(example_path / csv_dir / x, end_time=ecg.time.iloc[-1])
        #     else:
        #         ann_dict_temp = read_ann(example_path / csv_dir / x)
        #     ann_dict = ann_dict.append(ann_dict_temp)
        # ann_dict.reset_index(inplace=True)
        ecg_waveform = ecg['data_ch1'].iloc[2:].values
        ecg_waveform = ecg_waveform.astype(float)
        fs=1/125
        # plot_ecg(ecg_waveform, ann_dict, fs, len_n=60, out_dir=example_path / (ecg_example + "_example_plot.png"))

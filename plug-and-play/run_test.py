import os
import sys
sys.path.append('/home/shanybiton/repos/Shany_Repo/utils/')
import subprocess
import pandas as pd
from pathlib import PurePath
import numpy as np
import wfdb


def comp_metrics(rec_length, events):
    """
    This function computes statistical metrics for exportation.
    :param directory: The directory under which the files will be saved.
    :param rec_length: overall time of recording in seconds.
    :param events: dataframe of AF events with starting and end time after concatenating close-by events.
    :param unique_name: file name.
    """
    if len(events) == 0:
        return 4*[0]
    events['duration'] = events['End'] - events['Beginning']
    n_events = len(events)
    max_event = max(events["duration"])
    min_event = min(events["duration"])
    afb = events['duration'].sum() / rec_length
    print(f'Number of AF events: {n_events},\n'
          f'Longest event: {round(max_event, 2)} sec,\n'
          f'Shortest event: {round(min_event, 2)} sec,\n'
          f'AF-burden: {round(100*afb, 2)}%')

    return n_events, max_event, min_event, afb
# Path to the directory where the files are stored
directory_path = "./test_files"

# List all the files in the directory (you can also filter for certain file types)
files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]  # Adjust extension as needed

# Loop through each file and run the script with that file as an argument
for i, file_name in enumerate(files):
    print(f'processing file: {file_name}')
    pat = file_name.split("_")[0]
    record = wfdb.rdrecord(f"/oriondata/AIMLab/Shany/databases/shdb_physionet/{pat}")
    file_path = os.path.join(directory_path, file_name)
    output_path = f'{pat}_df_test_proba'
    # Run the script with the file as an argument
    subprocess.run(['python', 'run_ArNet2.py', '--input_file', file_path, '--output_name', output_path])
    subprocess.run(['python', '/home/shanybiton/repos/Shany_Repo/utils/export_arnet2_output_to_physiozoo.py',
                    '--input_file', f"/home/shanybiton/repos/Shany_Repo/plug-and-play/{output_path}.csv",
                    '--output_name', output_path.replace('_df_test_proba.csv', 'physiozoo_rhythms')])
    df = pd.read_csv(output_path + '.txt', sep="\s+|;|,", error_bad_lines=False, skiprows=6,
                                         engine='python')

    n_events, max_event, min_event, afb = comp_metrics(len(record.p_signal) / 200, df)
    if i == 0:
        rhythm_df = pd.DataFrame({'pat': pat, 'n_events': [n_events], 'max_event': [max_event],
                                  'min_event': [min_event], 'afb': [100*afb]})
    else:
        rhythm_df = rhythm_df.append(
            pd.DataFrame({'pat': pat, 'n_events': [n_events], 'max_event': [max_event],
                          'min_event': [min_event], 'afb': [100*afb]})
        )
print(rhythm_df)
rhythm_df.to_csv('./rhythm_df.csv', index='False')
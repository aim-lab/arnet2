import pandas as pd
import os
import pathlib
import numpy as np
import argparse
from warnings import warn


def export_rhythms_to_physiozoo(directory, start_events, end_events, rhythms, unique_name):
    """
    This function exports a given recording to the physiozoo format for further investigation.
    The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
    The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
    :param directory: The directory under which the files will be saved.
    :param rhythms: bool array with True for AF and False for non-AF.
    :param start_events: time array of the start time of all windows.
    :param end_events: time array of the end time of all windows.
    """
    rhythms_header = ['---\n',
                      'type: rhythms annotation\n',
                      'source file: ',  # To be completed by filename
                      '\n',
                      '---\n',
                      '\n',
                      'Beginning\tEnd\t\tClass\n']
    diff = 10  # events with less then 10 seconds difference count as same event
    if not os.path.exists(directory):
        os.makedirs(directory)
    rhythms_full_path = pathlib.PurePath(directory) / (str(unique_name) + '.txt')
    # Rhythms
    with open(rhythms_full_path, 'w+') as rhythms_file:
        rhythms_header[2] = 'source file: rhythms.txt\n'
        rhythms_file.writelines(rhythms_header)
        final_rhythms = np.array(rhythms.astype(int))
        mask_rhythms = final_rhythms > 0  # We do not keep NSR as rhythm
        start_events, end_events = start_events[mask_rhythms], end_events[mask_rhythms]
        final_rhythms = final_rhythms[mask_rhythms]
        events = pd.DataFrame({'start_events': start_events, 'end_events': end_events, 'rhythms': final_rhythms})
        events = (events.groupby((events.start_events - events.end_events.shift() > diff).cumsum()).agg(
            {'start_events': 'min', 'end_events': 'max', 'rhythms': 'first'})[
            ['start_events', 'end_events', 'rhythms']])
        final_rhythms_str = np.array(['AFIB' for i in np.array(events.rhythms)])
        rhythms_file.write('\n'.join(
            ['%.5f\t%.5f\t%s' % (events.start_events.to_list()[i], events.end_events.to_list()[i], final_rhythms_str[i])
             for i
             in
             range(len(final_rhythms_str))]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Use ArNet2 in a plug-and-play manner.')
    parser.add_argument('--input_file', type=str, default='./df_test_proba.csv',
                        help='path to ArNet2 output file')
    parser.add_argument('--save_path', type=str, default='./',
                        help='location to save the output file')
    parser.add_argument('--output_name', type=str, default='physiozoo_rhythms',
                        help='A unique name for the output file')

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    print(args)

    output_df = pd.read_csv(args.input_file)
    export_rhythms_to_physiozoo(args.save_path, output_df.start_time, output_df.end_time, output_df.pred,
                                args.output_name)


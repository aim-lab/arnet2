import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import glob
import  io
from preprocessing.Feature_extractor import bandpass_filter
import consts as cts
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import ecg_plot
from math import ceil
def reorganize_data(data, lead_index, orig_index, actual_fs, orig_fs):
    from scipy import signal
    ecg = np.zeros(shape=(4096, 12))
    for i_lead, lead in enumerate(lead_index):
        ecg_lead = data[:, orig_index[lead]]
        ecg_lead = signal.resample(ecg_lead, int(len(ecg_lead) * actual_fs / orig_fs))
        ecg_lead = bandpass_filter(data=ecg_lead, id=i_lead, lead=lead, lowcut=0.67, highcut=100, signal_freq=400,
                                   filter_order=75, notch_freq=50, debug=False)
        padding = 4096-len(ecg_lead)
        offset = int(round(padding/2))
        ecg[offset:len(ecg_lead) + offset, i_lead] = ecg_lead
        # ecg[:, i_lead] = ecg_lead
    return ecg


def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_range=3.6, minor_time_step=0.04,
             minor_amp_step=0.1):
    minor_time_ticks = np.arange(0, 11, minor_time_step)
    major_time_ticks = np.arange(-11, 11, 5*minor_time_step)
    minor_amp_ticks = np.arange(-2, 2, minor_amp_step)
    major_amp_ticks = np.arange(-2, 2, 5*minor_amp_step)
    ax.set_xticks(major_time_ticks)
    ax.set_xticks(minor_time_ticks, minor=True)
    ax.set_yticks(major_amp_ticks)
    ax.set_yticks(minor_amp_ticks, minor=True)
    # ax.set_yticks(np.arange(ceil(np.mean(y)-amplitude_range), ceil(np.mean(y)+amplitude_range), 1.0))

    # ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))

    # ax.minorticks_on()
    ax.grid(which='both')
    # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylim(np.mean(y)-amplitude_range, np.mean(y)+amplitude_range)
    ax.set_xlim(0, secs)
    ax.grid(b=True, which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(b=True, which='minor', linestyle='-', linewidth='0.5', color='red', alpha=0.2)

    ax.plot(x, y, linewidth=lwidth, color='k')

def plot_ecg_fig(df, patient_id, exams_id, namedfile,
                 path_to_save='/home/shanybiton/AIMLabProjects/afib-prediction/outputs/examples/Grant/id_'):
    path_to_csv = '/MLdata/AIMLab/databases/tnmg/ecg-traces/ecg-traces/preprocessed/traces.hdf5'
    f = h5py.File(path_to_csv, 'r')
    # Get ids
    traces_ids = np.array(f['id_exam'])
    x = f['signal']

    signals_num = 12
    lead_index = ('DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

    for k in patient_id:
        path = path_to_save + str(k)

        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        exams_k = df[df['id_patient'] == k]
        # patients_id_exams[m] in i for id_k['id_exam']  in patients_id_exams
        contained = [x in exams_k['id_exam']._values for x in exams_id]
        exam_to_export = (np.asarray(exams_id))[contained]
        exams_id_to_export = [(np.where(traces_ids == exam_to_export[0]))[0][0]]
        # , (np.where(traces_ids == exam_to_export[1]))[0][0]]
        # for m in exams_id_to_export:
        data = x[exams_id_to_export[0], :, :]
        f = plt.figure()
        f.subplots_adjust(hspace=1, wspace=0.5)

        axes = []
        ax = plt.subplot(signals_num / 6, 1, 1)

        # ylims = (-1.5, 1.5)
        start_time = 0  # beginning position in seconds
        time = 10.24  # data length in seconds
        sample_length = int(time * fs)  # data length in sample units
        end_time = start_time + time
        t = np.arange(start_time, end_time, 1 / fs)
        vl1 = np.arange(start_time, end_time, 0.04)
        vl2 = np.arange(start_time, end_time, 0.2)

        for j in np.arange(0, 12, 2):
            f = plt.figure()
            f.subplots_adjust(hspace=1, wspace=0.5)
            for i in np.arange(0 + j, 2 + j):
                MaxAmp = round(np.max(data[:, i]) + 0.5)
                MinAmp = round(np.min(data[:, i]) - 0.5)
                ax = plt.subplot(signals_num / 6, 1, i + 1 - j)
                axes.append(ax)

                plt.ylabel('voltage (mV)')

                # draw pink grid
                hl1 = np.arange(MinAmp, MaxAmp, 0.1)
                hl2 = np.arange(MinAmp, MaxAmp, 0.5)

                ax.vlines(vl1, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
                ax.hlines(hl1, start_time, end_time, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
                ax.vlines(vl2, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
                ax.hlines(hl2, start_time, end_time, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)

                # read data for specified signal
                # equal to record.read(i, ...
                # draw signal
                ax.plot(t, data[:, i], linewidth=1, color='k', alpha=1.0)
                ax.tick_params(axis="y", labelsize=8)
                ax.set_title(str(lead_index[i]), color='k')
                # ax.text(0.55,0.02, ('patient id: %d, exam id: %d' %(patient['id_exam'],patient['id_patient'])), transform=ax.transAxes)
                ax.annotate(('patient id: %d, exam id: %d' % (k, exam_to_export[0])), xy=(1, 0),
                            xycoords='axes fraction',
                            fontsize=8,
                            horizontalalignment='right', verticalalignment='bottom')
                plt.ylim(MinAmp, MaxAmp)
                plt.xlim(start_time, end_time)

            plt.xlabel('time (s)')
            xticklabels = [a.get_xticklabels() for a in axes[:-1]]
            plt.setp(xticklabels, visible=False)
            plt.tight_layout()

            plt.show()

            f.savefig(path + '/' + str(exam_to_export[0]) + '_' + str(j / 2) + '.png', dpi=800)
        m = [(x / 2) for x in np.arange(0, 12, 2)]

        im_name = [path + '/' + str(exam_to_export[0]) + '_' + str(n) + '.png' for n in m]
        images1 = [Image.open(x) for x in im_name[:3]]
        images2 = [Image.open(x) for x in im_name[3:]]

        widths, heights = zip(*(i.size for i in images1))

        gap = 300
        max_width = max(widths)
        total_height = sum(heights) - (len(images1) - 1) * gap

        new_im1 = Image.new('RGB', (max_width, total_height))
        new_im2 = Image.new('RGB', (max_width, total_height))
        final_image = Image.new('RGB', (2 * max_width, total_height))
        y_offset = 0
        x_offset = max_width
        for im1, im2 in zip(images1, images2):
            new_im1.paste(im1, (0, y_offset))
            new_im2.paste(im2, (0, y_offset))
            y_offset += im1.size[1] - gap

        final_image.paste(new_im1, (0, 0))
        final_image.paste(new_im2, (x_offset, 0))
        final_image.save(path + '/' + str(exam_to_export[0]) + '.png')



def plot_ecg_fig_MD(
        ecg,
        lead_index,
        sample_rate=500,
        title='ECG 12',
        lead_order=None,
        style=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        peaks=False,
        peak_dict=None,
        detector='epltd0',
):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,  # the left side of the subplots of the figure
        right=1,  # the right side of the subplots of the figure
        bottom=0,  # the bottom of the subplots of the figure
        top=1
    )

    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if (style == 'bw'):
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0.7)

    if (show_grid):
        ax.set_xticks(np.arange(x_min, x_max, 0.2))
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height / 2) * ceil(i % rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if (c > 0):
                    x_offset = secs * c
                    if (show_separate_line):
                        ax.plot([x_offset, x_offset],
                                [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3],
                                linewidth=line_width * display_factor, color=color_line, zorder=2, )

                t_lead = lead_order[c * rows + i]

                step = 1.0 / sample_rate
                if (show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor, zorder=2)
                ax.plot(
                    np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_line, zorder=4
                )
                if peaks:
                    yvals = ecg[t_lead] + y_offset
                    tvals = np.arange(0, len(ecg[t_lead]) * step, step) + x_offset
                    peakvals = peak_dict[detector][str(t_lead)]
                    ax.scatter(tvals[peakvals], yvals[peakvals], zorder=3)

    # path_to_csv = '/MLdata/AIMLab/databases/tnmg/ecg-traces/ecg-traces/preprocessed/traces.hdf5'
    # f = h5py.File(path_to_csv, 'r')
    # # Get ids
    # traces_ids = np.array(f['id_exam'])
    # x = f['signal']
    #
    # signals_num = 12
    # rows = int(round(signals_num/columns))
    # row_height = 6
    # for idx, k in enumerate(exams_id):
    #     print("ploting MD ecg recording for recording id: " + str(k))
    #     path = path_to_save
    #     exams_id_to_export = [(np.where(traces_ids == k))[0][0]]
    #     data = x[exams_id_to_export[0], :, :]
    #     axes = []
    #     ecg_len = len(data[:, 0]) / fs
    #     f = plt.figure(figsize=(ecg_len*columns, rows * row_height / 5))
    #     gs = mpl.gridspec.GridSpec(3, 1)
    #     gs.update(wspace=0, hspace=0)
    #     # mpl.rcParams['axes.linewidth'] = 0.1  # set the value globally
    #     # f.subplots_adjust(hspace=0, wspace=0)
    #     idx_loc = 0
    #     for i in np.arange(rows):
    #         ldl = lead_index[i::rows]
    #         line = data[:, i::rows]  # get only leads {I, aVR, V1, V4}
    #         start = (line[:, 0] != 0).argmax(axis=0)
    #         end = len(line) - (line[::-1, 1] != 0).argmax(axis=0)
    #         if filt:
    #             filtered = np.zeros(shape=(line.shape))
    #             for d in range(4):
    #                 ecg_lead = line[:, d]
    #                 ecg_filtered = bandpass_filter(ecg_lead, k, d+idx_loc, 0.67, 100, fs, 75, debug=False)
    #                 # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 48, 52, fs, 3)
    #                 # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 58, 62, fs, 3)
    #                 filtered[:,d] = ecg_filtered
    #             line=filtered
    #         line = (line[start:end, :]).flatten('F')
    #         time = (end - start) / fs  # data length in seconds
    #         start_time = 0  # beginning position in seconds
    #         end_time = start_time + time
    #         end_time_of_plot = 4 * end_time
    #         MaxAmp = round(np.max(line) + 0.5)
    #         MinAmp = round(np.min(line) - 0.5)
    #         ax = plt.subplot(gs[i])
    #         axes.append(ax)
    #         # plt.ylabel('Amplitude (mV)')
    #         t = np.arange(start_time, end_time_of_plot, 1 / fs)
    #
    #         # draw pink grid
    #         vl1 = np.arange(start_time, end_time_of_plot, 0.04)
    #         vl2 = np.arange(start_time, end_time_of_plot, 0.2)
    #
    #         hl1 = np.arange(MinAmp, MaxAmp, 0.1)
    #         hl2 = np.arange(MinAmp, MaxAmp, 0.5)
    #
    #         ax.vlines(vl1, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
    #         ax.hlines(hl1, start_time, end_time_of_plot, colors='r', linestyles='-', alpha=0.2, linewidth=0.4)
    #         ax.vlines(vl2, MinAmp, MaxAmp, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
    #         ax.hlines(hl2, start_time, end_time_of_plot, colors='r', linestyles='-', alpha=0.6, linewidth=0.4)
    #
    #         # draw lead location
    #         ld = np.arange(0, end_time_of_plot, end_time)
    #         ld[0] += 0.3
    #         ax.vlines(ld, line.mean() - 0.5, line.mean() + 0.5, colors='b', linestyles='-', alpha=1, linewidth=1.2)
    #         # read data for specified signal
    #         # equal to record.read(i, ...
    #         # draw signal
    #         ax.plot(t.astype(float)[:len(line)], line, linewidth=0.5, color='k', alpha=1.0)
    #         plt.setp(ax.spines.values(), color='r', alpha=0.6, linewidth=0.4)
    #         # ax.set(frame_on=False)
    #         # ax.tick_params(axis="y", labelsize=8)
    #         # ax.set_title(idx, color='k')
    #         # plt.text(0.55, 0.02, ('exam id: %d' % k), transform=ax.transAxes)
    #         if i == 2:
    #             ax.annotate(('exam id: %d' % (k, )), xy=(1, 0),
    #                         xycoords='axes fraction',
    #                         color='b', fontsize=16,
    #                         horizontalalignment='right', verticalalignment='bottom')
    #         for mm, loc in enumerate(ld):
    #             ax.annotate(ldl[mm], xy=(loc, line.mean() - 0.8), textcoords="offset points",
    #                         color='b', fontsize=18, xytext=(0.2, -2))
    #         plt.ylim(MinAmp, MaxAmp)
    #         plt.xlim(start_time, end_time_of_plot)
    #
    #         # plt.xlabel('time (s)')
    #         # xticklabels = [a.get_xticklabels() for a in axes[:-1]]
    #         plt.xticks([])
    #         plt.yticks([])
    #         idx_loc = idx_loc+4
    #     plt.suptitle(idx, color='k', fontsize=20)
    #     plt.tight_layout()
    #
    #     f.savefig(path + '/' + str(idx) + '_' + str(k) + '_' + namedfile + '.' + str(format), dpi=400, transparent=True)
    #     plt.close()
    # return

def plot_ecg_as_Antonio(ax, data, start, fs, filt=False, length_ecg=10,
                    savefig=False, savedir=None, dpi=400):
    start_sample = start * fs
    end_sample = round(start_sample + length_ecg*fs)
    if filt:
        ecg_filtered = bandpass_filter(data, 0, j, 0.67, 99, fs, 75, debug=False)
        data=ecg_filtered
    data = (data[start_sample:end_sample]).flatten('F')
    MaxAmp = round(np.max(line) + 0.5)
    MinAmp = round(np.min(line) - 0.5)
    seconds = len(data) / fs
    step = 1.0 / fs
    _ax_plot(ax=ax, x=np.arange(0, len(data) * step, step), y=data, secs=seconds, amplitude_range=1.5,)
    plt.ylabel('Amplitude [mV]')
    plt.xlabel('Time [sec]')
    ax.tick_params(axis='both', which='major', labelsize=6, pad=-1)
    plt.tight_layout()
    if savefig:
        plt.tight_layout()
        lab = 'ecg_example.png'
        plt.savefig(savedir + lab, dpi=dpi, transparent=True)
        plt.close()

def plot_rr(ax, data, start, fs, end=None, length_ecg=10, lwidth=0.5, savefig=False, savedir=None, dpi=400, format='png'):
    start_sample = round(start * fs)

    if end is None:
        end_sample = round(start_sample + length_ecg * fs)
    else:
        end_sample = round(end * fs)
        length_ecg = (end_sample-start_sample)/fs
    timeline = np.arange(0, length_ecg + 2, 1 / fs)
    ann = data[np.where(np.logical_and(data >= start_sample-200, data < end_sample+200))]
    ann -= (start_sample - 200)
    rr = np.diff(ann) / fs

    ax.set_xticklabels([])
    # ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.plot(timeline[ann][:-1], rr, linewidth=lwidth, color='k')
    ax.set_xlim(1, length_ecg)
    ax.set_ylim(0.2, 1.2)
    ax.set_ylabel('RR-interval [sec]')

    ax.tick_params(axis='y', which='major', labelsize=6, pad=-2)

    plt.tight_layout()
    if savefig:
        plt.tight_layout()
        lab = 'RR_example.' + format
        plt.savefig(savedir + lab, dpi=dpi, transparent=True)
        plt.close()

def plot_example(recordings, ecg_path, ann_path, windows, lab=None, savefig=False, savedir=None, dpi=400, format='png'):
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(len(recordings), 2, width_ratios=[0.7, 0.3], hspace=0.3, wspace=0.1)
    for i, pat in enumerate(recordings):
        t_ecg = np.sort(glob.glob(str(ecg_path / str(pat) / "*ecg*")))
        f_ecg = open(t_ecg[0])
        lines = f_ecg.readlines()[23:]
        data = pd.read_csv(io.StringIO('\n'.join(lines)), delim_whitespace=True, header=None)
        data = pd.DataFrame.to_numpy(data)
        line = data[:, 0]  # get only leads from lead_index
        ax = fig.add_subplot(gs[i, 0])  # row 0, col 0
        ax.set_ylabel('Amplitude [mv]')
        plot_ecg_as_Antonio(ax, data=line, start=windows[pat][0], fs=cts.EPLTD_FS, filt=False)
        t_ann = glob.glob(str(ann_path / str(pat) / "*peaks*"))
        f_ann = open(t_ann[0])
        ann = f_ann.readlines()[13:]
        ann = pd.read_csv(io.StringIO('\n'.join(ann)), delim_whitespace=True, header=None)
        ann = pd.DataFrame.to_numpy(ann).astype(int).reshape(1,-1).flatten()
        ax = fig.add_subplot(gs[i, 1])  # row 0, col 0
        plot_rr(ax=ax, data=ann, start=windows[pat][0], fs=cts.EPLTD_FS) # end=windows[pat][1],
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.1)
    if savefig:
        lab = lab + '.' + format
        plt.savefig(savedir / lab, dpi=dpi, transparent=True)
        plt.close()
    return

if __name__ == '__main__':
    # format = '.pdf'
    # lead_index = {0: '1', 1: '2'}
    # recordings={}
    # windows={}
    # recordings['FP']= ['O518B978', '111'] #'0440',
    # recordings['FN']= ['0004', 'V720Gd6e']
    # recordings['start'] = [20437, 4859] #16378,
    # recordings['start'] = [60385, 40877]
    # # windows['0440'] = [16377.25750, 16408.84500]
    # windows['O518B978'] = [20417.41250, 20476.40750]
    # windows['111'] = [4849.21750, 4880.71000]
    # # windows['0440'] = [16377.25750, 16408.84500]
    # windows['0004'] = [60375.11250, 60403.43500]
    # windows['V720Gd6e'] = [40867.20250, 40890.45000]
    # gs = gridspec.GridSpec(len(recordings['FP']), 2, width_ratios=[0.7, 0.3])
    # gs.update(hspace=0.3)

    # for Jonathan
    # circ_rec = ['N520960f', 'L4198c15', 'R3209148', 'U5219a9c']
    # circ_windows={}
    # circ_windows['N520960f'] = [14267]
    # circ_windows['L4198c15'] = [10489]
    # circ_windows['R3209148'] = [45900]
    # circ_windows['U5219a9c'] = [46803]
    # plot_example(recordings=circ_rec, ecg_path=cts.BASE_DIR / 'Shany/medAIM/RBAFDB-annotated/',
    #              ann_path=cts.BASE_DIR / 'Shany/medAIM/RBAFDB-annotated/',
    #              windows=circ_windows, lab='circ_examples', savefig=True,
    #              savedir=pathlib.PurePath('/home/shanybiton/repos/CircadianAF/'),
    #              format='pdf')

    # plot_example(recordings=recordings['FP'], ecg_path=cts.REPO_DIR / 'figs/error_analysis/recordings/FP',
    #              ann_path=cts.REPO_DIR / 'figs/error_analysis/recordings/FP',
    #              windows=windows, lab='FP_examples', savefig=True, savedir=cts.REPO_DIR / 'figs/error_analysis',
    #              format='pdf')
    #
    # plot_example(recordings=recordings['FN'], ecg_path=cts.REPO_DIR / 'figs/error_analysis/recordings/FN',
    #              ann_path=cts.REPO_DIR / 'figs/error_analysis/recordings/FN',
    #              windows=windows, lab='FN_examples', savefig=True, savedir=cts.REPO_DIR / 'figs/error_analysis',
    #              format='pdf')


    format = '.pdf'
    lead_index = {0: '1', 1: '2'}
    recordings = {}
    windows = {}
    recordings['FP'] = ['U419B33b', '111']  # '0440',
    recordings['FN'] = ['0004', 'V720Gd6e']
    recordings['FP start'] = [7059, 4859]  # 16378,
    recordings['FN start'] = [60385, 40877]
    # windows['0440'] = [16377.25750, 16408.84500]
    windows['U419B33b'] = [7059.41250, 7069.41250]
    windows['O518B978'] = [20417.41250, 20476.40750]
    windows['111'] = [4849.21750, 4880.71000]
    # windows['0440'] = [16377.25750, 16408.84500]
    windows['0004'] = [60375.11250, 60403.43500]
    windows['V720Gd6e'] = [40867.20250, 40890.45000]
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(recordings['FP']) + len(recordings['FN']) + 1, 2, width_ratios=[0.7, 0.3],
                           height_ratios=[4, 4, 0.02, 4, 4], hspace=0.7, wspace=0.12)
    for i, pat in enumerate(recordings['FP']):
        t_ecg = glob.glob(str(cts.REPO_DIR / 'figs/error_analysis/recordings/FP' / str(pat) / "*ecg*"))
        f_ecg = open(t_ecg[0])
        lines = f_ecg.readlines()[23:]
        data = pd.read_csv(io.StringIO('\n'.join(lines)), delim_whitespace=True, header=None)
        data = pd.DataFrame.to_numpy(data)
        line = data[:, 0]  # get only leads from lead_index
        ax = fig.add_subplot(gs[i, 0])  # row 0, col 0
        plot_ecg_as_Antonio(ax, data=line, start=recordings['FP start'][i], fs=cts.EPLTD_FS, filt=False)
        t_ann = glob.glob(str(cts.REPO_DIR / 'figs/error_analysis/recordings/FP' / str(pat) / "*peaks*"))
        f_ann = open(t_ann[0])
        ann = f_ann.readlines()[13:]
        ann = pd.read_csv(io.StringIO('\n'.join(ann)), delim_whitespace=True, header=None)
        ann = pd.DataFrame.to_numpy(ann).astype(int).reshape(1, -1).flatten()
        ax = fig.add_subplot(gs[i, 1])  # row 0, col 0
        plot_rr(ax=ax, data=ann, start=windows[pat][0], fs=cts.EPLTD_FS)  # end=windows[pat][1],
    ax = fig.add_subplot(gs[3, 0])
    ax.remove()
    for i, pat in enumerate(recordings['FN']):
        t_ecg = glob.glob(str(cts.REPO_DIR / 'figs/error_analysis/recordings/FN' / str(pat) / "*ecg*"))
        f_ecg = open(t_ecg[0])
        lines = f_ecg.readlines()[23:]
        data = pd.read_csv(io.StringIO('\n'.join(lines)), delim_whitespace=True, header=None)
        data = pd.DataFrame.to_numpy(data)
        line = data[:, 0]  # get only leads from lead_index
        ax = fig.add_subplot(gs[i + 3, 0])  # row 0, col 0
        plot_ecg_as_Antonio(ax, data=line, start=recordings['FN start'][i], fs=cts.EPLTD_FS, filt=False)
        t_ann = glob.glob(str(cts.REPO_DIR / 'figs/error_analysis/recordings/FN' / str(pat) / "*peaks*"))
        f_ann = open(t_ann[0])
        ann = f_ann.readlines()[13:]
        ann = pd.read_csv(io.StringIO('\n'.join(ann)), delim_whitespace=True, header=None)
        ann = pd.DataFrame.to_numpy(ann).astype(int).reshape(1, -1).flatten()
        ax = fig.add_subplot(gs[i + 3, 1])  # row 0, col 0
        plot_rr(ax=ax, data=ann, start=windows[pat][0], fs=cts.EPLTD_FS)  # end=windows[pat][1]
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.1)

    lab = 'examples_4' + format
    fig.savefig(cts.REPO_DIR / 'figs/error_analysis' / lab, dpi=400, transparent=True)


import sys

sys.path.append("/home/shanybiton/repos/afib-prediction/preprocessing")
import csv
import os
import h5py
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import physionet_matlab
import scipy
import wfdb.processing
import consts as cts
import pathlib
from scipy.signal import savgol_filter, butter, sosfreqz, sosfiltfilt
from scipy.spatial import cKDTree
from multiprocessing import Process
from preprocessing.Bundle_Branch_Block import extraction_feature_wavedet_BBB

from preprocessing.Intervals_duration import extract_intervals_duration
from preprocessing.Waves_characteristics import extract_waves_characteristics


##################################################
############# Data Processing & FE ###############
##################################################

def get_trace(traces_ids, x, id):
    loc = np.where(traces_ids == int(id))  # find the location of the exam
    trace = x[loc].reshape(-1, x.shape[-1])  # shape: (4096,12)
    return trace


def get_data(y, traces_ids):
    # ----- Data settings ----- #
    diagnosis = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    # ------------------------- #
    y.set_index('id_exam', drop=True, inplace=True)
    y = y.reindex(traces_ids, copy=False)
    df_diagnosis = y.reindex(columns=[d for d in diagnosis])
    y = df_diagnosis.values
    return y


def bandpass_filter(data, id, lead, lowcut, highcut, signal_freq, filter_order,  notch_freq=50, debug=False):
    """This function uses a Butterworth filter. The coefficoents are computed automatically. Lowcut and highcut are in Hz"""
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    sos = butter(filter_order, [low, high], btype="band", output='sos', analog=False)
    y = sosfiltfilt(sos, data)
    y = mne.filter.notch_filter(y.astype(np.float), signal_freq, freqs=notch_freq, verbose=debug)
    if debug:
        filename_freq = "exam_" + str(id) + "_lead_" + str(lead) + ".png"
        filename_spect = "exam_" + str(id) + "_lead_" + str(lead) + "_spect.png"

        # get_freq_plot(data, y, sos, filter_order, signal_freq, filename_freq)
        get_spect_plot(data, y, signal_freq, filename_spect, dpi=400)
    return y


def get_freq_plot(y_orig, y_filt, coefs, order, fs, filename):
    w, h = sosfreqz(coefs, worN=2000)
    t = np.linspace(0, len(y_orig) / fs, len(y_orig))
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), '#3465a4')
    # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5 * fs)
    plt.title("Bandpass Filter Frequency Response, order=" + str(order))
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), '#3465a4')
    # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5 * fs)
    plt.xlim(0, 3)
    plt.title("Bandpass Filter Frequency Response, order=" + str(order))
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(cts.REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename, dpi=400)
    plt.close()
    return


def get_spect_plot(y_orig, y_filt, fs, filename, dpi=400):
    labels = ["(a)", "(b)"]
    # get the FFT of the signals
    ps_orig = np.abs(np.fft.fft(y_orig)) ** 2
    ps_filt = np.abs(np.fft.fft(y_filt)) ** 2
    t = np.linspace(0, len(y_orig) / fs, len(y_orig))
    time_step = 1 / fs
    freqs_orig = np.fft.fftfreq(ps_orig.size, time_step)
    idx_orig = np.argsort(freqs_orig)
    freqs_filt = np.fft.fftfreq(ps_filt.size, time_step)
    idx_filt = np.argsort(freqs_filt)
    ps_orig_log = 10 * np.log10(ps_orig)
    ps_filt_log = 10 * np.log10(ps_filt)
    # plt.plot(freqs_filt[idx_filt], ps_filt_log[idx_filt], color='#3465a4', linestyle='--', label='Filtered')

    # Get the PSD of the signals using welch method
    fx, Pxx = scipy.signal.welch(y_orig, fs, nperseg=len(y_orig))
    fy, Pyy = scipy.signal.welch(y_filt, fs, nperseg=len(y_filt))
    ps_orig_log = 10 * np.log10(Pxx)
    ps_filt_log = 10 * np.log10(Pyy)
    fig = plt.figure(dpi=400)
    # plt.semilogy(fx, Pxx, label="Raw data", color='r')
    # plt.semilogy(fy, Pyy, label="Filtered", color='#3465a4', linewidth=1)
    ax0 = plt.subplot(2, 1, 1)
    ax0.semilogy(fx, Pxx, color='r', linewidth=1, label='Raw ECG', )
    ax0.semilogy(fy, Pyy, color='#3465a4', linewidth=0.5, label='Filtered ECG', )

    ax0.set_title('Power spectrum density')
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_ylabel("PSD (V^2/Hz)")
    ax0.grid(True, which='both')
    ax0.legend(loc=1)

    ax1 = plt.subplot(2, 1, 2)
    ax1.plot(t, y_orig * 10, color='r', linewidth=1)
    ax1.plot(t, y_filt * 10, color='#3465a4', linewidth=0.8)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('V [mv]')
    ax1.grid(True, which='both')

    Y1 = ax0.get_tightbbox(fig.canvas.get_renderer())
    for a, label in zip([ax0, ax1], labels):
        bbox = a.get_tightbbox(fig.canvas.get_renderer())
        fig.text(Y1.x0 - 50, bbox.y1 + 100, label, fontsize=14, va="top", ha="left",
                 transform=None)
    ax1.set_xlim(1,5)
    plt.tight_layout()
    plt.savefig(cts.REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename, dpi=dpi)
    print(cts.REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename)
    plt.close()
    return


def get_label(integer):
    return annot_to_patho[str(integer)]


def extraction_feature_wavedet(filename, list_lead, ecg, freq, runtime, qrs_peaks=[]):
    """This function processes the output of wavedet into a dataframe, for all of the leads considered"""
    wavedet_3D_dict, wavedet_3D_dict_challenge = wavedet_3D_wrapper(filename=filename, list_lead=list_lead,
                                                                    qrs_peaks=qrs_peaks, runtime=runtime, ecg=ecg, fs=freq)
    offset = 12
    MOR_feat = pd.DataFrame([])
    for i, lead in enumerate(list_lead):
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        extracted_features = [(key, np.asarray(wavedet_3D_dict_i[key], dtype=np.int32)) for key in cts.interesting_keys]
        dict_features = {key: value for key, value in extracted_features}
        MOR_feat_lead = extraction_MOR_features(ecg[lead + offset], freq, dict_features, lead)
        MOR_feat = pd.concat([MOR_feat, MOR_feat_lead], axis=1)
    HRV_features = extraction_HRV_features(freq, wavedet_3D_dict)
    return MOR_feat, HRV_features, wavedet_3D_dict


def challenge_wrapper(ecg, list_lead, runtime, dict):
    """This function processes the output of the challenge matlab function, into a dataframe. The challenge function required
    qrs points. Inside this function, we also compute the bsqi index, which allows later on to discard ecgs."""
    """
    # ------------------------------------------------------------------------------- #
    # All bsqi related calculation were removed (#) by Shany #
    # Those were carefully calculated previously  #
    # ------------------------------------------------------------------------------- #
    """
    challenge_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        points_i = dict[i]
        if hasattr(points_i["R"], "__len__"):
            qrs_points = np.array(points_i["R"])
            qrs_points[np.isnan(qrs_points)] = 0
            qrs_points = np.asarray(qrs_points, dtype=np.int64)
            challenge_result_i = runtime.challenge(ecg[lead].tolist(), lead, 500, qrs_points.tolist()[0], dict[lead])
        else:
            qrs_points = np.array(points_i["R"]).reshape(1, 1)[0]
            qrs_points[np.isnan(qrs_points)] = 0
            qrs_points = np.asarray(qrs_points, dtype=np.int64)
            challenge_result_i = runtime.challenge(ecg[lead].tolist(), lead, 500, qrs_points.tolist(), dict[lead])
        # R_points = wfdb.processing.gqrs_detect(fs=500, sig=ecg[lead], adc_gain=1000, adc_zero=0)
        # ind = bsqi(qrs_points, R_points)
        # challenge_result_i['bsqi_' + str(lead)] = ind
        for key, values in challenge_result_i.items():
            if np.isnan(values):
                values = 0
            challenge_result_i[key] = [values]
        challenge_result_i = pd.DataFrame.from_dict(challenge_result_i)
        challenge_DataFrame = pd.concat([challenge_DataFrame, challenge_result_i], axis=1)
    return challenge_DataFrame


def wavedet_3D_wrapper(filename, list_lead, qrs_peaks, runtime, ecg, fs):  # engine does not support structure array
    """Gets the output of the wavedet algorithm on Matlab. The raw wavedet algorithm had issues with detecting Q and S
    points so this is something I edited manually"""
    ecg_matlab = []
    import matlab
    l_peaks = min([len(qrs_peaks[i]) for i in range(12)])
    if l_peaks==0:
      qrs_peaks_matlab = []
    else:
      qrs_peaks_matlab = matlab.double([qrs_peaks[i][:l_peaks].tolist() for i in range(12)])
    for i in range(len(ecg)):
        ecg_matlab.append(ecg[i].tolist())
    ret_val = runtime.python_wrap_wavedet_3D(str(input_dir), ecg_matlab, filename, list_lead, qrs_peaks_matlab, fs,
                                             nargout=12)
    ret_val_matlab = [dict(ret_val[i]) for i in range(len(ret_val))]
    # print(ret_val)
    for i, lead in enumerate(list_lead):
        for key in (ret_val[i].keys()):
            if cts.interesting_keys.__contains__(key):
                if hasattr(ret_val[i][key], "__len__"):
                    ret_val[i][key] = np.array(ret_val[i][key]._data).reshape(ret_val[i][key].size, order='F')[0]
                    ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                    ret_val[i][key] = np.asarray(ret_val[i][key][1:], dtype=np.int64)
                else:
                    ret_val[i][key] = np.array(ret_val[i][key]).reshape(1, 1, order='F')[0]
                    ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                    ret_val[i][key] = np.asarray(ret_val[i][key][0:], dtype=np.int64)
        R_ref = ret_val[i]['R']
        Poff = ret_val[i]['Poff']
        Ton = ret_val[i]['Ton']
        if cts.interesting_keys.__contains__('Q'):  # choice
            for p, ind in enumerate(ret_val[i]['QRSon']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['Q'][p] == 0:
                        if R_ref[p] > ind + 1:
                            candidate = np.argmin(
                                ecg_lead[ind:R_ref[p]]) + ind  # todo: à remplacer par compute_argmin?
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind:ind + 5]))

                                if Poff[p] > 0:
                                    indice_minimal = (candidate + Poff[p]) / 2
                                else:
                                    indice_minimal = 0
                                while ind > 0 and np.abs((ecg_lead[
                                                              ind] - ref_value) / ref_value) < 0.2 and ind > indice_minimal:  # todo: divide by zero encountered
                                    ind = ind - 1
                                ret_val[i]['QRSon'][p] = ind
                            ret_val[i]['Q'][p] = candidate  # in order to correct when we do not detect any Q points
                        else:
                            continue
        if cts.interesting_keys.__contains__('S'):  # choice
            for p, ind in enumerate(ret_val[i]['QRSoff']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['S'][p] == 0:
                        if ind > R_ref[p] + 1:
                            candidate = np.argmin(ecg_lead[R_ref[p]:ind]) + R_ref[
                                p]
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind - 5:ind]))

                                if Ton[p] > 0:
                                    indice_maximal = (candidate + Ton[p]) / 2
                                else:
                                    indice_maximal = len(ecg_lead)
                                while ind < len(ecg_lead) and (
                                        np.abs((ecg_lead[ind] - ref_value) / ref_value)) and ind < indice_maximal:
                                    ind = ind + 1
                                ret_val[i]['QRSoff'][p] = ind
                            ret_val[i]['S'][p] = candidate  # in order to correct when we do not detect any S points
                        else:
                            continue
        # ref = ["Ton", "Poff", "R", "T"]
        # ref_full = [key for ind, key in enumerate(ref) if ~np.isnan(ret_val_matlab[i][key]).any()]
        # if len(ref_full) > 0:
        #     key_ref = [key for ind, key in enumerate(ref_full) if np.count_nonzero(ret_val[i][key]==0)==0]
        #     key_ref = key_ref[0]
        #     # key_ref = ref_full[0]
        #     key_len = len(ret_val[i][key_ref][ret_val[i][key_ref] > decg])
        # else:
        #     key_ref_loc = np.argmin([np.count_nonzero(ret_val[i][key] == 0) for key in ref])
        #     key_ref = ref[key_ref_loc]
        #     key_len = len(ret_val[i][key_ref][ret_val[i][key_ref] > decg])
        # for key in (ret_val[i].keys()):
        #     if cts.interesting_keys.__contains__(key):
        #         if hasattr(ret_val[i][key], "__len__") and key not in ['Ttipo', 'Ttipoon', 'Ttipooff']:
        #             ret_val[i][key] = ret_val[i][key][-key_len::] - decg
        #             ret_val[i][key][ret_val[i][key]<0] = 0
        #             temp_arr = ret_val[i][key].astype('float')
        #             temp_arr[temp_arr == 0] = np.nan
        #             ret_val_matlab[i][key] = matlab.double(list(temp_arr))
        #         if key in ['Ttipo', 'Ttipoon', 'Ttipooff']:
        #             T_len = len(ret_val[i]["T"])
        #             ret_val[i][key] = ret_val[i][key][-T_len::]
        #             ret_val_matlab[i][key] = matlab.double(list(ret_val[i][key]))
    return ret_val, ret_val_matlab


def bsqi(refqrs, testqrs, agw=0.05, fs=200):
    """
    This function is based on the following paper:
        Li, Qiao, Roger G. Mark, and Gari D. Clifford.
        "Robust heart rate estimation from multiple asynchronous noisy sources
        using signal quality indices and a Kalman filter."
        Physiological measurement 29.1 (2007): 15.

    The implementation itself is based on:
        Behar, J., Oster, J., Li, Q., & Clifford, G. D. (2013).
        ECG signal quality during arrhythmia and its application to false alarm reduction.
        IEEE transactions on biomedical engineering, 60(6), 1660-1666.

    :param refqrs:  Annotation of the reference peak detector (Indices of the peaks).
    :param testqrs: Annotation of the test peak detector (Indices of the peaks).
    :param agw:     Agreement window size (in seconds)
    :param fs:      Sampling frquency [Hz]
    :returns F1:    The 'bsqi' score, between 0 and 1.
    """

    agw *= fs
    if len(refqrs) > 0 and len(testqrs) > 0:
        NB_REF = len(refqrs)
        NB_TEST = len(testqrs)

        tree = cKDTree(refqrs.reshape(-1, 1))
        Dist, IndMatch = tree.query(testqrs.reshape(-1, 1))
        IndMatchInWindow = IndMatch[Dist < agw]
        NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
        TP = NB_MATCH_UNIQUE
        FN = NB_REF - TP
        FP = NB_TEST - TP
        Se = TP / (TP + FN)
        PPV = TP / (FP + TP)
        if (Se + PPV) > 0:
            F1 = 2 * Se * PPV / (Se + PPV)
            _, ind_plop = np.unique(IndMatchInWindow, return_index=True)
            Dist_thres = np.where(Dist < agw)[0]
            meanDist = np.mean(Dist[Dist_thres[ind_plop]]) / fs
        else:
            return 0

    else:
        F1 = 0
        IndMatch = []
        meanDist = fs
    return F1


def get_features_from_sqi(sqi_dict, leads, fs, agw, types=cts.ANNOTATION_TYPES, sqi_ref_ann="epltd0"):
    test_ann = types[types != sqi_ref_ann]
    sqi_dataframe = pd.DataFrame([])
    for ann in test_ann:
        signal_quality_dict = {}
        refqrs = sqi_dict['epltd0']
        testqrs = sqi_dict[ann]
        for lead in leads:
            len_ann = min([len(refqrs[str(lead)]), len(testqrs[str(lead)])])
            if len_ann <= 5:
                signal_quality_dict[lead] = 0
            else:
                signal_quality_dict[lead] = bsqi(refqrs[str(lead)], testqrs[str(lead)][:len_ann],
                                                 agw * np.ones(len_ann),
                                                 fs * np.ones(len_ann)
                                                 )
        sqi_dataframe[ann] = pd.Series(signal_quality_dict)
        col_names = ["sqi_" + str(lead) for lead in leads]
        sqi_features = pd.DataFrame(columns=col_names, data=sqi_dataframe.max(axis=1).values.reshape(1, -1))
        return sqi_features


def ecgtovcg(signal):
    """
    This function implementation is based on the paper: Linear affine transformations
    between 3-lead (Frank XYZ leads) vectorcardiogram and 12-lead electrocardiogram signals,
    Drew Dawson, Hui Yang, Milind Malshe.
    We convert an 8-leads ecg (leads I, II, V1, V2, V3, V4, V5, V6 to a planar VCG representation.
    We will only use a single plane projection of this VCG representation.
    Input: 12-leads ecg.
    Output: Planar representation of the ecg.
    """
    leads_to_be_transformed = [0, 1, 6, 7, 8, 9, 10, 11]
    ecg_toconvert = np.asarray(np.asarray(signal)[leads_to_be_transformed])
    # I       II      V1      V2     V3     V4     V5     V6
    invDower = np.array([[0.156, -0.010, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194],  # X transformation
                         [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],  # Y transformation
                         [-0.022, -0.102, 0.229, 0.310, 0.246, 0.063, -0.055, -0.108]])  # Z transformation
    vcg = np.dot(invDower, ecg_toconvert)
    return vcg


def maximal_amplitude(vcg):
    """
    This function allows to find the point in the plane where the vcg amplitude is maximum.
    This will allow to compute the VECG Angle right after
    """
    time = np.argmax(np.linalg.norm(vcg[0:2], axis=0))
    maximal_coordinates = vcg[0][time], vcg[1][time]
    return maximal_coordinates


def VECGAng(ecg):
    """
    This function computes the VECG Angle, the feature we needed according to the paper:
    Automatic detection of premature atrial contractions in the electrocardiogram
    Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov
    """
    vcg = ecgtovcg(ecg)
    coordinates = maximal_amplitude(vcg)
    VECGang_max = np.angle(complex(coordinates[0], coordinates[1]), deg=True)
    if VECGang_max < 0:
        VECGang_max = 360 + VECGang_max
    return VECGang_max


def extraction_MOR_features(ecg, freq, features_dict, lead):
    """This function produces the MOR features for a 12-leads ecg and concatenate them into a df"""
    # features_global = pd.DataFrame()
    interval_durations_feats = extract_intervals_duration(freq, features_dict, lead)
    waves_characteristics_feats = extract_waves_characteristics(ecg, freq, features_dict, lead)
    # features_AF = extraction_feature_AF(ecg, freq, features_dict, lead)
    # features_AVB = extraction_feature_AVB(ecg, freq, features_dict, lead)
    # features_PAC = extraction_feature_PAC(ecg, freq, features_dict, lead)
    # features_PVC = extraction_feature_PVC(ecg, freq, features_dict, lead)
    # features_ST = extraction_feature_ST(ecg, freq, features_dict, lead)
    # features_global = pd.concat([features_global, features_AF], axis=1)
    # features_global = pd.concat([features_global, features_AVB], axis=1)
    # features_global = pd.concat([features_global, features_PAC], axis=1)
    # features_global = pd.concat([features_global, features_PVC], axis=1)
    # features_global = pd.concat([features_global, features_ST], axis=1)
    MOR_features = pd.concat([interval_durations_feats, waves_characteristics_feats], axis=1)
    return MOR_features


def extraction_HRV_features(freq, dict_features, factor=1000, lead=6):
    """This function produces the HRV features for lead V1 concatenate them into a df"""
    # features_global = pd.DataFrame()
    wavedet_3D_dict_i = dict_features[lead]
    extracted_features = [(key, np.asarray(wavedet_3D_dict_i[key], dtype=np.int32)) for key in cts.interesting_keys]
    dict_features = {key: value for key, value in extracted_features}
    HRV_features = extraction_feature_AF(freq, dict_features, factor, lead=lead)
    return HRV_features


def extraction_HRV_features_temp(freq, dict_features, factor=1000, lead=6):
    """This function produces the HRV features for lead V1 concatenate them into a df"""
    HRV_features = extraction_feature_AF(freq, dict_features, factor, lead=lead)
    return HRV_features


def frequency_regularity(data, lead, freq, eng):
    """
    This function computes the frequence deviation, in order to discriminate AF and AFL
    """
    try:
        ecg_lead = data[lead]
        ecgs = np.array_split(ecg_lead, 6)
        dominant_frequencies = list()
        for i, ecg_window in enumerate(ecgs):
            qrs = wfdb.processing.gqrs_detect(fs=500, sig=np.asarray(ecg_window.tolist() * 10), adc_gain=1000,
                                              adc_zero=0)
            max_freq = eng.f_wave_detection(matlab.double(ecg_window.tolist()), matlab.double(qrs.tolist()),
                                            matlab.double([500]))
            dominant_frequencies.append(max_freq)
        std = np.std(dominant_frequencies)
    except:
        std = 0
        print('We could not manage to get the std for the lead', lead)
    return std


def preprocessing_check(ecg):
    """
    This function allows you to select ecgs based on 2 criteria:
    - A signal quality threshold (median signal quality accross the leads). TODO later"Automatic diagnosis of the 12-lead ECG using a dee
    - The preprocessing check from Ribeiro, Antônio H., et al. p neural network."
    """
    Keep = (np.max(np.abs(np.array(ecg[0]) + np.array(ecg[2]) - np.array(ecg[1]))) == 0) & (
        np.max(np.abs(np.array(ecg[3]) + np.array(ecg[4]) + np.array(ecg[5])) == 0))
    return Keep


def get_features_from_QRS(id, orig_fs, data=None, qrs_peaks=[]):
    physionet_matlab.initialize_runtime([])
    matlab_runtime = physionet_matlab.initialize()
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ecg = []
    fs = orig_fs
    for i_lead, lead in enumerate(list_lead):
        ecg_lead = data[:, lead]
        ecg.append(ecg_lead)
    for l in range(12):
        ecg_lead = ecg[l]
        ecg_filtered = bandpass_filter(ecg_lead, id, l, 0.67, 100, fs, 75, debug=False)
        # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 48, 52, fs, 3)
        # ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 58, 62, fs, 3)
        ecg.append(ecg_filtered)

    MOR_feat, HRV_feat, wavedet_3D_dict = extraction_feature_wavedet(filename='', list_lead=list_lead, ecg=ecg,
                                                                     qrs_peaks=qrs_peaks,
                                                                     freq=fs, runtime=matlab_runtime)
    DataFrame_sample_i_BBB = extraction_feature_wavedet_BBB(filename='',
                                                            ecg=ecg, list_lead=list_lead,
                                                            freq=fs, wavedet_3D_dict=wavedet_3D_dict,
                                                            runtime=matlab_runtime)
    DataFrame_sample_i = pd.concat([MOR_feat, DataFrame_sample_i_BBB, HRV_feat], axis=1)
    # Ang = VECGAng(ecg)
    # DataFrame_sample_i['VECGAng'] = Ang
    return DataFrame_sample_i


def process_id(start_index, step):
    # Get path
    path_to_hdf5 = cts.ECG_DATA_DIR / "preprocessed" / "traces.hdf5"
    # path_to_csv = cts.ECG_DATA_DIR / "annotations.csv"
    dataset_name = "signal"
    input_dir = DIR / "features"
    peak_dir = pathlib.PurePath("/ann/C2")
    # Get tracings
    f = h5py.File(path_to_hdf5, "r")
    x = f[dataset_name]
    traces_ids = np.array(f['id_exam'])
    orig_fs = 400
    
    # Get annotations
    # y_csv = pd.read_csv(path_to_csv)
    # y = get_data(y_csv, traces_ids)
    
    # Set saving path
    naming = "C2_features_4_" + str(start_index) + ".csv"
    filename = DIR / "add_features" / naming
    ids = traces_ids
    # exam = pd.read_csv(input_dir / "exam_C2.csv")
    # feat_C2 = pd.read_csv(DIR / "features" / "for_use" / "features_C2.csv")
    # ids = feat_C2["id_exam"]
    # files = os.listdir(peak_dir)
    # f = [int(x.split("_")[0]) for x in files]
    # Get HRV and morphological features
    # ids_tot = exam.sort_values(by="id_patient").id_exam
    # patient_extracted = exam.loc[exam.id_exam.isin(ids), "id_exam"]
    # ids_to_export = ids_tot[~np.isin(ids_tot, patient_extracted.values)]
    # ids_to_export = ids_to_export[np.isin(ids_to_export, f)]
    # ids_to_export = ids_to_export[:100000]
    # np.save(DIR / "ids_proceesed_by_119.npy", ids_to_export, allow_pickle=True)
    
    # Get HRV and morphological features
    # ids = exam.sort_values(by="id_patient").id_exam
    # ids = np.load(cts.REPO_DIR / "ids_for_119.npy", allow_pickle=True)
    # sub_ids = ids[70000:100000]
    for i, id_ in enumerate(ids[start_index:start_index+step]):
        data = get_trace(traces_ids, x, id_)
        peaks_filename = str(id_) + '_peaks_dict.npy'
        peaks = np.load(peak_dir / peaks_filename, allow_pickle=True)
        qrs_peaks = list(peaks.item()["epltd0"].values())
        DataFrame_sample_i = get_features_from_QRS(id=id_, data=data, qrs_peaks=qrs_peaks, orig_fs=400)
        DataFrame_sqi = get_features_from_sqi(peaks.item(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fs=orig_fs,
                                              agw=0.05, types=np.array(['epltd0', 'xqrs']))
        DataFrame_sample_i = pd.concat([DataFrame_sample_i, DataFrame_sqi], axis=1)
        # DataFrame_sample_i['Age'] = y_csv.loc[id_]['age']
        # DataFrame_sample_i['Gender'] = y_csv.loc[id_]['sex']
        DataFrame_sample_i['id_exam'] = id_
        if i == 0:
          DataFrame_sample_i.to_csv(filename, index=False, header=True)
        else:
          DataFrame_sample_i.to_csv(filename, index=False, header=False, mode='a')


if __name__ == "__main__":

    # set path
    path_to_hdf5 = cts.ECG_DATA_DIR / "preprocessed" / "traces.hdf5"
    path_to_csv = cts.ECG_DATA_DIR / "annotations.csv"
    input_dir = cts.REPO_DIR / "features"

    # Get tracings
    f = h5py.File(path_to_hdf5, "r")
    x = f[cts.dataset_name]
    traces_ids = np.array(f['id_exam'])
    orig_fs = 400

    # Get annotations
    y_csv = pd.read_csv(path_to_csv)
    y = get_data(y_csv, traces_ids)

    filename = cts.REPO_DIR / "example.csv"
    exam = pd.read_csv(cts.REPO_DIR / "features" / "exam_C2.csv")

    # Get HRV and morphological features
    ids = exam.sort_values(by="id_patient").id_exam
    for i, id_ in enumerate(ids[:2]):
        data = get_trace(traces_ids, x, id_)
        peaks_filename = str(id_) + '_peaks_dict.npy'
        peaks = np.load(cts.ANN_DIR / "TNMGDB" / "C2" / peaks_filename, allow_pickle=True)
        qrs_peaks = list(peaks.item()["epltd0"].values())
        DataFrame_sample_i = get_features_from_QRS(id=id_, data=data, qrs_peaks=qrs_peaks, orig_fs=400)
        DataFrame_sqi = get_features_from_sqi(peaks.item(), [0,1,2,3,4,5,6,7,8,9,10,11], fs=orig_fs, agw=0.07, types=np.array(['epltd0', 'xqrs']))
        DataFrame_sample_i = pd.concat([DataFrame_sample_i, DataFrame_sqi], axis=1)
        DataFrame_sample_i['Age'] = y_csv.loc[id_]['age']
        DataFrame_sample_i['Gender'] = y_csv.loc[id_]['sex']
        DataFrame_sqi['id_exam'] = id_
        if i == 0:
            DataFrame_sqi.to_csv(filename, index=False, header=True)
        else:
            DataFrame_sqi.to_csv(filename, index=False, header=False, mode='a')

    # multiprocessing
    # input_dir = cts.REPO_DIR / "features"
    # nb_files = 100000
    # nb_process = 10
    # step = nb_files // nb_process
    # lp = []
    # for start_index in range(0, nb_files, step):
    #     p = Process(target=process_id, args=[start_index, step])
    #     p.start()
    #     lp.append(p)
    # for p in lp:
    #     p.join()


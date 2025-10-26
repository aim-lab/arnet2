import pandas as pd
from Preprocessing_features import compute_mean, compute_median, compute_std, minimum, maximum
import numpy as np


def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg


def compute_statistics(freq, features_dict, lead, interval, factor=1000):
    begin_fiducial = features_dict[0]
    end_fiducial = features_dict[1]
    indexes_effectives = begin_fiducial * end_fiducial > 0
    begin_fiducial = begin_fiducial[indexes_effectives]
    end_fiducial = end_fiducial[indexes_effectives]
    units = (factor / freq)  # ms
    Dival = (end_fiducial - begin_fiducial) * units
    feat = pd.DataFrame({'Dmed_' + str(interval) + '_' + str(lead): [compute_median(Dival)],
                         # 'Dmean_' + str(interval) + '_' + str(lead): [compute_mean(Dival)],
                         'Dstd_' + str(interval) + '_' + str(lead): [compute_std(Dival)],
                         # 'Dmax_' + str(interval) + '_' + str(lead): [maximum(Dival)],
                         })
    return feat


def compute_RR_IR_statistics(freq, features_dict, lead, interval, factor=1000):
    R_points = features_dict[0]
    R_locations = R_points[R_points > 0]
    RR_index = R_locations[:-1] * R_locations[1:] > 0
    rr = np.asarray(comp_diff(R_locations))
    units = (factor / freq)  # ms
    Dival = (rr[RR_index]) * units
    IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
    feat = pd.DataFrame({'Dmed_' + str(interval) + '_' + str(lead): [compute_median(Dival)],
                         # 'Dmean_' + str(interval) + '_' + str(lead): [compute_mean(Dival)],
                         'Dstd_' + str(interval) + '_' + str(lead): [compute_std(Dival)],
                         # 'Dmax_' + str(interval) + '_' + str(lead): [maximum(Dival)],
                         'IRmedian_' + str(lead): [compute_median(IR)],
                         # 'IRmean_' + str(lead): [compute_mean(IR)],
                         'IRstd_' + str(lead): [compute_std(IR)],
                         # 'IRmax_' + str(lead): [maximum(IR)],
                         })
    return feat


def compute_PR2_statistics(freq, features_dict, lead, interval, factor=1000):
    P_locations = features_dict[0]
    R_points = features_dict[1]
    indexes_effectives = R_points * P_locations > 0
    RR_index = (R_points[:-1] * R_points[1:] > 0)
    R_points_effective, P_locations_effective = R_points[indexes_effectives], P_locations[indexes_effectives]
    units = (factor / freq)  # ms
    PR_interval = (R_points_effective - P_locations_effective) * units
    RR_interval = (np.asarray(comp_diff(R_points))) * units
    final_index = RR_index * indexes_effectives[1:] > 0
    PR_interval_final = (R_points[1:][final_index] - P_locations[1:][final_index]) * units
    if len(PR_interval_final) == len(RR_interval[final_index]) and len(RR_interval[final_index]) > 0:
        RAPR = compute_mean(PR_interval_final / RR_interval[final_index])
    elif len(RR_interval[final_index]) > 0:
        RAPR = compute_mean(PR_interval_final[1:] / RR_interval[final_index])
    else:
        RAPR = 0.0
    feat = pd.DataFrame({'MDPR_' + str(lead): [compute_median(PR_interval)],
                         # 'MAPR_' + str(lead): [compute_mean(PR_interval)],
                         'RAPR_' + str(lead): [RAPR]})
    return feat


def corrected_qt(DmedQT, DmedRR, lead, factor=1000):
    # Factor uses to convert between units used for morphological
    # calculation (i.e ms) to sec, which needed for QTc calculation
    # self check: (1) https://www.clinigate.com/clinicalc/corrected-qt-interval-qtc.php
    #             (2) https://www.thecalculator.co/health/QTc-Calculator-385.html
    if (DmedRR==0):
        QTc_b, QTc_frid, QTc_fra, QTc_hod = [0, 0, 0, 0]
    else:
        RR = DmedRR / factor
        QTc_b = DmedQT / np.sqrt(RR)  # ms
        QTc_frid = DmedQT / RR**(1/float(3))  # ms
        QTc_fra = DmedQT + (0.154 * (1 - RR) * factor)  # ms
        QTc_hod = DmedQT + (0.00175 * (60 / RR - 60) * factor) # ms
    feat = pd.DataFrame({'DmedQTc_b_' + str(lead): [QTc_b],
                         'DmedQTc_frid_' + str(lead): [QTc_frid],
                         'DmedQTc_fra_' + str(lead): [QTc_fra],
                         'DmedQTc_hod_' + str(lead): [QTc_hod]})
    return feat


def extract_intervals_duration(freq, features_dict, lead, factor=1000):
    feat_df = pd.DataFrame()
    intervals = dict(Pwave=[features_dict['Pon'], features_dict['Poff']],
                     PR=[features_dict['Pon'], features_dict['QRSon']],
                     PRseg=[features_dict['Poff'], features_dict['QRSon']],
                     PR2=[features_dict['P'], features_dict['R']],
                     RR=[features_dict['R']],
                     QRS=[features_dict['QRSon'], features_dict['QRSoff']],
                     QT=[features_dict['QRSon'], features_dict['Toff']],
                     Twave=[features_dict['Ton'], features_dict['Toff']])
    Classics = ['Pwave', 'PR', 'PRseg', 'QRS', 'QT', 'Twave']
    for ival in Classics:
        feats = compute_statistics(freq, intervals[ival], lead, ival, factor)
        feat_df = pd.concat([feat_df, feats], axis=1)

    featsRR = compute_RR_IR_statistics(freq, intervals["RR"], lead, 'RR', factor)
    featsPR2 = compute_PR2_statistics(freq, intervals["PR2"], lead, 'PR2', factor)
    QTc = corrected_qt(feat_df.loc[:, feat_df.columns.str.startswith('Dmed_QT')].values.flatten()[0],
                       featsRR.loc[:, featsRR.columns.str.startswith('Dmed_RR')].values.flatten()[0], lead, factor)
    feat_df = pd.concat([feat_df, featsRR, featsPR2, QTc], axis=1)
    return feat_df

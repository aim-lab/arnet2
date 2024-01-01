ecg = load("matlab_matrix.mat")
[out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12] = python_wrap_wavedet_3D([], ecg.ecg, [], [1,2,3,4,5,6,7,8,9,10,11,12], [])

% [template] = FECGSYN_tgen(ecg,qrs_select,500)
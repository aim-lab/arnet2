clear all;
ecg = load('/home/shanybiton/AIMLabProjects/afib-prediction/preprocessing/matlab_matrix.mat');
list_lead = [0,1,2,3,4,5,6,7,8,9,10,11];
[l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,l11,l12]=wrapper_for_wavedat3(ecg, list_lead, []);
s=struct('a',l1,'b',l2, 'c', l3, 'd', l4, 'e', l5', 'f', l6, 'g', l7, 'h', l8, 'i', l9, 'j', l10, 'k', l11, 'l', l12)
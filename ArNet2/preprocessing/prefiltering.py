import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt


lead=[0, 2, 3, 4, 6, 11]
#lead=[12, 14, 15, 16, 18, 23]
k=0
t = np.linspace(0, len(signal[0]) / freq, len(signal[0]))
plt.figure(figsize=(20,10))
for i in range(1, 7):
    l=lead[i-1]
    l_f = l+12
    #l_f = l
    sig = signal[l_f]
    plt.subplot(2, 3, i)
    plt.plot(t, sig, color="grey")
    plt.scatter(wavedet_3D_dict[l]["P"]/freq, sig[wavedet_3D_dict[l]["P"]], color="C8", marker="x", label="P")
    plt.scatter(wavedet_3D_dict[l]["Pon"]/freq, sig[wavedet_3D_dict[l]["Pon"]], color="C0", marker="x", label="Pon")
    plt.scatter(wavedet_3D_dict[l]["Poff"]/freq, sig[wavedet_3D_dict[l]["Poff"]], color="C1", marker="x", label="Poff")
    plt.scatter(wavedet_3D_dict[l]["Q"]/freq, sig[wavedet_3D_dict[l]["Q"]], color="C2", marker="x",label="Q")
    plt.scatter(wavedet_3D_dict[l]["R"]/freq, sig[wavedet_3D_dict[l]["R"]], color="C3", marker="x", label="R")
    plt.scatter(wavedet_3D_dict[l]["S"]/freq, sig[wavedet_3D_dict[l]["S"]], color="C4", marker="x", label="S")
    plt.scatter(wavedet_3D_dict[l]["Ton"]/freq, sig[wavedet_3D_dict[l]["Ton"]], color="C5", marker="x", label="Ton")
    plt.scatter(wavedet_3D_dict[l]["Toff"]/freq, sig[wavedet_3D_dict[l]["Toff"]], color="C6", marker="x", label="Toff")
    plt.scatter(wavedet_3D_dict[l]["T"]/freq, sig[wavedet_3D_dict[l]["T"]], color="C9", marker="x", label="T")
    plt.xlim(4,6)
    plt.title("Lead " + lead_to_func_ending_dict[l])
    # plt.legend()
plt.legend()
plt.tight_layout()
filename = "fiducials_id_" + str(id_) + "_filtered.png"
plt.savefig(REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename, dpi=400)

lead=[6, 11]
#lead=[12, 14, 15, 16, 18, 23]
k=0
t = np.linspace(0, len(signal[0]) / freq, len(signal[0]))
plt.figure(figsize=(20,10))
for i in range(1, 3):
    l=lead[i-1]
    l_f = l+12
    #l_f = l
    sig = signal[l_f]
    plt.subplot(2, 1, i)
    plt.plot(t, sig, color="grey")
    plt.scatter(wavedet_3D_dict[l]["P"]/freq, sig[wavedet_3D_dict[l]["P"]], color="C8", marker="x", label="P")
    plt.scatter(wavedet_3D_dict[l]["Pon"]/freq, sig[wavedet_3D_dict[l]["Pon"]], color="C0", marker="x", label="Pon")
    plt.scatter(wavedet_3D_dict[l]["Poff"]/freq, sig[wavedet_3D_dict[l]["Poff"]], color="C1", marker="x", label="Poff")
    plt.scatter(wavedet_3D_dict[l]["Q"]/freq, sig[wavedet_3D_dict[l]["Q"]], color="C2", marker="x",label="Q")
    plt.scatter(wavedet_3D_dict[l]["R"]/freq, sig[wavedet_3D_dict[l]["R"]], color="C3", marker="x", label="R")
    plt.scatter(wavedet_3D_dict[l]["S"]/freq, sig[wavedet_3D_dict[l]["S"]], color="C4", marker="x", label="S")
    plt.scatter(wavedet_3D_dict[l]["Ton"]/freq, sig[wavedet_3D_dict[l]["Ton"]], color="C5", marker="x", label="Ton")
    plt.scatter(wavedet_3D_dict[l]["Toff"]/freq, sig[wavedet_3D_dict[l]["Toff"]], color="C6", marker="x", label="Toff")
    plt.scatter(wavedet_3D_dict[l]["T"]/freq, sig[wavedet_3D_dict[l]["T"]], color="C9", marker="x", label="T")
    plt.title("Lead " + lead_to_func_ending_dict[l])
    # plt.legend()
plt.legend()
plt.tight_layout()
filename = "fiducials_id_" + str(id_) + "_full.png"
plt.savefig(REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename, dpi=400)


def BPFilter(data,lowcut,highcut, signal_freq):
    L = data.size
    nyquist_freq = 0.5 * signal_freq

    Wp = 0.67 / nyquist_freq
    Ws = 0.6 / nyquist_freq
    Rp = 1
    Rs = 50
    n, Ws = cheb2ord(Wp, Ws, Rp, Rs)
    z, p, k = cheby2(n, Rs, Ws, 'high', output='zpk')
    sos = zpk2sos(z, p, k)
    y = sosfiltfilt(sos, data.astype(float))



    fc = 0.67 / nyquist_freq
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)

    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = np.blackman(N)
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    # reverse function
    sinc_func = -sinc_func
    sinc_func[int((N - 1) / 2)] += 1

    s = list(data['10 Min Std Dev'])
    new_signal = np.convolve(data, sinc_func, "same")

    n, Wn = scipy.signal.buttord([low, high], [low * 0.8, high / 0.8], 1, 10)
    sos = scipy.signal.butter(n, Wn, 'band', output='sos')

    k = 0.7 # cut - off
    alpha = (1 - k * cos(2 * pi * fc) - sqrt(2 * k * (1 - cos(2 * pi * fc)) - k ^ 2 * sin(2 * pi * fc) ^ 2)) / (1 - k)
    y = zeros(size(x))

    return y




id_AF = 3 # sqi=0.969697
id_AF = 2042341 # sqi=0.693878
id_AF = 2012017 # sqi=0.8

id_NA = 3627670 # sqi=0.774194
id_NA = 1940708 # sqi=0.842105
id_NA = 1237278 # sqi=1.0


lead=0
plt.plot(ecg[lead])
plt.scatter(ret_val[lead]["Pon"], ecg[lead][ret_val[lead]["Pon"]], color="r", marker="x")
plt.scatter(ret_val[lead]["Poff"], ecg[lead][ret_val[lead]["Poff"]], color="r", marker="x")
plt.scatter(ret_val[lead]["Q"], ecg[lead][ret_val[lead]["Q"]], color="g", marker="x")
plt.scatter(ret_val[lead]["R"], ecg[lead][ret_val[lead]["R"]], color="y", marker="x")
plt.scatter(ret_val[lead]["S"], ecg[lead][ret_val[lead]["S"]], color="g", marker="x")
plt.scatter(ret_val[lead]["Ton"], ecg[lead][ret_val[lead]["Ton"]], color="g", marker="x")
plt.scatter(ret_val[lead]["Toff"], ecg[lead][ret_val[lead]["Toff"]], color="g", marker="x")
plt.show()




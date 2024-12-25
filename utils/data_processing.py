try:
    from utils.base_packages import *
except ModuleNotFoundError:
    from base_packages import *
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, sosfreqz, sosfiltfilt
import mne

def hyperparamaters_comb(dict_vals):
    """
    This function returns all the possible combinations of hyperparameters.
    Returns a list of dictionaries. Each dict contains the name and the value of the hyperparameter
    for the given combination.
    :param dict_vals: Dictionnary containing as keys the different hyperparameter names and as values the different values they take.
    :returns dict_comb: The output list.
    """
    hyper_names = list(dict_vals.keys())
    n_hyper = len(hyper_names)
    combinations = list(itertools.product(*list(dict_vals.values())))
    dict_comb = [{hyper_names[i]: comb[i] for i in range(n_hyper)} for comb in combinations]
    return dict_comb


def is_integer(float_num):
    """
     This function returns if the given number is an integer
    :param float_num: the input number
    :returns bool: boolean, True if the input is an integer, False otherwise.
    """
    return math.ceil(float_num) == float_num


def cumsum_reset(np_arr):
    """
    This function performs the 'cumsum' operation on an array, resetting itself each time a zero is
    found in the array.
    :param np_arr: Numpy Array containing the input.
    :returns res: Numpy Array containing the cumulative sum.
    """
    return np.array(pd.DataFrame(np_arr).astype(int).apply(lambda x: x.groupby((~x.astype(bool)).cumsum()).cumsum())).reshape(-1)


def resample_by_interpolation(signal, input_fs, output_fs):
    """
    This function interpolates a signal from an original to a desired sampling frequency.
    :param signal: The input signal.
    :param input_fs: The original sampling frequency.
    :param output_fs: The desired sampling frequency.
    :returns resampled_signal: The signal resampled to output_fs.
    """
    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def normalize_data(data_train, data_test):
    """
    This functions normalizes train and test sets (Z-normalization) based on the parameters (mean and std.)
    derived from the training set.
    :param data_train: Numpy array containing the training data (dimensions: (n_samples, n_features)).
    :param data_test: Numpy array containing the test data (dimensions: (n_samples, n_features)).
    """
    mean_train = data_train.mean(axis=0)
    std_train = data_train.std(axis=0)
    data_train = (data_train - mean_train) / std_train
    data_train[np.isnan(data_train)] = 0.0
    if len(data_test) > 0:
        data_test = (data_test - mean_train) / std_train
        data_test[np.isnan(data_test)] = 0.0
    return data_train, data_test, mean_train, std_train


def fillna(X):
    """
    Fills the input array colum-wise using the average value over the column.
    :param X: the input array.
    :returns X_new: the imputed array.
    :returns means: the means over the different columns.
    """
    if len(X) > 0:
        X_new = copy.deepcopy(X)
        means = np.nanmean(X_new, axis=0)
        for i in range(X_new.shape[1]):
            if np.any(np.isnan(X_new[:, i])):
                if means is None:
                    curr_mean = np.nanmean(X_new[:, i])
                    X_new[np.isnan(X_new[:, i]), i] = curr_mean
                    means[i] = curr_mean
                else:
                    X_new[np.isnan(X_new[:, i]), i] = means[i]
        return X_new, means
    else:
        return X, np.array([])


def check_stratification(X_train, X_test, y_train, y_test, plot=False, feats_name=None, n_points=50):
    """
    This function verifies the stratification between train and test data
    for each one of the different features (verification of the distributions) and for the labels
    (checks if the proportions for each class is similar)
    :param X_train: The training dataset.
    :param X_test: The test dataset.
    :param y_train: The training labels.
    :param y_test: The test labels.
    :param plot: (optional). Boolean value to indicate if the features histograms need to be plotted.
    :param feats_name : (optional). The name of the different features ordered in a list.
    :param n_points: (optional). Number of points required for the histograms binning.
    """
    if len(y_train) == 0 or len(y_test) == 0:
        return
    else:
        for index in range(X_train.shape[1]):

            data_train, data_test = X_train[:, index], X_test[:, index]
            max = np.max([data_train.max(), data_test.max()])
            min = np.min([data_train.min(), data_test.min()])
            if plot:
                fig, axes = graph.create_figure()
                labs = np.unique(y_train)
                for lab in labs:
                    weights_train = np.ones_like(data_train[y_train == lab]) / len(data_train[y_train == lab])
                    weights_test = np.ones_like(data_test[y_test == lab]) / len(data_test[y_test == lab])
                    axes[0][0].hist(data_train[y_train == lab], np.linspace(min, max, n_points), color=cts.COLORS[lab],
                                    label='Label: ' + str(lab), weights=weights_train, rwidth=1 - lab / (len(labs)))
                    axes[0][0].hist(data_test[y_test == lab], np.linspace(min, max, n_points), color=cts.COLORS[lab + 2],
                                    label='Label: ' + str(lab), weights=weights_test, rwidth=1 - lab / (len(labs)))
                if feats_name is not None:
                    graph.complete_figure(fig, axes, suptitle=feats_name[index])
                else:
                    graph.complete_figure(fig, axes)


def bandpass_filter(signal, id, lead, lowcut, highcut, signal_freq, filter_order,  notch_freq=50, debug=False):
    """
    Applies a Butterworth filter and a notch filter to a given signal. The coefficients are computed automatically.
    :param signal: The input signal.
    :param id: The id of the signal.
    :param lead: Lead number.
    :param lowcut: Low butterworth filter cutoff in Hz.
    :param highcut: High butterworth filter cutoff in Hz.
    :param signal_freq: The frequency of the signal in Hz.
    :param filter_order: Set the order of the butterworth filter.
    :param notch_freq: The frequencies for which to apply notch filter in z.
    :param debug: If true, plot the filtered signal and the spectrum.
    :returns y: Filtered signal.
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    sos = butter(filter_order, [low, high], btype="band", output='sos', analog=False)
    y = sosfiltfilt(sos, signal)
    y = mne.filter.notch_filter(y.astype(np.float), signal_freq, freqs=notch_freq, verbose=debug)
    if debug:
        filename_freq = "exam_" + str(id) + "_lead_" + str(lead) + ".png"
        filename_spect = "exam_" + str(id) + "_lead_" + str(lead) + "_spect.png"

        # get_freq_plot(data, y, sos, filter_order, signal_freq, filename_freq)
        get_spect_plot(signal, y, signal_freq, filename_spect, dpi=400)
    return y


def get_freq_plot(coefs, order, fs, filename):
    """
    Plots The frequency response given coefficients, order of a filter and frequency.
    :param coefs: Filter coefficients.
    :param order: Order of the filter.
    :param fs: Signal frequency.
    :param filename: Name for saving the plot.
    """
    w, h = sosfreqz(coefs, worN=2000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), '#3465a4')
    plt.title("Bandpass Filter Frequency Response, order=" + str(order))
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), '#3465a4')
    plt.xlim(0, 3)
    plt.title("Bandpass Filter Frequency Response, order=" + str(order))
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(cts.REPO_DIR / "AIMLab_report" / "MOR" / "Filtering" / filename, dpi=400)
    plt.close()
    return


def get_spect_plot(y_orig, y_filt, fs, filename, dpi=400):
    """
    Plots The PSD using Welch method and the signal (filtered on top the original).
    :param y_orig: original signal.
    :param y_filt: Filtered signal.
    :param fs: Signal frequency.
    :param filename: Name of the plot file.
    :param dpi: Plot resolution.
    """
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

    # Get the PSD of the signals using welch method
    fx, Pxx = scipy.signal.welch(y_orig, fs, nperseg=len(y_orig))
    fy, Pyy = scipy.signal.welch(y_filt, fs, nperseg=len(y_filt))
    ps_orig_log = 10 * np.log10(Pxx)
    ps_filt_log = 10 * np.log10(Pyy)
    fig = plt.figure(dpi=400)
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

from tensorflow.keras.utils import Sequence
import numpy as np


class DataGenerator(Sequence):
    """Generates temporal data based on the original features derived on the windows themselves.
    The generator will return for each time it is called an array of size (batch_size, history * n_features).
    """

    def __init__(self, orig_data, orig_labels=None, weights=None, to_fit=True, batch_size=1024, history=5, shuffle=True,
                 type='Conc', add_data=None, mask=None):
        """Initialization
        :param orig_data: The original features derived from the RR intervals.
        :param orig_labels: The original labels
        :param orig_prec_windows: For each window, the number of previous windows available.
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.orig_data = orig_data[:,
                         :-1]  # Not considering the last column - should be the preceeding available windows.
        if add_data is not None:
            self.orig_data = np.concatenate((self.orig_data, add_data), axis=1)
        self.orig_labels = orig_labels
        self.orig_prec_windows = orig_data[:, -1].astype(int)  # The last column is the preceeding available windows
        if mask is not None:
            self.orig_data = self.orig_data[mask, :]
            if self.orig_labels is not None:
                self.orig_labels = self.orig_labels[mask]
            self.orig_prec_windows = self.orig_prec_windows[mask]
        self.weights = weights
        self.history = history
        self.windows_to_keep = self.orig_prec_windows >= self.history
        self.n_samples = len(self.orig_data)  # np.sum(self.windows_to_keep)
        self.indexes = np.arange(self.n_samples)  # np.where(self.windows_to_keep)[0]
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type = type
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(
            np.ceil(self.n_samples / self.batch_size))  # Ceil to handle the last batch, smaller than the other ones

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # if self.to_fit:
        X = [np.concatenate((np.zeros(
            (self.history - min(self.history, self.orig_prec_windows[i], i+1), self.orig_data.shape[1])),
                             self.orig_data[(i - min(self.history, self.orig_prec_windows[i], i+1) + 1):(i + 1)]), axis=0)
             for i in indexes]
        # else:
        #     X = [self.orig_data[(i - self.orig_prec_windows[i] + 1):(i + 1)] for i in indexes]

        if self.type == 'Conc':
            X = np.concatenate(tuple([x.reshape(1, -1) for x in X]), axis=0)  # Row Stack
        elif self.type == 'LSTM':
            X = np.concatenate(tuple([x.reshape(1, self.history, -1) for x in X]),
                               axis=0)  # Row Stack for LSTM: (Samples, Timesteps, Features)
        elif self.type == 'ResCRNN':
            X = np.concatenate(tuple([x.reshape(1, -1) for x in X]),
                               axis=0)  # Row Stack for LSTM: (Samples, Timesteps * Features)
            X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')
        if self.to_fit:
            y = self.orig_labels[indexes]
            if self.weights is not None:
                weights = self.weights[indexes]  # Weight = 2 for AF, 1 for Non-AF
                return X, y, weights
            else:
                return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

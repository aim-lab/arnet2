from tensorflow.keras.utils import Sequence
import pandas as pd
import tensorflow as tf
import numpy as np


def make_synthetic(N=500, F=60, seed=42):
    rng = np.random.default_rng(seed)

    # Per-window features: [N, F], uniform [0,1]
    X = rng.random((N, F), dtype=np.float32)

    # Preceding windows available: [N, 1], integer; pw[i] in [0, i]
    pw = np.array([rng.integers(0, i + 1) for i in range(N)], dtype=np.int32).reshape(-1, 1)

    # Old generator expects orig_data with LAST COLUMN = prec_windows
    orig_data = np.concatenate([X, pw.astype(np.float32)], axis=1)  # [N, F+1]  (pw kept as float column like before)

    return orig_data


def check_parity(orig_data, history, dg_type='LSTM', add_data=None, mask=None, batch_size=8):
    old = DataGenerator(orig_data, to_fit=False, batch_size=batch_size, history=history,
                 shuffle=False, type=dg_type, add_data=add_data, mask=mask)
    new = TFDataGenerator(orig_data, to_fit=False, batch_size=batch_size, history=history,
                          shuffle=False, dg_type=dg_type, add_data=add_data, mask=mask)

    # take first few batches
    it_old = iter(range(len(old)))
    it_new = iter(new.as_dataset())

    for _ in range(3):
        try:
            i = next(it_old)
            X_old = old[i]                     # numpy array
            X_new = next(it_new).numpy()       # tf -> numpy
        except StopIteration:
            break

        assert X_old.shape == X_new.shape, f"shape mismatch {X_old.shape} vs {X_new.shape}"
        max_abs = np.max(np.abs(X_old - X_new))
        print(f"batch {_}: shape={X_old.shape}, max_abs_diff={max_abs:.3e}")


class TFDataGenerator:
    """
    A tf.data-based replacement for DataGenerator.
    Replicates the packing rules:
      k = min(history, prec_windows[i], i+1)
      seq = zeros(history-k, F) + features[i-k+1 : i+1]
    and returns shapes depending on `dg_type`:
      - 'LSTM'    -> [history, F]
      - 'Conc'    -> [history*F]
      - 'ResCRNN' -> [history*F, 1]

    Inputs:
      orig_data:  [N, F+1] float (last column = prec_windows)
      orig_labels: [N] (optional, for training)
      weights:     [N] (optional, sample weights)
      history:     int
      batch_size:  int
      shuffle:     bool
      dg_type:     'LSTM' | 'Conc' | 'ResCRNN'
      add_data:    [N, F_add] or None (concatenated to per-window features)
      mask:        boolean [N] or None (row filter applied before packing)
    """
    def __init__(self,
                 orig_data,
                 orig_labels=None,
                 weights=None,
                 to_fit=True,
                 batch_size=1024,
                 history=5,
                 shuffle=True,
                 dg_type='LSTM',
                 add_data=None,
                 mask=None):
        self.to_fit = to_fit
        self.batch_size = int(batch_size)
        self.history = int(history)
        self.shuffle = bool(shuffle)
        self.dg_type = dg_type

        # Convert to tensors
        data = tf.convert_to_tensor(orig_data)
        feats = tf.cast(data[:, :-1], tf.float32)     # per-window features (exclude prec_windows col)
        precw = tf.cast(data[:, -1],  tf.int32)       # last col = prec_windows

        if add_data is not None:
            feats = tf.concat([feats, tf.cast(add_data, tf.float32)], axis=1)

        if mask is not None:
            mask_t = tf.convert_to_tensor(mask)
            feats = tf.boolean_mask(feats, mask_t)
            precw = tf.boolean_mask(precw, mask_t)
            if orig_labels is not None:
                orig_labels = tf.boolean_mask(tf.convert_to_tensor(orig_labels), mask_t)
            if weights is not None:
                weights = tf.boolean_mask(tf.convert_to_tensor(weights), mask_t)

        self.N = tf.shape(feats)[0]
        self.F = tf.shape(feats)[1]

        # Build dataset of indices so we can map the packing function
        # Make range() happy with int64, then cast back to int32 for consistent math
        ds = tf.data.Dataset.range(tf.cast(self.N, tf.int64)).map(
            lambda i: tf.cast(i, tf.int32), num_parallel_calls=tf.data.AUTOTUNE
        )

        if self.shuffle:
            buffer_size = tf.cast(tf.minimum(self.N, 100000), tf.int64)
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        # map index -> sequence (and label/weight if provided)
        H = tf.cast(self.history, tf.int32)
        F = self.F
        precw = precw  # closure capture
        feats = feats  # closure capture

        def make_seq(i):
            # k = min(history, prec_windows[i], i+1)
            k = tf.minimum(H, tf.minimum(precw[i], i + 1))
            start = i - k + 1
            tail = feats[start:i+1]                           # [k, F]
            pad = tf.zeros([H - k, F], feats.dtype)           # left pad
            seq = tf.concat([pad, tail], axis=0)              # [H, F]

            if self.dg_type == 'LSTM':
                x = seq                                      # [H, F]
            elif self.dg_type == 'Conc':
                x = tf.reshape(seq, [H * F])                 # [H*F]
            elif self.dg_type == 'ResCRNN':
                x = tf.reshape(seq, [H * F, 1])              # [H*F, 1]
            else:
                raise ValueError(f"Unsupported dg_type: {self.dg_type}")

            return x

        ds_x = ds.map(make_seq, num_parallel_calls=tf.data.AUTOTUNE)

        if self.to_fit and orig_labels is not None:
            ds_y = tf.data.Dataset.from_tensor_slices(tf.cast(tf.convert_to_tensor(orig_labels), tf.float32))
            if weights is not None:
                ds_w = tf.data.Dataset.from_tensor_slices(tf.cast(tf.convert_to_tensor(weights), tf.float32))
                # Keras expects: dataset yields (X, y, sample_weight) when using fit() with dataset
                ds = tf.data.Dataset.zip((ds_x, ds_y, ds_w))
            else:
                ds = tf.data.Dataset.zip((ds_x, ds_y))
        else:
            ds = ds_x

        self.dataset = ds.batch(self.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    def as_dataset(self):
        return self.dataset


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
            X = np.stack([x.reshape(self.history, -1) for x in X], axis=0).astype(
                'float32')  # Row Stack for LSTM: (Samples, Timesteps, Features)
        elif self.type == 'ResCRNN':
            X = np.stack([x.reshape(-1, 1) for x in X], axis=0).astype(
                'float32')  # Row Stack for LSTM: (Samples, Timesteps * Features)
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

if __name__ == '__main__':
    X = make_synthetic()

    check_parity(X, history=10, dg_type='LSTM')

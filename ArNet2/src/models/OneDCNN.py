import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool1D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
import copy


class OneDCNN:

    def __init__(self, window_size=60, n_filters_start=64, n_hidden_start=512, batch_size=1024,
                 activation='relu', len_sub_window=10, n_jobs=1, dropout=0.2, af_weight=1):
        tf.compat.v1.keras.backend.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.n_features = 1
        self.patience = 50  # High to allow running for the whole 30 epochs.
        self.len_sub_window = len_sub_window
        # self.batch_size = batch_size
        self.model = Sequential()
        self.window_size = window_size
        self.n_filters_start = n_filters_start
        self.n_hidden_start = n_hidden_start
        self.activation = activation
        self.history = None
        self.dropout = dropout
        self.af_weight = af_weight
        # Update batch size value
        self.batch_size = batch_size

        self.model.add(Conv1D(self.n_filters_start, self.len_sub_window, input_shape=(window_size, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(2 * self.n_filters_start, self.len_sub_window))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPool1D())
        self.model.add(Conv1D(4 * self.n_filters_start, self.len_sub_window))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())
        self.model.add(Dense(self.n_hidden_start, activation='relu'))
        self.model.add(Dense(int(self.n_hidden_start / 2), activation='relu'))
        self.model.add(Dense(int(self.n_hidden_start / 4), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.init_weights = copy.deepcopy(self.model.get_weights())

        self.loss_train = None
        self.loss_valid = None
        self.n_epochs_train = 0

    def fit(self, X, y, validation_data=None, warm_start=False, n_epochs=30):

        if not warm_start:
            self.model.set_weights(self.init_weights)
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
        X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')
        sample_weight = np.array([self.af_weight if lab == True else 1 for lab in y])
        if validation_data is not None:
            X_valid, y_valid = validation_data
            sample_weight_valid = np.array([self.af_weight if lab == True else 1 for lab in y_valid])
            X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
            self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
                                          validation_data=(X_valid, y_valid, sample_weight_valid), epochs=n_epochs,  # Rk : we use the training sample_weight for validation
                                          callbacks=[EarlyStopping(patience=self.patience, min_delta=1e-3,
                                                                   restore_best_weights=True)])
            self.loss_train = self.history.history['loss']
            self.loss_valid = self.history.history['val_loss']
            self.n_epochs_train = len(self.loss_train)

        else:
            self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
                                          epochs=n_epochs)
            self.loss_train = self.history.history['loss']
            self.n_epochs_train = len(self.loss_train)

    def predict(self, X, th=0.5):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X, batch_size=self.batch_size).reshape(-1) > th

    def predict_layer(self, X, layer_name='dense_1'):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(X, batch_size=self.batch_size)
        return intermediate_output

    def predict_proba(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')
        res = self.model.predict(X, batch_size=self.batch_size)
        res = np.concatenate((1 - res, res), axis=1)  # For sklearn compatibility
        return res

    def get_state_dict(self):
        res = {
            'weights': self.model.get_weights(),
            'window_size': self.window_size,
            'n_filters': self.n_filters_start,
            'n_hidden_start': self.n_hidden_start,
            'n_features': self.n_features,
            'batch_size': self.batch_size,
            'activation': self.activation,
            'len_sub_window': self.len_sub_window,
            'init_weights': self.init_weights,
            'dropout': self.dropout,
            "af_weight": self.af_weight,
            'loss_train': self.loss_train,
            'loss_valid': self.loss_valid,
            'n_epochs_train': self.n_epochs_train
        }
        return res

    def set_state_dict(self, state):
        for key, val in state.items():
            if key == 'weights':
                self.model.set_weights(val)
            else:
                self.__dict__[key] = val

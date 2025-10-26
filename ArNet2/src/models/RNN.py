import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, Bidirectional

from ArNet2.src.models.datagen import DataGenerator


class RNN:

    def __init__(self, window_size=60, time_history=10, activation='relu', n_units=8, n_dense=1, dropout_fcn=0.5, learning_rate=0.001, af_weight=3,
                 batch_size=1024):
        tf.compat.v1.keras.backend.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.n_features = 1
        self.patience = 20  # High to allow running for at least 20 epochs.
        self.window_size = window_size
        self.time_history = time_history
        self.activation = activation
        self.n_units = n_units
        self.n_dense = n_dense
        self.dropout_fcn = dropout_fcn
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.batch_size = batch_size
        self.af_weight = af_weight

        self.params = {"num_categories": 1, "init": "he_normal", "num_skip": 2,
                       "increase_channels_at": 2, "compile": True}
        self.params.update({"input_shape": (self.time_history, 30),  # could be changed if I manage to create window_size dependent LatEntSMOTE
                            "conv_activation": activation,
                            "n_units": n_units, "n_dense": n_dense,
                            "fcn_dropout": dropout_fcn, "conv_learning_rate": learning_rate,
                            "conv_window_size": window_size, "conv_batch_size": batch_size})

        self.model = self.build_network(**self.params)

        self.loss_train = None
        self.loss_valid = None
        self.n_epochs_train = 0

    def add_rnn_layer(self, layer, **params):
        layer = Bidirectional(GRU(params["n_units"]))(layer)
        return layer

    def add_output_layer(self, layer, **params):
        for i in range(params["n_dense"]):
            layer = (Dense(params["n_units"]//2**i, activation='relu'))(layer)  # divide n hidden layers by 2 at each block
            layer =  (Dropout(params["fcn_dropout"]))(layer)
        layer = (Dense(params["num_categories"], activation='sigmoid'))(layer)
        return layer

    def add_compile(self, model, **params):
        optimizer = tf.keras.optimizers.Adam(
            lr=params["conv_learning_rate"],
            clipnorm=params.get("clipnorm", 1))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', 'AUC'])

    def build_network(self, **params):
        inputs = Input(shape=params['input_shape'],
                       dtype='float32',
                       name='inputs')

        layer = self.add_rnn_layer(inputs, **params)
        output = self.add_output_layer(layer, **params)
        model = Model(inputs=[inputs], outputs=[output])
        if params.get("compile", True):
            self.add_compile(model, **params)
        return model

    def temporal_concat(self, X, y, history=10):
        """ Transform each batch of consecutive windows into equal-length batches (can probably be drastically cleaned).
        """
        # Get indexes of beginnings of consecutive windows
        begs = np.where(X[:, -3] == 1)[0]
        begs = np.append(begs, len(X))

        X_batches = []
        y_batches = []
        is_sample = []
        sample_weight_batches = []
        # Transform each batch of consecutive windows into equal-length batches, possibly padding them
        for beg, end in zip(begs[:-1], begs[1:]):
            # If batch length is inferior to time_history, pad it with zeros
            if end - beg <= history:
                X_batches.append(
                    np.concatenate((X[beg: end][:, :-3], np.zeros((history - (end - beg), X[:, :-3].shape[1]))),
                                   axis=0))
                y_batches.append(np.concatenate((y[beg: end], np.zeros(history - (end - beg))), axis=0))
                sample_weight_batches.append(np.concatenate(([self.af_weight if lab == True else 1 for lab in y[beg: end]], np.zeros(history - (end - beg))), axis=0))
                is_sample.append(np.concatenate((np.ones(end-beg), np.zeros(history - (end - beg))), axis=0))
            # If batch length is superior to time_history,
            else:
                # cut it into consecutive time_history batches,
                for i in range((end - beg) // history):
                    X_batches.append(X[beg + history * i: beg + history * (i + 1)][:, :-3])
                    y_batches.append(y[beg + history * i: beg + history * (i + 1)])
                    sample_weight_batches.append([self.af_weight if lab == True else 1 for lab in y[beg + history * i: beg + history * (i + 1)]])
                    is_sample.append(np.ones(history))

                # and possibly pad the last one
                if (end - beg) // history != (end - beg) / history:
                    X_batches.append(np.concatenate((X[beg + history * ((end - beg) // history): end][:, :-3], np.zeros(
                        (history - (end - (beg + history * ((end - beg) // history))), X[:, :-3].shape[1]))), axis=0))
                    y_batches.append(np.concatenate((y[beg + history * ((end - beg) // history): end], np.zeros(
                        history - (end - (beg + history * ((end - beg) // history))))), axis=0))
                    sample_weight_batches.append(np.concatenate(([self.af_weight if lab == True else 1 for lab in y[beg + history * ((end - beg) // history): end]], np.zeros(
                        history - (end - (beg + history * ((end - beg) // history))))), axis=0))
                    is_sample.append(
                        np.concatenate((np.ones(end - (beg + history * ((end - beg) // history))), np.zeros(
                        history - (end - (beg + history * ((end - beg) // history))))),
                                       axis=0))

        X_batches = np.array(X_batches).astype('float32')
        y_batches = np.array(y_batches).astype('int')
        sample_weight_batches = np.array(sample_weight_batches).astype('int')
        is_sample = np.array(is_sample).astype('int')
        return X_batches, y_batches, sample_weight_batches, is_sample

    def fit(self, X, y, validation_data=None, n_epochs=30):
        sample_weight = np.array([self.af_weight if lab == True else 1 for lab in y])  # TODO : add sample_weight

        # training_gen = TimeConcatGen(X, y, batch_size=512, shuffle=True)
        # self.history = self.model.fit_generator(training_gen, epochs=5, verbose=1)

        # X_batches, y_batches, sample_weight_batches, _ = self.temporal_concat(X, y, history=self.time_history)
        # self.history = self.model.fit(X_batches, y_batches, sample_weight=None, batch_size=self.batch_size, epochs=n_epochs,
        #                callbacks=[
        #                    # ReduceLROnPlateau(factor=0.1, patience=2, min_lr=self.params["conv_learning_rate"] * 0.001),
        #                    # EarlyStopping(monitor='val_auc', mode='max', patience=self.patience, min_delta=1e-3, restore_best_weights=True, verbose=1)
        #                ])

        training_generator = DataGenerator(X[:, :-2].astype('float32'), y, weights=sample_weight,
                                           batch_size=self.batch_size, history=self.time_history, type='LSTM')
        if validation_data is not None:
            X_valid, y_valid = validation_data
            sample_weight_valid = np.array([self.af_weight if lab == True else 1 for lab in y_valid])
            valid_generator = DataGenerator(X_valid[:, :-2].astype('float32'), y_valid, weights=sample_weight_valid,
                                            batch_size=self.batch_size, history=self.time_history, type='LSTM')
            self.history = self.model.fit_generator(training_generator, validation_data=valid_generator,
                                                    epochs=n_epochs, workers=5, callbacks=[
                                                        # ReduceLROnPlateau(factor=0.1, patience=2, min_lr=self.params["conv_learning_rate"] * 0.001),
                                                        # EarlyStopping(monitor='val_auc', mode='max', patience=self.patience, min_delta=1e-3, restore_best_weights=True, verbose=1)
                                                    ])
            self.loss_train = self.history.history['loss']
            self.loss_valid = self.history.history['val_loss']
            self.n_epochs_train = len(self.loss_train)

        else:
            self.history = self.model.fit_generator(training_generator, epochs=n_epochs)
            self.loss_train = self.history.history['loss']
            self.n_epochs_train = len(self.loss_train)

    def predict_proba(self, X):
        """
        Predict probability of AF
        """
        test_generator = DataGenerator(X[:, :-2].astype('float32'), weights=None, batch_size=self.batch_size,
                                       history=self.time_history, to_fit=False, shuffle=False, type='LSTM')
        # X_batches, _, _, is_sample = self.temporal_concat(X, np.zeros(X.shape[0]), history=self.time_history)
        # pred = self.model.predict(X_batches, batch_size=self.batch_size)
        # res = pred.flatten()[is_sample.flatten()==1]  # keep only the prediction of the true inputs
        res = self.model.predict_generator(test_generator)
        # res = res.reshape(-1, 1)
        res = np.concatenate((1 - res, res), axis=1)  # for sklearn compatibility
        return res

    def predict(self, X, th=0.5):
        """
        Predict binary AF.
        """
        return self.predict_proba(X)[:, 1] > th

    def get_state_dict(self):
        res = {
            'weights': self.model.get_weights(),
            "window_size": self.window_size,
            "time_history": self.time_history,
            'activation': self.activation,
            "n_units": self.n_units,
            "n_dense": self.n_dense,
            "dropout_fcn": self.dropout_fcn,
            "learning_rate": self.learning_rate,
            "af_weight": self.af_weight,
            "batch_size": self.batch_size,
            'n_features': self.n_features,
            'loss_train': self.loss_train,
            'loss_valid': self.loss_valid,
            'n_epochs_train': self.n_epochs_train,
        }
        return res

    def set_state_dict(self, state):
        for key, val in state.items():
            if key == 'weights':
                self.model.set_weights(val)
            else:
                self.__dict__[key] = val
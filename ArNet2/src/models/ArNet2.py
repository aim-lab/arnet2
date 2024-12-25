"""
ArNet2: not cleaned because hopefully doomed to disappear.
"""

# General imports
import os
import numpy as np
import pandas as pd
import pickle
import copy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L1L2

# Relative imports
import utils.consts as cts
from OneDCNN import OneDCNN
from ResNet import ResNet
from datagen import DataGenerator


def fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta ** 2
    true_positive = np.sum(y_pred * y_true)
    precision = true_positive / (np.sum(y_pred) + eps)
    recall = true_positive / (np.sum(y_true) + eps)
    res = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    return res


def custom_loss(y_true, y_pred, threshold=0.5, lamb=0.5):
    res_bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return (1 - lamb) * res_bce - lamb * fbeta_score(y_true, y_pred, 3, threshold)


def load_feature_extractor(path):
    """
    Load 1D-CNN feature extractor to transform the input into features that will be given to the GRUs.
    """
    _, filename = os.path.split(path)
    model_full_name = filename.split(".")[0]

    with open(path, 'rb') as file:
        clf_dict = pickle.load(file)
        kwargs = clf_dict['hyperparameters']
        if "1D-CNN" in model_full_name:
            new_model = OneDCNN(**kwargs)
        elif "ResNet" in model_full_name:  #  note : it could be ResNet_DA, hence the importance of 'in'
            new_model = ResNet(**kwargs)
        new_model.set_state_dict(clf_dict['classifier'])
        feature_extractor = new_model

    return feature_extractor


class ArNet2:
    """
    ArNet2 model, that first transforms inputs into features with its 1D-CNN feature extractor, and then uses those
    features as well as preceding windows and global labels to predict the AF thourgh adapted GRUs.
    """

    def __init__(self, path_feature_extractor, window_size=60, time_history=10, extract_level='dense_1', n_units=8,
                 dropout=0.1, learning_rate=0.001, af_weight=3, batch_size=1024, n_patients=None,
                 n_jobs=1, n_additional_feat=0):
        tf.compat.v1.keras.backend.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.n_patients = n_patients
        self.patience = 10
        self.window_size = window_size
        self.time_history = time_history
        self.extract_level = extract_level
        self.feature_extractor = load_feature_extractor(path=path_feature_extractor)
        self.single_input_size = self.feature_extractor.model.get_layer(extract_level).output_shape[
                                     1] + n_additional_feat
        self.total_input_size = self.single_input_size * time_history
        self.n_units = n_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.af_weight = af_weight
        self.batch_size = batch_size
        self.labels = np.array([cts.PATIENT_LABEL_NON_AF, cts.PATIENT_LABEL_AF_MILD, cts.PATIENT_LABEL_AF_MODERATE,
                                cts.PATIENT_LABEL_AF_SEVERE])
        self.init_weights = {}
        self.models = {}
        self.histories = {}
        for lab in self.labels:
            model = Sequential()
            # model.add(Bidirectional(GRU(self.n_units, activation='relu'), input_shape=(None, self.single_input_size)))
            model.add(GRU(self.n_units, activation='relu', input_shape=(None, self.single_input_size)))
            model.add(Dense(self.n_units//2, activation='relu'))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
            self.models[lab] = model
            self.init_weights[lab] = copy.deepcopy(model.get_weights())
        self.losses_train = {}
        self.losses_valid = {}
        self.loss_train = None
        self.loss_valid = None
        self.n_epochs_train = {}

    def fit(self, X, y, validation_data=None, warm_start=False, n_epochs=5, n_epochs_flex=None,
            add_X=None, add_X_valid=None):

        if not warm_start:
            for lab, model in self.models.items():
                model.set_weights(self.init_weights[lab])
                model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
        X, prec_windows, global_label, ids = X[:, :-3].astype('float32'), X[:, -3].astype('float32'), X[:, -2].astype(
            'float32'), X[:, -1]
        features_X = self.feature_extractor.predict_layer(X, layer_name=self.extract_level)
        features_X = np.concatenate((features_X, prec_windows.reshape(-1, 1)), axis=1)
        if n_epochs_flex is None:
            n_epochs_flex = [n_epochs] * len(self.labels)
        sample_weight = np.array([self.af_weight if lab == True else 1 for lab in y])

        for lab, n_epochs_lab in zip(self.labels, n_epochs_flex):
            print("Training GRU for label: " + str(lab))
            mask = global_label == lab
            # mask = global_label == global_label
            if not np.any(mask):
                continue
            training_generator = DataGenerator(features_X, y, sample_weight, batch_size=self.batch_size,
                                               history=self.time_history, type='LSTM', add_data=add_X, mask=mask)
            if validation_data is not None:
                X_valid, y_valid = validation_data
                sample_weight_valid = np.array([self.af_weight if lab == True else 1 for lab in y_valid])
                X_valid, prec_windows_valid, global_label_valid, ids_valid = X_valid[:, :-3].astype('float32'), \
                                                                             X_valid[:, -3].astype('float32'), \
                                                                             X_valid[:, -2].astype('float32'), \
                                                                             X_valid[:, -1]
                # global_label_valid = self.predict_global_label(X_valid, ids_valid)
                mask_valid = global_label_valid == lab
                if not np.any(mask_valid):
                    continue
                features_X_valid = self.feature_extractor.predict_layer(X_valid, layer_name=self.extract_level)
                features_X_valid = np.concatenate((features_X_valid, prec_windows_valid.reshape(-1, 1)), axis=1)
                valid_generator = DataGenerator(features_X_valid, y_valid, sample_weight_valid, batch_size=self.batch_size,
                                                history=self.time_history, type='LSTM', add_data=add_X_valid,
                                                mask=mask_valid)
                self.histories[lab] = self.models[lab].fit_generator(training_generator,
                                                              validation_data=valid_generator, epochs=n_epochs_lab,
                                                              workers=5,
                                                              callbacks=[
                                                                  EarlyStopping(patience=self.patience, min_delta=1e-3,
                                                                                restore_best_weights=True)])
                self.losses_train[lab] = self.histories[lab].history['loss']
                self.losses_valid[lab] = self.histories[lab].history['val_loss']
                self.n_epochs_train[lab] = len(self.losses_train[lab])

            else:
                self.histories[lab] = self.models[lab].fit_generator(training_generator, epochs=n_epochs_lab)
                self.losses_train[lab] = self.histories[lab].history['loss']
                self.n_epochs_train[lab] = len(self.losses_train[lab])
        order_losses = [cts.PATIENT_LABEL_AF_MODERATE, cts.PATIENT_LABEL_AF_MILD, cts.PATIENT_LABEL_AF_SEVERE,
                        cts.PATIENT_LABEL_NON_AF]
        for lab in order_losses:
            try:
                self.loss_train = self.losses_train[lab]
                if len(self.losses_valid) > 0:
                    self.loss_valid = self.losses_valid[
                        lab]  # Selecting the moderates to set up the optimal number of epochs.
            except KeyError:
                continue
        # break
        # # order_losses = [cts.PATIENT_LABEL_AF_MODERATE, cts.PATIENT_LABEL_AF_MILD, cts.PATIENT_LABEL_AF_SEVERE,
        # #                 cts.PATIENT_LABEL_NON_AF]
        # # for lab in order_losses:
        # #     try:
        # #         self.loss_train = self.losses_train[lab]
        # #         if len(self.losses_valid) > 0:
        # #             self.loss_valid = self.losses_valid[
        # #                 lab]  # Selecting the moderates to set up the optimal number of epochs.
        # #     except KeyError:
        # #         continue

    def predict_proba(self, X, add_X=None):
        """
        Predict probability of AF
        """
        X, prec_windows, glob_lab, ids = X[:, :-3].astype('float32'), X[:, -3].astype('float32'), X[:, -2].astype(
            'float32'), X[:, -1]
        res = np.zeros(len(X))
        glob_lab = self.predict_global_label(X, ids)
        features_X = self.feature_extractor.predict_layer(X, layer_name=self.extract_level)
        features_X = np.concatenate((features_X, prec_windows.reshape(-1, 1)), axis=1)
        for lab in self.labels:
            mask = glob_lab == lab
            # mask = glob_lab == glob_lab
            if not np.any(mask):
                continue
            test_generator = DataGenerator(features_X, batch_size=self.batch_size,
                                           history=self.time_history, to_fit=False, shuffle=False, type='LSTM',
                                           add_data=add_X, mask=mask)
            res[mask] = self.models[lab].predict_generator(test_generator).reshape(-1)
            # break
        res = res.reshape(-1, 1)
        res = np.concatenate((1 - res, res), axis=1)  # For sklearn compatibility
        return res

    def predict(self, X, ids, th=0.5, add_X=None):
        """
        Predict binary AF.
        """
        return self.predict_proba(X, ids, add_X)[:, 1] > th

    def predict_global_label(self, X, ids):
        """
        Use whole 1D-CNN to predict the AF global label, before giving the extracted features to the adapted GRU.
        """
        y_pred = self.feature_extractor.predict_proba(X)[:, 1]
        id_df = pd.DataFrame({'id': ids, 'len_rr': np.sum(X, axis=1), 'y_pred': y_pred})
        id_df['time_in_af_pred'] = id_df['y_pred'] * id_df['len_rr']
        id_df['time_in_af_pred'] = id_df['time_in_af_pred'].astype(float)
        id_df['len_rr'] = id_df['len_rr'].astype(float)
        res = id_df.groupby('id').agg('sum')
        times_in_af = {pat: res.loc[pat]['time_in_af_pred'] for pat in np.unique(ids)}
        af_burdens = {pat: times_in_af[pat] / res.loc[pat]['len_rr'] for pat in np.unique(ids)}
        glob_labs_dict = {pat: cts.PATIENT_LABEL_NON_AF for pat in af_burdens.keys()}
        for pat in af_burdens.keys():
            if af_burdens[pat] > cts.AF_SEVERE_THRESHOLD:
                glob_labs_dict[pat] = cts.PATIENT_LABEL_AF_SEVERE
            elif af_burdens[pat] > cts.AF_MODERATE_THRESHOLD:
                glob_labs_dict[pat] = cts.PATIENT_LABEL_AF_MODERATE
            elif times_in_af[pat] > cts.AF_MILD_THRESHOLD:
                glob_labs_dict[pat] = cts.PATIENT_LABEL_AF_MILD
        global_labs = np.array([glob_labs_dict[pat] for pat in ids])
        return global_labs

    def get_state_dict(self):
        """
        Get some of model hyperparameters.
        """
        res = {
            'weights': [model.get_weights() for model in self.models.values()],
            'window_size': self.window_size,
            'time_history': self.time_history,
            'extract_level': self.extract_level,
            'n_units': self.n_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'af_weight': self.af_weight,
            'batch_size': self.batch_size,
            'n_patients': self.n_patients,
            'init_weights': self.init_weights,
            'loss_train': self.loss_train,
            'n_epochs_train': self.n_epochs_train
        }
        return res

    def set_state_dict(self, state):
        """
        Set model hyperparameters.
        """
        for key, val in state.items():
            if key == 'weights':
                for i, w in enumerate(val):
                    self.models[i].set_weights(w)
            else:
                self.__dict__[key] = val
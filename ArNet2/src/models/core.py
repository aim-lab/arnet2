
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
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# Relative imports
import utils.consts as cts
from ArNet2.src.models.OneDCNN import OneDCNN
from ArNet2.src.models.ResNet import ResNet
from ArNet2.src.models.datagen import TFDataGenerator


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
                 n_additional_feat=0):
        tf.keras.backend.clear_session()
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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

        self.labels = np.array([
            cts.PATIENT_LABEL_NON_AF,
            cts.PATIENT_LABEL_AF_MILD,
            cts.PATIENT_LABEL_AF_MODERATE,
            cts.PATIENT_LABEL_AF_SEVERE
        ])

        self.init_weights = {}
        self.models = {}
        self.histories = {}
        for lab in self.labels:
            model = Sequential([
                GRU(self.n_units, activation='relu', input_shape=(None, self.single_input_size)),
                Dense(self.n_units // 2, activation='relu'),
                Dropout(self.dropout),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            self.models[lab] = model
            self.init_weights[lab] = copy.deepcopy(model.get_weights())
        self.losses_train = {}
        self.losses_valid = {}
        self.loss_train = None
        self.loss_valid = None
        self.n_epochs_train = {}

    def _parse_mixed_X(self, X):
        """
        Accepts X as either:
          • tf/numpy string matrix [N, 63] (your old mixed layout), OR
          • numeric matrix where [:, :-3] are rr, [-3] is prec_windows, [-1] are ids (numeric).

        Returns:
          rr:       tf.float32 [N, 60]
          prec_win: tf.int32   [N]
          ids:      tf.string  [N]
        """
        X = tf.convert_to_tensor(X)  # keep dtype as-is first

        if X.dtype == tf.string:
            rr = tf.strings.to_number(X[:, :-3], tf.float32)
            prec_win = tf.cast(tf.strings.to_number(X[:, -3], tf.float32), tf.int32)
            ids = X[:, -1]
        else:
            rr = tf.cast(X[:, :-3], tf.float32)
            prec_win = tf.cast(X[:, -3], tf.int32)
            # ids could be numeric; normalize to string so downstream stays pure TF
            ids = tf.as_string(X[:, -1])

        return rr, prec_win, ids

    def _make_tf_dataset(self, orig_data, orig_labels=None, weights=None,
                         to_fit=True, shuffle=True, add_data=None, mask=None):
        gen = TFDataGenerator(
            orig_data=orig_data,  # [N, F+1], last col = prec_windows
            orig_labels=orig_labels,  # [N] or None
            weights=weights,  # [N] or None
            to_fit=to_fit,
            batch_size=self.batch_size,
            history=self.time_history,
            shuffle=shuffle,
            dg_type='LSTM',  # same as before
            add_data=add_data,
            mask=mask
        )
        return gen.as_dataset()

    def fit(self, X, y, validation_data=None, warm_start=False, n_epochs=5, n_epochs_flex=None,
            add_X=None, add_X_valid=None):

        if not warm_start:
            for lab, model in self.models.items():
                model.set_weights(self.init_weights[lab])
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC()]
                )
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

            training_ds = self._make_tf_dataset(
                orig_data=features_X,  # [N, F+1], last col = prec_windows
                orig_labels=y,
                weights=sample_weight,
                to_fit=True,
                shuffle=True,
                add_data=add_X,
                mask=mask
            )

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

                valid_ds = self._make_tf_dataset(
                    orig_data=features_X_valid,
                    orig_labels=y_valid,
                    weights=sample_weight_valid,
                    to_fit=True,
                    shuffle=False,
                    add_data=add_X_valid,
                    mask=mask_valid
                )

                self.histories[lab] = self.models[lab].fit(
                    training_ds,
                    validation_data=valid_ds,
                    epochs=n_epochs_lab,
                    callbacks=[EarlyStopping(patience=self.patience, min_delta=1e-3,
                                             restore_best_weights=True)]
                )
                self.losses_train[lab] = self.histories[lab].history['loss']
                self.losses_valid[lab] = self.histories[lab].history['val_loss']
                self.n_epochs_train[lab] = len(self.losses_train[lab])

            else:
                self.histories[lab] = self.models[lab].fit(training_ds, epochs=n_epochs_lab)
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

    def predict_proba_tf(self, X, add_X=None):
        """
        Pure-TF inference.

        X layout:
          [:, :60] -> rr
          [:, -3]  -> prec_windows
          [:, -2]  -> glob_lab (ignored)
          [:, -1]  -> ids (string or numeric)
        Returns tf.float32 [N, 2] = [1-p, p].
        """
        # --- unified parsing: works for string-mixed OR numeric X
        rr, prec_win, ids = self._parse_mixed_X(X)  # rr:[N,60] float32, prec_win:[N] int32, ids:[N] string

        # global labels (batched)
        row_lab = self.predict_global_label_tf(rr, ids)  # [N] int32

        # features at extract_level (TF-safe)
        feat = self.feature_extractor.predict_layer(rr, layer_name=self.extract_level)  # [N, F0], batched inside ResNet
        feat = tf.concat([feat, tf.cast(prec_win[:, None], tf.float32)], axis=1)  # [N, F0+1]
        if add_X is not None:
            feat = tf.concat([feat, tf.cast(add_X, tf.float32)], axis=1)

        N = tf.shape(feat)[0]
        probs = tf.zeros([N], tf.float32)

        # Per-label masked inference via your TFDataGenerator; concatenate batches
        for lab in self.labels:
            mask_bool = tf.equal(row_lab, tf.cast(lab, tf.int32))
            if not tf.reduce_any(mask_bool):
                continue

            ds = self._make_tf_dataset(
                orig_data=feat,
                orig_labels=None,
                weights=None,
                to_fit=False,
                shuffle=False,
                add_data=None if add_X is None else add_X,
                mask=mask_bool
            )

            preds_lab = tf.zeros([0], dtype=tf.float32)  # grow as 1-D vector
            for xb in ds:  # xb: [batch, H, F]
                yb = self.models[lab](xb, training=False)  # [batch,1]
                yb = tf.squeeze(yb, axis=-1)  # [batch]
                preds_lab = tf.concat([preds_lab, yb], axis=0)

            idx_full = tf.reshape(tf.where(mask_bool), [-1])  # [K]
            probs = tf.tensor_scatter_nd_update(
                probs,
                indices=tf.expand_dims(idx_full, 1),
                updates=preds_lab
            )

        return tf.stack([1.0 - probs, probs], axis=1)

    def predict_proba(self, X, add_X=None):
        """
        Backward-compatible wrapper that returns NumPy like your original.
        Accepts either string-mixed [N,63] or numeric matrix with the same column layout.
        """
        # Do NOT force dtype here—let predict_proba_tf parse robustly.
        X_tf = tf.convert_to_tensor(X)
        return self.predict_proba_tf(X_tf, add_X).numpy()

    def predict(self, X, ids=None, th=0.5, add_X=None):
        """
        Binary AF prediction (NumPy output). 'ids' kept for API parity; not used.
        """
        p = self.predict_proba(X, add_X)[:, 1]  # NumPy [N]
        return (p > float(th)).astype(np.int32).reshape(-1)

    # Inside ArNet2 class

    def predict_global_label_tf(self, X, ids):
        """
        Pure-TF version. Returns tf.int32 tensor [N] of label codes.
        """
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)  # [N, 60] (or your window size)

        ids_tf = tf.convert_to_tensor(ids)
        if ids_tf.dtype != tf.string:
            ids_tf = tf.as_string(ids_tf)  # [N] string

        # P(AF) per row via TF path on feature extractor
        if hasattr(self.feature_extractor, "predict_proba_tf"):
            probs2 = self.feature_extractor.predict_proba_tf(X_tf)  # [N, 2], batched inside ResNet
            y_pred = probs2[:, 1]  # [N]
        else:
            fe_out = self.feature_extractor.model(X_tf, training=False)  # [N,1] or [N,2]
            fe_out = tf.convert_to_tensor(fe_out, dtype=tf.float32)
            if fe_out.shape.rank == 2 and fe_out.shape[-1] == 2:
                y_pred = tf.nn.softmax(fe_out, axis=-1)[:, 1]
            else:
                y_pred = tf.nn.sigmoid(tf.squeeze(fe_out, axis=-1))

        len_rr = tf.reduce_sum(X_tf, axis=1)  # [N]
        time_in_af = y_pred * len_rr  # [N]

        unique_ids, row_to_bucket = tf.unique(ids_tf)  # [B], [N]
        B = tf.shape(unique_ids)[0]

        sum_len_rr = tf.math.unsorted_segment_sum(len_rr, row_to_bucket, num_segments=B)  # [B]
        sum_time = tf.math.unsorted_segment_sum(time_in_af, row_to_bucket, num_segments=B)  # [B]
        burden = sum_time / (sum_len_rr + 1e-9)  # [B]

        severe = burden > tf.constant(cts.AF_SEVERE_THRESHOLD, tf.float32)
        moderate = burden > tf.constant(cts.AF_MODERATE_THRESHOLD, tf.float32)
        mild = sum_time > tf.constant(cts.AF_MILD_THRESHOLD, tf.float32)

        lab_non = tf.fill([B], tf.cast(cts.PATIENT_LABEL_NON_AF, tf.int32))
        lab_mild = tf.fill([B], tf.cast(cts.PATIENT_LABEL_AF_MILD, tf.int32))
        lab_moder = tf.fill([B], tf.cast(cts.PATIENT_LABEL_AF_MODERATE, tf.int32))
        lab_severe = tf.fill([B], tf.cast(cts.PATIENT_LABEL_AF_SEVERE, tf.int32))

        bucket_label = tf.where(severe, lab_severe,
                                tf.where(moderate, lab_moder,
                                         tf.where(mild, lab_mild, lab_non)))  # [B] int32

        row_labels = tf.gather(bucket_label, row_to_bucket)  # [N] int32
        return row_labels

    def predict_global_label(self, X, ids):
        """
        Backward-compatible wrapper returning numpy(), for places that expect numpy.
        Do NOT call this inside @tf.function.
        """
        return self.predict_global_label_tf(X, ids).numpy()

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
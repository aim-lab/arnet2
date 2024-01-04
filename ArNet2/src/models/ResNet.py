import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, BatchNormalization, Activation, MaxPooling1D, Lambda, Add, \
    GRU, TimeDistributed, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

import utils.consts as cts


class ResNet:

    def __init__(self, window_size=60, n_blocks=6, n_filters_start=64, filter_length=10, activation='relu',
                 dropout_conv=0.2, n_hidden_start=512, dropout_fcn=0.5, learning_rate=0.001, af_weight=3,
                 batch_size=1024):
        tf.compat.v1.keras.backend.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        self.n_features = 1
        self.patience = 20  # High to allow running for at least 20 epochs.
        self.window_size = window_size
        self.n_blocks = n_blocks
        self.subsample_lengths = [1, 2] * (n_blocks // 2) + [1] * (n_blocks % 2)
        self.n_filters_start = n_filters_start
        self.filter_length = filter_length
        self.activation = activation
        self.dropout_conv = dropout_conv
        self.n_hidden_start = n_hidden_start
        self.dropout_fcn = dropout_fcn
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.batch_size = batch_size
        self.af_weight = af_weight

        self.params = {"input_shape": (window_size, 1), "num_categories": 1, "init": "he_normal", "num_skip": 2,
                       "increase_channels_at": 2, "compile": True}
        self.params.update({"conv_subsample_lengths": self.subsample_lengths, "conv_n_filters_start": n_filters_start,
                            "conv_filter_length": filter_length, "conv_activation": activation,
                            "conv_dropout": dropout_conv, "n_hidden_start": n_hidden_start, "fcn_dropout": dropout_fcn,
                            "conv_learning_rate": learning_rate, "conv_window_size": window_size,
                            "conv_batch_size": batch_size})

        self.model = self.build_network(**self.params)
        # print(self.model.summary())

        self.loss_train = None
        self.loss_valid = None
        self.n_epochs_train = 0

    def _bn_relu(self, layer, dropout=0, **params):

        layer = BatchNormalization()(layer)
        layer = Activation(params["conv_activation"])(layer)

        if dropout > 0:
            layer = Dropout(params["conv_dropout"])(layer)

        return layer

    def add_conv_weight(self,
                        layer,
                        filter_length,
                        num_filters,
                        subsample_length=1,
                        **params):
        layer = Conv1D(
            filters=num_filters,
            kernel_size=filter_length,
            strides=subsample_length,
            padding='same',
            kernel_initializer=params["init"])(layer)
        return layer

    def add_conv_layers(self, layer, **params):
        for subsample_length in params["conv_subsample_lengths"]:
            layer = self.add_conv_weight(
                layer,
                params["conv_filter_length"],
                params["conv_n_filters_start"],
                subsample_length=subsample_length,
                **params)
            layer = self._bn_relu(layer, **params)
        return layer

    def resnet_block(self,
                     layer,
                     num_filters,
                     subsample_length,
                     block_index,
                     **params):

        def zeropad(x):
            y = K.zeros_like(x)
            return K.concatenate([x, y], axis=2)

        def zeropad_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3
            shape[2] *= 2
            return tuple(shape)

        shortcut = MaxPooling1D(pool_size=subsample_length, padding='same')(layer)
        zero_pad = (block_index % params["increase_channels_at"]) == 0 \
                   and block_index > 0
        if zero_pad is True:
            shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

        for i in range(params["num_skip"]):
            if not (block_index == 0 and i == 0):
                layer = self._bn_relu(
                    layer,
                    dropout=params["conv_dropout"] if i > 0 else 0,
                    **params)
            layer = self.add_conv_weight(
                layer,
                params["conv_filter_length"],
                num_filters,
                subsample_length if i == 0 else 1,
                **params)
        layer = Add()([shortcut, layer])
        return layer

    def get_num_filters_at_index(self, index, num_start_filters, **params):
        return 2 ** int(index / params["increase_channels_at"]) \
               * num_start_filters

    def add_resnet_layers(self, layer, **params):
        layer = self.add_conv_weight(
            layer,
            params["conv_filter_length"],
            params["conv_n_filters_start"],
            subsample_length=1,
            **params)
        layer = self._bn_relu(layer, **params)
        for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
            num_filters = self.get_num_filters_at_index(
                index, params["conv_n_filters_start"], **params)
            layer = self.resnet_block(
                layer,
                num_filters,
                subsample_length,
                index,
                **params)
        layer = self._bn_relu(layer, **params)
        return layer

    def add_lstm_layer(self, layer, **params):
        layer = GRU(64)(layer)
        layer = Dropout(params["conv_dropout"])(layer)
        layer = BatchNormalization()(layer)
        return layer

    def add_output_layer(self, layer, **params):
        layer = Flatten()(layer)
        layer = Dense(params["n_hidden_start"], activation='relu')(layer)
        layer = Dropout(params["fcn_dropout"])(layer)
        layer = Dense(params["n_hidden_start"] // 2, activation='relu')(layer)
        layer = Dropout(params["fcn_dropout"])(layer)
        layer = Dense(params["n_hidden_start"] // 4, activation='relu')(layer)
        layer = Dropout(params["fcn_dropout"])(layer)
        layer = Dense(params["num_categories"], activation='sigmoid')(layer)
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

        if params.get('is_regular_conv', False):
            layer = self.add_conv_layers(inputs, **params)
        else:
            layer = self.add_resnet_layers(inputs, **params)
        # layer = self.add_lstm_layer(layer, **params)
        output = self.add_output_layer(layer, **params)
        model = Model(inputs=[inputs], outputs=[output])
        if params.get("compile", True):
            self.add_compile(model, **params)
        return model

    def fit(self, X, y, validation_data=None, n_epochs=30):
        X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')
        sample_weight = np.array([self.af_weight if lab == True else 1 for lab in y])
        if validation_data is not None:
            X_valid, y_valid = validation_data
            sample_weight_valid = np.array([self.af_weight if lab == True else 1 for lab in y_valid])
            X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1).astype('float32')
            self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
                                          validation_data=(X_valid, y_valid, sample_weight_valid), epochs=n_epochs,  # Rk : we use the training sample_weight for validation
                                          callbacks=[
                                              # ReduceLROnPlateau(factor=0.1, patience=2, min_lr=self.params["conv_learning_rate"] * 0.001),
                                              # EarlyStopping(monitor='val_auc', mode='max', patience=self.patience, min_delta=1e-3, restore_best_weights=True, verbose=1)
                                          ])
            self.loss_train = self.history.history['loss']
            self.loss_valid = self.history.history['val_loss']
            self.n_epochs_train = len(self.loss_train)

        else:
            self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
                                          epochs=n_epochs)
            self.loss_train = self.history.history['loss']
            self.n_epochs_train = len(self.loss_train)

    def fit_smote(self, X, y, sample_weight=None, validation_data=None, n_epochs=30):

        def loss(model, x, y, training, sample_weight=None):
            # training=training is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_ = model(x, training=training)
            return loss_object(y_true=y, y_pred=y_, sample_weight=sample_weight)

        def grad(model, inputs, targets, sample_weight=None):
            with tf.GradientTape() as tape:
                loss_value = loss(model, inputs, targets, training=True, sample_weight=sample_weight)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

        X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')

        loss_object = tf.keras.losses.BinaryCrossentropy()
        # accuracy_object = tf.keras.metrics.Accuracy()
        # auc_object = tf.keras.metrics.AUC()
        optimizer = tf.keras.optimizers.Adam(lr=self.params["conv_learning_rate"],
                                             clipnorm=self.params.get("clipnorm", 1))

        # Keep results for plotting
        train_loss_results = []
        train_accuracy_results = []

        batch_losses = []
        # batch_accuracies = []
        # batch_aucs = []
        for epoch in range(n_epochs):
            epoch_starttime = np.datetime64('now')
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()
            epoch_auc = tf.keras.metrics.AUC()
            train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
            train_dataset = train_dataset.shuffle(X.shape[0] // 10, seed=epoch * cts.SEED).batch(self.batch_size,
                                                                                                 drop_remainder=True)  # change of seed so that each epoch has a new shuffle, but controlled by the based SEED for reproducibility

            # Training loop
            for i, (xx, yy) in enumerate(train_dataset):
                # Optimize the model
                loss_value, grads = grad(self.model, xx, yy, sample_weight=sample_weight)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(yy, self.model(xx, training=True))
                epoch_auc.update_state(yy, self.model(xx, training=True))
                if i % 100 == 0:
                    print("Epoch: {:03d} Batch: {:03d} Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3%}".format(epoch, i,
                                                                                                           epoch_loss_avg.result(),
                                                                                                           epoch_accuracy.result(),
                                                                                                           epoch_auc.result()))

            # Training loop adding SMOTE
            print("SMOTE POST-TRAINING...")
            tf.compat.v1.keras.backend.clear_session()
            # Oversample the feature layer outputs
            pre_smote_model = Model(inputs=self.model.input, outputs=self.model.get_layer('flatten').output)
            from sklearn.utils import shuffle
            X_shuffled, y_shuffled = shuffle(X, y, random_state=cts.SEED)
            len_X = X.shape[0]
            generated_features = []
            y_generated = []
            for i in range(11):
                print(i)
                X_batch, y_batch = X_shuffled[i * (len_X // 10): (i + 1) * (len_X // 10)], y_shuffled[
                                                                                           i * (len_X // 10): (
                                                                                                                          i + 1) * (
                                                                                                                          len_X // 10)]
                if np.sum(y_batch) <= len(X_batch) // 50:
                    print("no enough AF examples in this batch, continue")
                    continue
                pre_smote_features_batch = pre_smote_model.predict(X_batch, batch_size=self.batch_size)
                oversample = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=cts.SEED)
                generated_features_batch, y_generated_batch = oversample.fit_resample(pre_smote_features_batch, y_batch)
                # generated_features_batch, y_generated_batch = post_smote_features_batch[pre_smote_features_batch.shape[0]:], y_smote_batch[pre_smote_features_batch.shape[0]:]  # keep only the generated samples
                generated_features.append(generated_features_batch)
                y_generated.append(y_generated_batch)
                tf.compat.v1.keras.backend.clear_session()
            generated_features = np.concatenate(generated_features)
            y_generated = np.concatenate(y_generated)
            generated_dataset = tf.data.Dataset.from_tensor_slices((generated_features, y_generated))
            generated_dataset = generated_dataset.shuffle(generated_features.shape[0] // 10,
                                                          seed=epoch * cts.SEED).batch(self.batch_size,
                                                                                       drop_remainder=True)

            input = tf.keras.layers.Input((generated_features.shape[1]))
            l = self.model.layers[-4:]  # from dense_1 to output
            output = input
            for lay in l:
                output = lay(output)
            post_smote_model = Model(inputs=input, outputs=output)

            for i, (xx, yy) in enumerate(generated_dataset):
                # Optimize the model
                loss_value, grads = grad(post_smote_model, xx, yy, sample_weight=sample_weight)
                optimizer.apply_gradients(zip(grads, post_smote_model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(yy, post_smote_model(xx, training=True))
                epoch_auc.update_state(yy, post_smote_model(xx, training=True))
                if i % 100 == 0:
                    print(
                        "Epoch: {:03d} SMOTE POST-TRAINING Batch: {:03d} Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3%}".format(
                            epoch, i,
                            epoch_loss_avg.result(),
                            epoch_accuracy.result(),
                            epoch_auc.result()))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            epoch_endtime = np.datetime64('now')
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3%} --- {}".format(epoch,
                                                                                            epoch_loss_avg.result(),
                                                                                            epoch_accuracy.result(),
                                                                                            epoch_auc.result(),
                                                                                            str(
                                                                                                epoch_endtime - epoch_starttime)))

        # self.history = self.model.fit(train_dataset, epochs=n_epochs)

        # if validation_data is not None:
        #     X_valid, y_valid, sample_weight_valid = validation_data
        #     X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1).astype('float32')
        #     self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
        #                                   validation_data=(X_valid, y_valid, sample_weight_valid), epochs=n_epochs,
        #                                   callbacks=[
        #                                       # ReduceLROnPlateau(factor=0.1, patience=2, min_lr=self.params["conv_learning_rate"] * 0.001),
        #                                       # EarlyStopping(monitor='val_auc', mode='max', patience=self.patience, min_delta=1e-3, restore_best_weights=True, verbose=1)
        #                                   ])
        #     self.loss_train = self.history.history['loss']
        #     self.loss_valid = self.history.history['val_loss']
        #     self.n_epochs_train = len(self.loss_train)
        #
        # else:
        #     self.history = self.model.fit(X, y, sample_weight=sample_weight, batch_size=self.batch_size,
        #                                   epochs=n_epochs)
        #     self.loss_train = self.history.history['loss']
        #     self.n_epochs_train = len(self.loss_train)

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
            "window_size": self.window_size,
            "n_blocks": self.n_blocks,
            "n_filters_start": self.n_filters_start,
            "filter_length": self.filter_length,
            "dropout_conv": self.dropout_conv,
            "n_hidden_start": self.n_hidden_start,
            "dropout_fcn": self.dropout_fcn,
            "learning_rate": self.learning_rate,
            "af_weight": self.af_weight,
            "batch_size": self.batch_size,
            'n_features': self.n_features,
            'activation': self.activation,
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


if __name__ == '__main__':
    model = ResNet(window_size=100)
    print(model.model.summary())
"""
Drawn from DeepSMOTE paper : https://arxiv.org/abs/2105.02340
PyTorch code : https://github.com/dd1github/DeepSMOTE
"""
import argparse
import os
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Input, Flatten, LeakyReLU, Reshape, Conv1DTranspose
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import utils.consts as cts
from data import data_loading
from parsing.db_loader import *

class LatentSMOTE(Model):
    """ Conditional AutoEncoder for rr windows."""

    def __init__(self):
        super(LatentSMOTE, self).__init__()
        tf.compat.v1.keras.backend.clear_session()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        # Convolutional Encoder
        self.encoder = tf.keras.Sequential([
            Input(shape=(60, 1)),
            Conv1D(filters=32, kernel_size=5, strides=2, padding='same', use_bias=False),
            LeakyReLU(alpha=0.2),
            Conv1D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(30)])

        # Convolutional Decoder
        self.decoder = tf.keras.Sequential([
            Input(shape=(30,)),
            Dense(15 * 64),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Reshape((15, 64)),
            Conv1DTranspose(64, 5, strides=1, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1DTranspose(32, 5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv1DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='relu')])  # 'relu' / 'linear'

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        """ Training step for one batch of data, implementing the two-parts loss."""

        # Get the data
        if len(data) == 3:
            rr, y, sample_weight = data
        else:
            sample_weight = None
            rr, y = data

        # Define the losses
        with tf.GradientTape() as tape:

            # Reconstruction loss = MSE (Mean Squared Error) between input and output
            generated = self(rr)
            reconstruction_loss = self.loss(rr, generated, sample_weight=sample_weight)

            # Penalty Loss = MSSD (Mean Square of the Successive Differences )
            generated_non_af, generated_af = generated[y==False], generated[y==True]
            mean_af_entropy = tf.reduce_mean((generated_af[:, 1:] - generated_af[:, :-1])**2)
            mean_nonaf_entropy = tf.reduce_mean((generated_non_af[:, 1:] - generated_non_af[:, :-1])**2)
            entropy_loss = 0.01 * (mean_nonaf_entropy / mean_af_entropy)

            # LatentSMOTE loss = sum of the two
            latentsmote_loss = reconstruction_loss + entropy_loss

        grads = tape.gradient(latentsmote_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"latentsmote_loss": latentsmote_loss, "reconstruction_loss": reconstruction_loss, "entropy_loss": entropy_loss}

def generate_smote(rr_latent, y, n_neighbors=5, kneighbors=None, extrapolation=False, speed_NN=False):
    """ Apply SMOTE in the latent space to generate new samples.
    First, compute the Nearest Neighbors matrix of the latent samples.
    Then for each point in the latent space, randomly choose one of its `n_neighbors` closest neighbors.
    Finally, perform either interpolation (classic SMOTE) or extrapolation between the original and the neighbor point,
    to generate a "pattern mixed" point.
    As the NearestNeighbors search can become highly computationally expensive with too much samples,
    we develop a speed up version, where we cut the training set into 100,000 samples batches, on which we perform NearestNeighbors Search.
    Also, to speed up, there is the possibility to feed the already computed kneighbors matrix, if we want to SMOTE under different settings, but with the same kneighbors matrix.

    :param rr_latent: the training set, projected into the latent space via the LatenSMOTE encoder
    :param y: their AF labels
    :param n_neighbors: the number of  closest neighbors we must look into for the NearestNeighbors Search
    :param kneighbors: the possible pre-computed kneighbors matrix. If None, we re-compute it from scratch.
    :param extrapolation: whether we want to perform extrapolation. If False, perform interpolaiton (classical SMOTE)
    :param speed_NN: whether we want to speed the kneighbors matrix computation, by cutting the training set into smaller batches
    :return rr_latent_smote: the latent space points generated by SMOTE
    :return kneighbors: the kneighbors matrix
    :return neighbor_indices: the indices of the randomly chosen closest neighbors
    """

    # Fit the NN model
    print("Fitting NearestNeighbors...")
    if kneighbors is None:
        def fit_class_kneighbors(n_neighbors, rr_latent, y):
            """ Fit NearestNeighbors for each label (AF/nonAF) and build a global kneighbors matrix with both AF and nonAF samples."""
            kneighbors = np.zeros((rr_latent.shape[0], n_neigh)).astype(int)
            for lab in [False, True]:
                nn_lab = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
                rr_latent_lab = rr_latent[y == lab]
                nn_lab.fit(rr_latent_lab)
                kneighbors_lab_relative = nn_lab.kneighbors(rr_latent_lab, return_distance=False)  # here the indices of the kneighbors are relative of the label (so from 0 to n_samples_lab)
                kneighbors_lab_absolute = (lambda x: np.arange(y.shape[0])[y == lab][x])(kneighbors_lab_relative)  # here the indices of the kneighbors are absolute (so from 0 to n_samples)
                kneighbors[y == lab] = kneighbors_lab_absolute
            return kneighbors

        if not speed_NN:
            kneighbors = fit_class_kneighbors(n_neighbors, rr_latent, y)
        else:
            # Fit NearestNeighbors to each batch
            kneighbors = []
            for i in range(rr_latent.shape[0] // 100000 + 1):
                print(i)
                rr_latent_sub = rr_latent[100000 * i:100000 * (i + 1)]
                y_sub = y[100000 * i:100000 * (i + 1)]
                kneighbors_sub = fit_class_kneighbors(n_neighbors, rr_latent_sub, y_sub)  # relative indicies withing the 100,000-sized batch
                kneighbors_sub = kneighbors_sub + 100000 * i  # absolute indices, adding 100,000 * i to the previous ones
                kneighbors.append(kneighbors_sub)
            kneighbors = np.concatenate(kneighbors, axis=0)

    # Generate samples
    print("Generating Samples...")
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), rr_latent.shape[0])
    rr_latent_neighbor = rr_latent[kneighbors[np.arange(rr_latent.shape[0]), neighbor_indices]]
    rr_latent_smote = rr_latent + (-1)**extrapolation * np.multiply(np.random.rand(rr_latent.shape[0], 1),
                                 rr_latent_neighbor - rr_latent)

    return rr_latent_smote, kneighbors, neighbor_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data augmentation for AF classification')
    parser.add_argument('--skip_plots', action='store_true',
                        help='dont generate plots')
    parser.add_argument('--path_models', default=cts.BASE_DIR / 'Shany'/ "Tom" / "generative_models",
                        help='where the models are saved')
    parser.add_argument('--save_path', default=cts.REPO_DIR / "data" / "splits" / "model_input" / "LatentSMOTE",
                        help='where to save augmented data')
    parser.add_argument('--task', default='train',
                        help='train / eval')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    """ the train pipeline trains the autoencoder and the NearestNeighbors, and saves the results.
        the load pipeline loads the trained autoencoder and kneighbors matrix, performs SMOTE under different parameters, and saves the results. """

    # Define model loading and saving path
    folder_date = '2022-01-27T15:51:23' #"2021-08-11T15:50:26" / np.datetime64('now') / None
    if folder_date is None:
        folder_dates = [np.datetime64(x) for x in os.listdir(args.path_models)]
        folder_date = np.max(folder_dates)
    path_models = args.path_models / str(folder_date)

    # Load Data
    print("Loading Data...")
    data_train, test_dict = data_loading.return_input_model(parser_mapping_dict=PARSER_MAP, test_set_list=cts.test_set_list)

    X_train, y_train, t_s_train = data_train
    rr_train = X_train[:,:-3].reshape(-1, 60, 1).astype('float32')

    if args.task == "train":
        # Create and fit AutoEncoder (or load trained model directly)
        autoencoder = LatentSMOTE()
        autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(), run_eagerly=True)
        reduce_lr = ReduceLROnPlateau(monitor='latentsmote_loss', mode='min', factor=0.5, patience=5, min_delta=0.0005,
                                      min_lr=0.00001, verbose=1)
        early_stopping = EarlyStopping(monitor='latentsmote_loss', mode='min', patience=10, min_delta=0,
                                       restore_best_weights=True, verbose=1)
        # sample_weight = np.ones(shape=(len(y_train),))
        # sample_weight[y_train == True] = 5
        autoencoder.fit(rr_train, y_train, epochs=50, batch_size=4096, shuffle=True, sample_weight=None,
                        validation_data=None, callbacks=[reduce_lr, early_stopping])

        # autoencoder = tf.keras.models.load_model(path_models / 'autoencoder')

        # Generate samples with SMOTE
        print("Projecting AF samples into latent space...")
        rr_train_latent = autoencoder.encoder.predict(rr_train, batch_size=4096).reshape((-1, 30))
        print("SMOTing the latent space...")
        n_neigh = 100
        rr_train_latent_smote, kneighbors, _ = generate_smote(rr_train_latent, y_train, n_neighbors=n_neigh, speed_NN=True)
        print("Decoding SMOTEd samples...")
        rr_gen = autoencoder.decoder.predict(rr_train_latent_smote, batch_size=4096).reshape((-1, 60))

        # Save model and generated data
        print("Saving model and generated data...")
        if not os.path.exists(path_models):
            os.makedirs(path_models)
        np.save(path_models / "X_train_latent", np.concatenate((rr_train_latent, X_train[:, -3:]), axis=1))
        np.save(path_models / "kneighbors", kneighbors)
        autoencoder.save(path_models / 'autoencoder')

    else:
        autoencoder = tf.keras.models.load_model(path_models / 'autoencoder')
        X_train_latent = np.load(path_models / "X_train_latent.npy", allow_pickle=True)
        rr_train_latent = X_train_latent[:, :-3].astype('float32')
        kneighbors = np.load(path_models / "kneighbors.npy")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        for n_neigh in [5, 25, 50, 100]:
            for extrapolation in [False, True]:
                rr_train_latent_smote, _, neighbor_indices = generate_smote(rr_train_latent, y_train, n_neighbors=n_neigh, kneighbors=kneighbors, extrapolation=extrapolation)
                rr_neighbor = rr_train[kneighbors[np.arange(rr_train.shape[0]), neighbor_indices]]
                rr_gen = autoencoder.decoder.predict(rr_train_latent_smote, batch_size=4096).reshape((-1, 60))
                X_latent_smote = np.concatenate((rr_train_latent_smote, X_train[:, -3:]), axis=1)  # we assume that prec_win, glob_lab and id are preserved
                X_gen = np.concatenate((rr_gen, X_train[:, -3:]), axis=1)
                np.save(args.save_path / (f"X_latent_smote_{'extra' if extrapolation else 'inter'}" + "_" + str(n_neigh)), X_latent_smote)
                np.save(args.save_path / (f"rr_neighbor_{'extra' if extrapolation else 'inter'}"+"_"+str(n_neigh)), rr_neighbor)
                np.save(args.save_path / (f"X_gen_{'extra' if extrapolation else 'inter'}" + "_" + str(n_neigh)), X_gen)
            np.save(args.save_path / "y_gen", y_train)
            np.save(args.save_path / "t_s_gen", t_s_train)

    # PLOTS
    # import matplotlib.pyplot as plt
    # rr_train_reconstructed = autoencoder.predict(rr_train, batch_size=4096).reshape((-1, 60))
    # for rr, rr_reconstructed, label in zip(rr_train[::1000000], rr_train_reconstructed[::1000000], y_train[::1000000]):
    #     plt.plot(rr, label="orig", color='blue')
    #     plt.plot(rr_reconstructed, label="reconstructed", color='red', ls=':')
    #     plt.legend()
    #     plt.title(("" if label else "non") + "AF")
    #     plt.show()
    # for rr, rr_reconstructed, label in zip(rr_train[y_train==True][::50000], rr_train_reconstructed[y_train==True][::50000], y_train[y_train==True][::50000]):
    #     plt.plot(rr, label="orig", color='blue')
    #     plt.plot(rr_reconstructed, label="reconstructed", color='red', ls=':')
    #     plt.legend()
    #     plt.title(("" if label else "non") + "AF")
    #     plt.show()

    # for rr, lab in zip(rr_gen[::1000000], y_train[::1000000]):
    #     plt.plot(rr, color='green')
    #     plt.title("AF sample generated via LatentSMOTE")
    #     plt.show()

    # from sklearn.decomposition import PCA, KernelPCA
    # rr_train_embedding = autoencoder.encoder.predict(rr_train, batch_size=4096).reshape((-1, 30))
    # rr_train_proj = PCA(n_components=2).fit_transform(rr_train_embedding)
    # plt.scatter(*rr_train_proj.T, c=y_train)
    # plt.show()
    # np.save(main_path / "X_test_reconstructed", autoencoder(rr_test).numpy().reshape((-1, 60))))
    # output = autoencoder(X_test[:,:-3].reshape(-1, 60, 1).astype('float32')).reshape((-1, 60))

    # folder_date_latentsmote = "2021-08-11T15:50:26"  # np.datetime64('now') / "2021-04-21T10:00:00" / None
    # if folder_date_latentsmote is None:
    #     folder_dates_latentsmote = [np.datetime64(x) for x in os.listdir(cts.BASE_DIR / "Tom" / "generative_models")]
    #     folder_date_latentsmote = np.max(folder_dates_latentsmote)
    # path_models_latentsmote = cts.BASE_DIR / "Tom" / "generative_models" / str(folder_date_latentsmote)
    # X_train_latent = np.load(path_models_latentsmote / "X_train_latent.npy", allow_pickle=True)
    # X_gen, y_gen = np.load(path_models_latentsmote / "X_latent_smote_inter_5.npy", allow_pickle=True), y_train
    # print(X_gen.shape)
    # # X_gen = X_gen[np.random.choice(list(range(len(X_gen))), 1 * np.sum(y_train==True), replace=False)]
    # # y_gen = np.array([True for i in range(X_gen.shape[0])])
    # X_gen, y_gen = X_gen[X_gen[:, -2] >= 1], y_gen[X_gen[:, -2] >= 1]
    # print(X_gen.shape)
    # X_train, y_train = np.concatenate((X_train_latent, X_gen), axis=0), np.concatenate((y_train, y_gen), axis=0)

    # import tensorflow as tf
    # autoencoder = tf.keras.models.load_model(path_models_latentsmote / 'autoencoder')
    # X_test = np.concatenate((autoencoder.encoder.predict(X_test[:, :-3].reshape(-1, 60, 1).astype('float32'), batch_size=4096).reshape((-1, 30)), X_test[:, -3:]), axis=1)
    # X_ltaf = np.concatenate((autoencoder.encoder.predict(X_ltaf[:, :-3].reshape(-1, 60, 1).astype('float32'), batch_size=4096).reshape((-1, 30)), X_ltaf[:, -3:]), axis=1)
    # X_jpaf = np.concatenate((autoencoder.encoder.predict(X_jpaf[:, :-3].reshape(-1, 60, 1).astype('float32'), batch_size=4096).reshape((-1, 30)), X_jpaf[:, -3:]), axis=1)

    # X_gen_inter_5 = np.load(path_models / "X_gen_inter_5.npy", allow_pickle=True)
    # X_gen_inter_100 = np.load(path_models / "X_gen_inter_100.npy", allow_pickle=True)
    # X_gen_extra_5 = np.load(path_models / "X_gen_extra_5.npy", allow_pickle=True)
    # X_gen_extra_100 = np.load(path_models / "X_gen_extra_100.npy", allow_pickle=True)
    # rr_neighbor_inter_5 = np.load(path_models / "rr_neighbor_inter_5.npy", allow_pickle=True)
    # rr_neighbor_inter_100 = np.load(path_models / "rr_neighbor_inter_100.npy", allow_pickle=True)
    # rr_neighbor_extra_5 = np.load(path_models / "rr_neighbor_extra_5.npy", allow_pickle=True)
    # rr_neighbor_extra_100 = np.load(path_models / "rr_neighbor_extra_100.npy", allow_pickle=True)
    # import matplotlib.pyplot as plt
    # for x, neigh_inter_5, x_gen_inter_5, neigh_inter_100, x_gen_inter_100, neigh_extra_5, x_gen_extra_5, neigh_extra_100, x_gen_extra_100, label in zip(X_train[::500000], rr_neighbor_inter_5[::500000], X_gen_inter_5[::500000], rr_neighbor_inter_100[::500000], X_gen_inter_100[::500000], rr_neighbor_extra_5[::500000], X_gen_extra_5[::500000], rr_neighbor_extra_100[::500000], X_gen_extra_100[::500000], y_train[::500000]):
    #     plt.plot(x[:-3], lw=3, label="input")
    #     plt.plot(x_gen_inter_5[:-3], ls='--', lw=1, label="x_gen_inter_5")
    #     plt.plot(x_gen_inter_100[:-3], ls='--', lw=1, label="x_gen_inter_100")
    #     plt.plot(x_gen_extra_5[:-3], ls='--', lw=1, label="x_gen_extra_5")
    #     plt.plot(x_gen_extra_100[:-3], ls='--', lw=1, label="x_gen_extra_100")
    #     plt.title(f'AF = {label}')
    #     plt.legend()
    #     plt.show()
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #     fig.set_size_inches(10, 10)
    #     ax1.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax1.plot(neigh_inter_5, 'c-', lw=1, label="inter_5")
    #     ax1.plot(x_gen_inter_5[:-3], 'g--', lw=1, label="gen_inter_5")
    #     ax1.legend()
    #     ax2.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax2.plot(neigh_inter_100, 'c-', lw=1, label="inter_100")
    #     ax2.plot(x_gen_inter_100[:-3], 'g--', lw=1, label="gen_inter_100")
    #     ax2.legend()
    #     ax3.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax3.plot(neigh_extra_5, 'c-', lw=1, label="extra_5")
    #     ax3.plot(x_gen_extra_5[:-3], 'g--', lw=1, label="gen_extra_5")
    #     ax3.legend()
    #     ax4.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax4.plot(neigh_extra_100, 'c-', lw=1, label="extra_100")
    #     ax4.plot(x_gen_extra_100[:-3], 'g--', lw=1, label="gen_extra_100")
    #     ax4.legend()
    #     plt.suptitle(f'AF = {label}')
    #     plt.show()
    # for x, neigh_inter_5, x_gen_inter_5, neigh_inter_100, x_gen_inter_100, neigh_extra_5, x_gen_extra_5, neigh_extra_100, x_gen_extra_100, label in zip(X_train[y_train==True][::10000], rr_neighbor_inter_5[y_train==True][::10000], X_gen_inter_5[y_train==True][::10000], rr_neighbor_inter_100[y_train==True][::10000], X_gen_inter_100[y_train==True][::10000], rr_neighbor_extra_5[y_train==True][::10000], X_gen_extra_5[y_train==True][::10000], rr_neighbor_extra_100[y_train==True][::10000], X_gen_extra_100[y_train==True][::10000], y_train[y_train==True][::10000]):
    #     plt.plot(x[:-3], lw=3, label="input")
    #     plt.plot(x_gen_inter_5[:-3], ls='--', lw=1, label="x_gen_inter_5")
    #     plt.plot(x_gen_inter_100[:-3], ls='--', lw=1, label="x_gen_inter_100")
    #     plt.plot(x_gen_extra_5[:-3], ls='--', lw=1, label="x_gen_extra_5")
    #     plt.plot(x_gen_extra_100[:-3], ls='--', lw=1, label="x_gen_extra_100")
    #     plt.title(f'AF = {label}')
    #     plt.legend()
    #     plt.show()
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #     fig.set_size_inches(10, 10)
    #     ax1.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax1.plot(neigh_inter_5, 'c-', lw=1, label="inter_5")
    #     ax1.plot(x_gen_inter_5[:-3], 'g--', lw=1, label="gen_inter_5")
    #     ax1.legend()
    #     ax2.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax2.plot(neigh_inter_100, 'c-', lw=1, label="inter_100")
    #     ax2.plot(x_gen_inter_100[:-3], 'g--', lw=1, label="gen_inter_100")
    #     ax2.legend()
    #     ax3.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax3.plot(neigh_extra_5, 'c-', lw=1, label="extra_5")
    #     ax3.plot(x_gen_extra_5[:-3], 'g--', lw=1, label="gen_extra_5")
    #     ax3.legend()
    #     ax4.plot(x[:-3], 'b-', lw=3, label="input")
    #     ax4.plot(neigh_extra_100, 'c-', lw=1, label="extra_100")
    #     ax4.plot(x_gen_extra_100[:-3], 'g--', lw=1, label="gen_extra_100")
    #     ax4.legend()
    #     plt.suptitle(f'AF = {label}')
    #     plt.show()
from tensorflow.keras.utils import Sequence
import numpy as np

from src.data import data_loading
from src.parsing.ltaf_parser import LTAFDB_Parser


class TimeConcatGen(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        begs = np.where(self.X[:, -3] == 1)[0]
        begs = np.append(begs, len(self.X))
        self.begs = begs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.begs) - 1

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):

        beg, end = self.begs[index], self.begs[index + 1]
        Xb = np.array([self.X[beg: end][:, :-3].astype('float32')])
        yb = np.array([self.y[beg: end]])
        return np.array([Xb]), np.array([yb])



if __name__ == '__main__':
    ltaf_db = LTAFDB_Parser()
    ltaf_pat = np.intersect1d(ltaf_db.non_corrupted_ecg_patients(), ltaf_db.high_sqi_patients())
    X_ltaf, y_ltaf, _, _, feats_ltaf, _ = data_loading.return_train_test(ltaf_db, ltaf_pat, ltaf_pat,
                                                                         normalize=False)

    gen = TimeConcatGen(X_ltaf, y_ltaf, batch_size=1, shuffle=True)
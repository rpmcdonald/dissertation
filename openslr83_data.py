import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import random
import sys

class Mfcc():

    def __init__(self, folder, limit):
        self.folder = folder
        self.accent = folder[:2]
        self.limit = limit
        self.target_size = 16
        self.mfcc_size = 32

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13) # format is (n_mfcc, )
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        list_of_deltas = []
        list_of_d_deltas = []
        wavs = []
        for file in os.listdir(f"..\experiments_data\openslr_83\{self.folder}"):
            if file.endswith(".wav"):
                wavs.append(file)
        #random.seed(1)
        random.Random(4).shuffle(wavs)
        for wav in tqdm(wavs[:self.limit]):
            file_name = f"..\experiments_data\openslr_83\{self.folder}\{wav}"
            mfcc = self.wavtomfcc(file_name)
            delta = librosa.feature.delta(mfcc)
            d_delta = librosa.feature.delta(mfcc, order=2)
            list_of_mfccs.append(mfcc)
            list_of_deltas.append(delta)
            list_of_d_deltas.append(d_delta)
            
        self.list_of_mfccs = list_of_mfccs
        self.list_of_deltas = list_of_deltas
        self.list_of_d_deltas = list_of_d_deltas

    def resize_mfcc(self):
        resized_mfcc = [librosa.util.fix_length(mfcc, size=self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]

        resized_delta = [librosa.util.fix_length(delta, size=self.target_size, axis=1)
                         for delta in self.list_of_deltas]
        resized_delta = [np.vstack((np.zeros((3, self.target_size)), delta)) for delta in resized_delta]

        resized_d_delta = [librosa.util.fix_length(d_delta, size=self.target_size, axis=1)
                         for d_delta in self.list_of_d_deltas]
        resized_d_delta = [np.vstack((np.zeros((3, self.target_size)), d_delta)) for d_delta in resized_d_delta]

        combined = []
        for i in range(len(resized_mfcc)):
            combined.append(np.concatenate((resized_mfcc[i], resized_delta[i])))

        self.X = combined
        #self.X = resized_mfcc
        #sys.exit("Error message")

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS[self.accent], dtype=int)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=0.15)
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.15)
        self.X_train = np.array(X_train).reshape(-1, self.mfcc_size, self.target_size)
        print(self.X_train.shape)
        self.X_test = np.array(X_test).reshape(-1, self.mfcc_size, self.target_size)
        #print(self.X_test.shape)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)

    def standardize_mfcc(self):
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        self.X_train_std = (self.X_train-train_mean)/train_std
        self.X_test_std = (self.X_test-train_mean)/train_std
        #self.X_val_std = (self.X_val-train_mean)/train_std

    def pca(self):
        # test_size = len(self.X_test_std)
        # combined = np.concatenate((self.X_train_std, self.X_test_std))
        # pca = PCA(n_components=8)
        # comb_pca = [pca.fit_transform(x) for x in combined]
        # self.X_train_std = comb_pca[:-test_size]
        # self.X_test_std = comb_pca[-test_size:]

        
        x_train_pca = []
        for mfcc in self.X_train_std:
            pca = PCA()
            x_train_pca.append(pca.fit_transform(mfcc))
        x_train_pca = np.array(x_train_pca).reshape(-1, self.mfcc_size, self.target_size)
        self.X_train_std = x_train_pca

        x_test_pca = []
        for mfcc in self.X_test_std:
            pca = PCA()
            x_test_pca.append(pca.fit_transform(mfcc))
        x_test_pca = np.array(x_test_pca).reshape(-1, self.mfcc_size, self.target_size)
        self.X_test_std = x_test_pca

        # x_train_pca = [pca.fit_transform(x) for x in self.X_train_std]
        # x_train_pca = np.array(x_train_pca).reshape(-1, self.mfcc_size, self.target_size)
        # print(self.X_train_std.shape, x_train_pca.shape)
        # print(self.X_train_std[0][0][0], x_train_pca[0][0][0])
        #self.X_train_std = x_train_pca

        # pca = PCA(0.95)
        # x_test_pca = [pca.fit_transform(x) for x in self.X_test_std]
        # x_test_pca = np.array(x_test_pca).reshape(-1, int(self.mfcc_size/2), self.target_size)
        # self.X_test_std = x_test_pca

        # print(self.X_train_std.shape, principalComponents.shape)
        # print(len(principalComponents), len(principalComponents[0]), len(principalComponents[0][0]))
        # print(self.X_train_std[0][0][0], principalComponents[0][0][0])
        # sys.exit("Error message")

    def save_mfccs(self):
        MFCCS[self.accent] = {
            "x_train": self.X_train_std,
            "x_test": self.X_test_std,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }



if __name__ == '__main__':
    ACCENTS = {"we": 0, "ir": 1, "mi": 2, "no": 3, "sc": 4, "so": 5}
    MFCCS = {}
    random.seed(1)
    folders = [f.name for f in os.scandir("..\experiments_data\openslr_83") if f.is_dir()]
    for f in folders:
        if f == "indv" or f == "irish_english":
            continue
        print(f)
        # if f[-6:] == "female": # update this to be nicer but works for now
        #     continue
        mfcc = Mfcc(f, limit=650)
        mfcc.create_mfcc()
        mfcc.resize_mfcc()
        mfcc.label_samples()
        mfcc.split_data()
        mfcc.standardize_mfcc()
        mfcc.pca()
        mfcc.save_mfccs()

    keys = list(MFCCS.keys())

    X_train_std = MFCCS[keys[0]]["x_train"]
    X_test_std = MFCCS[keys[0]]["x_test"]
    y_train = MFCCS[keys[0]]["y_train"]
    y_test = MFCCS[keys[0]]["y_test"]

    for k in keys[1:]:
        X_train_std = np.concatenate((X_train_std, MFCCS[k]["x_train"]))
        X_test_std = np.concatenate((X_test_std, MFCCS[k]["x_test"]))
        y_train = np.concatenate((y_train, MFCCS[k]["y_train"]))
        y_test = np.concatenate((y_test, MFCCS[k]["y_test"]))

    np.save(f'mfccs/X_train_openslr83.npy', X_train_std)
    np.save(f'mfccs/X_test_openslr83.npy', X_test_std)
    np.save(f'mfccs/y_train_openslr83.npy', y_train)
    np.save(f'mfccs/y_test_openslr83.npy', y_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
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
            combined.append(np.concatenate((resized_mfcc[i], resized_delta[i], resized_d_delta[i])))

        #self.X = combined
        self.X = resized_mfcc
        #sys.exit("Error message")

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS[self.accent], dtype=int)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=0.15)
        #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle = False, test_size=0.25)
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.15)
        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.25)
        self.X_train = np.array(X_train).reshape(-1, 16, self.target_size)
        #print(self.X_train.shape)
        self.X_test = np.array(X_test).reshape(-1, 16, self.target_size)
        #print(self.X_test.shape)
        #self.X_val = np.array(X_val).reshape(-1, 16, self.target_size)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)
        #self.y_val = np.array(y_val).reshape(-1,1)

    def standardize_mfcc(self):
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        self.X_train_std = (self.X_train-train_mean)/train_std
        self.X_test_std = (self.X_test-train_mean)/train_std
        #self.X_val_std = (self.X_val-train_mean)/train_std

    def oversample(self):
        temp = pd.DataFrame({'mfcc_id':range(self.X_train_std.shape[0]), 'accent':self.y_train.reshape(-1)})
        temp_1 = temp[temp['accent']==1]
        idx = list(temp_1['mfcc_id'])*3
        idx = idx + list(temp_1.sample(frac=.8)['mfcc_id'])
        print("oversample")
        print(self.X_train_std.shape)
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        print(self.X_train_std.shape)
        print(self.y_train.shape)
        self.y_train = np.vstack((self.y_train, np.ones(232).reshape(-1,1)))
        print(self.y_train.shape)

    def save_mfccs(self):
        MFCCS[self.accent] = {
            "x_train": self.X_train_std,
            "x_test": self.X_test_std,
            #"x_val": self.X_val_std,
            "y_train": self.y_train,
            "y_test": self.y_test,
            #"y_val": self.y_val
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
        # mfcc.oversample()
        mfcc.save_mfccs()

    keys = list(MFCCS.keys())

    X_train_std = MFCCS[keys[0]]["x_train"]
    X_test_std = MFCCS[keys[0]]["x_test"]
    #X_val_std = MFCCS[keys[0]]["x_val"]
    y_train = MFCCS[keys[0]]["y_train"]
    y_test = MFCCS[keys[0]]["y_test"]
    #y_val = MFCCS[keys[0]]["y_val"]

    for k in keys[1:]:
        X_train_std = np.concatenate((X_train_std, MFCCS[k]["x_train"]))
        X_test_std = np.concatenate((X_test_std, MFCCS[k]["x_test"]))
        #X_val_std = np.concatenate((X_val_std, MFCCS[k]["x_val"]))
        y_train = np.concatenate((y_train, MFCCS[k]["y_train"]))
        y_test = np.concatenate((y_test, MFCCS[k]["y_test"]))
        #y_val = np.concatenate((y_val, MFCCS[k]["y_val"]))

    np.save(f'mfccs/X_train_openslr83.npy', X_train_std)
    np.save(f'mfccs/X_test_openslr83.npy', X_test_std)
    #np.save(f'mfccs/X_val_openslr83.npy', X_val_std)
    np.save(f'mfccs/y_train_openslr83.npy', y_train)
    np.save(f'mfccs/y_test_openslr83.npy', y_test)
    #np.save(f'mfccs/y_val_openslr83.npy', y_val)
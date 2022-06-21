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

class Mfcc():

    def __init__(self, folder):
        self.folder = folder
        self.accent = folder[:2]

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13)
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        wavs = []
        for file in os.listdir(f"..\experiments_data\openslr_83\{self.folder}"):
            if file.endswith(".wav"):
                wavs.append(file)
        
        for wav in tqdm(wavs):
            file_name = f"..\experiments_data\openslr_83\{self.folder}\{wav}"
            mfcc = self.wavtomfcc(file_name)
            list_of_mfccs.append(mfcc)
        self.list_of_mfccs = list_of_mfccs

    def resize_mfcc(self):
        self.target_size = 64
        resized_mfcc = [librosa.util.fix_length(mfcc, size=self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        self.X = resized_mfcc

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS[self.accent], dtype=int)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.15)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.15)
        self.X_train = np.array(X_train).reshape(-1, 16, self.target_size)
        self.X_test = np.array(X_test).reshape(-1, 16, self.target_size)
        self.X_val = np.array(X_val).reshape(-1, 16, self.target_size)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)
        self.y_val = np.array(y_val).reshape(-1,1)

    def standardize_mfcc(self):
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        self.X_train_std = (self.X_train-train_mean)/train_std
        self.X_test_std = (self.X_test-train_mean)/train_std
        self.X_val_std = (self.X_val-train_mean)/train_std

    def oversample(self):
        temp = pd.DataFrame({'mfcc_id':range(self.X_train_std.shape[0]), 'accent':self.y_train.reshape(-1)})
        temp_1 = temp[temp['accent']==1]
        idx = list(temp_1['mfcc_id'])*3
        idx = idx + list(temp_1.sample(frac=.8)['mfcc_id'])
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        self.y_train = np.vstack((self.y_train, np.ones(232).reshape(-1,1)))

    def save_mfccs(self):
        # np.save(f'../experiments_data/mozilla/mfccs/X_train_moz_{self.accent}.npy', self.X_train_std)
        # np.save(f'../experiments_data/mozilla/mfccs/X_test_moz_{self.accent}.npy', self.X_test_std)
        # np.save(f'../experiments_data/mozilla/mfccs/X_val_moz_{self.accent}.npy', self.X_val_std)
        # np.save(f'../experiments_data/mozilla/mfccs/y_train_moz_{self.accent}.npy', self.y_train)
        # np.save(f'../experiments_data/mozilla/mfccs/y_test_moz_{self.accent}.npy', self.y_test)
        # np.save(f'../experiments_data/mozilla/mfccs/y_val_moz_{self.accent}.npy', self.y_val)
        MFCCS[self.accent] = {
            "x_train": self.X_train_std,
            "x_test": self.X_test_std,
            "x_val": self.X_val_std,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_val": self.y_val
        }




if __name__ == '__main__':
    ACCENTS = {"we": 0, "ir": 1, "mi": 2, "no": 3, "sc": 4, "so": 5}
    MFCCS = {}
    folders = [f.name for f in os.scandir("..\experiments_data\openslr_83") if f.is_dir()]
    for f in folders:
        print(f)
        if f[-6:] == "female": # update this to be nicer but works for now
            continue
        mfcc = Mfcc(f)
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
    X_val_std = MFCCS[keys[0]]["x_val"]
    y_train = MFCCS[keys[0]]["y_train"]
    y_test = MFCCS[keys[0]]["y_test"]
    y_val = MFCCS[keys[0]]["y_val"]

    for k in keys[1:]:
        X_train_std = np.concatenate((X_train_std, MFCCS[k]["x_train"]))
        X_test_std = np.concatenate((X_test_std, MFCCS[k]["x_test"]))
        X_val_std = np.concatenate((X_val_std, MFCCS[k]["x_val"]))
        y_train = np.concatenate((y_train, MFCCS[k]["y_train"]))
        y_test = np.concatenate((y_test, MFCCS[k]["y_test"]))
        y_val = np.concatenate((y_val, MFCCS[k]["y_val"]))

    np.save(f'mfccs/X_train_openslr83.npy', X_train_std)
    np.save(f'mfccs/X_test_openslr83.npy', X_test_std)
    np.save(f'mfccs/X_val_openslr83.npy', X_val_std)
    np.save(f'mfccs/y_train_openslr83.npy', y_train)
    np.save(f'mfccs/y_test_openslr83.npy', y_test)
    np.save(f'mfccs/y_val_openslr83.npy', y_val)
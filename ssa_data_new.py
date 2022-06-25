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


def clean_df(file):
    df = pd.read_csv(file)
    output_df = pd.DataFrame()
    for accent in ACCENTS:
        output_df = pd.concat([output_df, df[df['native_language']==accent]], ignore_index=True)
    output_df.drop(['age', 'age_onset', 'birthplace', 'sex', 'speakerid', 'country', 'file_missing?'], axis=1, inplace=True)
    output_df.rename(columns={"native_language": "accent"}, inplace=True)
    return output_df


class Mfcc():

    def __init__(self, df, accent, limit, test_size):
        self.df = df
        self.accent = accent
        self.col = "filename"
        self.limit = limit
        self.target_size = 8
        self.mfcc_size = 26
        self.test_size = test_size

    def mp3towav(self):
        print(accent)
        accent_df = self.df[self.df['accent']==self.accent]
        for filename in tqdm(accent_df[self.col]):
            pydub.AudioSegment.from_mp3(f"../experiments_data/ssa/recordings/{filename}.mp3").export(f"../experiments_data/ssa/wavs/{self.accent}/{filename}.wav", format="wav")

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13) # format is (n_mfcc, )
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        list_of_deltas = []
        list_of_d_deltas = []
        wavs = []
        for file in os.listdir(f"..\experiments_data\ssa\wavs\{self.accent}"):
            if file.endswith(".wav"):
                wavs.append(file)
        random.shuffle(wavs)
        for wav in tqdm(wavs[:self.limit]):
            file_name = f"..\experiments_data\ssa\wavs\{self.accent}\{wav}"
            mfcc = self.wavtomfcc(file_name)
            delta = librosa.feature.delta(mfcc)
            d_delta = librosa.feature.delta(mfcc, order=2)
            list_of_mfccs.append(mfcc)
            list_of_deltas.append(delta)
            list_of_d_deltas.append(d_delta)
            
        self.list_of_mfccs = list_of_mfccs
        self.list_of_deltas = list_of_deltas
        self.list_of_d_deltas = list_of_d_deltas

    def save_mfccs(self):
        MFCCS[self.accent] = {
            "x_train": self.X_train_std,
            "x_test": self.X_test_std,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }



if __name__ == '__main__':
    ACCENTS = ["english", "arabic", "spanish"]
    df = clean_df('../experiments_data/ssa/speakers_all.csv')
    print("DF created")

    MFCCS = {}
    random.seed(1)
    for accent in ACCENTS:
        mfcc = Mfcc(df=df, accent=accent, limit=20, test_size=10)
        # mfcc.mp3towav()
        mfcc.create_mfcc()
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

    print(X_train_std[0][7])
    print(X_train_std[0][22])

    np.save(f'mfccs/X_train_ssa.npy', X_train_std)
    np.save(f'mfccs/X_test_ssa.npy', X_test_std)
    np.save(f'mfccs/y_train_ssa.npy', y_train)
    np.save(f'mfccs/y_test_ssa.npy', y_test)
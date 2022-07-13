from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as py_go
import plotly.offline as py_o
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
import speechpy
from tqdm import tqdm
import os
import random
import sys
import math


def clean_df(file):
    df = pd.read_csv(file)
    output_df = pd.DataFrame()
    for accent in ACCENTS:
        output_df = pd.concat([output_df, df[(df['accent']==accent)]], ignore_index=True)
    output_df.drop(['text', 'up_votes', 'down_votes', 'age', 'gender', 'duration'], axis=1, inplace=True)
    return output_df


class Spectrogram():

    def __init__(self, df, accent, limit, test_size, target_size, randomise):
        self.df = df
        self.accent = accent
        self.col = "filename"
        self.limit = limit
        self.target_size = target_size
        self.test_size = test_size
        self.randomise = randomise

    def mp3towav(self):
        accent_df = self.df[self.df['accent']==self.accent]
        for filename in tqdm(accent_df[self.col]):
            pydub.AudioSegment.from_mp3(f"../experiments_data/mozilla/org_mp3/{filename}.mp3").export(f"../experiments_data/mozilla/wavs/{filename}.wav", format="wav")


    def wav_to_spectrogram(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr)
        return spectrogram

    def create_spectrograms(self):
        accent_df = self.df[self.df['accent']==self.accent]
        if self.randomise:
            accent_df = accent_df.sample(frac=1)
        else:
            accent_df = accent_df.sample(frac=1, random_state=0)
        
        list_of_specs = []
        for wav in accent_df[self.col][:self.limit]:
            file_name = f"..\experiments_data\mozilla\wavs\{wav}.wav"
            spectrogram = self.wav_to_spectrogram(file_name)
            list_of_specs.append(spectrogram)
        self.list_of_spectrograms = list_of_specs

        print("len of spectrograms", len(self.list_of_spectrograms), "size:", self.list_of_spectrograms[0].shape)
        # librosa.display.specshow(librosa.power_to_db(self.list_of_spectrograms[0], ref=np.max))
        # plt.show()

    def resize_spectrograms(self):
        resized = [librosa.util.fix_length(mfcc, size=self.target_size, axis=1) for mfcc in self.list_of_spectrograms]
        self.X = resized

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS.index(self.accent), dtype=int)

    def split_data(self):
        if self.randomise:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=self.test_size)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=int(self.test_size/2))
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=self.test_size)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle = False, test_size=int(self.test_size/2))
        self.X_train = np.array(X_train).reshape(-1, 128, self.target_size)
        print("X_train shape", self.X_train.shape)
        self.X_test = np.array(X_test).reshape(-1, 128, self.target_size)
        print("X_test shape", self.X_test.shape)
        self.X_val = np.array(X_val).reshape(-1, 128, self.target_size)
        print("X_val shape", self.X_val.shape)
        self.y_train = np.array(y_train).reshape(-1,1)
        self.y_test = np.array(y_test).reshape(-1,1)
        self.y_val = np.array(y_val).reshape(-1,1)

    def save_mfccs(self):
        SPECTROGRAMS[self.accent[0]] = {
            "x_train": self.X_train,
            "x_test": self.X_test,
            "x_val": self.X_val,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_val": self.y_val
        }


if __name__ == '__main__':
    #ACCENTS = ["canada", "australia", "indian"]
    ACCENTS = ["canada", "australia"]
    df = clean_df('..\experiments_data\mozilla\\validated_full.csv')
    print("DF created")

    SPECTROGRAMS = {}
    target_size = 256
    randomise = False

    if randomise == False:
        random.seed(1)

    for accent in ACCENTS:
        print(accent)
        spectrograms = Spectrogram(df=df, 
                    accent=accent, 
                    limit=4000, 
                    test_size=600, 
                    target_size=target_size, 
                    randomise=randomise
                    )
        # spectrograms.mp3towav()
        spectrograms.create_spectrograms()
        spectrograms.resize_spectrograms()
        spectrograms.label_samples()
        spectrograms.split_data()
        spectrograms.save_mfccs()

    keys = list(SPECTROGRAMS.keys())

    X_train = SPECTROGRAMS[keys[0]]["x_train"]
    X_test = SPECTROGRAMS[keys[0]]["x_test"]
    X_val = SPECTROGRAMS[keys[0]]["x_val"]
    y_train = SPECTROGRAMS[keys[0]]["y_train"]
    y_test = SPECTROGRAMS[keys[0]]["y_test"]
    y_val = SPECTROGRAMS[keys[0]]["y_val"]

    for k in keys[1:]:
        X_train = np.concatenate((X_train, SPECTROGRAMS[k]["x_train"]))
        X_test = np.concatenate((X_test, SPECTROGRAMS[k]["x_test"]))
        X_val = np.concatenate((X_val, SPECTROGRAMS[k]["x_val"]))
        y_train = np.concatenate((y_train, SPECTROGRAMS[k]["y_train"]))
        y_test = np.concatenate((y_test, SPECTROGRAMS[k]["y_test"]))
        y_val = np.concatenate((y_val, SPECTROGRAMS[k]["y_val"]))

    # ---Standardise
    # Whiten over each file seperately
    X_train_std=whiten(X_train.transpose()).transpose()
    X_test_std=whiten(X_test.transpose()).transpose()
    X_val_std=whiten(X_val.transpose()).transpose()

    np.save(f'spectrograms/X_train_moz.npy', X_train_std)
    np.save(f'spectrograms/X_test_moz.npy', X_test_std)
    np.save(f'spectrograms/X_val_moz.npy', X_val_std)
    np.save(f'spectrograms/y_train_moz.npy', y_train)
    np.save(f'spectrograms/y_test_moz.npy', y_test)
    np.save(f'spectrograms/y_val_moz.npy', y_val)
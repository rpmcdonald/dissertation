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


class Mfcc():

    def __init__(self, df, accent, limit, test_size, target_size, randomise):
        self.df = df
        self.accent = accent
        self.col = "filename"
        self.limit = limit
        self.target_size = target_size
        self.test_size = test_size
        self.randomise = randomise
        self.names = []

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
            self.names.append(wav)
            file_name = f"..\experiments_data\mozilla\wavs\{wav}.wav"
            spectrogram = self.wav_to_spectrogram(file_name)
            list_of_specs.append(spectrogram)
        self.list_of_spectrograms = list_of_specs

        print("len of spectrograms", len(self.list_of_spectrograms))
        # librosa.display.specshow(librosa.power_to_db(self.list_of_spectrograms[0], ref=np.max))
        # plt.show()

    def resize_spectrograms(self):
        combined = []
        for i in range(len(self.list_of_mfccs)):
            comb = np.concatenate((self.list_of_mfccs[i], self.list_of_deltas[i], self.list_of_d_deltas[i]))
            if comb.shape[1] < target_size: # This will remove anything which is smaller than the target size so it isn't padded with zeroes
                pass
            else:
                combined.append(comb)
        if not self.gkf:
            resized = [librosa.util.fix_length(mfcc, size=self.target_size, axis=1) for mfcc in combined]
        else:
            resized = [librosa.util.fix_length(mfcc, size=self.frames, axis=1) for mfcc in combined]
        print("len of mfccs reshaped", len(self.list_of_mfccs))
        print("Single mfcc reshaped shape:", resized[0].shape)
        
        # resized = np.array(resized).reshape(-1, self.mfcc_size, self.target_size)
        # print(resized.shape)
        if self.cmvn:
            resized = [speechpy.processing.cmvn(x, variance_normalization=False) for x in resized]
        self.X = resized

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS.index(self.accent), dtype=int)

    def split_data(self):
        if self.randomise:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=self.test_size)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=int(self.test_size/3))
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=self.test_size)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle = False, test_size=int(self.test_size/3))
        
        if not self.gkf:
            self.X_train = np.array(X_train).reshape(-1, self.mfcc_size, self.target_size)
            print("X_train shape", self.X_train.shape)
            self.X_test = np.array(X_test).reshape(-1, self.mfcc_size, self.target_size)
            print("X_test shape", self.X_test.shape)
            self.X_val = np.array(X_val).reshape(-1, self.mfcc_size, self.target_size)
            print("X_val shape", self.X_val.shape)
        else:
            self.X_train = np.array(X_train).reshape(-1, self.mfcc_size, self.frames)
            print("X_train shape", self.X_train.shape)
            self.X_test = np.array(X_test).reshape(-1, self.mfcc_size, self.frames)
            print("X_test shape", self.X_test.shape)
            self.X_val = np.array(X_val).reshape(-1, self.mfcc_size, self.frames)
            print("X_val shape", self.X_val.shape)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)
        self.y_val = np.array(y_val).reshape(-1,1)

    def save_mfccs(self):
        MFCCS[self.accent[0]] = {
            "x_train": self.X_train,
            "x_test": self.X_test,
            "x_val": self.X_val,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "y_val": self.y_val
        }
    
    def return_names(self):
        return self.names


if __name__ == '__main__':
    #ACCENTS = ["canada", "australia", "indian"]
    ACCENTS = ["canada", "australia"]
    df = clean_df('..\experiments_data\mozilla\\validated_full.csv')
    print("DF created")

    MFCCS = {}
    names = []
    target_size=128
    randomise = False

    if randomise == False:
        random.seed(1)

    for accent in ACCENTS:
        print(accent)
        mfcc = Mfcc(df=df, 
                    accent=accent, 
                    limit=200, 
                    test_size=30, 
                    target_size=target_size, 
                    randomise=randomise
                    )
        # mfcc.mp3towav()
        mfcc.create_spectrograms()
        # mfcc.resize_spectrograms()
        # mfcc.label_samples()
        # mfcc.split_data()
        # mfcc.save_mfccs()
        # names += mfcc.return_names()
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm
import os
import random
import sys


def clean_df(file):
    df = pd.read_csv(file)
    output_df = pd.DataFrame()
    for accent in ACCENTS:
        output_df = pd.concat([output_df, df[df['country']==accent]], ignore_index=True)
    output_df.drop(['age', 'age_onset', 'birthplace', 'sex', 'speakerid', 'native_language', 'file_missing?'], axis=1, inplace=True)
    output_df.rename(columns={"country": "accent"}, inplace=True)
    return output_df


class Mfcc():

    def __init__(self, df, accent, limit, test_size, mfcc_size, target_size):
        self.df = df
        self.accent = accent
        self.col = "filename"
        self.limit = limit
        self.target_size = target_size
        self.mfcc_size = mfcc_size
        self.test_size = test_size

    def mp3towav(self):
        accent_df = self.df[self.df['accent']==self.accent]
        for filename in tqdm(accent_df[self.col]):
            pydub.AudioSegment.from_mp3(f"../experiments_data/ssa/recordings/{filename}.mp3").export(f"../experiments_data/ssa/wavs/{self.accent}/{filename}.wav", format="wav")

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        n_fft=2048
        hop_length=512
        win_length=hop_length*4
        # This is the same as not changing anything
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, win_length=win_length, hop_length=hop_length, n_fft=n_fft) # format is (n_mfcc, )
        delta = librosa.feature.delta(mfcc)
        d_delta = librosa.feature.delta(mfcc, order=2)
        #print(mfcc.shape)
        return mfcc, delta, d_delta

    def create_mfcc(self):
        list_of_mfccs = []
        list_of_deltas = []
        list_of_d_deltas = []
        wavs = []
        for file in os.listdir(f"..\experiments_data\ssa\wavs\{self.accent}"):
            if file.endswith(".wav"):
                wavs.append(file)
        random.shuffle(wavs)
        #for wav in tqdm(wavs[:self.limit]):
        for wav in wavs[:self.limit]:
            file_name = f"..\experiments_data\ssa\wavs\{self.accent}\{wav}"
            mfcc, delta, d_delta = self.wavtomfcc(file_name)

            list_of_mfccs.append(mfcc)
            list_of_deltas.append(delta)
            list_of_d_deltas.append(d_delta)
            
        self.list_of_mfccs = list_of_mfccs
        self.list_of_deltas = list_of_deltas
        self.list_of_d_deltas = list_of_d_deltas

        print(self.list_of_mfccs[0].shape, self.list_of_mfccs[0][0][0])

    def resize_mfcc(self):
        # First part sets the length of all mfccs to the target_size and cuts off all data after that
        resized_mfcc = [librosa.util.fix_length(mfcc, size=self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        print(resized_mfcc[0].shape, resized_mfcc[0][0][0], resized_mfcc[0][0][-1])
        # Second part adds in 3 arrays of zeros to the start of the mfcc
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        # print(resized_mfcc[0].shape, resized_mfcc[0][0][0], resized_mfcc[0][3][0], resized_mfcc[0][0][-1])
        
        
        resized_delta = [librosa.util.fix_length(delta, size=self.target_size, axis=1)
                         for delta in self.list_of_deltas]
        resized_delta = [np.vstack((np.zeros((3, self.target_size)), delta)) for delta in resized_delta]

        # resized_d_delta = [librosa.util.fix_length(d_delta, size=self.target_size, axis=1)
        #                  for d_delta in self.list_of_d_deltas]
        # resized_d_delta = [np.vstack((np.zeros((3, self.target_size)), d_delta)) for d_delta in resized_d_delta]

        combined = []
        for i in range(len(resized_mfcc)):
            combined.append(np.concatenate((resized_mfcc[i], resized_delta[i])))

        self.X = combined
        # self.X = resized_mfcc
        #sys.exit("Error message")

    def label_samples(self):
        self.y = np.full(shape=len(self.X), fill_value=ACCENTS.index(self.accent), dtype=int)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=self.test_size)
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=self.test_size)
        self.X_train = np.array(X_train).reshape(-1, self.mfcc_size, self.target_size)
        print(self.X_train.shape)
        self.X_test = np.array(X_test).reshape(-1, self.mfcc_size, self.target_size)
        print(self.X_test.shape)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)

    def save_mfccs(self):
        MFCCS[self.accent] = {
            "x_train": self.X_train,
            "x_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }



if __name__ == '__main__':
    #ACCENTS = ["usa", "china", "uk", "india", "canada", "south korea"]
    ACCENTS = ["usa", "china", "india"]
    df = clean_df('../experiments_data/ssa/speakers_all.csv')
    print("DF created")

    MFCCS = {}
    #random.seed(1)
    target_size=128
    mfcc_size=32
    run_pca = True

    for accent in ACCENTS:
        print(accent)
        mfcc = Mfcc(df=df, accent=accent, limit=50, test_size=10, target_size=target_size, mfcc_size=mfcc_size)
        # mfcc.mp3towav()
        mfcc.create_mfcc()
        mfcc.resize_mfcc()
        mfcc.label_samples()
        mfcc.split_data()
        #mfcc.standardize_mfcc()
        mfcc.save_mfccs()

    keys = list(MFCCS.keys())

    X_train = MFCCS[keys[0]]["x_train"]
    X_test = MFCCS[keys[0]]["x_test"]
    y_train = MFCCS[keys[0]]["y_train"]
    y_test = MFCCS[keys[0]]["y_test"]

    for k in keys[1:]:
        X_train = np.concatenate((X_train, MFCCS[k]["x_train"]))
        X_test = np.concatenate((X_test, MFCCS[k]["x_test"]))
        y_train = np.concatenate((y_train, MFCCS[k]["y_train"]))
        y_test = np.concatenate((y_test, MFCCS[k]["y_test"]))

    # Standardise
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train_std = (X_train-train_mean)/train_std
    X_test_std = (X_test-train_mean)/train_std
    print("std:", X_train_std.shape)
    # print(X_train_std[0].shape, X_train_std[0][0][0], X_train_std[0][0][-1])
    # print(X_train[0][3])
    # print(X_train_std[0][3])
    #sys.exit("Error message")

    # PCA
    if run_pca == True:
        # n_components = int((target_size*mfcc_size))     # If you do not put in a n_components it will give target_size*mfcc_size, this maybe isnt true
        #n_components = 10
        #pca = PCA(n_components=n_components)
        pca = PCA()

        nsamples, nx, ny = X_train_std.shape
        X_train_reshape = X_train_std.reshape((nsamples,nx*ny))
        x_train_pca = pca.fit_transform(X_train_reshape)
        print(X_train_std.shape, x_train_pca.shape)
        #x_train_pca = np.array(x_train_pca).reshape(-1, mfcc_size, target_size)
        # print(self.X_train_std.shape, x_train_pca.shape)
        # print(self.X_train_std[0][0][0], x_train_pca[0][0][0])
        X_train_std = x_train_pca

        nsamples, nx, ny = X_test_std.shape
        X_test_reshape = X_test_std.reshape((nsamples,nx*ny))
        x_test_pca = pca.transform(X_test_reshape)
        #x_test_pca = np.array(x_test_pca).reshape(-1, mfcc_size, target_size)
        X_test_std = x_test_pca

    plt.scatter(x_train_pca[:,0], x_train_pca[:,1], c=y_train)
    plt.show()

    plt.scatter(X_train_std[:,0], X_train_std[:,1], c=y_train)
    plt.show()


    np.save(f'mfccs/X_train_ssa.npy', X_train_std)
    np.save(f'mfccs/X_test_ssa.npy', X_test_std)
    np.save(f'mfccs/y_train_ssa.npy', y_train)
    np.save(f'mfccs/y_test_ssa.npy', y_test)
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

    def __init__(self, df, accent, limit, test_size, mfcc_size, target_size, randomise, get_key_frames, remove_silence, remove_silence_percent, cmvn):
        self.df = df
        self.accent = accent
        self.col = "filename"
        self.limit = limit
        self.target_size = target_size
        self.mfcc_size = mfcc_size
        self.test_size = test_size
        self.randomise = randomise
        self.gkf = get_key_frames
        self.rem_silence = remove_silence
        self.rem_silence_percent = remove_silence_percent
        self.cmvn = cmvn
        self.frames = 10
        self.names = []

    def mp3towav(self):
        accent_df = self.df[self.df['accent']==self.accent]
        for filename in tqdm(accent_df[self.col]):
            pydub.AudioSegment.from_mp3(f"../experiments_data/mozilla/org_mp3/{filename}.mp3").export(f"../experiments_data/mozilla/wavs/{filename}.wav", format="wav")

    def get_key_frames(self, mfcc, delta, d_delta, rms, n):
        rms = rms.reshape(-1)
        posns = list(zip(*sorted( [(x,i) for (i,x) in enumerate(rms)], 
                    reverse=True )[:n*3] ))[1] 
        # Make sure highest values are not next to each other
        cleaned_pos = []
        for i in posns:
            if len(cleaned_pos) == n:
                break
            for j in cleaned_pos:
                if i == j+1 or i == j-1:
                    break
            else:
                cleaned_pos.append(i)
        return_mfcc = []
        return_delta = []
        return_d_delta = []
        for i in range(len(mfcc)):
            temp_mfcc = []
            temp_delta = []
            temp_d_delta = []
            for n in cleaned_pos:
                temp_mfcc.append(mfcc[i][n])
                temp_delta.append(delta[i][n])
                temp_d_delta.append(d_delta[i][n])
            return_mfcc.append(temp_mfcc)
            return_delta.append(temp_delta)
            return_d_delta.append(temp_d_delta)

        return np.array(return_mfcc), np.array(return_delta), np.array(return_d_delta)

    def remove_silence(self, mfcc, delta, d_delta, rms, rem_silence_percent):
        rms = rms.reshape(-1)
        _, data_length = mfcc.shape
        data_length = math.ceil(data_length * (1-rem_silence_percent))
        posns = list(zip(*sorted( [(x,i) for (i,x) in enumerate(rms)], 
                    reverse=True )[:data_length] ))[1]
        posns = sorted(posns)
        return_mfcc = []
        return_delta = []
        return_d_delta = []
        for i in range(len(mfcc)):
            temp_mfcc = []
            temp_delta = []
            temp_d_delta = []
            for n in posns:
                temp_mfcc.append(mfcc[i][n])
                temp_delta.append(delta[i][n])
                temp_d_delta.append(d_delta[i][n])
            return_mfcc.append(temp_mfcc)
            return_delta.append(temp_delta)
            return_d_delta.append(temp_d_delta)
            
        return np.array(return_mfcc), np.array(return_delta), np.array(return_d_delta)

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        n_fft=2048
        hop_length=512
        win_length=2048
        # defaults n_fft=2048 hop_length=512 win_length=2048
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, win_length=win_length, hop_length=hop_length, n_fft=n_fft) # format is (n_mfcc, )
        delta = librosa.feature.delta(mfcc)
        d_delta = librosa.feature.delta(mfcc, order=2)
        if self.gkf:
            rms = librosa.feature.rms(y=wave)
            mfcc, delta, d_delta = self.get_key_frames(mfcc, delta, d_delta, rms, self.frames)
        if self.rem_silence:
            rms = librosa.feature.rms(y=wave)
            mfcc, delta, d_delta = self.remove_silence(mfcc, delta, d_delta, rms, self.rem_silence_percent)
        return mfcc, delta, d_delta

    def create_mfcc(self):
        list_of_mfccs = []
        list_of_deltas = []
        list_of_d_deltas = []
        accent_df = self.df[self.df['accent']==self.accent]
        if self.randomise:
            accent_df = accent_df.sample(frac=1)
        else:
            accent_df = accent_df.sample(frac=1, random_state=0)
        for wav in accent_df[self.col][:self.limit]:
            self.names.append(wav)
            file_name = f"..\experiments_data\mozilla\wavs\{wav}.wav"
            mfcc, delta, d_delta = self.wavtomfcc(file_name)

            list_of_mfccs.append(mfcc)
            list_of_deltas.append(delta)
            list_of_d_deltas.append(d_delta)
            
        self.list_of_mfccs = list_of_mfccs
        self.list_of_deltas = list_of_deltas
        self.list_of_d_deltas = list_of_d_deltas

        print("len of mfccs", len(self.list_of_mfccs))
        print("Single mfcc shape:", self.list_of_mfccs[0].shape)

    def resize_mfcc(self):
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
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=int(self.test_size/2))
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle = False, test_size=self.test_size)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle = False, test_size=int(self.test_size/2))
        
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
    target_size=256
    mfcc_size=39
    randomise = False
    get_key_frames = False
    remove_silence = False
    remove_silence_percent = 0.15
    cmvn = False

    split_files = False
    split_size=64

    if randomise == False:
        random.seed(1)

    for accent in ACCENTS:
        print(accent)
        mfcc = Mfcc(df=df, 
                    accent=accent, 
                    limit=4000, 
                    test_size=600, 
                    target_size=target_size, 
                    mfcc_size=mfcc_size, 
                    randomise=randomise, 
                    get_key_frames=get_key_frames,
                    remove_silence=remove_silence,
                    remove_silence_percent=remove_silence_percent,
                    cmvn=cmvn)
        # mfcc.mp3towav()
        mfcc.create_mfcc()
        mfcc.resize_mfcc()
        mfcc.label_samples()
        mfcc.split_data()
        mfcc.save_mfccs()
        names += mfcc.return_names()

    keys = list(MFCCS.keys())

    X_train = MFCCS[keys[0]]["x_train"]
    X_test = MFCCS[keys[0]]["x_test"]
    X_val = MFCCS[keys[0]]["x_val"]
    y_train = MFCCS[keys[0]]["y_train"]
    y_test = MFCCS[keys[0]]["y_test"]
    y_val = MFCCS[keys[0]]["y_val"]

    for k in keys[1:]:
        X_train = np.concatenate((X_train, MFCCS[k]["x_train"]))
        X_test = np.concatenate((X_test, MFCCS[k]["x_test"]))
        X_val = np.concatenate((X_val, MFCCS[k]["x_val"]))
        y_train = np.concatenate((y_train, MFCCS[k]["y_train"]))
        y_test = np.concatenate((y_test, MFCCS[k]["y_test"]))
        y_val = np.concatenate((y_val, MFCCS[k]["y_val"]))

    # Split the files into smaller chunks
    if split_files:
        # Split X_train and X_test into 128 frames, and then update y_train and y_test so they match the new results
        n_splits = int(target_size/split_size)

        # y_train and y_test
        new_y_train = []
        for y in y_train:
            new_y_train.extend([y for i in range(n_splits)])

        new_y_test = []
        for y in y_test:
            new_y_test.extend([y for i in range(n_splits)])

        new_y_val = []
        for y in y_val:
            new_y_val.extend([y for i in range(n_splits)])
        
        # X_train and X_test
        def splitter(X_set):
            output = []
            for x in X_set:
                new_x = []
                low = 0
                high = split_size
                while len(new_x) < n_splits:
                    temp_mfccs = []
                    for mfcc in x:
                        temp_mfccs.append(mfcc[low:high])
                    low += split_size
                    high += split_size
                    new_x.append(temp_mfccs)
                #print(len(new_x), len(new_x[0]), len(new_x[0][0]), len(new_x[1]), len(new_x[1][0]))
                output.extend(new_x)
            return output

        new_X_train = splitter(X_train)
        new_X_test = splitter(X_test)
        new_X_val = splitter(X_val)

        y_train = np.array(new_y_train)
        y_test = np.array(new_y_test)
        y_val = np.array(new_y_val)
        print(X_train.shape, X_test.shape)
        print(len(new_X_train), len(new_X_train[0]), len(new_X_train[0][0]))
        print(len(new_X_test), len(new_X_test[0]), len(new_X_test[0][0]))

        X_train = np.array(new_X_train).reshape(-1, mfcc_size, split_size)
        X_test = np.array(new_X_test).reshape(-1, mfcc_size, split_size)
        X_val = np.array(new_X_val).reshape(-1, mfcc_size, split_size)
    
    # ---Standardise
    # Whiten over each file seperately
    X_train_std=whiten(X_train.transpose()).transpose()
    X_test_std=whiten(X_test.transpose()).transpose()
    X_val_std=whiten(X_val.transpose()).transpose()

    np.save(f'mfccs/X_train_moz.npy', X_train_std)
    np.save(f'mfccs/X_test_moz.npy', X_test_std)
    np.save(f'mfccs/X_val_moz.npy', X_val_std)
    np.save(f'mfccs/y_train_moz.npy', y_train)
    np.save(f'mfccs/y_test_moz.npy', y_test)
    np.save(f'mfccs/y_val_moz.npy', y_val)
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


def clean_df(file):
    df = pd.read_csv(file)
    output_df = pd.DataFrame()
    for accent in mozilla_categories:
        output_df = pd.concat([output_df, df[df['accent']==accent]], ignore_index=True)
    output_df.drop(['text', 'up_votes', 'down_votes', 'age', 'gender', 'duration'], axis=1, inplace=True)
    return output_df

class Mfcc():

    def __init__(self, df, col, accent, limit):
        self.df = df
        self.col = col
        self.accent = accent
        self.limit = limit

    def mp3towav(self):
        accent_df = self.df[self.df['accent']==self.accent]
        for filename in tqdm(accent_df[self.col]):
            pydub.AudioSegment.from_mp3(f"../experiments_data/mozilla/org_mp3/{filename}.mp3").export(f"../experiments_data/mozilla/wavs/{filename}.wav", format="wav")

    def wavtomfcc(self, file_path):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13)
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        accent_df = self.df[self.df['accent']==self.accent]
        for wav in tqdm(accent_df[self.col]):
            file_name = f'../experiments_data/mozilla/wavs/{wav}.wav'
            mfcc = self.wavtomfcc(file_name)
            list_of_mfccs.append(mfcc)
        self.list_of_mfccs = list_of_mfccs

    def resize_mfcc(self):
        self.target_size = 64
        resized_mfcc = [librosa.util.fix_length(mfcc, self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        self.X = resized_mfcc

    def label_samples(self):
        accent_df = self.df[self.df['accent']==self.accent]
        y_labels = np.array(accent_df['accent'])
        y = np.where(y_labels==self.accent, mozilla_categories.index(self.accent), 0)
        self.y = y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.15)
        #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.15)
        self.X_train = np.array(X_train).reshape(-1, 16, self.target_size)
        self.X_test = np.array(X_test).reshape(-1, 16, self.target_size)
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
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        self.y_train = np.vstack((self.y_train, np.ones(232).reshape(-1,1)))

    def save_mfccs(self):
        # np.save(f'../experiments_data/mozilla/mfccs/X_train_moz_{self.accent}.npy', self.X_train_std)
        # np.save(f'../experiments_data/mozilla/mfccs/X_test_moz_{self.accent}.npy', self.X_test_std)
        # np.save(f'../experiments_data/mozilla/mfccs/X_val_moz_{self.accent}.npy', self.X_val_std)
        # np.save(f'../experiments_data/mozilla/mfccs/y_train_moz_{self.accent}.npy', self.y_train)
        # np.save(f'../experiments_data/mozilla/mfccs/y_test_moz_{self.accent}.npy', self.y_test)
        # np.save(f'../experiments_data/mozilla/mfccs/y_val_moz_{self.accent}.npy', self.y_val)
        mfccs[self.accent] = {
            "x_train": self.X_train_std,
            "x_test": self.X_test_std,
            #"x_val": self.X_val_std,
            "y_train": self.y_train,
            "y_test": self.y_test,
            #"y_val": self.y_val
        }


mozilla_categories = ["us", "england", "australia", "indian", "canada", "malaysia", "ireland", "bermuda", "scotland", "african", "newzealand", "wales", "philippines", "singapore", "hongkong", "southatlandtic"]
mozilla_categories  = ["us", "england", "indian", "canada", "australia"]

mfccs = {}

# 354, 293, 61
if __name__ == '__main__':
    df = clean_df('../experiments_data/mozilla/validated.csv')
    print("DF created")
    print(len(df))
    #print(df)
    
    for accent in mozilla_categories:
        mfcc = Mfcc(df, 'filename', accent, limit=-1)
        # mfcc.mp3towav()
        mfcc.create_mfcc()
        mfcc.resize_mfcc()
        mfcc.label_samples()
        mfcc.split_data()
        mfcc.standardize_mfcc()
        # mfcc.oversample()
        mfcc.save_mfccs()

    X_train_std = mfccs[mozilla_categories[0]]["x_train"]
    X_test_std = mfccs[mozilla_categories[0]]["x_test"]
    #X_val_std = mfccs[mozilla_categories_small[0]]["x_val"]
    y_train = mfccs[mozilla_categories[0]]["y_train"]
    y_test = mfccs[mozilla_categories[0]]["y_test"]
    #y_val = mfccs[mozilla_categories_small[0]]["y_val"]

    for accent in mozilla_categories[1:]:
        X_train_std = np.concatenate((X_train_std, mfccs[accent]["x_train"]))
        X_test_std = np.concatenate((X_test_std, mfccs[accent]["x_test"]))
        #X_val_std = np.concatenate((X_val_std, mfccs[accent]["x_val"]))
        y_train = np.concatenate((y_train, mfccs[accent]["y_train"]))
        y_test = np.concatenate((y_test, mfccs[accent]["y_test"]))
        #y_val = np.concatenate((y_val, mfccs[accent]["y_val"]))

    np.save(f'mfccs/X_train_moz.npy', X_train_std)
    np.save(f'mfccs/X_test_moz.npy', X_test_std)
    #np.save(f'mfccs/X_val_moz.npy', X_val_std)
    np.save(f'mfccs/y_train_moz.npy', y_train)
    np.save(f'mfccs/y_test_moz.npy', y_test)
    #np.save(f'mfccs/y_val_moz.npy', y_val)
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as py_go
import plotly.offline as py_o
import librosa
import librosa.display
import IPython.display
import speechpy


def wavtomfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    n_fft=2048
    hop_length=512
    win_length=2048
    # defaults n_fft=2048 hop_length=512 win_length=2048
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13, win_length=win_length, hop_length=hop_length, n_fft=n_fft) # format is (n_mfcc, )
    spectro = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=128)
    mfcc = resize_mfcc(mfcc, 70)
    spectro = resize_mfcc(spectro, 70)

    return mfcc, spectro

def resize_mfcc(mfcc, frames):
    mfcc = librosa.util.fix_length(mfcc, size=frames, axis=1)
    return mfcc

file = "../experiments_data/clips_for_analysis/aus_male_nine_teen.wav"

mfcc, spectro = wavtomfcc(file)
#print(len(mfcc), len(mfcc[0]), len(mfcc[0]))

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(spectro, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

plt.show()

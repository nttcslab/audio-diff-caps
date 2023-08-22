import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy
from scipy.io import wavfile
import librosa
import shutil
from utils import FSD50K, ESC50

duration = 10


def wavwrite(fn, data, fs):
    data = np.array(np.around(data * 2**(15)), dtype = "int16")
    wavfile.write(fn, fs, data)


df = pd.read_csv('audio_csv/bg/fsd50k.csv')
for row in tqdm(df.itertuples()):
    fn = FSD50K+f'/FSD50K.{row.split}_audio/{row.fname}'
    audio, fs = librosa.load(fn, sr=None)
    rep = (duration*2*fs)//row.length+1
    rep_audio = np.repeat(audio, rep)
    save_dir = f'source/{row.split}/background/{row.label}/'
    os.makedirs(save_dir, exist_ok=True)
    wavwrite(save_dir+row.fname, rep_audio, fs)

df = pd.read_csv('audio_csv/ev/esc50.csv')
for row in tqdm(df.itertuples()):
    fn = ESC50+f'/{row.fname}'
    audio, fs = librosa.load(fn, sr=None)
    audio = audio[row.st:row.ed]
    # import pdb; pdb.set_trace()
    save_dir = f'source/{row.split}/foreground/{row.label}/'
    os.makedirs(save_dir, exist_ok=True)
    wavwrite(save_dir+row.fname, audio, fs)

for split in ['dev', 'eval']:
    for label in ['rain', 'car_passing']:
        shutil.copytree(
            f'source/{split}/background/{label}',
            f'source/{split}/foreground/{label}'
        )
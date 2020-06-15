import os
import sys
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random
import config

import soundfile

mini_data=True
labels=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
metadata_path = "/home/liuzhuangzhuang/Free sound dataset/FSDKaggle2018.meta/train_post_competition.csv"
feature_path = "/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/features/logmel_64frames_64melbins/pre_train.h5"
audio_path="/home/hxf/dcase2018Task2Data/freesound-audio-tagging/audio_train/"
create_folder(os.path.dirname(feature_path))


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file.

        Args:
          audio: (samples,)

        Returns:
          feature: (frames_num, freq_bins)
        '''

        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func

        # Compute short-time Fourier transform                     经过STFT
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=window_size,
            hop_length=hop_size,
            window=window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram#np.dot 矩阵乘积
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)  # 将STFT平方点乘mel谱

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32)

        return logmel_spectrogram
# 计算特征
def calculate_feature_for_all_audio_files(mini_data):
    '''Calculate feature of audio files and write out features to a single hdf5
    file.

    Args:
      dataset_dir: string
      workspace: string
      data_type: 'train' | 'validate' | 'evaluate'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    metadata_path = '/home/liuzhuangzhuang/Free sound dataset/FSDKaggle2018.meta/train_1.csv'


    audios_dir ="/home/hxf/dcase2018Task2Data/freesound-audio-tagging/audio_train/"

    feature_path = "/home/liuzhuangzhuang/pycharm_P/task5-多任务0.0/work_space/features/logmel_64frames_64melbins/pre_train.h5"

    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax)

    # Read metadata
    print('Extracting features of all audio files ...')
    extract_time = time.time()

    df = pd.read_csv(metadata_path, sep=',')
    audio_names = np.array(sorted(list(set(df['fname']))))
    label_targets = []
    if mini_data:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(audio_names)
        audio_names = audio_names[0: 10]
    for audio_name in audio_names:
        label_target = np.zeros(len(labels))
        df_label = df[df['fname'] == audio_name]
        for i, label in enumerate(labels):
            df_ = df_label[df_label['label'] == label]  # 名字和标签对应则不为空
            if not df_.empty:
                label_target[i] = 1

        label_targets.append(label_target)

    label_targets = np.array(label_targets)
    meta_dict = {
        'audio_name': audio_names,
        'label_targetst': label_targets}

    # Hdf5 containing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name',
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']],
        dtype='S32')


    if 'label_targetst' in meta_dict.keys():
        hf.create_dataset(
            name='label_targetst',
            data=meta_dict['label_targetst'],
            dtype=np.float32)

    hf.create_dataset(
        name='feature',
        shape=(0, frames_num, mel_bins),
        maxshape=(None, frames_num, mel_bins),
        dtype=np.float32)

    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)

        # Read audio
        (audio, fs) = soundfile.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)#一维
        if sample_rate is not None and fs != sample_rate:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)#重新采样


        # Pad or truncate audio recording
        if len(audio) < total_samples:
            audio = np.concatenate((audio, np.zeros(total_samples - len(audio))))
        else:
            audio =audio[0: total_samples]

        # Extract feature
        feature = feature_extractor.transform(audio)

        # Remove the extra frames caused by padding zero
        feature = feature[0: frames_num]

        hf['feature'].resize((n + 1, frames_num, mel_bins))
        hf['feature'][n] = feature

    hf.close()

    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))

if __name__ == '__main__':
    calculate_feature_for_all_audio_files(False)
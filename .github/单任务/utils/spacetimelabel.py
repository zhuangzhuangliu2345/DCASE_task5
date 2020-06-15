import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
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

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor,
    pad_truncate_sequence)
import config

NUM_YEARS = 4
NUM_HOURS = 4
NUM_DAYS = 2
NUM_WEEKS = 4
space_time_path = '/home/fangjunyan/task5-多任务0.0/work_space/features/logmel_64frames_64melbins/'
annotation_path = '/home/fangjunyan/2020shuju/annotations2020.csv'
time_labels = ['week', 'day', 'hour']
space_labels = ['latitude', 'longitude']


def one_hot(idx, num_items):
    return [(0.0 if n != idx else 1.0) for n in range(num_items)]


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets

    Parameters
    ----------
    annotation_data

    Returns
    -------
    train_idxs
    valid_idxs

    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename', 'annotator_id']]\
                          .groupby(by=['split', 'audio_filename'], as_index=False)\
                          .min()\
                          .sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []

    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        elif row['split'] == 'validate' :
            # For validation examples, only use verified annotations
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
annotation_data_trunc = annotation_data[['audio_filename',
                                             'latitude',
                                             'longitude',
                                             'year',
                                             'week',
                                             'day',
                                             'hour']].drop_duplicates()
file_list = annotation_data_trunc['audio_filename'].to_list()
latitude_list = annotation_data_trunc['latitude'].to_list()
longitude_list = annotation_data_trunc['longitude'].to_list()
year_list = annotation_data_trunc['year'].to_list()
week_list = annotation_data_trunc['week'].to_list()
day_list = annotation_data_trunc['day'].to_list()
hour_list = annotation_data_trunc['hour'].to_list()

for idx in range(0, len(year_list)):
    if year_list[idx] == 2016:
        year_list[idx] = 0
    if year_list[idx] == 2017:
        year_list[idx] = 1
    if year_list[idx] == 2018:
        year_list[idx] = 2
    if year_list[idx] == 2019:
        year_list[idx] = 3

for idx in range(0, len(week_list)):
    if 13 >= week_list[idx] >= 1:
        week_list[idx] = 1
    if 26 >= week_list[idx] >= 14:
        week_list[idx] = 2
    if 39 >= week_list[idx] >= 27:
        week_list[idx] = 3
    if 52 >= week_list[idx] >= 40:
        week_list[idx] = 4

for idx in range(0, len(day_list)):
    if 4 >= day_list[idx] >= 0:
        day_list[idx] = 0
    if 6 >= day_list[idx] >= 5:
        day_list[idx] = 1

for idx in range(0, len(hour_list)):
    if 5 >= hour_list[idx] >= 0:
        hour_list[idx] = 0
    if 11 >= hour_list[idx] >= 6:
        hour_list[idx] = 1
    if 17 >= hour_list[idx] >= 12:
        hour_list[idx] = 2
    if 23 >= hour_list[idx] >= 18:
        hour_list[idx] = 3

train_idxs, valid_idxs = get_subset_split(annotation_data)

X_train_time = np.array([
        one_hot(year_list[idx], NUM_YEARS) \
          + one_hot(week_list[idx] - 1, NUM_WEEKS) \
          + one_hot(day_list[idx], NUM_DAYS) \
          + one_hot(hour_list[idx], NUM_HOURS)
        for idx in train_idxs])
X_valid_time = np.array([
    one_hot(year_list[idx], NUM_YEARS) \
    + one_hot(week_list[idx] - 1, NUM_WEEKS) \
    + one_hot(day_list[idx], NUM_DAYS) \
    + one_hot(hour_list[idx], NUM_HOURS)
    for idx in valid_idxs])
X_train_loc = np.array([[latitude_list[idx],
                             longitude_list[idx]]
                            for idx in train_idxs])
X_valid_loc = np.array([[latitude_list[idx],
                             longitude_list[idx]]
                            for idx in valid_idxs])

X_train_spacetime = np.concatenate((X_train_time, X_train_loc), axis=-1)
X_valid_spacetime = np.concatenate((X_valid_time, X_valid_loc), axis=-1)

hf1 = h5py.File(os.path.join(space_time_path, 'trainspacetime.h5'), 'w')
hf2 = h5py.File(os.path.join(space_time_path, 'validatespacetime.h5'), 'w')

hf1.create_dataset(
    name='spacetime',
    data=X_train_spacetime,
    dtype=np.float32
)

hf2.create_dataset(
    name='spacetime',
    data=X_valid_spacetime,
    dtype=np.float32
)









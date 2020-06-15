import pandas as pd


pd.read_csv(annotation_path).sort_values('audio_filename')
annotation_data_trunc = annotation_data[['audio_filename',
                                         'latitude',
                                         'longitude',
                                         'week',
                                         'day',
                                         'hour']].drop_duplicates()
file_list = annotation_data_trunc['audio_filename'].to_list()
latitude_list = annotation_data_trunc['latitude'].to_list()
longitude_list = annotation_data_trunc['longitude'].to_list()
week_list = annotation_data_trunc['week'].to_list()
day_list = annotation_data_trunc['day'].to_list()
hour_list = annotation_data_trunc['hour'].to_list()

num_frames = X_train_emb.shape[1]

X_train_loc = np.array([[[latitude_list[idx],
                          longitude_list[idx]]] * num_frames
                        for idx in train_file_idxs])
X_valid_loc = np.array([[[latitude_list[idx],
                          longitude_list[idx]]] * num_frames
                        for idx in valid_file_idxs])

X_train_time = np.array([
    [one_hot(week_list[idx] - 1, NUM_WEEKS) \
     + one_hot(day_list[idx], NUM_DAYS) \
     + one_hot(hour_list[idx], NUM_HOURS)] * num_frames
    for idx in train_file_idxs])
X_valid_time = np.array([
    [one_hot(week_list[idx] - 1, NUM_WEEKS) \
     + one_hot(day_list[idx], NUM_DAYS) \
     + one_hot(hour_list[idx], NUM_HOURS)] * num_frames
    for idx in valid_file_idxs])

X_train_cts = np.concatenate((X_train_emb, X_train_loc), axis=-1)
X_valid_cts = np.concatenate((X_valid_emb, X_valid_loc), axis=-1)

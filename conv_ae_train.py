import dataset
import nets
import numpy as np
import pandas as pd
import os
import pickle
import utils

save_dir = 'cached/fma_small_mfcc_conv_m10_fps5_test'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
data_prep_params_save_path = os.path.join(save_dir, 'data_prep_params')
training_params_save_path = os.path.join(save_dir, 'training_params')
model_save_path = os.path.join(save_dir, 'model')
encoder_save_path = os.path.join(save_dir, 'encoder')

num_net_scale_downs = 3
data_split_ratio = 0.8

try:
    dataset.load_fma(sample_size=10, save_dir=save_dir, fps=5, num_segments=20)
except:
    print('Data already loaded')

if os.path.isfile(model_save_path):
    raise Exception('Already trained')

# Load mfccs, prams and norms

x = np.load(mfcc_save_path)
print('MFCCs data loaded. Data size', x.size)

tracks = pd.read_pickle(tracks_save_path)
print('Tracks data loaded. Data size', tracks.shape)

with open(data_prep_params_save_path, 'rb') as pf:
    data_prep_params = pickle.load(pf)
num_tracks, sr, fps, num_mfcc, num_segments_per_track, save_dir = data_prep_params
print('Data prep params loaded', data_prep_params)

# Shape for training
num_mfcc_frames = int(x.size / (num_tracks * num_segments_per_track * num_mfcc))
x = x.reshape(num_tracks * num_segments_per_track, num_mfcc, num_mfcc_frames, 1)
print('Data reshaped', x.shape)

# Pad
x, num_mfcc_new, num_mfcc_frames_new = utils.pad_mfccs(x, num_net_scale_downs, num_tracks * num_segments_per_track,
                                                       num_mfcc, num_mfcc_frames)

# Split
x_train, x_test = dataset.split_data(x, data_split_ratio)
print('Training shape', x_train.shape)
print('Test shape', x_test.shape)

# Normalize training set
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std

# Save training params
training_params = mean, std, data_split_ratio, num_net_scale_downs
with open(training_params_save_path, 'wb') as tf:
    pickle.dump(training_params, tf)

# Train
ae, encoder = nets.conv_ae(num_mfcc_new, num_mfcc_frames_new)
ae.summary()
print('Training...')
ae.fit(x_train, x_train,
       epochs=50,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, x_test))

ae.save(model_save_path)
encoder.save(encoder_save_path)

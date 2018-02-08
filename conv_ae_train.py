import dataset
import autoencoders
import numpy as np
import pandas as pd
import os
import pickle

save_dir = 'cached/fma_small_mfcc_conv_m6000_fps5'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
norms_save_path = os.path.join(save_dir, 'norms')
ae_save_path = os.path.join(save_dir, 'ae')
encoder_save_path = os.path.join(save_dir, 'encoder')

try:
    dataset.load_fma(sample_size=6000, save_dir=save_dir, fps=5, num_segments=20)
except:
    print('Data already loaded')

if os.path.isfile(ae_save_path):
    raise Exception('Already trained')

x = np.load(mfcc_save_path)
tracks = pd.read_pickle(tracks_save_path)
with open(params_save_path, 'rb') as pf:
    sample_size, sr, fps, mfcc, num_segments, save_dir = pickle.load(pf)

# Shape for training
num_frames = int(x.shape[0] / (sample_size * num_segments * mfcc))
x = x.reshape(sample_size * num_segments, mfcc, num_frames, 1)

# Pad
scale = 2 ** 3
pad_frames = (int(num_frames / scale) + 1) * scale - num_frames
x_pad_frames = np.zeros((sample_size * num_segments, mfcc, pad_frames, 1))
x = np.concatenate((x, x_pad_frames), axis=2)
pad_mfcc = (int(mfcc / scale) + 1) * scale - mfcc
x_pad_mfcc = np.zeros((sample_size * num_segments, pad_mfcc, x.shape[2], 1))
x = np.concatenate((x, x_pad_mfcc), axis=1)

# Split
x_train, x_test = dataset.split_data(x)
print('Training shape', x_train.shape)
print('Test shape', x_test.shape)

# Normalize training set
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
with open(norms_save_path, 'wb') as nf:
    pickle.dump((mean, std), nf)

# Train
inpdimx = x_train.shape[1]
inpdimy = x_train.shape[2]
ae, encoder = autoencoders.conv_ae(inpdimx, inpdimy)
ae.summary()
print('Training...')
ae.fit(x_train, x_train,
       epochs=30,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, x_test))

ae.save(ae_save_path)
encoder.save(encoder_save_path)

import dataset
import autoencoders
import numpy as np
import pandas as pd
import os
import pickle
import sklearn

save_dir = 'cached/fma_small_mfcc_conv_m6000_fps1'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
net_save_path = os.path.join(save_dir, 'net')

try:
    dataset.load_fma(sample_size=6000, save_dir=save_dir, fps=3, num_segments=20)
except:
    print('Data already loaded')

if os.path.isfile(net_save_path):
    raise Exception('Already trained')

x = np.load(mfcc_save_path)
tracks = pd.read_pickle(tracks_save_path)
with open(params_save_path, 'rb') as pf:
    sample_size, sr, fps, mfcc, num_segments, save_dir = pickle.load(pf)

# Normalize
x = x.reshape((x.shape[0], 1))
x = sklearn.preprocessing.normalize(x)
x = x.reshape((x.shape[0],))

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

# Train
inpdimx = x_train.shape[1]
inpdimy = x_train.shape[2]
ae, encoder = autoencoders.conv_ae(inpdimx, inpdimy)
ae.summary()
print('Training...')
ae.fit(x_train, x_train,
       epochs=20,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, x_test))

ae.save(net_save_path)

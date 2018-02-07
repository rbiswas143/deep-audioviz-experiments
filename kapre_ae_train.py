import dataset
import autoencoders
import numpy as np
import pandas as pd
import os
import pickle

save_dir = 'cached/fma_small_frames_conv_m10'
frames_save_path = os.path.join(save_dir, 'frames.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
nets_save_path = os.path.join(save_dir, 'net')

try:
    dataset.load_fma_tracks(sample_size=10, num_secs=20, save_dir=save_dir)
except:
    print('Data already loaded')

if os.path.isfile(nets_save_path):
    raise Exception('Already trained')

x = np.load(frames_save_path)
tracks = pd.read_pickle(tracks_save_path)
with open(params_save_path, 'rb') as pf:
    sample_size, sr, num_secs, save_dir = pickle.load(pf)

# Reshape
fps = 1
segment_size = sr * fps
x = x.reshape(int(x.shape[0] / segment_size), segment_size)

# Split
x_train, x_test = dataset.split_data(x)
print('Training shape', x_train.shape)
print('Test shape', x_test.shape)

ae, encoder = autoencoders.kapre_conv_ae(segment_size)
ae.summary()
print('Training...')
ae.fit(x_train, x_train,
       epochs=20,
       batch_size=128,
       shuffle=True)

ae.save(nets_save_path)

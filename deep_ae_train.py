import dataset
import autoencoders
import numpy as np
import pandas as pd
import os
import pickle
import sklearn

save_dir = 'cached/fma_small_mfcc_deep_m2000_fps1'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
net_save_path = os.path.join(save_dir, 'net')

try:
    dataset.load_fma(sample_size=2000, save_dir=save_dir, fps=1, num_segments=10)
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
input_size = int(x.shape[0] / (sample_size * num_segments))
x = x.reshape(sample_size * num_segments, input_size)

# Split
x_train, x_test = dataset.split_data(x)
print('Training shape', x_train.shape)
print('Test shape', x_test.shape)

# Train
ae, encoder = autoencoders.deep_ae(input_size)
ae.summary()
print('Training...')
ae.fit(x_train, x_train,
       epochs=20,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, x_test))

ae.save(net_save_path)

import dataset
import classifiers
import numpy as np
import pandas as pd
import os
import pickle
import sklearn

save_dir = 'cached/fma_small_mfcc_conv_m6000_fps1_genre'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
net_save_path = os.path.join(save_dir, 'net')

try:
    dataset.load_fma(sample_size=20, save_dir=save_dir, fps=1, num_segments=20)
except:
    print('Data already loaded')

if os.path.isfile(net_save_path):
    raise Exception('Already trained')

x = np.load(mfcc_save_path)
tracks = pd.read_pickle(tracks_save_path)
with open(params_save_path, 'rb') as pf:
    sample_size, sr, fps, mfcc, num_segments, save_dir = pickle.load(pf)

# Genres
genre_map = dataset.get_genre_map()
genre_map_rev = dataset.get_genre_map(return_reverse=True)
y = np.zeros((tracks.shape[0], len(genre_map)))
for i, idx in enumerate(tracks.index):
    y[i][genre_map_rev[tracks['track', 'genre_top'][idx]]] = 1
y = np.repeat(y, num_segments, axis=0)
y_train, y_test = dataset.split_data(y)

# Normalize
x = x.reshape((x.shape[0], 1))
x = sklearn.preprocessing.normalize(x)
x = x.reshape((x.shape[0],))

# Shape for training
num_frames = int(x.shape[0] / (sample_size * num_segments * mfcc))
x = x.reshape(sample_size * num_segments, mfcc, num_frames, 1)

# Pad
scale = 2 ** 2
pad_frames = (int(num_frames / scale) + 1) * scale - num_frames
x_pad_frames = np.zeros((sample_size * num_segments, mfcc, pad_frames, 1))
x = np.concatenate((x, x_pad_frames), axis=2)
pad_mfcc = (int(mfcc / scale) + 1) * scale - mfcc
x_pad_mfcc = np.zeros((sample_size * num_segments, pad_mfcc, x.shape[2], 1))
x = np.concatenate((x, x_pad_mfcc), axis=1)

# Split
x_train, x_test = dataset.split_data(x)
print('Training input shape', x_train.shape)
print('Training target shape', y_train.shape)
print('Test input shape', x_test.shape)
print('Test target shape', y_test.shape)

# Train
inpdimx = x_train.shape[1]
inpdimy = x_train.shape[2]
model = classifiers.genre_classifier_conv(inpdimx, inpdimy, len(genre_map))
model.summary()
print('Training...')
model.fit(x_train, y_train,
       epochs=30,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, y_test))

model.save(net_save_path)

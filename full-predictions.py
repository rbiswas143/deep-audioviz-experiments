import dataset
import importlib
import autoencoders
import numpy as np
import pandas as pd
import os
import pickle
import sklearn
import librosa
import fma_utils
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

importlib.reload(dataset)

save_dir = 'cached/fma_small_mfcc_conv_m6000_fps1'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
params_save_path = os.path.join(save_dir, 'params')
predictions_save_path = os.path.join(save_dir, 'full_predictions.json')
net_save_path = os.path.join(save_dir, 'net')

num_tracks_to_predict = 40
dim_red_pca = 5
dim_red_kmeans = 5

if os.path.isfile(predictions_save_path):
    Exception('Predictions already saved')

tracks = pd.read_pickle(tracks_save_path)
with open(params_save_path, 'rb') as pf:
    sample_size, sr, num_secs, mfcc, num_segments, save_dir = pickle.load(pf)
window_size = sr * num_secs

train_idx, test_idx = dataset.split_data(tracks.index)

train_tracks = tracks.loc[train_idx, :].sample(int(num_tracks_to_predict / 2))
test_tracks = tracks.loc[test_idx, :].sample(int(num_tracks_to_predict / 2))

tracks_data = []
all_data = {
    'sr': sr,
    'segment_time': num_secs,
    'mfcc': mfcc,
    'pca_enc_len': dim_red_pca,
    'kmeans_enc_len': dim_red_kmeans,
    'tracks': tracks_data
}

ae = encoder = None

encoded = np.array([])
encoding_length = None
counter = 0
for mode, indices in enumerate([train_tracks.index, test_tracks.index]):

    for track_index in indices:

        counter += 1
        print('Processing track {} of {}'.format(counter, num_tracks_to_predict))

        data = {
            'mode': 'train' if mode == 0 else 'test',
            'index': track_index,
            'name': tracks['track', 'title'][track_index],
            'genre': tracks['track', 'genre_top'][track_index],
            'artist': tracks['artist', 'name'][track_index],
            'album': tracks['album', 'title'][track_index],
        }
        tracks_data.append(data)

        track_file = fma_utils.get_audio_path(dataset.audio_path, track_index)
        data['path'] = track_file
        loaded, track_sr = librosa.load(track_file, sr=None, mono=True)

        actual_segments = int(loaded.shape[0] / window_size)
        if loaded.shape[0] > actual_segments * window_size:
            actual_segments += 1
        data['num_segments'] = actual_segments

        # Pad
        padded = np.zeros(actual_segments * window_size)
        padded[:loaded.shape[0]] = loaded

        # Split tracks
        reshaped = padded.reshape(actual_segments, window_size)

        # mfcc
        converted_all = np.array([])
        for segment in reshaped:
            converted = librosa.feature.mfcc(y=segment, n_mfcc=mfcc, sr=sr)
            converted_all = np.concatenate((converted_all, converted.reshape(converted.size)))

        # Normalize
        scaled = sklearn.preprocessing.scale(converted_all)

        # Shape for training
        num_frames = int(scaled.shape[0] / (actual_segments * mfcc))
        reshaped = scaled.reshape(actual_segments, mfcc, num_frames, 1)

        # Pad
        scale = 2 ** 3
        pad_frames = (int(num_frames / scale) + 1) * scale - num_frames
        x_pad_frames = np.zeros((actual_segments, mfcc, pad_frames, 1))
        x = np.concatenate((reshaped, x_pad_frames), axis=2)
        pad_mfcc = (int(mfcc / scale) + 1) * scale - mfcc
        x_pad_mfcc = np.zeros((actual_segments, pad_mfcc, x.shape[2], 1))
        x = np.concatenate((x, x_pad_mfcc), axis=1)

        mfcc_new, num_frames_new = x.shape[1], x.shape[2]

        # Load trained ae
        # ae = keras.models.load_model(net_save_path)
        if encoder is None:
            ae, encoder = autoencoders.conv_ae(mfcc_new, num_frames_new)
            ae.load_weights(net_save_path)

        # Predict
        y = encoder.predict(x)
        encoding_length = int(y.size / actual_segments)
        all_data['raw_enc_len'] = encoding_length

        # Flatten
        y = y.reshape(actual_segments, encoding_length)
        data['raw_enc'] = y.tolist()

        encoded = np.concatenate((encoded, y.reshape(y.size)))

# Reshape all encodings
total_segments = int(encoded.size / encoding_length)
encoded = encoded.reshape(total_segments, encoding_length)

# Fit PCA for all segments
pca = PCA(n_components=dim_red_pca)
pca.fit(encoded)
print('Variance retained: {}%'.format(pca.explained_variance_ratio_.sum() * 100))

# Fit PCA for all segments
kmeans = KMeans(n_clusters=dim_red_kmeans, random_state=0)
kmeans.fit(encoded)

for data in tracks_data:
    y = np.array(data['raw_enc'])

    # Dimension reduction with PCA
    y_pca = pca.transform(y)
    # Normalize each feature across all samples
    y_pca = sklearn.preprocessing.normalize(y_pca, axis=0)
    data['pca_enc'] = y_pca.tolist()

    # Dimension reduction with k-means
    y_kmeans = kmeans.transform(y)
    y_kmeans = 1 / (1 + y_kmeans)
    # Normlaize each sample
    y_kmeans_scaled = y_kmeans / y_kmeans.sum(axis=1)[:, None]
    data['kmeans_enc'] = y_kmeans.tolist()
    data['kmeans_enc_scaled'] = y_kmeans_scaled.tolist()

with open(predictions_save_path, 'w') as pf:
    json.dump(all_data, pf)

print('Predictions have been saved')

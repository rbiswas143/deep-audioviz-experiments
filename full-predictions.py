import dataset
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
import keras
import utils

save_dir = 'cached/final_2k_fps5_genre_9'
mfcc_save_path = os.path.join(save_dir, 'mfcc.npy')
tracks_save_path = os.path.join(save_dir, 'tracks')
data_prep_params_save_path = os.path.join(save_dir, 'data_prep_params')
training_params_save_path = os.path.join(save_dir, 'training_params')
encoder_save_path = os.path.join(save_dir, 'encoder')
model_save_path = os.path.join(save_dir, 'model')
predictions_save_path = os.path.join(save_dir, 'full_predictions.json')

mode = 'genre'  # autoencoder, genre, genre_multi
num_tracks_to_predict = 40
dim_red_pca = None
dim_red_kmeans = 2
pca_scale_range = (0, 1)

# Load all data

if os.path.isfile(predictions_save_path):
    Exception('Predictions already saved')

tracks = pd.read_pickle(tracks_save_path)
print('Tracks data loaded. Data size', tracks.shape)

with open(data_prep_params_save_path, 'rb') as pf:
    data_prep_params = pickle.load(pf)
num_tracks, sr, fps, num_mfcc, num_segments_per_track, save_dir = data_prep_params
print('Data prep params loaded', data_prep_params)

with open(training_params_save_path, 'rb') as nf:
    training_params = pickle.load(nf)
mean, std, data_split_ratio, num_net_scale_downs = training_params
print('Training params loaded', training_params)

window_size = int(sr / fps)

train_idx, test_idx = dataset.split_data(tracks.index, data_split_ratio)
train_tracks = tracks.loc[train_idx, :].sample(int(num_tracks_to_predict / 2))
test_tracks = tracks.loc[test_idx, :].sample(int(num_tracks_to_predict / 2))

tracks_data = []
all_data = {
    'sr': sr,
    'fps': fps,
    'num_mfcc': num_mfcc,
    'tracks': tracks_data
}

encoder = None

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
        split = padded.reshape(actual_segments, window_size)

        # mfcc
        converted_all = np.array([])
        for segment in split:
            converted = librosa.feature.mfcc(y=segment, n_mfcc=num_mfcc, sr=sr)
            converted_all = np.concatenate((converted_all, converted.reshape(converted.size)))

        # Shape for training
        num_frames = int(converted_all.shape[0] / (actual_segments * num_mfcc))
        reshaped = converted_all.reshape(actual_segments, num_mfcc, num_frames, 1)

        # Pad
        x, num_mfcc_new, num_frames_new = utils.pad_mfccs(reshaped, num_net_scale_downs, actual_segments, num_mfcc, num_frames)

        # Normalize
        x = (x - mean) / std

        # Load trained ae
        # ae = keras.models.load_model(net_save_path)
        if encoder is None:
            encoder = keras.models.load_model(encoder_save_path)

        # Predict
        y = encoder.predict(x)
        if type(y) is list:
            print('Multi layer encoder. Layers', len(y))
            y_all = np.array([])
            for y_layer in y:
                y_all = np.concatenate((y_all, y_layer.reshape(-1)))
            y = y_all
        else:
            print('Prediction shape', y.shape)

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
if dim_red_pca is None:
    dim_red_pca = encoding_length
pca = PCA(n_components=dim_red_pca)
pca.fit(encoded)
all_data['pca_enc_len'] = dim_red_pca
print('Variance retained: {}%'.format(pca.explained_variance_ratio_.sum() * 100))

# Fit kmeans for all segments
if dim_red_kmeans is None:
    dim_red_kmeans = encoding_length
kmeans = KMeans(n_clusters=dim_red_kmeans, random_state=0)
kmeans.fit(encoded)
all_data['kmeans_enc_len'] = dim_red_kmeans

for data in tracks_data:
    y = np.array(data['raw_enc'])
    if mode == 'genre_multi':
        data['raw_enc'] = None

    # Dimension reduction with PCA
    y_pca = pca.transform(y)
    scaler = sklearn.preprocessing.MinMaxScaler(pca_scale_range)
    scaler.fit(y_pca.reshape(-1, 1))
    y_pca = scaler.transform(y_pca)
    # Normalize each feature across all samples
    # y_pca = sklearn.preprocessing.normalize(y_pca, axis=0)
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

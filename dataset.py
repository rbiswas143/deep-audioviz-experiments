import fma_utils
import librosa
import numpy as np
import pickle
import os

# FMA params
tracks_path = 'datasets/fma/fma_metadata/tracks.csv'
audio_path = 'datasets/fma/fma_small'


# Loads FMA small dataset
def load_fma_tracks(sample_size=2000, sr=44100, num_secs=20, save_dir='cached/fma_small'):

    num_frames = int(sr * num_secs)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        raise Exception("Save directory already exists")

    # Get small
    tracks = fma_utils.load(tracks_path)
    small = tracks[tracks['set', 'subset'] <= 'small'].sample(frac=1)

    # mfcc of all tracks
    data = np.array([])
    track_idx = []

    done_count = 0
    for i, track_id in enumerate(small['track'].index):

        if done_count >= sample_size:
            print('All files read')
            break

        if done_count % 10 == 0:
            print('Read {3} files. Now reading file {0} of {1}. Track ID: {2}'.format(i + 1, len(small['track']),
                                                                                      track_id, done_count))

        # Load track
        track_file = fma_utils.get_audio_path(audio_path, track_id)
        try:
            loaded, track_sr = librosa.load(track_file, sr=None, mono=True)
        except Exception as ex:
            print("Error loading track", track_file)
            print(ex)
            continue
        if sr != track_sr:
            print('Incorrect sample rate {0} for file {1}'.format(track_sr, track_file))
            continue

        if loaded.shape[0] < num_frames:
            print('Not enough frames {0} for file {1}'.format(loaded.shape[0], track_file))
            continue

        # Trim
        trimmed = loaded[:num_frames]
        data = np.concatenate((data, trimmed))

        # Save index
        track_idx.append(track_id)
        done_count += 1

    # Save everything
    frames_save_path = os.path.join(save_dir, 'frames')
    np.save(frames_save_path, data)

    tracks_save_path = os.path.join(save_dir, 'tracks')
    small = small.loc[track_idx, :]
    small.to_pickle(tracks_save_path)

    params_save_path = os.path.join(save_dir, 'params')
    params = sample_size, sr, num_secs, save_dir
    with open(params_save_path, 'wb') as pf:
        pickle.dump(params, pf)

    print('Data saved')


# Loads FMA small dataset and converts it into an mfcc
def load_fma(sample_size=2000, sr=44100, fps=5, mfcc=20, num_segments=50, save_dir='cached/fma_small_mfcc'):

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        raise Exception("Save directory already exists")

    # Num samples in a segment
    window_size = int(sr / fps)

    # Get small
    tracks = fma_utils.load(tracks_path)
    small = tracks[tracks['set', 'subset'] <= 'small'].sample(frac=1)

    # mfcc of all tracks
    data = np.array([])
    track_idx = []

    done_count = 0
    for i, track_id in enumerate(small['track'].index):

        if done_count >= sample_size:
            print('All files read')
            break

        if done_count % 10 == 0:
            print('Read {3} files. Now reading file {0} of {1}. Track ID: {2}'.format(i + 1, len(small['track']),
                                                                                      track_id, done_count))

        # Load track
        track_file = fma_utils.get_audio_path(audio_path, track_id)
        try:
            loaded, track_sr = librosa.load(track_file, sr=None, mono=True)
        except Exception as ex:
            print("Error loading track", track_file)
            print(ex)
            continue
        if sr != track_sr:
            print('Incorrect sample rate {0} for file {1}'.format(track_sr, track_file))
            continue

        actual_segments = int(len(loaded) / window_size)
        if actual_segments < num_segments:
            print('Not enough segments {0} for file {1}'.format(actual_segments, track_file))
            continue

        # Split tracks
        reshaped = loaded[:actual_segments * window_size]
        reshaped = reshaped.reshape(actual_segments, window_size)

        # Sample
        sample_idx = np.arange(actual_segments)
        np.random.shuffle(sample_idx)
        sampled = reshaped[sample_idx[:num_segments], :]

        # mfcc
        for segment in sampled:
            converted = librosa.feature.mfcc(y=segment, n_mfcc=mfcc, sr=sr)
            feature_size = converted.shape[0] * converted.shape[1]
            data = np.concatenate((data, converted.reshape(feature_size)))

        # Save index
        track_idx.append(track_id)
        done_count += 1

    # Save everything
    mfcc_save_path = os.path.join(save_dir, 'mfcc')
    np.save(mfcc_save_path, data)

    tracks_save_path = os.path.join(save_dir, 'tracks')
    small = small.loc[track_idx, :]
    small.to_pickle(tracks_save_path)

    params_save_path = os.path.join(save_dir, 'params')
    params = sample_size, sr, fps, mfcc, num_segments, save_dir
    with open(params_save_path, 'wb') as pf:
        pickle.dump(params, pf)

    print('Data saved')


# Split train and test
def split_data(data, ratio=0.8):
    div = int(data.shape[0] * ratio)
    return data[:div], data[div:]
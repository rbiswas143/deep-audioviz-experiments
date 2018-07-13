""" Pre-process and partition the FMA DataSet for later training
    
    Module Contents:
        A CLI for creating processed audio partitions
        DataSet partitioning logic and helpers
        Configuration classes with default config (overridden by JSON config via CLI) driving the processing of tracks
        Multiple processing modes: End to End (e2e), MFCCs
        Other pre-processing related utilities
        Sanity tests for all functionality
    
    Notes:
        The data processing is resumable. Feel free to kill the script and resume
        Data (processes data, config, etc) is stored in an hdf5 file and a reader is provided to abstract data access
"""

import fma_utils
import utils

import librosa
import numpy as np
import pandas as pd
import h5py
import os
import unittest
import json
import argparse
import shutil
import atexit
import traceback
import time


##
# Config
##

class BaseConfig:

    @classmethod
    def load_from_file(cls, path):
        """Initializes default config and overrides it with config dictionary obtained from a json file"""
        config = cls()
        with open(path, 'rb') as cf:
            config.update(json.load(cf))
        return config

    def update(self, dict_):
        """Overrides config with dictionary"""
        for key in dict_.keys():
            setattr(self, key, dict_[key])

    def get_dict(self):
        """Returns all config as a dictionary"""
        return self.__dict__


class DataPrepConfig(BaseConfig):
    """Base config class for all pre-processors"""

    def __init__(self):
        self.name = 'data_{0}'.format(int(time.time()))
        self.fma_audio_dir = 'datasets/fma/fma_small'
        self.fma_meta_dir = 'datasets/fma/fma_metadata'
        self.fma_type = 'small'  # small/medium
        self.num_tracks = 7500
        self.sr = 44100
        self.datasets_dir = 'datasets/processed'
        self.do_re_sample = True
        self.test_split = 0.1
        self.cv_split = 0.1
        self.num_train_partitions = 6
        self.scaler = 'standard'

    def get_dataset_path(self):
        return os.path.join(self.datasets_dir, self.name + '.h5')

    @staticmethod
    def load_from_dataset(path):
        mode = read_h5_attrib('mode', path)
        config = get_config_cls(mode)()
        config.update(read_h5_attrib('config', path, deserialize=True))
        return config


class MfccDataPrepConfig(DataPrepConfig):
    """Config for MFCC Pre-processor"""

    def __init__(self):
        DataPrepConfig.__init__(self)
        self.segments_per_track = None
        self.frames_per_segment = 90
        self.num_mfcc = 20
        self.mfcc_hops = 512
        self.n_fft = 2048


class E2eDataPrepConfig(DataPrepConfig):
    """Config for End to End Pre-processor"""

    def __init__(self):
        DataPrepConfig.__init__(self)
        self.segments_per_track = None
        self.frames_per_segment = self.sr * 1  # sec


def get_config_cls(mode):
    """Returns config class for a pre-processing mode"""
    return {
        'e2e': E2eDataPrepConfig,
        'mfcc': MfccDataPrepConfig
    }[mode]


# @unittest.skip
class DataPrepConfigTests(unittest.TestCase):

    def test_update(self):
        config = DataPrepConfig()
        test_config = {
            'name': 'temp',
            'num_tracks': -1
        }
        config.update(test_config)
        # Original Attribs
        self.assertEqual(config.fma_type, 'small')
        self.assertEqual(config.sr, 44100)
        # Updated Attribs
        self.assertEqual(config.name, 'temp')
        self.assertEqual(config.num_tracks, -1)


###
# h5 I/O wrappers
###


def read_h5_data(key, dataset_path, dtype=np.float32):
    """Returns a DataSet or None if it does not exist"""
    with h5py.File(dataset_path, 'r') as dataset:
        return dataset[key][()].astype(dtype) if key in dataset else None


def write_h5_data(key, data, dataset_path, dtype=np.float32):
    """Writes a DataSet. Creates the file/directory tree if it does not exist"""
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    data = np.array(data)
    with h5py.File(dataset_path, 'a') as dataset:
        if key in dataset and dataset[key].shape != data.shape:
            del dataset[key]
        if key not in dataset:
            dataset.create_dataset(key, shape=data.shape, dtype=dtype)
        dataset[key][...] = data.astype(dtype)


def read_h5_data_shape(key, dataset_path):
    """Returns the shape of the data at the specified key. Returns None if it does not exist"""
    with h5py.File(dataset_path, 'r') as dataset:
        return dataset[key].shape if key in dataset else None


def read_h5_attrib(key, dataset_path, deserialize=False):
    """Returns a top level attribute or None if it does not exist"""
    with h5py.File(dataset_path, 'r') as dataset:
        attrib = dataset.attrs[key] if key in dataset.attrs else None
    if attrib is None or not deserialize:
        return attrib
    return json.loads(attrib)


def write_h5_attrib(key, value, dataset_path, serialize=False):
    """Writes a tip level attribute value.. Creates the file/directory tree if it does not exist"""
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if serialize:
        value = json.dumps(value)
    with h5py.File(dataset_path, 'a') as dataset:
        dataset.attrs[key] = value


# @unittest.skip
class H5HelperTests(unittest.TestCase):

    def setUp(self):
        self.root = 'test'
        self.h5_file = os.path.join(self.root, 'test.h5')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_file(self):
        # File should create on creating a dataset
        self.assertFalse(os.path.isfile(self.h5_file))
        write_h5_data('test_data', [], self.h5_file)
        self.assertTrue(os.path.isfile(self.h5_file))
        os.unlink(self.h5_file)

        # File should create on writing an attribute
        self.assertFalse(os.path.isfile(self.h5_file))
        write_h5_attrib('test_attrib', 'test_value', self.h5_file)
        self.assertTrue(os.path.isfile(self.h5_file))

    def test_data_io(self):
        stored_data = np.random.rand(10, 10).astype(np.float32)
        write_h5_data('test_data', stored_data, self.h5_file)
        ret_data = read_h5_data('test_data', self.h5_file)
        self.assertTrue(np.array_equal(stored_data, ret_data))

        # Test data shape
        ret_shape = read_h5_data_shape('test_data', self.h5_file)
        self.assertEqual(ret_shape, (10, 10))

        # Test multiple same write
        write_h5_data('test_data', stored_data, self.h5_file)
        ret_data = read_h5_data('test_data', self.h5_file)
        self.assertTrue(np.array_equal(stored_data, ret_data))

        # Test multiple distiinct write persistence
        stored_data_dup = np.random.rand(10, 10).astype(np.float32)
        write_h5_data('test_data_dup', stored_data_dup, self.h5_file)
        ret_data_dup = read_h5_data('test_data_dup', self.h5_file)
        self.assertTrue(np.array_equal(stored_data_dup, ret_data_dup))
        ret_data = read_h5_data('test_data', self.h5_file)
        self.assertTrue(np.array_equal(stored_data, ret_data))

        # Test multiple same writes with different shapes
        stored_data_reshaped = np.random.rand(12, 8).astype(np.float32)
        write_h5_data('test_data', stored_data_reshaped, self.h5_file)
        ret_data_reshaped = read_h5_data('test_data', self.h5_file)
        self.assertTrue(np.array_equal(stored_data_reshaped, ret_data_reshaped))

    def test_attrib_io(self):
        # Simple attrib I/O
        attrib1, value1 = 'test', 20
        write_h5_attrib(attrib1, value1, self.h5_file)
        ret_value = read_h5_attrib(attrib1, self.h5_file)
        self.assertEqual(ret_value, value1)

        # Dictionary I/O
        attrib2, value2 = 'test_dict', {'a': 10, 'b': True}
        write_h5_attrib(attrib2, value2, self.h5_file, serialize=True)
        ret_value = read_h5_attrib(attrib2, self.h5_file, deserialize=True)
        self.assertEqual(ret_value, value2)

        # Test multiple write persistence
        ret_value = read_h5_attrib(attrib1, self.h5_file)
        self.assertEqual(ret_value, value1)


##
# Processing Utils
##


def cached(func):
    """Decorator for caching function output"""
    cache = {}

    def get_key(*args, **kwargs):
        key = 'cache'
        hshs = [str(arg.__hash__()) for arg in args]
        hshs += [str(arg.__hash__()) for arg in kwargs.values()]
        hshs.sort()
        return '_'.join([key] + hshs)

    def wrapper(*args, **kwargs):
        nonlocal cache
        key = get_key(*args, **kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


@cached
def get_fma_meta(meta_dir, fma_type):
    """Fetches meta data for all tracks of an FMA type as a Pandas DataFrame"""
    all_tracks = fma_utils.load(os.path.join(meta_dir, 'tracks.csv'))
    return all_tracks[all_tracks['set', 'subset'] == fma_type]


@cached
def get_fma_genres(meta_dir):
    """Fetches meta data for all genres as a Pandas DataFrame"""
    return fma_utils.load(os.path.join(meta_dir, 'genres.csv'))


def sample_segments(segments, req_samples):
    """Returns a random sample form a list of segments"""
    sample_idx = np.arange(segments.shape[0])
    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:req_samples]
    sample_idx.sort()
    return segments[sample_idx]


def load_track(path, sr, do_re_sample=True):
    """Loads a track, re-samples and converts to mono using librosa"""
    try:
        audio_data, track_sr = librosa.load(path, sr=sr if do_re_sample else None, mono=True)
    except Exception as ex:
        print("Error loading audio file (Skipping): ", path)
        traceback.print_exc()
        return
    if sr != track_sr:
        print('Incorrect sample rate {0} for file {1} (Skipping)'.format(track_sr, path))
        return
    return audio_data


# @unittest.skip
class UtilsTests(unittest.TestCase):

    def test_sample_segments(self):
        segments = np.random.rand(10, 10)
        sampled = sample_segments(segments, 5)
        self.assertEqual(sampled.shape, (5, 10))
        segments = np.random.rand(10, 4, 4, 4)
        sampled = sample_segments(segments, 2)
        self.assertEqual(sampled.shape, (2, 4, 4, 4))

    def test_cache(self):
        func = cached(lambda x: np.random.rand(x))
        data1 = func(10)
        self.assertEqual(data1.shape, (10,))
        data2 = func(10)
        self.assertIs(data1, data2)
        data3 = func(5)
        self.assertIsNot(data1, data3)

    def test_get_fma_meta(self):
        meta_dir = 'datasets/fma/fma_metadata'
        meta = get_fma_meta(meta_dir, 'small')
        self.assertIsInstance(meta, pd.DataFrame)
        meta = get_fma_genres(meta_dir)
        self.assertIsInstance(meta, pd.DataFrame)

    def test_load_track(self):
        # Valid sr
        track_path = 'datasets/fma/fma_small/000/000002.mp3'
        track_data = load_track(track_path, 44100, do_re_sample=False)
        self.assertIsInstance(track_data, np.ndarray)
        # Invalid sr
        track_data = load_track(track_path, 20000, do_re_sample=False)
        self.assertIsNone(track_data)
        # Resampling
        track_data = load_track(track_path, 20000, do_re_sample=True)
        self.assertIsInstance(track_data, np.ndarray)


##
# Pre-precess single tracks
##


def pre_process_track_e2e(audio_path, config):
    """Processes an audio track for End to End training
        Each track is broken down into #segments=segments_per_track of length #frames=frames_per_segment
    """
    # Load audio data
    audio_data = load_track(audio_path, config.sr, config.do_re_sample)
    if audio_data is None:
        return None

    # Check if sufficient frames are available
    possible_segments = int(audio_data.size / config.frames_per_segment)
    total_segments = possible_segments if config.segments_per_track is None else config.segments_per_track
    if possible_segments < total_segments:
        print('Not enough segments ({0})for file {1} (Skipping)'.format(possible_segments, audio_path))
        return

    # Trim and split the track into segments
    trimmed = audio_data[:possible_segments * config.frames_per_segment]
    segments = trimmed.reshape(possible_segments, config.frames_per_segment)

    # Sample required segments
    sampled = sample_segments(segments, total_segments)

    return sampled


def pre_process_track_mfcc(audio_path, config):
    """Processes an audio track for training in frequency domain using MFCCs
        Each track is broken down into #segments=segments_per_track each of shape (num_mfcc, frames_per_segment)
    """
    # Load audio data
    audio_data = load_track(audio_path, config.sr, config.do_re_sample)
    if audio_data is None:
        return None

    mfcc = librosa.feature.mfcc(y=audio_data, n_mfcc=config.num_mfcc, sr=config.sr, n_fft=config.n_fft,
                                hop_length=config.mfcc_hops)

    # Check if sufficient frames are available
    possible_segments = int(mfcc.shape[1] / config.frames_per_segment)
    total_segments = possible_segments if config.segments_per_track is None else config.segments_per_track
    if total_segments > possible_segments:
        print('Not enough segments {0} for file {1} (Skipping)'.format(possible_segments, audio_path))
        return

    # Trim and split into segments
    trimmed = mfcc[:, :possible_segments * config.frames_per_segment]
    segments = trimmed.reshape(config.num_mfcc, possible_segments, config.frames_per_segment)

    # Rearrange axes and sample segments
    segments = segments.swapaxes(0, 1)
    return sample_segments(segments, total_segments)


def pre_process_track_mfcc_old(audio_path, config):
    """Processes an audio track for training in frequency domain using MFCCs
        Each track is broken down into #segments=segments_per_track each of shape (num_mfcc, frames_per_segment)
    """
    # Load audio data
    audio_data = load_track(audio_path, config.sr, config.do_re_sample)
    if audio_data is None:
        return None

    # Determine the frames per segment so that the desired mfcc feature length can be obtained
    frames_per_segment = (config.mfcc_frames_per_segment - 1) * config.mfcc_hops

    # Check if sufficient frames are available
    possible_segments = int(audio_data.size / frames_per_segment)
    total_segments = possible_segments if config.segments_per_track is None else config.segments_per_track
    total_frames = frames_per_segment * total_segments
    if audio_data.size < total_frames:
        print('Not enough frames {0} for file {1} (Skipping)'.format(audio_data.size, audio_path))
        return

    # Trim and split the track into segments
    trimmed = audio_data[:possible_segments * frames_per_segment]
    segments = trimmed.reshape(possible_segments, frames_per_segment)

    # Sample required segments
    sampled = sample_segments(segments, total_segments)

    # mfcc
    mfcc = np.array([])
    for segment in sampled:
        converted = librosa.feature.mfcc(y=segment, n_mfcc=config.num_mfcc, sr=config.sr, hop_length=config.mfcc_hops)
        mfcc = np.concatenate((mfcc, converted.reshape(-1)))

    # Check mfcc size
    expected_size = total_segments * config.num_mfcc * config.mfcc_frames_per_segment
    try:
        assert mfcc.size == expected_size
    except AssertionError:
        print('Error in computing MFFCs. Expected Size: {0}*{1}*{2}={3}. Actual Size: {4}'
              .format(total_segments, config.num_mfcc, config.mfcc_frames_per_segment, expected_size, mfcc.size))
        return None

    # Reshape mfcc
    reshaped = mfcc.reshape(total_segments, config.num_mfcc, config.mfcc_frames_per_segment)
    return reshaped


def pre_process_track(track_index, mode, config):
    """Assigns the processing of a track to an appropriate processor using mode"""
    audio_path = fma_utils.get_audio_path(config.fma_audio_dir, track_index)
    if mode == 'e2e':
        track_data = pre_process_track_e2e(audio_path, config)
    elif mode == 'mfcc':
        track_data = pre_process_track_mfcc(audio_path, config)
    else:
        raise ('Invalid mode: {}'.format(mode))
    return track_data


class PreProcessTests(unittest.TestCase):

    def setUp(self):
        self.track_index = 2

    def test_single_track_e2e(self):
        config = E2eDataPrepConfig()
        config.segments_per_track = 10
        config.frames_per_segment = 1024
        audio_path = fma_utils.get_audio_path(config.fma_audio_dir, self.track_index)

        # Fixed no of segments
        segment_data = pre_process_track_e2e(audio_path, config)
        self.assertIsInstance(segment_data, np.ndarray)
        self.assertEqual(segment_data.shape, (10, 1024))

        # Get all possible segments
        config.segments_per_track = None
        segment_data = pre_process_track_e2e(audio_path, config)
        self.assertIsInstance(segment_data, np.ndarray)
        self.assertEqual(segment_data.shape, (segment_data.shape[0], 1024))
        self.assertGreater(segment_data.shape[0], 10)

    def test_single_track_mfcc(self):
        config = MfccDataPrepConfig()
        config.segments_per_track = 10
        config.frames_per_segment = 40
        config.num_mfcc = 32
        config.mfcc_hops = 128
        config.n_fft = 512
        audio_path = fma_utils.get_audio_path(config.fma_audio_dir, self.track_index)

        # Fixed no of segments
        segment_data = pre_process_track_mfcc(audio_path, config)
        self.assertIsInstance(segment_data, np.ndarray)
        self.assertEqual(segment_data.shape, (10, 32, 40))

        # Get all possible segments
        config.segments_per_track = None
        segment_data = pre_process_track_mfcc(audio_path, config)
        self.assertIsInstance(segment_data, np.ndarray)
        self.assertEqual(segment_data.shape, (segment_data.shape[0], 32, 40))
        self.assertGreater(segment_data.shape[0], 10)


###
# Partition Scaling
###

def scale_partition_with_standard_scaler(partition, mean, std):
    """Sales a partition with loaded data"""
    if not partition.is_loaded():
        raise Exception('Data for partition {} is not loaded'.format(partition.key))
    if partition.segment_data.size == 0:
        return
    data = partition.segment_data.reshape(partition.segment_data.shape[0], -1)
    # data -  mean.reshape(1, -1)
    # data = data / std.reshape(1, -1)
    data -= mean
    data /= std


def fit_standard_scaler_on_partitions(partitions, dataset_path):
    norms = load_scaler_norms(dataset_path)
    if norms is not None:
        mean, std = norms
    else:
        print('Computing Norms')
        count = 0
        total_sum = None
        total_square_sum = None
        for partition in partitions:
            if partition.key in [get_partition_key(part_type) for part_type in ['cv', 'test']]:
                print('Ignoring partition:', partition.key)
                continue
            partition.load_data()
            data = partition.segment_data.reshape(partition.segment_data.shape[0], -1)
            count += partition.segment_data.shape[0]
            if total_sum is None:
                total_sum = np.zeros(data.shape[1])
                total_square_sum = np.zeros(data.shape[1])
            total_sum += data.sum(axis=0)
            total_square_sum += np.square(data).sum(axis=0)
            partition.flush_data()
        mean = total_sum / count
        std = np.sqrt(((total_square_sum - (2 * mean * total_sum)) / count) + np.square(mean))
        norms = (mean, std)
        save_scaler_norms(norms, dataset_path)
    for partition in partitions:
        if partition.scaled:
            continue
        print('Scaling Partition', partition.key)
        partition.load_data()
        scale_partition_with_standard_scaler(partition, mean, std)
        partition.dirty = True
        partition.scaled = True
        partition.save()
        partition.flush_data()


def scale_partition_with_minmax_scaler(partition, abs_min, abs_max):
    """Sales a partition with loaded data"""
    if not partition.is_loaded():
        raise Exception('Data for partition {} is not loaded'.format(partition.key))
    data = partition.segment_data.reshape(partition.segment_data.shape[0], -1)
    data -= abs_min
    data /= (abs_max - abs_min)


def fit_minmax_scaler_on_partitions(partitions, dataset_path):
    norms = load_scaler_norms(dataset_path)
    if norms is not None:
        abs_min, abs_max = norms
    else:
        print('Computing Norms')
        abs_min = None
        abs_max = None
        for partition in partitions:
            if partition.key in [get_partition_key(part_type) for part_type in ['cv', 'test']]:
                print('Ignoring partition:', partition.key)
                continue
            partition.load_data()
            data = partition.segment_data.reshape(partition.segment_data.shape[0], -1)
            if abs_min is None:
                abs_min = data.min(axis=0)
                abs_max = data.max(axis=0)
            else:
                abs_min = np.minimum(abs_min, data.min(axis=0))
                abs_max = np.maximum(abs_max, data.max(axis=0))
            partition.flush_data()
        norms = (abs_min, abs_max)
        save_scaler_norms(norms, dataset_path)
    for partition in partitions:
        if partition.scaled:
            continue
        print('Scaling Partition', partition.key)
        partition.load_data()
        scale_partition_with_minmax_scaler(partition, abs_min, abs_max)
        partition.dirty = True
        partition.scaled = True
        partition.save()
        partition.flush_data()


def fit_scaler_on_partitions(partitions, dataset_path, scaler='standard'):
    if scaler == 'standard':
        fit_standard_scaler_on_partitions(partitions, dataset_path)
    elif scaler == 'minmax':
        fit_minmax_scaler_on_partitions(partitions, dataset_path)
    else:
        raise Exception('Invalid scaler: {}'.format(scaler))


def scale_partitions(partitions, norms, scaler='standard'):
    if scaler == 'standard':
        scale_partition_with_minmax_scaler(partitions, dataset_path)
    elif scaler == 'minmax':
        fit_minmax_scaler_on_partitions(partitions, dataset_path)
    else:
        raise Exception('Invalid scaler: {}'.format(scaler))


def save_scaler_norms(norms, dataset_path):
    norms_proc = []
    for norm in norms:
        norms_proc.append(norm.tolist())
    write_h5_attrib('scaler_norms', norms_proc, dataset_path, serialize=True)


def load_scaler_norms(dataset_path):
    norms_proc = read_h5_attrib('scaler_norms', dataset_path, deserialize=True)
    if norms_proc is None:
        return None
    norms = []
    for norm in norms_proc:
        norms.append(np.array(norm))
    return norms


# @unittest.skip
class PartitionScalerTests(unittest.TestCase):

    def setUp(self):
        self.root = 'test'
        self.dataset_path = os.path.join(self.root, 'dataset')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # @unittest.skip
    def test_standard_scaler(self):
        test_data = np.random.randint(100, size=(10, 10)).astype(np.float32)
        test_data_scaled = test_data - test_data[:8, :].mean(axis=0)
        test_data_scaled /= test_data[:8, :].std(axis=0)
        pkeys = ['p1', 'p2', get_partition_key('cv')]
        pdata = [test_data[:5, :], test_data[5:8, :], test_data[8:, :]]
        partitions = []
        for i in range(3):
            partition = Partition(pkeys[i], np.array([]), self.dataset_path)
            partition.segment_data = pdata[i]
            partition.segment_indices = []
            partition.dirty = True
            partition.save()
            partitions.append(partition)
        fit_standard_scaler_on_partitions(partitions, self.dataset_path)
        scaled_data = []
        for i in range(3):
            partitions[i].load_data()
            scaled_data.extend(partitions[i].segment_data.tolist())
        scaled_data = np.array(scaled_data).astype(np.float32)
        self.assertEqual(np.rint(scaled_data - test_data_scaled).sum(), 0)

    def test_minmax_scaler(self):
        test_data = np.random.randint(100, size=(10, 10)).astype(np.float32)
        test_data_scaled = test_data - test_data[:7, :].min(axis=0)
        test_data_scaled /= (test_data[:7, :].max(axis=0) - test_data[:7, :].min(axis=0))
        pkeys = ['p1', 'p2', get_partition_key('test')]
        pdata = [test_data[:5, :], test_data[5:7, :], test_data[7:, :]]
        partitions = []
        for i in range(3):
            partition = Partition(pkeys[i], np.array([]), self.dataset_path)
            partition.segment_data = pdata[i]
            partition.segment_indices = []
            partition.dirty = True
            partition.save()
            partitions.append(partition)
        fit_minmax_scaler_on_partitions(partitions, self.dataset_path)
        scaled_data = []
        for i in range(3):
            partitions[i].load_data()
            scaled_data.extend(partitions[i].segment_data.tolist())
        scaled_data = np.array(scaled_data).astype(np.float32)
        # self.assertEqual(np.rint(scaled_data - test_data_scaled).sum(), 0
        self.assertTrue(np.array_equal(scaled_data, test_data_scaled))


###
# Partitions
###


class Partition:
    """Represent a single partiton of the DataSet that can be loaded and processed in memory"""

    def __init__(self, key, indices, dataset_path):
        self.key = key  # Unique
        self.track_indices = indices  # FMA track indices
        self.dataset_path = dataset_path
        self.segment_data = None  # np array (num_segments x feature1 x feature2)
        self.segment_indices = None  # np array (num_segments x 1)
        self.cached = False  # Is data processed and saved?
        self.dirty = False  # Need to save data?
        self.scaled = False  # Scaling complete?

    def process_data(self, mode, config):
        """Start/Resume processing tracks in the current partition"""

        # print()
        done_indices = [] if self.segment_indices is None else set(np.unique(self.segment_indices))
        if self.segment_data is None:
            self.segment_data = []
            self.segment_indices = []
            self.dirty = True
        else:
            self.segment_data = self.segment_data.tolist()
            self.segment_indices = self.segment_indices.tolist()

        progress = utils.ProgressBar(
            len(self.track_indices),
            status="Processing {0} tracks in partition {1}".format(len(self.track_indices), self.key)
        )
        for curr_track, track_index in enumerate(self.track_indices):

            # Track already processed?
            if track_index not in done_indices:
                # Pre-process and store in cache
                segment_data = pre_process_track(track_index, mode, config)
                if segment_data is None:
                    continue
                segment_count = segment_data.shape[0]
                self.segment_data.extend(segment_data)
                self.segment_indices.extend([track_index] * segment_count)
                self.dirty = True

            # Print status
            # if curr_track % 30 == 0:
            #     print('Processed {0} of {1} tracks'.format(curr_track + 1, self.track_indices.size))
            progress.update(curr_track)
        progress.complete(status='Partition {} has been processed'.format(self.key))

        # Convert processed data to numpy ndarray
        self.segment_data = np.array(self.segment_data).astype(np.float32)
        self.segment_indices = np.array(self.segment_indices)

        # Were the required number of files read
        total_tracks_read = np.unique(self.segment_indices).size
        if total_tracks_read < len(self.track_indices):
            print('WARNING: Only {0} of {1} files were read'.format(total_tracks_read, len(self.track_indices)))

        # print("All tracks processed in partition", self.key)

    def load_data(self):
        """Load the processed partition data"""
        self.segment_data = read_h5_data(self.key + '/segment_data', self.dataset_path)
        self.segment_indices = read_h5_data(self.key + '/segment_indices', self.dataset_path, dtype=np.int32)
        self.dirty = False

    def save_data(self, force_save=False):
        """Save the processed partition data. Proceeds to save only if dirty flag is set. Can optionally be forced"""
        if not force_save and not self.dirty:
            return
        assert self.segment_data is not None
        write_h5_data(self.key + '/segment_data', self.segment_data, self.dataset_path)
        assert self.segment_indices is not None
        write_h5_data(self.key + '/segment_indices', self.segment_indices, self.dataset_path, dtype=np.int32)
        self.cached = True
        write_h5_attrib(self.key + '_cached', self.cached, self.dataset_path)
        self.dirty = False

    def save(self):
        """Saves all attributes of this class along with the processed data"""
        write_h5_data(self.key + '/track_indices', self.track_indices, self.dataset_path, dtype=np.int32)
        write_h5_attrib(self.key + '_cached', self.cached, self.dataset_path)
        write_h5_attrib(self.key + '_scaled', self.scaled, self.dataset_path)
        self.save_data()

    @staticmethod
    def load(key, dataset_path):
        """Load a partition (metadata only) from a h5 file using its unique key"""
        indices = read_h5_data(key + '/track_indices', dataset_path, dtype=np.int32)
        part = Partition(key, indices, dataset_path)
        part.cached = read_h5_attrib(key + '_cached', dataset_path)
        part.scaled = read_h5_attrib(key + '_scaled', dataset_path)
        return part

    def is_loaded(self):
        return self.segment_data is not None

    def flush_data(self):
        """Save partition data and release from memory"""
        self.save_data()
        self.segment_data = None
        self.segment_indices = None

    def get_num_segments(self):
        shape = read_h5_data_shape(self.key + '/segment_data', self.dataset_path)
        return 0 if shape is None else shape[0]


def create_partitions(tracks, num_tracks, num_train_partitions, cv_split, test_split, dataset_path):
    """Create partitions for train, test and cross-validation"""
    indices = np.array(tracks.index)
    np.random.shuffle(indices)
    indices = indices[:num_tracks]

    # CV
    if not 0 <= cv_split <= 1:
        raise ('Invalid cv_split: {}'.format(cv_split))
    cv_split_index = int(indices.size * cv_split)
    cv_part = Partition(get_partition_key('cv'), indices[:cv_split_index], dataset_path)

    # Test
    if not 0 <= test_split <= 1:
        raise ('Invalid test_split: {}'.format(test_split))
    test_split_index = cv_split_index + int(indices.size * test_split)
    test_part = Partition(get_partition_key('test'), indices[cv_split_index: test_split_index], dataset_path)

    # Train
    train_parts = []
    train_part_size = int((indices.size - test_split_index) / num_train_partitions)
    for part_num in range(num_train_partitions):
        part_start = test_split_index + (part_num * train_part_size)
        part_end = test_split_index + (
            (part_num + 1) * train_part_size if part_num < (num_train_partitions - 1) else indices.size)
        train_parts.append(Partition(get_partition_key('train', part_num), indices[part_start: part_end], dataset_path))

    return train_parts, cv_part, test_part


def load_created_partitions(dataset_path):
    with h5py.File(dataset_path, 'r') as dataset:
        partition_keys = list(dataset.keys())
    parts = [Partition.load(key, dataset_path) for key in partition_keys]
    train_parts = []
    cv_part = None
    test_part = None
    for part in parts:
        if 'train' in part.key:
            train_parts.append(part)
        elif 'cv' in part.key:
            cv_part = part
        elif 'test' in part.key:
            test_part = part
    return train_parts, cv_part, test_part


def get_partition_key(partition_type, partition_num=None):
    """Generates unique key for a partition using its attributes"""
    key = 'partition'
    if partition_type in ['cv', 'test']:
        return '_'.join([key, partition_type])
    elif partition_type == 'train':
        assert partition_num is not None
        return '_'.join([key, partition_type, str(partition_num)])
    else:
        raise Exception('Invalid partition type {}'.format(partition_type))


class PartitionBatchGenerator:

    def __init__(self, partitions, batch_size, mode='train', post_process=None):
        assert mode in ['train', 'cv']
        self.partitions = partitions if mode == 'train' else [partitions]
        self.batch_size = batch_size if batch_size is not None else self.partitions[0].get_num_segments()
        self.post_process = post_process if post_process is not None else lambda *_: _

    def __len__(self):
        size = 0
        for partition in self.partitions:
            size += np.ceil(partition.get_num_segments() / self.batch_size)
        return int(size)

    def __iter__(self):
        for partition in self.partitions:
            partition.load_data()
            indices = np.arange(partition.segment_data.shape[0])
            np.random.shuffle(indices)
            partition.segment_data[:] = partition.segment_data[indices]
            partition.segment_indices[:] = partition.segment_indices[indices]
            curr_batch_start = 0
            while curr_batch_start < partition.segment_data.shape[0]:
                curr_batch_end = min(curr_batch_start + self.batch_size, partition.segment_data.shape[0])
                curr_batch_segments = partition.segment_data[curr_batch_start: curr_batch_end]
                curr_batch_segment_indices = partition.segment_indices[curr_batch_start: curr_batch_end]
                curr_batch_start += self.batch_size
                yield self.post_process(curr_batch_segments, curr_batch_segment_indices)
            partition.flush_data()


# @unittest.skip
class PartitionTests(unittest.TestCase):

    def setUp(self):
        self.root = 'temp'
        self.dataset_path = os.path.join(self.root, 'dataset')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_partition_key(self):
        self.assertEqual(get_partition_key('cv'), 'partition_cv')
        self.assertEqual(get_partition_key('test'), 'partition_test')
        self.assertEqual(get_partition_key('train', 10), 'partition_train_10')
        with self.assertRaises(AssertionError):
            get_partition_key('train')
        with self.assertRaises(Exception):
            get_partition_key('unknown')

    def test_create_partitions(self):
        total_tracks = 200
        tracks = pd.DataFrame(np.ones((total_tracks, 5)))
        train_parts, cv_part, test_part = create_partitions(tracks, 100, 4, 0.1, 0.1, self.dataset_path)
        parts = train_parts + [cv_part, test_part]
        self.assertEqual(len(parts), 6)
        for part in parts:
            if 'train' in part.key:
                self.assertEqual(len(part.track_indices), 20)
            else:
                self.assertEqual(len(part.track_indices), 10)
        # Empty test partition
        train_parts, cv_part, test_part = create_partitions(tracks, 100, 3, 0.25, 0, self.dataset_path)
        parts = train_parts + [cv_part, test_part]
        self.assertEqual(len(parts), 5)
        for part in parts:
            if 'test' in part.key:
                self.assertEqual(len(part.track_indices), 0)
            else:
                self.assertEqual(len(part.track_indices), 25)

    def test_all_partitions_meta_io(self):
        total_tracks = 200
        tracks = pd.DataFrame(np.ones((total_tracks, 5)))
        train_parts, cv_part, test_part = create_partitions(tracks, 100, 4, 0.1, 0.1, self.dataset_path)
        parts = train_parts + [cv_part, test_part]
        for part in parts:
            part.save()
        # Load
        train_parts, cv_part, test_part = load_created_partitions(self.dataset_path)
        loaded_parts = train_parts + [cv_part, test_part]
        loaded_parts = {p.key: p for p in loaded_parts}
        for part in parts:
            self.assertIn(part.key, loaded_parts)
            loaded = loaded_parts[part.key]
            self.assertTrue(np.array_equal(part.track_indices, loaded.track_indices))

    def test_partition_meta_io(self):
        part = Partition('temp_key', np.arange(10), self.dataset_path)
        self.assertFalse(os.path.isfile(self.dataset_path))
        part.save()
        loaded = Partition.load('temp_key', self.dataset_path)
        self.assertEqual(loaded.key, part.key)
        self.assertTrue(np.array_equal(loaded.track_indices, part.track_indices))
        self.assertFalse(loaded.cached)
        self.assertFalse(loaded.dirty)

    def test_partition_data_io(self):
        # Save data
        part = Partition('temp_key', np.arange(10), self.dataset_path)
        self.assertFalse(os.path.isfile(self.dataset_path))
        part.save()
        part.segment_indices = np.ones((5,))
        part.segment_data = np.ones((5, 5))
        part.dirty = True
        part.flush_data()
        self.assertIsNone(part.segment_data)
        self.assertIsNone(part.segment_indices)
        self.assertTrue(part.cached)
        self.assertFalse(part.dirty)

        # Load Data
        loaded = Partition.load('temp_key', self.dataset_path)
        self.assertTrue(loaded.cached)
        self.assertFalse(loaded.dirty)
        self.dirty = True
        loaded.load_data()
        self.assertTrue(np.array_equal(loaded.segment_indices, np.ones((5,))))
        self.assertTrue(np.array_equal(loaded.segment_data, np.ones((5, 5))))
        self.assertFalse(loaded.dirty)
        self.assertTrue(loaded.cached)

        # Num Segments
        self.assertEqual(part.get_num_segments(), 5)

    def test_partition_processing(self):
        tracks_to_process = [2]
        config = MfccDataPrepConfig()
        config.segments_per_track = 10
        config.frames_per_segment = 40
        config.num_mfcc = 32
        config.mfcc_hops = 128

        part = Partition('temp_key', np.array(tracks_to_process), self.dataset_path)
        part.save()
        part.process_data('mfcc', config)

        self.assertIsInstance(part.segment_data, np.ndarray)
        self.assertEqual(part.segment_data.shape, (10, 32, 40))
        self.assertIsInstance(part.segment_indices, np.ndarray)
        self.assertEqual(part.segment_indices.shape, (10,))

    def test_generate_batch(self):
        # Save data
        pkeys = ['p1', 'p2']
        pdata = [np.ones(5), np.ones(7)]
        parts = []
        for i in range(2):
            part = Partition(pkeys[i], np.array([]), self.dataset_path)
            part.segment_data = pdata[i]
            part.segment_indices = pdata[i]
            part.dirty = True
            part.flush_data()
            parts.append(part)

        # Generator length
        batch_size = 2
        gen = PartitionBatchGenerator(parts, batch_size)
        gen_cv = PartitionBatchGenerator(parts[0], batch_size, mode='cv')
        self.assertEqual(len(gen), 7)
        self.assertEqual(len(gen_cv), 3)

        # Generated data
        count = 0
        for gn in [gen, gen_cv]:
            for gen_data, gen_indices in gn:
                if count >= 2:
                    break
                count += 1
                self.assertTrue(np.array_equal(gen_data, np.ones(batch_size)))

        # Single Batch
        batch_size = None
        gen = PartitionBatchGenerator(parts, batch_size)
        gen_cv = PartitionBatchGenerator(parts[0], batch_size, mode='cv')
        self.assertEqual(len(gen), 3)
        self.assertEqual(len(gen_cv), 1)

        # Generated data
        for gn in [gen, gen_cv]:
            for gen_data, gen_indices in gn:
                self.assertTrue(np.array_equal(gen_data, np.ones(5)))
                break


##
# Exec
##


def run():
    """Executes CLI
        Example: "python data_processor.py -m mfcc -c config.json -o" (Create partitions in MFCC mode)
        Example: "python data_processor.py -t" (Run Unit Tests)
    """

    # Arguments Parser
    parser = argparse.ArgumentParser(description='Prepare DataSet Partitions for later training')
    parser.add_argument('-m', '--mode', choices=['e2e', 'mfcc'], default='mfcc', help='Processing Mode')
    # parser.add_argument('-s', '--scaler', choices=['standard', 'minmax'], default='standard', help='Partition Scaler')
    parser.add_argument('-c', '--config_path', help='Path to config JSON')
    parser.add_argument('-o', '--override', action='store_true', help='Discard previous processing and start again')
    parser.add_argument('-t', '--tests', action='store_true', help='Run unit tests')

    # Parse arguments
    args = parser.parse_args()
    # mode, scaler, config_path, override, tests = args.mode, args.scaler, args.config_path, args.override, args.tests
    mode, config_path, override, tests = args.mode, args.config_path, args.override, args.tests
    print('Arguments: Mode:{0}\tOverride:{1}\tTest:{2}\tConfig Path:{3}'.format(mode, override, tests, config_path))

    # Run tests and exit
    if tests:
        print('Running Tests')
        suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
        unittest.TextTestRunner().run(suite)
        return

    # Load pre-processing specific default config and override with json config
    config = get_config_cls(mode)() if config_path is None else get_config_cls(mode).load_from_file(config_path)

    # Create or Load dataset file
    dataset_exists = os.path.isfile(config.get_dataset_path())
    print('Dataset already exists?', dataset_exists)
    if dataset_exists and not override:
        mode = read_h5_attrib('mode', config.get_dataset_path())
        config.update(read_h5_attrib('config', config.get_dataset_path(), deserialize=True))
        # scaler = read_h5_attrib('scaler', config.get_dataset_path())
        print('Read Mode:', mode)
        print('Read Config:', config.get_dict())
        # print('Read Scaler:', scaler)
    else:
        if dataset_exists:
            print('Deleted existing dataset file')
            os.unlink(config.get_dataset_path())
        print('Creating new dataset file')
        write_h5_attrib('mode', mode, config.get_dataset_path())
        write_h5_attrib('config', config.get_dict(), config.get_dataset_path(), serialize=True)
        # write_h5_attrib('scaler', scaler, config.get_dataset_path())

    # Load existing partitions from dataset or create new ones
    train_parts, cv_part, test_part = load_created_partitions(config.get_dataset_path())
    parts = train_parts[:]
    if cv_part is not None:
        parts.append(cv_part)
    if test_part is not None:
        parts.append(test_part)
    if len(parts) == 0:
        print('Creating new partitions')
        tracks = get_fma_meta(config.fma_meta_dir, config.fma_type)
        train_parts, cv_part, test_part = create_partitions(tracks, config.num_tracks, config.num_train_partitions,
                                                            config.cv_split, config.test_split,
                                                            config.get_dataset_path())
        parts = train_parts + [cv_part, test_part]
        for part in parts:
            part.save()

    # Start/Resume processing
    print('Processing partitions data')
    for part in parts:
        def _save_on_exit():
            print("Saving unsaved changes in Partition: {0} before exiting".format(part.key))
            part.save()
            print('Save complete')

        atexit.register(_save_on_exit)
        part.load_data()
        part.process_data(mode, config)
        part.flush_data()
        atexit.unregister(_save_on_exit)

    print('All partitions processed')

    # Scale partitions
    print('Scaling partitions using scaler', config.scaler)
    fit_scaler_on_partitions(parts, config.get_dataset_path(), config.scaler)
    print('Done')


if __name__ == '__main__':
    run()

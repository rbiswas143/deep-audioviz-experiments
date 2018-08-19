"""Classes and methods used across the project"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

import fma_utils

"""Generic Utils"""


class ProgressBar:
    """Terminal progress bar
    Arguments:
        total: Relative size corresponding to completed task
        status: In progress status message
        bar_len: Actual size of progress bar when full
        max_len: Anything beyond this is truncated
        update_freq: Write to terminal st this rate (increase to save time on i/o)
    """

    def __init__(self, total, status='', **kwargs):
        self.total = total
        self.status = status
        self.params = {
            'max_len': 150,
            'bar_len': 60,
            'update_freq': 2
        }
        self.params.update(kwargs)
        self.update_time = None
        self.completed = False

    def update(self, count):
        """Updates the progress bar with a new progress value"""

        curr_time = time.time()
        if self.update_time is not None and self.update_time + self.params['update_freq'] > curr_time:
            return
        self.update_time = curr_time

        filled_len = self.params['bar_len'] if self.total == 0 else \
            int(round(self.params['bar_len'] * count / float(self.total)))
        percents = 100.0 if self.total == 0 else round(100.0 * count / float(self.total), 1)

        clear = ' ' * self.params['max_len']  # Blank line is used to clear previous content on the terminal
        bar = '=' * filled_len + '-' * (self.params['bar_len'] - filled_len)
        output = ('[%s] %s%s ...%s' % (bar, percents, '%', self.status))
        if len(output) > self.params['max_len']:
            output = output[:self.params['max_len']]

        sys.stdout.write('%s\r' % clear)
        sys.stdout.write(output)
        sys.stdout.write('\n' if self.completed else '\r')
        sys.stdout.flush()

    def complete(self, status=''):
        """Fills the progress bar to completion and updates the status message"""
        self.completed = True
        self.status = status
        self.update_time = None
        self.update(self.total)


def cached(key_fn=lambda *a, **k: str(a + tuple(k.values()))):
    """Decorator for caching method outputs by key"""

    def args_wrap(fn):
        cache = {}

        def fn_wrap(*args, **kwargs):
            key = None if key_fn is None else key_fn(*args, **kwargs)
            if key not in cache:
                cache[key] = fn(*args, **kwargs)
            return cache[key]

        return fn_wrap

    return args_wrap


def plot_learning_curve(data_a, data_b, title='Learning Curve', x_label='Epoch', y_label='Loss', legend_a='Train',
                        legend_b='CV', block=False, close=False):
    """Plots learning curve for a training process in a non-blocking way
    Arguments:
        data_a: Numpy Array or list values to plot (typically training set errors)
        data_b: Optional Numpy Array or list values to plot (typically cv set errors)
        block: Set True to block code execution while the plot is displayed
        close: Set True to close an existing plot before plotting. Can be useful for showing plots successively
    """
    if close:
        plt.close()
    data_a = np.array(data_a).reshape(-1)
    plt.plot(np.arange(1, data_a.size + 1), data_a, label=legend_a)
    if data_b is not None:
        data_b = np.array(data_b).reshape(-1)
        plt.plot(np.arange(1, data_a.size + 1), data_b, label=legend_b)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show(block=block)


def save_learning_curve(data_a, data_b, path, title='Learning Curve', x_label='Epoch', y_label='Loss', legend_a='Train',
                        legend_b='CV'):
    """Saves learning curve for a training process in a non-blocking way
    Arguments:
        data_a: Numpy Array or list values to plot (typically training set errors)
        data_b: Optional Numpy Array or list values to plot (typically cv set errors)
        path: Save path
    """
    data_a = np.array(data_a).reshape(-1)
    plt.plot(np.arange(1, data_a.size + 1), data_a, label=legend_a)
    if data_b is not None:
        data_b = np.array(data_b).reshape(-1)
        plt.plot(np.arange(1, data_a.size + 1), data_b, label=legend_b)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(path)
    plt.close()


"""FMA DataSet Utils"""


@cached()
def get_fma_meta(meta_dir, fma_type):
    """Fetches meta data for all tracks of an FMA type as a Pandas DataFrame"""
    all_tracks = fma_utils.load(os.path.join(meta_dir, 'tracks.csv'))
    return all_tracks[all_tracks['set', 'subset'] == fma_type]


@cached()
def get_fma_genres(meta_dir):
    """Fetches meta data for all genres as a Pandas DataFrame"""
    return fma_utils.load(os.path.join(meta_dir, 'genres.csv'))


@cached()
def get_genres_map(meta_dir, fma_type, reverse=False):
    """Gets unique genres from the FMA DataSet, deterministically assigns them indices, and returns a map"""
    tracks = get_fma_meta(meta_dir, fma_type)
    unique = enumerate(sorted([g for g in tracks['track', 'genre_top'].unique() if g is not np.nan]))
    if reverse:
        return {g: i for i, g in unique}
    else:
        return {i: g for i, g in unique}


def map_indices_to_genre(seg_indices, meta_dir, fma_type):
    """Maps a list of track indices to genre indices (as assigned by 'get_genres_map')"""
    genre_map = get_genres_map(meta_dir, fma_type, reverse=True)
    tracks = get_fma_meta(meta_dir, fma_type)
    tracks = tracks['track'].loc[seg_indices]
    return np.array(list(map(lambda g: genre_map[g], tracks['genre_top']))).astype(np.long)


"""Base Classes"""


class BaseConfig:
    """Base class for configurations with some helping utilities"""

    @classmethod
    def load_from_file(cls, path):
        """Initializes default config and overrides it with config dictionary obtained from a json file"""
        config = cls()
        with open(path, 'r') as cf:
            config.update(json.load(cf))
        return config

    def update(self, dict_):
        """Overrides config with dictionary"""
        for key in dict_.keys():
            setattr(self, key, dict_[key])

    def get_dict(self):
        """Returns all config as a dictionary"""
        return self.__dict__


"""PyTorch Utils"""


def get_trainable_params(model):
    """Counts the trainable parameters of a PyTorch model"""
    count = 0
    for param in list(model.parameters()):
        if param.requires_grad:
            count += np.prod(param.size())
    return count

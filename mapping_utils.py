"""Utils for mapping extracted features to visual parameters"""

import data_processor as dp
import models
import commons

import torch
import numpy as np
import os
import pickle


class MappingConfig(commons.BaseConfig):
    """Configuration for mapping features to visual parameters"""

    def __init__(self):
        self.model = 'conv_ae_shared'
        self.train_config_path = models.trained_model_configs[self.model]
        self.feature_mapping = 'raw'  # options: raw, pca, kmeans, kmeans-pca
        self.feature_scaling = 'across'  # options: features, across
        self.scaling_method = 'standard'  # options: standard, minmax
        self.classifier_layer = None


def get_enc_scaled(enc, mode='across', method='standard', std_scale=0.4, save_dir=None, prefix=None):
    """Scale encodings using a specified technique (Scaler norms must be precomputed)
    Arguments:
        enc: Encoding to scale
        mode: Use 'across' to scale across all features and 'features' to scale each feature independently
        method: Use 'standard' for Standard Normalization and 'minmax' for Minmax Normalization
        std_scale: Scale standard deviation to this value after scaling (only applicable with method 'standard')
        save_dir: Used to build scaler path
        prefix: Used to build scaler path
    """

    assert mode in ['features', 'across']
    assert method in ['minmax', 'standard']

    enc_shape = enc.shape
    if mode == 'across':
        enc = enc.reshape(-1, 1)

    scaler_path = os.path.join(save_dir, '{}.{}.{}.scaler'.format(prefix, mode, method))
    with open(scaler_path, 'rb') as modfile:
        print('Loading saved scaler {}'.format(scaler_path))
        scaler = pickle.load(modfile)

    enc = scaler.transform(enc)
    if method == 'standard':
        # Scale between 0 and 1
        enc = (enc * std_scale) + 0.5
        enc = np.clip(enc, 0, 1)

    return enc.reshape(enc_shape)


def get_enc_pca(enc, analysis_dir, prefix):
    """PCA transform encodings using saved PCA model"""
    scaler_path = os.path.join(analysis_dir, '{}.pca.scaler'.format(prefix))
    pca_model_path = os.path.join(analysis_dir, '{}.pca.model'.format(prefix))
    with open(scaler_path, 'rb') as modfile:
        print('Loading saved scaler {}'.format(scaler_path))
        scaler = pickle.load(modfile)
        enc_scaled = scaler.transform(enc)
    with open(pca_model_path, 'rb') as modfile:
        print('Loading saved model {}'.format(pca_model_path))
        pca = pickle.load(modfile)
    return pca.transform(enc_scaled)


def get_enc_kmeans(enc, analysis_dir, prefix):
    """K-Means transform encodings using saved PCA model"""
    scaler_path = os.path.join(analysis_dir, '{}.kmeans.scaler'.format(prefix))
    model_path = os.path.join(analysis_dir, '{}.kmeans.model'.format(prefix))
    with open(scaler_path, 'rb') as modfile:
        print('Loading saved scaler {}'.format(scaler_path))
        scaler = pickle.load(modfile)
        enc_scaled = scaler.transform(enc)
    with open(model_path, 'rb') as modfile:
        print('Loading saved model {}'.format(model_path))
        kmeans = pickle.load(modfile)
    # Similarity: Inverse of Eucledian distances from cluster centroids
    enc_kmeans = kmeans.transform(enc_scaled)
    enc_kmeans = 1 / (enc_kmeans)
    return enc_kmeans


def encode(model, batches, train_config, request_config):
    """Encodes a track using the specified model"""
    enc = None
    for x, y in batches:
        with torch.no_grad():
            if train_config.model == 'cnn_classifier':
                classifier_block, classifier_layer_index = models.encoding_layer_options[
                    request_config.model][request_config.classifier_layer]
                batch_enc = model.encode(x, classifier_block, classifier_layer_index)
            elif train_config.model == 'conv_autoencoder':
                batch_enc = model.encode(x)
            enc = batch_enc.cpu().numpy() if enc is None else np.concatenate([enc, batch_enc.cpu().numpy()])
    enc = enc.reshape(enc.shape[0], -1)
    return enc


def map_and_scale(enc, request_config, train_config):
    """Maps and scales a raw encoding using the specified configuration"""

    # For classifiers, analysis directory is based on extraction layer
    analysis_dir = os.path.join(train_config.models_dir, 'analysis')
    if train_config.model == 'cnn_classifier':
        analysis_dir = os.path.join(analysis_dir, request_config.classifier_layer)

    if request_config.feature_mapping == 'pca':
        analysis_dir = os.path.join(analysis_dir, 'pca')
        prefix = train_config.name
        enc = get_enc_pca(enc, analysis_dir, prefix)
    elif request_config.feature_mapping == 'kmeans':
        analysis_dir = os.path.join(analysis_dir, 'kmeans')
        prefix = train_config.name
        enc = get_enc_kmeans(enc, analysis_dir, prefix)
    elif request_config.feature_mapping == 'kmeans-pca':
        analysis_dir_kmeans = os.path.join(analysis_dir, 'kmeans')
        prefix = train_config.name
        enc = get_enc_kmeans(enc, analysis_dir_kmeans, prefix)
        analysis_dir = os.path.join(analysis_dir, 'kmeans-pca')
        prefix = "{}.kmeans-pca".format(train_config.name)
        enc = get_enc_pca(enc, analysis_dir, prefix)
        prefix = train_config.name
    elif request_config.feature_mapping == 'raw':
        analysis_dir = os.path.join(analysis_dir, 'raw')
        prefix = 'stats'
    else:
        raise Exception('Invalid feature mapping: {}'.format(request_config.feature_mapping))

    # Scale
    return get_enc_scaled(enc, mode=request_config.feature_scaling, method=request_config.scaling_method, save_dir=analysis_dir, prefix=prefix)


def generate_partition(track_path, dataset_mode, dataset_config):
    """Pre-process a track and create a dummy data partition while preserving the order of samples"""
    processed = dp.pre_process_track(track_path, dataset_mode, dataset_config, sample=False)
    partition = dp.Partition('track', None, None)
    partition.segment_data = processed.astype(np.float32)
    partition.segment_indices = np.arange(processed.shape[0])
    norms = dp.load_scaler_norms(dataset_config.get_dataset_path())
    if dataset_config.scaler == 'standard':
        dp.scale_partition_with_standard_scaler(partition, *norms)
    elif dataset_config.scaler == 'minmax':
        dp.scale_partition_with_minmax_scaler(partition, *norms)
    else:
        raise Exception('Invalid scaler: {}'.format(dataset_config.scaler))
    return partition

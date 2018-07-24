import data_processor as dp
import models

import torch
import numpy as np
import os
import sklearn
import pickle


class MappingConfig(dp.BaseConfig):

    def __init__(self):
        self.model = 'conv_ae_shared'
        self.train_config_path = models.trained_model_configs[self.model]
        self.feature_mapping = 'normal'
        self.kmeans_softmax = True
        self.feature_scaling = 'across'
        self.classifier_layer = None


def get_enc_scaled(enc, mode='across'):
    assert mode in ['features', 'across']
    if mode == 'features':
        return sklearn.preprocessing.MinMaxScaler().fit_transform(enc)
    else:
        return sklearn.preprocessing.MinMaxScaler().fit_transform(enc.reshape(-1, 1)).reshape(enc.shape)


def get_enc_pca(enc, analysis_dir, prefix):
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


def get_enc_kmeans(enc, analysis_dir, prefix, softmax=True):
    scaler_path = os.path.join(analysis_dir, '{}.kmeans.scaler'.format(prefix))
    model_path = os.path.join(analysis_dir, '{}.kmeans.model'.format(prefix))
    with open(scaler_path, 'rb') as modfile:
        print('Loading saved scaler {}'.format(scaler_path))
        scaler = pickle.load(modfile)
        enc_scaled = scaler.transform(enc)
    with open(model_path, 'rb') as modfile:
        print('Loading saved model {}'.format(model_path))
        kmeans = pickle.load(modfile)
    enc_kmeans = kmeans.transform(enc_scaled)
    enc_kmeans = 1 / (1 + enc_kmeans)
    if softmax:
        enc_kmeans = np.exp(enc_kmeans) / np.exp(enc_kmeans).sum(axis=1, keepdims=True)
    return enc_kmeans


def encode(model, batch, train_config, request_config):
    enc = None
    for x, y in batch:
        with torch.no_grad():
            if train_config.model == 'cnn_classifier':
                classifier_block, classifier_layer_index = models.encoding_layer_options[
                    request_config.model][request_config.classifier_layer]
                enc = model.encode(x, classifier_block, classifier_layer_index)
            elif train_config.model == 'conv_autoencoder':
                enc = model.encode(x)
            enc = enc.cpu().numpy() if enc is None else np.concatenate([enc, enc.cpu().numpy()])
    enc = enc.reshape(enc.shape[0], -1)
    return enc


def get_mapping(enc, request_config, train_config):
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
        enc = get_enc_kmeans(enc, analysis_dir, prefix, softmax=request_config.kmeans_softmax)
    elif request_config.feature_mapping == 'kmeans-pca':
        analysis_dir_kmeans = os.path.join(analysis_dir, 'kmeans')
        prefix = train_config.name
        enc = get_enc_kmeans(enc, analysis_dir_kmeans, prefix, softmax=request_config.kmeans_softmax)
        analysis_dir = os.path.join(analysis_dir, 'kmeans-pca')
        prefix = "{}.kmeans{}-pca".format(train_config.name, '-softmax' if request_config.kmeans_softmax else '')
        enc = get_enc_pca(enc, analysis_dir, prefix)
    elif request_config.feature_mapping == 'normal':
        pass
    else:
        raise Exception('Invalid feature mapping: {}'.format(request_config.feature_mapping))
    return enc


def generate_partition(track_path, dataset_mode, dataset_config):
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

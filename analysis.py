
# coding: utf-8

# In[ ]:


# IMPORTS

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import train
import data_processor as dp
import utils
import models

import torch
import numpy as np
import pandas as pd
import sklearn
import importlib
import pylab
import matplotlib.pyplot as plt
import os
import pickle


# In[ ]:


# DATA LOADER

# Block Vars
_quiet = True

# Load Model
model_name = 'conv_ae_shared'
classifier_block = 'features'
classifier_layer_index = 30
train_config_path = {
  'conv_ae_shared_test': 'models/test/conv_autoencoder_shared/config.json',
  'classifier_test': 'models/test/classifier_vgg16/config.json',
  'conv_ae_shared' : 'models/hp_tune_ae/conv_ae_mix/shared/config.json',
  'conv_ae_skip' : 'models/hp_tune_ae/conv_ae_mix/skip/config.json',
  'conv_ae_not_shared' : 'models/hp_tune_ae/conv_ae_mix/not_shared/config.json',
  'alexnet' : 'models/hp_tune_classifier/classifier_mix/alexnet_2/config.json',
  'vgg11' : 'models/hp_tune_classifier/classifier_mix/vgg11_2/config.json',
  'vgg13' : 'models/hp_tune_classifier/classifier_mix/vgg13_2/config.json',
  'vgg16' : 'models/hp_tune_classifier/classifier_mix/vgg16_2/config.json'
}[model_name]
train_config = train.TrainingConfig.load_from_file(train_config_path)
cuda = torch.cuda.is_available()

model = train_config.get_by_model_key(cuda)
checkpoint = models.ModelCheckpoint(model)
model.load_state(train_config.get_model_path('state_best'))
checkpoint.load(train_config.get_model_path('checkpoint_best'))
if not _quiet:
    print('Model [{}] loaded with weights. Cuda:{}.\nConfig:\n{}\nCheckpoint:\n{}\n'
          .format(train_config.name, cuda, train_config.get_dict(), checkpoint.get_dict()))
    
# Analysis Dir
analysis_dir = os.path.join(train_config.models_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)
if not _quiet:
    print("Analysis dir: {}".format(analysis_dir))

# Load Dataset
dataset_config = dp.DataPrepConfig.load_from_dataset(train_config.dataset_path)
train_parts, cv_part, test_part = dp.load_created_partitions(train_config.dataset_path)
if test_part.get_num_segments() == 0:
    raise Exception('No data in test set')
if not _quiet:
    print('Dataset [{}] loaded. Config:\n{}\n'.format(dataset_config.name, dataset_config.get_dict()))

test_set = dp.PartitionBatchGenerator(test_part, train_config.batch_size, mode='test')
test_set_len = len(test_set)
if not _quiet:
    print('Test Set Loaded. Batch Size:{} Num Batches:{}'.format(test_set.batch_size, test_set_len))

# Load Tracks
tracks = utils.get_fma_meta(dataset_config.fma_meta_dir, dataset_config.fma_type)
if not _quiet:
    print('FMA metadata loaded. Shape {}'.format(tracks.shape))


# In[ ]:


model.model._modules


# In[ ]:


# EVALUATION
def eval_test():
    loss_test = 0
    model.begin_evaluation()
    for x_test, y_test in test_set:
        loss_batch_test = model.evaluate(x_test, y_test)
        loss_test += loss_batch_test
    avg_loss_test = loss_test / test_set_len
    print('Average test loss per batch:', avg_loss_test)
    model.post_evaluation()
if False:
    eval_test()


# In[ ]:


# GENERATE ENCODINGS
def get_test_enc(train_config, test_set, classifier_block=None, classifier_layer_index=None, quiet=False):
    test_enc = None
    enc_segs = None
    for x_test, y_test in test_set:
        with torch.no_grad():
            if train_config.model == 'cnn_classifier':
                enc = model.encode(x_test, classifier_block, classifier_layer_index)
            elif train_config.model == 'conv_autoencoder':
                enc = model.encode(x_test)
            test_enc = enc.cpu().numpy() if test_enc is None else np.concatenate([test_enc, enc.cpu().numpy()])
            enc_segs = y_test if enc_segs is None else np.concatenate([enc_segs, y_test])
    if not quiet: print('Test set encoding shape: {}'.format(test_enc.shape))
    test_enc = test_enc.reshape(test_enc.shape[0], -1)
    if not quiet: print('Test set encoding reshaped: {}'.format(test_enc.shape))
    return test_enc, enc_segs
    
if True:
    _load_cached = True
    _cache_dir = os.path.join(analysis_dir, 'cached')
    os.makedirs(_cache_dir, exist_ok=True)
    _enc_file = os.path.join(_cache_dir, 'test_enc.npy')
    _segs_file = os.path.join(_cache_dir, 'enc_segs.npy')
    if _load_cached and os.path.isfile(_enc_file) and os.path.isfile(_segs_file):
        print('Loading saved encodings')
        test_enc, enc_segs = np.load(_enc_file), np.load(_segs_file)
    else:
        print('Generating encodings')
        test_enc, enc_segs = get_test_enc(train_config, test_set, classifier_block, classifier_layer_index)
        np.save(_enc_file, test_enc), np.save(_segs_file, enc_segs)
    print(test_enc.shape, enc_segs.shape)


# In[ ]:


# SCALE ENCODINGS
def get_enc_scaled(enc, mode='all'):
    assert mode in ['features', 'across']
    enc_min = enc.min(axis=0) if mode == 'features' else enc.reshape(-1).min()
    enc_max = enc.max(axis=0) if mode == 'features' else enc.reshape(-1).max()
    return (enc - enc_min) / (enc_max - enc_min)


# In[ ]:


# ENCODING STATS

# Overall stats
def print_enc_stats(enc, max_segs=1000000, max_encs=200, save_plots=False, save_dir=None, save_file_prefix=None):
    print('Num segments:', enc.shape[0])
    print('Distribution across entire encoding')
    print(pd.Series(enc.reshape(-1)).describe())
    
    pylab.rcParams['figure.figsize'] = (14,8)
    
    enc_idx = np.arange(enc.shape[1])
    if enc_idx.size > max_encs:
        print('Keeping only {} components'.format(max_encs))
        np.random.shuffle(enc_idx)
        enc_idx = enc_idx[:max_encs]
        enc_idx.sort()
        enc = enc[:, enc_idx]
    if enc.shape[0] > max_segs:
        print('Keeping only {} segments'.format(max_segs))
        idx = np.arange(enc.shape[0])
        np.random.shuffle(idx)
        idx = idx[:max_segs]
        enc = enc[:max_segs, :]

    print('Plotting stats for {} components'.format(enc_idx.size))

    x_label = 'Encoding Component'
    
    plt.subplot(2, 2, 1)
    plt.xlabel(x_label)
    plt.ylabel('Mean')
    plt.bar(enc_idx, enc.mean(axis=0))

    plt.subplot(2, 2, 2)
    plt.xlabel(x_label)
    plt.ylabel('Min')
    plt.bar(enc_idx, enc.min(axis=0))

    plt.subplot(2, 2, 4)
    plt.xlabel(x_label)
    plt.ylabel('Max')
    plt.bar(enc_idx, enc.max(axis=0))

    plt.subplot(2, 2, 3)
    plt.xlabel(x_label)
    plt.ylabel('Variance')
    plt.bar(enc_idx, enc.var(axis=0))

    if save_plots:
        path = os.path.join(analysis_dir, save_dir, "{}.desc.jpg".format(save_file_prefix))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        print('Plots saved to: {}'.format(path))
    plt.show()
    
    print('Plotting percentiles {} components'.format(enc_idx.size))
    pylab.rcParams['figure.figsize'] = (14,12)
    percentiles = [10, 30, 50 ,70, 90, 100]
    for i, p in enumerate(percentiles):
        plt.subplot(3, 2, i+1)
        plt.xlabel(x_label)
        plt.ylabel('{} Percentile'.format(p))
        plt.bar(enc_idx, np.percentile(enc, p, axis=0))
    if save_plots:
        path = os.path.join(analysis_dir, save_dir, "{}.percetiles.jpg".format(save_file_prefix))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        print('Plots saved to: {}'.format(path))
    plt.show()


# In[ ]:


# RAW ENCODING ANALYSIS
if True:
    _stats_dir = 'raw'
    for _scale in [None, 'features', 'across']:
        print('\n\nRaw Encoding Analysis. Scale: {}\n\n'.format(_scale))
        if _scale is None:
            _file_prefix = 'unscaled.stats'
            enc_scaled = test_enc
        else:
            enc_scaled = get_enc_scaled(test_enc, _scale)
            _file_prefix = 'scaled_{}.stats'.format(_scale)
        print_enc_stats(enc_scaled, save_plots=True, save_dir=_stats_dir, save_file_prefix=_file_prefix)


# In[ ]:


# ENCODING SCATTER PLOTS

def show_enc_scatter(enc, num_plots=10):
    pylab.rcParams['figure.figsize'] = (20, 20)
    dims_x = np.random.randint(0, enc.shape[1], num_plots)
    dims_y = np.random.randint(0, enc.shape[1], num_plots)
    for i in range(num_plots):
        dim1, dim2 = dims_x[i], dims_y[i]
        x = np.transpose(enc[:, dim1])
        y = np.transpose(enc[:, dim2])
        plt.subplot(int(num_plots/3)+1, 3, i+1)
        plt.xlabel('Dim {0}'.format(dim1))
        plt.ylabel('Dim {0}'.format(dim2))
        plt.scatter(x, y, marker='^', c='blue')
if False:
    show_enc_scatter(enc_scaled, num_plots=20)


# In[ ]:


# PCA

def get_enc_pca(enc, reduced_dims, save=False, load=False, save_dir=None, save_file_prefix=None):

    model_path = os.path.join(analysis_dir, save_dir, '{}.pca.model'.format(save_file_prefix))
    if load and os.path.isfile(model_path):
        with open(model_path, 'rb') as modfile:
            print('Loading saved model {}'.format(model_path))
            pca = pickle.load(modfile)
    else:
        pca = sklearn.decomposition.PCA(n_components=reduced_dims)
        pca.fit(enc)
        if save:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as modfile:
                pickle.dump(pca, modfile)
                print('Model saved to: {}'.format(model_path))
            
    enc_pca = pca.transform(enc)
    print('Variance retained: {}%'.format(pca.explained_variance_ratio_.sum()*100))
    if True:
        print('Variance by components')
        print(pca.explained_variance_ratio_.cumsum())
    return enc_pca


# In[ ]:


# PCA ANALYSIS
if True:
    _pca_model_prefix = train_config.name
    _save_dir = 'pca'
    enc_pca = get_enc_pca(test_enc, 10, save=True, load=True, save_dir=_save_dir, save_file_prefix=_pca_model_prefix)
    for _scale in [None, 'features', 'across']:
        print('\n\nPCA Analysis. Scale: {}\n\n'.format(_scale))
        if _scale is None:
            _stats_file_prefix = 'pca.unscaled.stats'
        else:
            enc_pca = get_enc_scaled(enc_pca, _scale)
            _stats_file_prefix = 'pca.scaled_{}.stats'.format(_scale)
        print_enc_stats(enc_pca, save_plots=True, save_dir=_save_dir, save_file_prefix=_stats_file_prefix)


# In[ ]:


# BEST CLUSTER

def get_best_cluster(enc, try_clusters=15):

    cluster_range = range( 1, try_clusters )
    cluster_errors = []

    for num_clusters in cluster_range:
        print('Checking cluster {} of {}'.format(num_clusters+1, try_clusters))
        clusters = sklearn.cluster.KMeans(num_clusters)
        clusters.fit(enc)
        cluster_errors.append(clusters.inertia_)

    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
    print('Cluster Errors')
    print(clusters_df)

    plt.figure(figsize=(12,6))
    plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
if False:
    get_best_cluster(enc_pca)


# In[ ]:


# KMEANS

def get_enc_kmeans(enc, reduced_dims, softmax=True, save=False, load=False, save_dir=None, save_file_prefix=None):
    
    model_path = os.path.join(analysis_dir, save_dir, '{}.kmeans.model'.format(save_file_prefix))
    if load and os.path.isfile(model_path):
        with open(model_path, 'rb') as modfile:
            print('Loading saved model {}'.format(model_path))
            kmeans = pickle.load(modfile)
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters=reduced_dims)
        kmeans.fit(enc)
        if save:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as modfile:
                pickle.dump(kmeans, modfile)
                print('Model saved to: {}'.format(model_path))
    
    enc_kmeans = kmeans.transform(enc)
    print('Score', kmeans.score(enc))
    print('Data transformed', pd.Series(enc_kmeans.reshape(-1)).describe())
    enc_kmeans = 1 / (1 + enc_kmeans)
    print('Data similarity', pd.Series(enc_kmeans.reshape(-1)).describe())
    if softmax:
        enc_kmeans = np.exp(enc_kmeans) / np.exp(enc_kmeans).sum(axis=1, keepdims=True)
        print('Softmax similarity', pd.Series(enc_kmeans.reshape(-1)).describe())
    return enc_kmeans


# In[ ]:


# KMEANS ANALYSIS
if True:
    _kmeans_model_prefix = train_config.name
    _save_dir = 'kmeans'
    for _kmeans_softmax in [True, False]:
        enc_kmeans = get_enc_kmeans(test_enc, 10, softmax=_kmeans_softmax, save=True, load=True, save_dir=_save_dir, save_file_prefix=_kmeans_model_prefix)
        for _scale in [None, 'features', 'across']:
            print('\n\nK-Means Analysis. Scale: {}\n\n'.format(_scale))
            if _scale is None:
                _stats_file_prefix = 'kmeans{}.unscaled.stats'.format('.softmax' if _kmeans_softmax else '')
            else:
                enc_kmeans = get_enc_scaled(enc_kmeans, _scale)
                _stats_file_prefix = 'kmeans{}.scaled_{}.stats'.format('.softmax' if _kmeans_softmax else '', _scale)
            print_enc_stats(enc_kmeans, save_plots=True, save_dir=_save_dir, save_file_prefix=_stats_file_prefix)


# In[ ]:


# KMEANS PCA ANALYSIS
if True:
    _kmeans_model_prefix = train_config.name
    _save_dir_kmeans = 'kmeans'
    _save_dir_kmeans_pca = 'kmeans-pca'
    for _kmeans_softmax in [True, False]:
        _pca_model_prefix = "{}.kmeans{}-pca".format(train_config.name, '-softmax' if _kmeans_softmax else '')
        enc_kmeans = get_enc_kmeans(test_enc, 10, softmax=_kmeans_softmax, save=True, load=True, save_dir=_save_dir_kmeans, save_file_prefix=_kmeans_model_prefix)
        enc_pca = get_enc_pca(enc_kmeans, 10, save=True, load=True, save_dir=_save_dir_kmeans_pca, save_file_prefix=_pca_model_prefix)
        for _scale in [None, 'features', 'across']:
            print('\n\nK-Means PCA Analysis. Scale: {}\n\n'.format(_scale))
            if _scale is None:
                _stats_file_prefix = 'kmeans-pca{}.unscaled.stats'.format('.softmax' if _kmeans_softmax else '')
            else:
                enc_pca = get_enc_scaled(enc_kmeans, _scale)
                _stats_file_prefix = 'kmeans-pca{}.scaled_{}.stats'.format('.softmax' if _kmeans_softmax else '', _scale)
            print_enc_stats(enc_pca, save_plots=True, save_dir=_save_dir_kmeans_pca, save_file_prefix=_stats_file_prefix)


# In[ ]:


# Done
import IPython
IPython.display.Audio("/home/rb/hdd/rsrcs/sounds/sms.mp3", autoplay=True)


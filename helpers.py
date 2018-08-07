"""Helper utils"""

import commons
import train
import models

import time
import os
import numpy as np
import json


def show_plot(train_config_file, opt=1):
    """Plot learning curve for a training process"""
    training_config = train.TrainingConfig.load_from_file(train_config_file)

    # Model initialization
    model = training_config.get_by_model_key(False)
    checkpoint = models.ModelCheckpoint(model)
    checkpoint.load(training_config.get_model_path('checkpoint'))
    if not checkpoint.loaded:
        print('Not evaluated')
        return

    if opt == 1:
        commons.plot_learning_curve(checkpoint.training_losses, checkpoint.cv_losses, close=True)
    elif opt == 2:
        commons.plot_learning_curve(checkpoint.cv_accuracies, checkpoint.model_specific['polled_accuracies'],
                                    close=True)
    else:
        return
    time.sleep(60)


def hp_grid_vgg16():
    """Generate random hyper parameters for Classifier"""
    size = 10
    lrs = 10 ** np.random.uniform(-5, 0, size).astype(np.float32)
    moms = np.random.choice([0.9, 0.9, 0.98], size)
    bns = np.random.choice(['true', 'false'], size)
    batch_sizes = np.random.choice([32, 64, 64, 128], size)

    hp_tune_dir = 'models/hp_tune_classifier/vgg13'

    for i, item in enumerate(zip(lrs, moms, bns, batch_sizes)):
        print(i, item)
        lr, mom, bn, batch_size = item
        data = {
            "name": "vgg13_hp_{}".format(i + 1),
            "num_epochs": 5,
            "batch_size": int(batch_size),
            "resume": True,
            "ignore": False,
            "models_dir": os.path.join(hp_tune_dir, 'hp_{}'.format(i + 1)),
            "dataset_path": "datasets/processed/timing/mfcc_classifier/mfcc_classifier_timing.h5",
            "model": "cnn_classifier",
            "model_params": {
                "input_dims": [
                    64,
                    96
                ],
                "num_classes": 8,
                "pretrained": True,
                "lr": float(lr),
                "momentum": float(mom),
                "batchnorm": bn,
                "arch": "vgg13"
            }
        }

        os.makedirs(os.path.join(hp_tune_dir, 'hp_{}'.format(i + 1)), exist_ok=True)
        with open(os.path.join(hp_tune_dir, 'hp_{}/config.json'.format(i + 1)), 'w') as cfile:
            json.dump(data, cfile, indent=2)
    configs = [os.path.join(hp_tune_dir, 'hp_{}/config.json'.format(i + 1)) for i in range(size)]
    with open(os.path.join(hp_tune_dir, 'hp_config.txt'), 'w') as cfile:
        cfile.write('\n'.join(configs))


def hp_grid_conv_ae():
    """Generate random hyper parameters for Autoencoder"""
    needed = 8
    size = 500

    lrs = 10 ** np.random.uniform(-4, -2, size).astype(np.float32)
    moms = np.random.choice([0.9, 0.95, 0.99], size)
    batch_sizes = np.random.choice([32, 64], size)
    num_init_filters = np.random.choice([16, 32, 32], size)
    num_pools = np.random.choice([4], size)
    num_fcs = np.random.choice([2, 2, 3, 4], size)
    fc_scale_downs = np.random.choice([2, 4, 4, 8, 8, 16], size)
    kernel_sizes = np.random.choice([3, 5], size)
    skip_conns = np.random.choice([False], size)

    param_range = [20000000, 30000000]

    hp_tune_dir = 'models/hp_tune_ae/conv_ae_shared_medium_final'

    done = 0
    for i, item in enumerate(
            zip(lrs, moms, batch_sizes, num_init_filters, num_pools, num_fcs, fc_scale_downs, kernel_sizes,
                skip_conns)):
        print(i, done, item)
        lr, mom, batch_size, num_init_filter, num_pool, num_fc, fc_scale_down, kernel_size, skip_conn = item
        data = {
            "name": "conv_ae_shared_medium_final_{}".format(done + 1),
            "num_epochs": 5,
            "batch_size": int(batch_size),
            "resume": True,
            "ignore": False,
            "models_dir": os.path.join(hp_tune_dir, 'hp_{}'.format(done + 1)),
            "dataset_path": "datasets/processed/medium/mfcc_ae/mfcc_ae_medium.h5",
            "model": "conv_autoencoder",
            "model_params": {
                "lr": float(lr),
                "momentum": float(mom),
                "input_dims": [64, 96],
                "enc_len": 10,
                "num_init_filters": int(num_init_filter),
                "num_pools": int(num_pool),
                "num_fc": int(num_fc),
                "fc_scale_down": int(fc_scale_down),
                "kernel_size": int(kernel_size),
                "padding": int(kernel_size / 2),
                "shared_weights": True,
                "skip_connections": bool(skip_conn),
                "enc_activation": "sigmoid"
            }
        }
        try:
            config = train.TrainingConfig()
            config.__dict__ = data
            model = config.get_by_model_key(False)
        except Exception as ex:
            print('Error:', ex)
            continue

        num_params = commons.get_trainable_params(model.model)
        if not param_range[0] <= num_params <= param_range[1]:
            print('Params not in range', num_params, 'less' if param_range[0] > num_params else 'more')
            continue

        os.makedirs(os.path.join(hp_tune_dir, 'hp_{}'.format(done + 1)), exist_ok=True)
        with open(os.path.join(hp_tune_dir, 'hp_{}/config.json'.format(done + 1)), 'w') as cfile:
            json.dump(data, cfile, indent=2)
        done += 1
        if done >= needed:
            break
    if done < needed:
        raise Exception('Could not complete')
    configs = [os.path.join(hp_tune_dir, 'hp_{}/config.json'.format(i + 1)) for i in range(needed)]
    with open(os.path.join(hp_tune_dir, 'hp_config.txt'), 'w') as cfile:
        cfile.write('\n'.join(configs))

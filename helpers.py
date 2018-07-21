import utils
import train
import models
import data_processor as dp

import torch
import time
import os
import numpy as np
import json


def show_plot(train_config_file, opt=1):
    training_config = train.TrainingConfig.load_from_file(train_config_file)

    # Model initialization
    model = training_config.get_by_model_key(False)
    checkpoint = models.ModelCheckpoint(model)
    checkpoint.load(training_config.get_model_path('checkpoint'))
    if not checkpoint.loaded:
        print('Not evaluated')
        return

    if opt == 1:
        utils.plot_learning_curve(checkpoint.training_losses, checkpoint.cv_losses, close=True)
    elif opt == 2:
        utils.plot_learning_curve(checkpoint.cv_accuracies, checkpoint.model_specific['polled_accuracies'], close=True)
    else:
        return
    time.sleep(60)


def hp_grid_vgg16():
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

        num_params = utils.get_trainable_params(model.model)
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


def encode_test_partition(train_config_path, output_dir, block=None, index=None):
    train_config = train.TrainingConfig.load_from_file(train_config_path)
    cuda = torch.cuda.is_available()

    model = train_config.get_by_model_key(cuda)
    model.load_state(train_config.get_model_path('state_best'))

    train_parts, cv_part, test_part = dp.load_created_partitions(train_config.dataset_path)
    test_set = dp.PartitionBatchGenerator(test_part, train_config.batch_size, mode='test')
    test_set_len = len(test_set)

    test_enc = torch.tensor([])
    progress = utils.ProgressBar(test_set_len, status='Encoding in progress')
    enc_start_time = time.time()
    for i, (x_test, y_test) in enumerate(test_set):
        with torch.no_grad():
            if train_config.model == 'cnn_classifier':
                assert block is not None and index is not None
                enc = model.encode(x_test, block, index)
            elif train_config.model == 'conv_autoencoder':
                enc = model.encode(x_test)
            test_enc = torch.cat([test_enc, enc.cpu()])
        progress.update(i)
    enc_stop_time = time.time()
    enc_time = enc_stop_time - enc_start_time
    progress.complete(status='Encoded in {} seconds'.format(enc_time))

    if train_config.model == 'cnn_classifier':
        output_path = os.path.join(output_dir, "{}_B{}_L{}.encoding".format(train_config.name, block, index))
    elif train_config.model == 'conv_autoencoder':
        output_path = os.path.join(output_dir, "{}.encoding".format(train_config.name))
    np.save(output_path, test_enc.numpy())

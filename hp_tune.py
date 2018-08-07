"""CLI and utils for training a batch of models and analysing hyper parameter tuning results"""

import train
import models
import data_processor as dp
import commons

import argparse
import torch
import os
import collections


def train_models(training_configs, email=False):
    """Train a batch of models"""
    for i, config in enumerate(training_configs):
        print('\nTraining Model {} of {}: {}'.format(i + 1, len(training_configs), config.name))
        train.train(config, plot_learning_curves=False, cuda=torch.cuda.is_available(), email=email)
    print('All Models have been evaluated')


def print_evaluation_report(training_config):
    """Print the training and evaluation results for a model"""
    # Training Config
    print('Training Config')
    for key, val in training_config.__dict__.items():
        print('{}\t{}'.format(key, val))
    print()

    # Checkpoint
    model = training_config.get_by_model_key(False)
    checkpoint = models.ModelCheckpoint(model)
    checkpoint.load(training_config.get_model_path('checkpoint'))
    if not checkpoint.loaded:
        print('Not evaluated')
        return

    print('Last checkpoint stats')
    for key, val in checkpoint.__dict__.items():
        print('{}\t{}'.format(key, val))


def _get_hps_for_autoencoder(training_config, checkpoint):
    hps = collections.OrderedDict()
    hps['name'] = training_config.name
    hps['trainable_params'] = checkpoint.trainable_params
    hps['epoch'] = checkpoint.epoch
    hps['best_loss'] = checkpoint.best_loss
    hps['batch_size'] = training_config.batch_size
    hps['lr'] = training_config.model_params['lr']
    hps['momentum'] = training_config.model_params['momentum']
    hps['num_init_filters'] = training_config.model_params['num_init_filters']
    hps['num_pools'] = training_config.model_params['num_pools']
    hps['num_fc'] = training_config.model_params['num_fc']
    hps['fc_scale_down'] = training_config.model_params['fc_scale_down']
    hps['kernel_size'] = training_config.model_params['kernel_size']
    hps['shared_weights'] = training_config.model_params['shared_weights']
    hps['skip_connections'] = training_config.model_params['skip_connections']
    return ["{} : {}".format(key, val) for key, val in hps.items()]


def _get_hps_for_classifier(training_config, checkpoint):
    hps = collections.OrderedDict()
    hps['name'] = training_config.name
    hps['trainable_params'] = checkpoint.trainable_params
    hps['epoch'] = checkpoint.epoch
    hps['best_loss'] = checkpoint.best_loss
    hps['batch_size'] = training_config.batch_size
    hps['lr'] = training_config.model_params['lr']
    hps['momentum'] = training_config.model_params['momentum']
    hps['arch'] = training_config.model_params['arch']
    hps['batchnorm'] = training_config.model_params['batchnorm']
    return ["{} : {}".format(key, val) for key, val in hps.items()]


def save_evaluation_report(training_configs, config_path):
    """Compile and save hyper-tuning report for all models in the batch"""
    hps = []
    for i, training_config in enumerate(training_configs):
        print('Saving report for Model {}: {}'.format(i + 1, training_config.name))
        model = training_config.get_by_model_key(False)
        checkpoint = models.ModelCheckpoint(model)
        checkpoint.load(training_config.get_model_path('checkpoint'))
        if not checkpoint.loaded:
            print('Not evaluated')
            continue
        if training_config.model == 'conv_autoencoder':
            hps.append(_get_hps_for_autoencoder(training_config, checkpoint))
        elif training_config.model == 'cnn_classifier':
            hps.append(_get_hps_for_classifier(training_config, checkpoint))
        else:
            raise Exception('Invalid model code: {}'.format(training_configs.model))
    with open(os.path.join(os.path.dirname(config_path), 'hps.txt'), 'w') as rep_file:
        rep_file.write('\n'.join(['\t'.join(hp) for hp in hps]))


def save_evaluation_plots(training_configs):
    """Create and save learning curves for all models in the batch"""
    for i, training_config in enumerate(training_configs):
        print('Saving plot for Model {}: {}'.format(i + 1, training_config.name))
        model = training_config.get_by_model_key(False)
        checkpoint = models.ModelCheckpoint(model)
        checkpoint.load(training_config.get_model_path('checkpoint'))
        if not checkpoint.loaded:
            print('Not evaluated')
            continue
        path = os.path.join(training_config.models_dir, "{}_lc.png".format(training_config.name))
        commons.save_learning_curve(checkpoint.training_losses, checkpoint.cv_losses, path)


def cli():
    """Runs CLI"""
    # Arguments Parser
    parser = argparse.ArgumentParser(description='Hyper Parameter tuning related actions')
    parser.add_argument('-c', '--config_files_path', help='Path to a file containing a list of training config files')
    parser.add_argument('-m', '--mode', choices=['train', 'print-report', 'save-hps', 'save-plots'], default='train',
                        help='Action to perform')
    parser.add_argument('-e', '--email', action='store_true', help='Send emails')
    parser.add_argument('-d', '--dataset', action='store_true', help='Print Dataset Details')

    # Parse arguments
    args = parser.parse_args()

    # Get model configs (read a single config file with newline separated paths to model configs)
    if args.config_files_path is None:
        raise Exception('Config file not specified')
    else:
        with open(args.config_files_path, 'r') as cfile:
            config_files = cfile.read().split('\n')
    train_configs = [train.TrainingConfig.load_from_file(fl) for fl in config_files]

    # Actions
    if args.mode == 'train':
        # Train a batch of models
        train_models(train_configs, email=args.email)
    elif args.mode == 'print-report':
        # Print report for all models
        for i, train_config in enumerate(train_configs):
            if args.dataset and i == 0:
                dataset_config = dp.DataPrepConfig.load_from_dataset(train_config.dataset_path)
                print('Dataset config for Model 1')
                for key, val in dataset_config.__dict__.items():
                    print('{}\t{}'.format(key, val))
            print()
            print('*' * 10 + 'Model {}: {}'.format(i + 1, train_config.name))
            print_evaluation_report(train_config)
            print()
    elif args.mode == 'save-hps':
        # Save hyper parameters for all models
        save_evaluation_report(train_configs, args.config_files_path)
    elif args.mode == 'save-plots':
        # Save learning curves for all models
        save_evaluation_plots(train_configs)
    else:
        raise Exception('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    cli()

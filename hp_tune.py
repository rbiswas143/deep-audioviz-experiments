import train
import models
import data_processor as dp
import emailer

import argparse
import torch
import traceback


def train_models(train_configs, email=False):
    for i, config in enumerate(train_configs):
        print('\nTraining Model {} of {}: {}'.format(i + 1, len(train_configs), config.name))
        try:
            train.train(config, plot_learning_curves=False, cuda=torch.cuda.is_available())
            if email:
                emailer.sendmail(
                    'HP Tuning - Model Trained - {}'.format(config.name),
                    str(config.get_dict())
                )
        except Exception as ex:
            if email:
                emailer.sendmail(
                    'HP Tuning - Model Training Failed - {}'.format(config.name),
                    'Model: {}\n\nError: {}'.format(str(config.get_dict()), traceback.format_exc())
                )
            traceback.print_exc()

    print('All Models have been evaluated')


def print_evaluation_report(training_configs):
    for i, training_config in enumerate(training_configs):
        if i > 0:
            print()
        else:
            dataset_config = dp.DataPrepConfig.load_from_dataset(training_config.dataset_path)
            print('Dataset config for Model 1')
            for key, val in dataset_config.__dict__.items():
                print('{}\t{}'.format(key, val))
            print()

        print('*' * 10 + 'Model {}: {}'.format(i + 1, training_config.name))

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
            continue

        print('Last checkpoint stats')
        for key, val in checkpoint.__dict__.items():
            print('{}\t{}'.format(key, val))
        print()


if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser(description='Hyper Parameter tuning related actions')
    parser.add_argument('-c', '--config_files_path', help='Path to a file containing a list of training config files')
    parser.add_argument('-m', '--mode', choices=['train', 'report'], default='report', help='Action to perform')
    parser.add_argument('-e', '--email', action='store_true', help='Send emails')

    # Parse arguments
    args = parser.parse_args()

    # Get model configs
    if args.config_files_path is None:
        raise Exception('Config file not specified')
    else:
        with open(args.config_files_path, 'r') as cfile:
            config_files = cfile.read().split('\n')
    train_configs = [train.TrainingConfig.load_from_file(fl) for fl in config_files]

    # Actions
    if args.mode == 'train':
        train_models(train_configs, email=args.email)
    elif args.mode == 'report':
        print_evaluation_report(train_configs)
    else:
        raise Exception('Invalid mode: ' + args.mode)

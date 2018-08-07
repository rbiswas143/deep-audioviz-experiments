"""Build and train a model"""
import data_processor as dp
import models
import emailer
import commons

import time
import os
import argparse
import torch
import traceback


"""Model Training Configuration """


class TrainingConfig(commons.BaseConfig):
    """All configuration and hyper parameters for creating and training a model"""

    def __init__(self):
        self.name = 'train_{0}'.format(int(time.time()))
        self.num_epochs = 30
        self.batch_size = 4
        self.resume = True  # Use False to train from scratch
        self.ignore = False  # Use False to skip training (Useful for ignoring a model while training multiple models)
        self.models_dir = 'models/VGG16_6k'
        self.dataset_path = 'datasets/processed/mfcc_vgg_6k/mfcc_vgg_6k'
        self.model = 'VGG16'
        self.model_params = {} # Model-specific parameters and hyper-parameters

    def get_by_model_key(self, cuda):
        """Initializes and returns a model using the current configuration"""
        if self.model == 'cnn_classifier':
            model = models.CNNClassifier(
                dp.DataPrepConfig.load_from_dataset(self.dataset_path) if os.path.isfile(self.dataset_path) else None,
                cuda=cuda, **self.model_params)
        elif self.model == 'conv_autoencoder':
            model = models.ConvAutoencoder(cuda=cuda, **self.model_params)
        else:
            raise Exception('Unknown model: ' + self.model)
        return model

    def get_model_path(self, path_type):
        """Returns a path for a specified model training related resource"""
        if path_type == 'state':
            return os.path.join(self.models_dir, self.name + '.state.torch')
        elif path_type == 'state_best':
            return os.path.join(self.models_dir, self.name + '.state.best.torch')
        elif path_type == 'checkpoint':
            return os.path.join(self.models_dir, self.name + '.checkpoint.torch')
        elif path_type == 'checkpoint_best':
            return os.path.join(self.models_dir, self.name + '.checkpoint.best.torch')
        else:
            raise Exception('Invalid path_type: {}'.format(path_type))


"""Model Training"""


def train(training_config, plot_learning_curves=False, cuda=False, email=False):
    """Train a model using the specified training configuration
    Arguments:
        training_config: Instance of TrainingConfig
        plot_learning_curves: Whether to plot learning curves at the end of each epoch (Useful for monitoring training)
        cuda: Use True to train on GPU
        email: Use True to send email notifications on training completion or failure
    """
    print('Training model {} [CUDA = {}, Plot = {}]'.format(training_config.name, cuda, plot_learning_curves))
    # Exception block to catch training failures and send email notifications
    try:

        if training_config.ignore:
            print('Ignoring model')
            return

        # Model initialization
        model = training_config.get_by_model_key(cuda)

        # Load checkpoint
        checkpoint = models.ModelCheckpoint(model)
        print('Model Size: {} params'.format(checkpoint.trainable_params))
        if training_config.resume:
            model.load_state(training_config.get_model_path('state'))
            checkpoint.load(training_config.get_model_path('checkpoint'))

        # Data generators for Training and Validation sets
        train_parts, cv_part, test_part = dp.load_created_partitions(training_config.dataset_path)
        if len(train_parts) == 0:
            raise Exception('No training partitions found')
        training_set = dp.PartitionBatchGenerator(train_parts, training_config.batch_size, mode='train')
        training_set_len = len(training_set)
        cv_set = dp.PartitionBatchGenerator(cv_part, training_config.batch_size, mode='cv')
        cv_set_len = len(cv_set)

        if checkpoint.epoch >= training_config.num_epochs:
            print('Already completed {} epochs'.format(checkpoint.epoch))
            return

        # Training loop
        for curr_epoch in range(checkpoint.epoch, training_config.num_epochs):

            # Plot learning curves after first epoch
            if plot_learning_curves and curr_epoch > 0:
                commons.plot_learning_curve(checkpoint.training_losses, checkpoint.cv_losses, close=True)

            # Train on training set
            model.begin_training()
            loss = 0
            train_start_time = time.time()
            progress = commons.ProgressBar(training_set_len, status='Training epoch %s' % str(curr_epoch + 1))
            for i, (x, y) in enumerate(training_set):
                loss += model.train_batch(x, y)
                progress.update(i)
            train_stop_time = time.time()
            training_time = train_stop_time - train_start_time
            checkpoint.training_times.append(training_time)
            progress.complete(status='Done training epoch {} in {} seconds'.format(str(curr_epoch + 1), training_time))
            avg_loss = loss / training_set_len
            checkpoint.training_losses.append(avg_loss)
            print('Average training loss per batch:', avg_loss)

            # Evaluate on validation set
            model.begin_evaluation()
            loss_cv = 0
            for i, (x_cv, y_cv) in enumerate(cv_set):
                loss_batch_cv = model.evaluate(x_cv, y_cv)
                loss_cv += loss_batch_cv
            avg_loss_cv = loss_cv / cv_set_len
            checkpoint.cv_losses.append(avg_loss_cv)
            checkpoint.best_loss = avg_loss_cv if checkpoint.best_loss is None else min(checkpoint.best_loss,
                                                                                        avg_loss_cv)
            print('Average validation loss per batch:', avg_loss_cv)
            print('Best Loss:', checkpoint.best_loss)

            # Post evaluation model specific actions
            model.post_evaluation(checkpoint)

            print()

            # Checkpoint
            checkpoint.epoch += 1
            model.save_state(training_config.get_model_path('state'))
            checkpoint.save(training_config.get_model_path('checkpoint'))
            if checkpoint.best_loss == avg_loss_cv:
                model.save_state(training_config.get_model_path('state_best'))
                checkpoint.save(training_config.get_model_path('checkpoint_best'))

        print('Training complete')

        # Success email
        if email:
            emailer.sendmail(
                'Model Training Complete: {}'.format(training_config.name),
                'Model Config: {}\n Model Checkpoint: {}'.format(
                    str(training_config.get_dict()), str(checkpoint.get_dict()))
            )
    except Exception as ex:
        print('Model Training Failed: {}'.format(str(ex)))
        # Failure email
        if email:
            emailer.sendmail(
                'Model Training Failed: {}'.format(training_config.name),
                'Error: {}'.format(traceback.format_exc())
            )
        raise


"""CLI"""


def cli():
    # Arguments Parser
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-c', '--config_path', help='Path to training config JSON')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot Curves')
    parser.add_argument('-e', '--email', action='store_true', help='Send emails')

    # Parse arguments
    args = parser.parse_args()
    training_config_path = args.config_path
    plot = args.plot
    email = args.email
    print('Arguments')
    print('Training config path:', training_config_path)
    print('Plot:', plot)
    print('Email:', email)

    # Load training config
    config = TrainingConfig.load_from_file(
        training_config_path) if training_config_path is not None else TrainingConfig()

    # Train
    train(config, plot_learning_curves=plot, cuda=torch.cuda.is_available(), email=email)


if __name__ == '__main__':
    cli()

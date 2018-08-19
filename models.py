"""PyTorch models for Deep Audio Viz

Module Contents:
    Base Model for any model compatible with Deep Audio Viz training
    CNN Classifier model - adaptation of VGG and AlexNet
    Convolution Autoencoder model - Normal, Shared Weights and Skip Connections
    Model related utils - Checkpoint, analysis config, etc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.tensor
import os
import commons
import math
import numpy as np

import alexnet  # torchvision implementation with minor edits

"""Model Analysis Config"""

# Dictionary defining trained models selected after hyper parameter tuning
trained_model_configs = {
    'conv_ae_shared_test': 'models/test/conv_autoencoder_shared/config.json',
    'classifier_test': 'models/test/classifier_vgg16/config.json',
    'conv_ae_shared': 'models/hp_tune_ae/conv_ae_mix/shared/config.json',
    'conv_ae_skip': 'models/hp_tune_ae/conv_ae_mix/skip/config.json',
    'conv_ae_not_shared': 'models/hp_tune_ae/conv_ae_mix/not_shared/config.json',
    'alexnet': 'models/hp_tune_classifier/classifier_mix/alexnet_2/config.json',
    'vgg11': 'models/hp_tune_classifier/classifier_mix/vgg11_2/config.json',
    'vgg13': 'models/hp_tune_classifier/classifier_mix/vgg13_2/config.json',
    'vgg16': 'models/hp_tune_classifier/classifier_mix/vgg16_2/config.json'
}

# For classifiers, the activations of these layers are used to get extract encodings
encoding_layer_options = {
    'vgg16': {
        'L13': ('features', 43),
        'L14': ('classifier', 1),
        'L15': ('classifier', 4),
    },
    'vgg13': {
        'L10': ('features', 34),
    },
    'vgg11': {
        'L8': ('features', 28)
    },
    'alexnet': {
        'L6': ('classifier', 1)
    }
}

"""Utils"""


def initialize_weights(module):
    """Initializes weights for various types of neural network layers"""
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.01)
        if module.bias is not None:
            module.bias.data.zero_()
    return module


"""Model helpers"""


class ModelBase:
    """Base class for all models defining core functionality and utilities"""

    def __init__(self, cuda=True):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = None
        self.optimizer = None

    def save_state(self, path):
        """Saves the model weights and optimizer state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load_state(self, path):
        """Loads the model weights and optimizer state"""
        if not os.path.isfile(path):
            return None
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def init_checkpoint(self):
        """Override this to add model-specific attributes to checkpoint"""
        return {}

    def post_evaluation(self, checkpoint=None):
        """Override this to perform a model-specific task at the end of each epoch after evaluation during training"""
        pass


class ModelCheckpoint:
    """Defines the state of model training at any given time/epoch"""

    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.best_loss = None
        self.training_losses = []
        self.cv_losses = []
        self.cv_accuracies = []
        self.training_times = []
        self.model_specific = self.model.init_checkpoint()
        self.trainable_params = commons.get_trainable_params(model.model)  # Transient (only for reference)
        self.loaded = False  # Useful for determining if the checkpoint was loaded successfully

    def save(self, path):
        """Saves the checkpoint to disc"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_losses': self.training_losses,
            'cv_losses': self.cv_losses,
            'cv_accuracies': self.cv_accuracies,
            'training_times': self.training_times,
            'model_specific': self.model_specific
        }, path)

    def load(self, path):
        """Loads the checkpoint from disc"""
        if not os.path.isfile(path):
            return
        config_dict = torch.load(path)
        self.epoch = config_dict['epoch']
        self.best_loss = config_dict['best_loss']
        self.training_losses = config_dict['training_losses']
        self.cv_losses = config_dict['cv_losses']
        self.cv_accuracies = config_dict['cv_accuracies']
        self.training_times = config_dict['training_times']
        self.model_specific = config_dict['model_specific']
        # Mark as loaded
        self.loaded = True

    def get_dict(self):
        return self.__dict__


"""CNN Classifier"""


class CNNClassifier(ModelBase):
    """CNN Classifier model providing adaptations of VGG and AlexNet
    Arguments:
        dataset_config: config used to create the dataset used for training (used for getting genres
          of the input samples)
        input_dims: dimensions (x, y) of each input sample (Note: no of channels = 1 for MFCC input data)
        pretrained: True downloads and loads pre-trained weights provided by torchvision
        cuda: Use True to train on GPU
        num_classes: No of output genre classes
        lr: Learning rate (float)
        batchnorm: Use True to use Batch Normalization
        arch: Model to load: One of (alexnet, vgg11, vgg13 and vgg16)
    """

    def __init__(self, dataset_config, input_dims=(64, 96), pretrained=True, cuda=True, num_classes=-1,
                 lr=0.001, momentum=0.9, batchnorm=False, arch='vgg16'):

        super(CNNClassifier, self).__init__(cuda)

        # Model Params
        self.input_dims = input_dims
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.batchnorm = batchnorm
        self.arch = arch

        # Dataset Params
        if dataset_config is None:
            print('Warning dataset config is not available')
        self.fma_meta_dir = dataset_config.fma_meta_dir if dataset_config is not None else None
        self.fma_type = dataset_config.fma_type if dataset_config is not None else None

        # Validate
        assert tuple(self.input_dims) == (64, 96)

        # Model
        self.model = self._build_model()

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Cache (used for polled evaluation)
        self.eval_cache = {}

    def _build_model(self):
        # Load a pretrained model using 'arch'
        if self.arch == 'vgg11':
            model = torchvision.models.vgg11_bn(pretrained=self.pretrained) if self.batchnorm else \
                torchvision.models.vgg11(pretrained=self.pretrained)
        elif self.arch == 'vgg13':
            model = torchvision.models.vgg13_bn(pretrained=self.pretrained) if self.batchnorm else \
                torchvision.models.vgg13(pretrained=self.pretrained)
        elif self.arch == 'vgg16':
            model = torchvision.models.vgg16_bn(pretrained=self.pretrained) if self.batchnorm else \
                torchvision.models.vgg16(pretrained=self.pretrained)
        elif self.arch == 'alexnet':
            model = alexnet.alexnet(pretrained=self.pretrained)
            if self.batchnorm:
                print('Warning: bathnorm not available for alexnet')
        else:
            raise Exception('Unidentified arch: {}'.format(self.arch))

        # Replace first layer to match single channel input
        if self.arch == 'alexnet':
            # Note: Stride of first layer has been changed to make the activations after the first layer
            # comparable to the corresponding  original activations of AlexNet
            first_layer = nn.Conv2d(1, 64, kernel_size=11, stride=(1, 2), padding=2)
        else:
            first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.features = nn.Sequential(*([initialize_weights(first_layer)]
                                         + list(model.features.children())[1:]))

        # Replace fully connected layers to match 'input_dims' (TODO: Remove hardcoded dimensions (64, 96))
        if self.arch == 'alexnet':
            fc_dim = 256 * 6 * 4
        else:
            fc_dim = 512 * 2 * 3
        model.classifier = nn.Sequential(
            initialize_weights(nn.Linear(fc_dim, 4096)),
            nn.ReLU(True),
            nn.Dropout(),
            initialize_weights(nn.Linear(4096, 4096)),
            nn.ReLU(True),
            nn.Dropout(),
            initialize_weights(nn.Linear(4096, self.num_classes)),
        )
        return model.to(self.device)

    def init_checkpoint(self):
        """Adds accuracy metrics to the checkpoint"""
        return {
            'cv_accuracies': [],
            'polled_accuracies': []
        }

    def begin_training(self):
        """To be called before training"""
        self.model.train()

    def _process_inputs(self, x):
        """Coverts a Numpy Array input batch to a PyTorch Tensor of the same"""
        return torch.from_numpy(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])).to(self.device)

    def _process_outputs(self, y):
        """Coverts a Numpy Array of track indices to a PyTorch Tensor of corresponding genre indices"""
        return torch.from_numpy(commons.map_indices_to_genre(y, self.fma_meta_dir, self.fma_type)).to(self.device)

    def train_batch(self, x, y):
        """Forward and backward pass on a single batch"""
        x = self._process_inputs(x)
        y = self._process_outputs(y)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def begin_evaluation(self):
        """To be called before evaluation. Also, resets the evaluation cache"""
        self.model.eval()
        self.eval_cache = {
            'segment_indices': [],
            'genre_correct': [],
            'genre_pred': [],
            'accuracy_sum': 0,
            'num_batches': 0
        }

    def encode(self, x, block, index):
        """Extracts the encodings for the given input x from the specified layer
        Arguments:
            x: batch obtained from a data partition
            block: 'features' or 'classifier'
            index: index of layer in the specified block to extract the encodings forom
        """
        assert block in ['classifier', 'features']
        x = self._process_inputs(x)
        if block == 'classifier':
            # Forward pass on classifier block
            x = self.model._modules['features'](x)
            x = x.view(x.size(0), -1)
        # Layer wise forward pass till the extraction layer is reached
        layers = self.model._modules[block]._modules.values()
        assert index < len(layers)
        for i, layer in enumerate(layers):
            x = layer(x)
            if index == i:
                return x

    def evaluate(self, x, y_indices):
        """Computes loss on an evaluation batch. Also, caches info for evaluating accuracies later"""
        x = self._process_inputs(x)
        y_genres = self._process_outputs(y_indices)
        self.eval_cache['segment_indices'].extend(y_indices)
        self.eval_cache['genre_correct'].extend(y_genres.cpu().numpy())
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y_genres).item()

            y_pred_idx = y_pred.max(1, keepdim=True)[1]
            self.eval_cache['genre_pred'].extend(y_pred_idx.cpu().numpy())
            correct = y_pred_idx.eq(y_genres.view_as(y_pred_idx))
            accuracy = correct.sum().item() / y_genres.shape[0]
            self.eval_cache['accuracy_sum'] += accuracy
            self.eval_cache['num_batches'] += 1

            return loss

    def post_evaluation(self, checkpoint=None):
        """Computes sample level and track level (polled) accuracies and optionally saves them to the checkpoint"""
        avg_accuracy = self.eval_cache['accuracy_sum'] / self.eval_cache['num_batches']
        if checkpoint is not None:
            checkpoint.model_specific['cv_accuracies'].append(avg_accuracy)
        print('Average accuracy per batch:', avg_accuracy)

        polled_accuracy = self._evaluate_polled()
        if checkpoint is not None:
            checkpoint.model_specific['polled_accuracies'].append(polled_accuracy)
        print('Polled accuracy:', polled_accuracy)

    def _evaluate_polled(self):
        """Computes track level (polled) accuracies by aggregating votes across samples"""
        segment_indices = np.array(self.eval_cache['segment_indices']).reshape(-1)
        genre_correct = np.array(self.eval_cache['genre_correct']).reshape(-1)
        genre_pred = np.array(self.eval_cache['genre_pred']).reshape(-1)

        # Track wise correct and predicted genres
        idx_correct = {}
        idx_preds = {}
        for idx, correct, pred in zip(segment_indices, genre_correct, genre_pred):
            idx_correct[idx] = correct
            if idx not in idx_preds:
                idx_preds[idx] = []
            idx_preds[idx].append(pred)

        # Compute correct predictions at track level
        correct_count = 0
        for idx, pred_list in idx_preds.items():
            pred_list_uniq, pred_list_count = np.unique(pred_list, return_counts=True)
            max_freq = np.max(pred_list_count)
            num_max = 0
            pred_max = None
            for pred_uniq, pred_count in zip(pred_list_uniq, pred_list_count):
                if pred_count == max_freq:
                    num_max += 1
                    pred_max = pred_uniq
            if num_max == 1 and pred_max == idx_correct[idx]:
                correct_count += 1

        return correct_count / len(idx_correct) if len(idx_correct) > 0 else 1


"""Convolutional Autoencoder"""


class _ConvAutoencoderModel(nn.Module):
    """PyTorch model providing convoultional autoencoder with shared weights and skip connections
    The model is composed of repating pattern of convolutional and pooling blocks and the encoder and decoder are
    symmetric. A block in the encoder is comprised of 2 convolutional layers with filter sizes N and 2*N followed
    by a pooling layer with stride (2x2). The covolutional blocks are followed by a number of fully connected layers
    Arguments:
        input_dims: dimensions (x, y) of each input sample (Note: no of channels = 1 for MFCC input data)
        enc_len: Size of middle/encoding layer
        num_init_filters: No of filters in the first convolutional layer (No of filters double/halve with each layer)
        num_pools: No of repeating blocks in the encoder (and similarly decoder)
        num_fc: No of fully connected layers in the encoder (and similarly decoder)
        fc_scale_down: Factor by which no of units in successive fully connected layers are reduced
        kernel_size: Kernel size used in the convolution and deconvolution operations
        padding: Padding used in the convolution and deconvolution operations
        shared_weights: If True, decoder does not have its own weights and it uses the encoder's weights instead
        skip_connections: If True, connections are added from the convolutional blocks to their corresponding
          deconvolutional counterparts
        enc_activation: Activation used in final layer (Intermediate layers use ReLU). Sigmoid (default) ensures that
          the range of predictions is same as the processed input data, i.e., between 0 and 1
    """

    def __init__(self, input_dims, enc_len, num_init_filters=16, num_pools=2,
                 num_fc=2, fc_scale_down=8, kernel_size=5, padding=2,
                 shared_weights=False, skip_connections=False, enc_activation='sigmoid'):
        super(_ConvAutoencoderModel, self).__init__()

        # Parameters
        self.input_dims = input_dims
        self.enc_len = enc_len
        self.num_init_filters = num_init_filters
        self.num_pools = num_pools
        self.num_fc = num_fc
        self.fc_scale_down = fc_scale_down
        self.kernel_size = kernel_size
        self.padding = padding
        self.shared_weights = shared_weights
        self.skip_connections = skip_connections
        self.enc_activation = enc_activation

        # Validate
        self._validate_hp()

        # Encoder
        curr_filter, next_filter = 1, num_init_filters
        for p in range(num_pools):
            self.add_module('conv{}a'.format(p),
                            initialize_weights(nn.Conv2d(curr_filter, next_filter, kernel_size, padding=padding)))
            self.add_module('conv{}b'.format(p),
                            initialize_weights(nn.Conv2d(next_filter, next_filter, kernel_size, padding=padding)))
            self.add_module('pool{}'.format(p), nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            curr_filter, next_filter = next_filter, next_filter * 2
        fc_inp_len = None
        for i in range(num_fc):
            fc_inp_len = self._get_fc_len() if fc_inp_len is None else fc_out_len
            fc_out_len = int(fc_inp_len / fc_scale_down) if i < num_fc - 1 else enc_len
            self.add_module('fc{}'.format(i), initialize_weights(nn.Linear(fc_inp_len, fc_out_len)))

        # Decoder
        dfc_inp_len = None
        for i in range(num_fc):
            dfc_inp_len = dfc_out_len if i > 0 else enc_len
            dfc_out_len = dfc_inp_len * fc_scale_down if i > 0 else fc_inp_len
            if self.shared_weights:
                self.register_parameter('dfc{}_bias'.format(i), nn.Parameter(torch.zeros(dfc_out_len)))
            else:
                self.add_module('dfc{}'.format(i), initialize_weights(nn.Linear(dfc_inp_len, dfc_out_len)))
        curr_filter = num_init_filters * (2 ** (num_pools - 1))
        next_filter = int(curr_filter / 2)
        for p in range(num_pools):
            curr_filter = next_filter if p > 0 else num_init_filters * (2 ** (num_pools - 1))
            next_filter = int(curr_filter / 2) if p < num_pools - 1 else 1

            self.add_module('unpool{}'.format(p), nn.MaxUnpool2d(kernel_size=2, stride=2))

            if self.shared_weights:
                # For shared weights, bias cannot be used from the encoder. Use own bias
                self.register_parameter('deconv{}a_bias'.format(p), nn.Parameter(torch.zeros(curr_filter)))
            else:
                self.add_module('deconv{}a'.format(p), initialize_weights(
                    nn.ConvTranspose2d(curr_filter, curr_filter, kernel_size, padding=padding)))

            if self.shared_weights:
                self.register_parameter('deconv{}b_bias'.format(p), nn.Parameter(torch.zeros(next_filter)))
            else:
                self.add_module('deconv{}b'.format(p), initialize_weights(
                    nn.ConvTranspose2d(curr_filter, next_filter, kernel_size, padding=padding)))

        # Lookup
        self.layer_dict = {name: layer for name, layer in self.named_children()}
        self.param_dict = {name: layer for name, layer in self.named_parameters()}

        # Cache (for skip connections)
        self.pool_idx = {}
        self.conv_outputs = {}

    def _validate_hp(self):
        """Checks that the current parameters will generate a valid mode"""
        fc_len = self._get_fc_len()
        assert self.input_dims[0] % (2 ** self.num_pools) == 0 and self.input_dims[1] % (2 ** self.num_pools) == 0
        assert fc_len % (self.fc_scale_down ** (self.num_fc - 1)) == 0
        assert fc_len / (self.fc_scale_down ** (self.num_fc - 1)) >= self.enc_len

    def _get_fc_len(self):
        """Computes the size of the first fully connected layer"""
        return (self.num_init_filters * (2 ** (self.num_pools - 1))) * int(
            self.input_dims[0] * self.input_dims[1] / (2 ** (self.num_pools * 2)))

    def _get_conv_dims(self):
        """Computes the dimensions of the last convolutional block"""
        return ((2 ** (self.num_pools - 1)) * self.num_init_filters,
                int(self.input_dims[0] / (2 ** self.num_pools)),
                int(self.input_dims[1] / (2 ** self.num_pools)))

    def encode(self, x):
        """Forward pass up to middle/encoding layer"""

        # Conv blocks
        for p in range(self.num_pools):
            k_conv_a = 'conv{}a'.format(p)
            x = self.conv_outputs[k_conv_a] = F.relu(self.layer_dict[k_conv_a](x))
            k_conv_b = 'conv{}b'.format(p)
            x = self.conv_outputs[k_conv_b] = F.relu(self.layer_dict[k_conv_b](x))
            k_pool = 'pool{}'.format(p)
            x, idx = self.layer_dict[k_pool](x)
            self.pool_idx[k_pool] = idx

        # FC blocks
        x = x.reshape(x.shape[0], -1)
        for i in range(self.num_fc):
            k_fc = 'fc{}'.format(i)
            x = self.layer_dict[k_fc](x)
            if i < self.num_fc - 1:
                x = F.relu(x)
            elif self.enc_activation == 'tanh':
                x = F.tanh(x)
            elif self.enc_activation == 'sigmoid':
                x = F.sigmoid(x)
            else:
                raise Exception('Invalid enc_activation: {}'.format(self.enc_activation))

        return x

    def decode(self, x):
        """Forward pass from middle/encoding layer to the model output/reconstructed input"""

        # FC blocks
        for i in range(self.num_fc):
            i_fwd = self.num_fc - i - 1
            k_fc = 'fc{}'.format(i_fwd)
            k_dfc = 'dfc{}'.format(i)
            k_dfc_bias = 'dfc{}_bias'.format(i)
            if self.shared_weights:
                # For shared weights, transpose weights of FC in encoder for use in decoder
                x = F.relu(F.linear(
                    x, self.layer_dict[k_fc].weight.transpose(0, 1),
                    bias=self.param_dict[k_dfc_bias]))
            else:
                x = F.relu(self.layer_dict[k_dfc](x))

        # Deconv blocks
        x = x.reshape(x.shape[0], *self._get_conv_dims())
        for p in range(self.num_pools):
            p_fwd = self.num_pools - p - 1
            k_pool = 'pool{}'.format(p_fwd)
            k_unpool = 'unpool{}'.format(p)
            x = self.layer_dict[k_unpool](x, self.pool_idx[k_pool])

            k_conv_b = 'conv{}b'.format(p_fwd)
            k_deconv_a = 'deconv{}a'.format(p)
            k_deconv_a_bias = 'deconv{}a_bias'.format(p)
            if self.skip_connections:
                # For skip connections, add the output of the corresponding convolutional block
                x = F.relu(x + self.conv_outputs[k_conv_b])
            if self.shared_weights:
                # For shared weights, use the weights of the corresponding convolutional block but not bias
                x = F.relu(F.conv_transpose2d(
                    x, self.layer_dict[k_conv_b].weight,
                    bias=self.param_dict[k_deconv_a_bias], padding=self.padding))
            else:
                x = F.relu(self.layer_dict[k_deconv_a](x))

            k_conv_a = 'conv{}a'.format(p_fwd)
            k_deconv_b = 'deconv{}b'.format(p)
            k_deconv_b_bias = 'deconv{}b_bias'.format(p)
            if self.skip_connections:
                x = F.relu(x + self.conv_outputs[k_conv_a])
            if self.shared_weights:
                x = F.conv_transpose2d(
                    x, self.layer_dict[k_conv_a].weight,
                    bias=self.param_dict[k_deconv_b_bias], padding=self.padding)
            else:
                x = self.layer_dict[k_deconv_b](x)

            x = F.relu(x) if p < self.num_pools - 1 else F.sigmoid(x)

        return x

    def forward(self, x):
        """Complete forward pass"""
        return self.decode(self.encode(x))


class ConvAutoencoder(ModelBase):
    """Wrapper around _ConvAutoencoderModel"""

    def __init__(self, cuda=True, input_dims=(64, 96), enc_len=10, lr=0.001, momentum=0.9, **kwargs):
        super(ConvAutoencoder, self).__init__(cuda)

        # Model Params
        self.lr = lr
        self.momentum = momentum

        # Model
        self.model = _ConvAutoencoderModel(input_dims, enc_len, **kwargs).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Loss
        self.loss_fn = nn.MSELoss()

    def begin_training(self):
        """To be called before training"""
        self.model.train()

    def _process_inputs(self, x):
        """Coverts a Numpy Array input batch to a PyTorch Tensor of the same"""
        return torch.from_numpy(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])).to(self.device)

    def train_batch(self, x, _):
        """Forward and backward pass on a single batch"""
        x = y = self._process_inputs(x)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def encode(self, x):
        """Extracts the encodings for the given input x from the encoding/middle layer"""
        x = self._process_inputs(x)
        return self.model.encode(x)

    def begin_evaluation(self):
        """To be called before evaluation"""
        self.model.eval()

    def evaluate(self, x, _):
        """Computes loss on an evaluation batch"""
        x = y = self._process_inputs(x)
        with torch.no_grad():
            y_pred = self.model(x)
            return self.loss_fn(y_pred, y).item()

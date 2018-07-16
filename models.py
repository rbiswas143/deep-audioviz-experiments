import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.tensor
import os
import utils
import math
import numpy as np

import alexnet


## Utils

def initialize_weights(module):
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


## Common model ops

class ModelBase:

    def __init__(self, cuda=True):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = None
        self.optimizer = None

    def save_state(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load_state(self, path):
        if not os.path.isfile(path):
            return None
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def init_checkpoint(self):
        return {}

    def post_evaluation(self, checkpoint):
        pass


class ModelCheckpoint:

    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.best_loss = None
        self.training_losses = []
        self.cv_losses = []
        self.cv_accuracies = []
        self.training_times = []
        self.model_specific = self.model.init_checkpoint()
        self.trainable_params = utils.get_trainable_params(model.model)
        self.loaded = False

    def save(self, path):
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
        self.loaded = True

    def get_dict(self):
        return self.__dict__


## VGG / Alex Net

class CNNClassifier(ModelBase):

    def __init__(self, dataset_config, input_dims=(64, 96), pretrained=True, cuda=True, num_classes=-1,
                 lr=0.001, momentum=0.9, batchnorm=False, arch='vgg16'):
        # print(num_classes, lr, momentum, batchnorm)
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
        self.fma_meta_dir = dataset_config.fma_meta_dir
        self.fma_type = dataset_config.fma_type

        # Validate
        assert tuple(self.input_dims) == (64, 96)

        self.model = self._build_model()

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Cache
        self.eval_cache = {}

        # Decay LR by a factor of 0.1 every 7 epochs
        # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def _build_model(self):
        # Load pretrained model
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

        # Replace first layer
        if self.arch == 'alexnet':
            first_layer = nn.Conv2d(1, 64, kernel_size=11, stride=(1, 2), padding=2)
        else:
            first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.features = nn.Sequential(*([initialize_weights(first_layer)]
                                         + list(model.features.children())[1:]))

        # Replace FC layers
        if self.arch == 'alexnet':
            fc_dim = 256 * 6 * 4
        else:
            fc_dim = 512 * 2 * 3
        model.classifier = nn.Sequential(
            initialize_weights(nn.Linear(fc_dim, 4096)),  # Assuming input dims is (64, 96)
            nn.ReLU(True),
            nn.Dropout(),
            initialize_weights(nn.Linear(4096, 4096)),
            nn.ReLU(True),
            nn.Dropout(),
            initialize_weights(nn.Linear(4096, self.num_classes)),
        )
        return model.to(self.device)

    def init_checkpoint(self):
        return {
            'cv_accuracies': [],
            'polled_accuracies': []
        }

    def begin_training(self):
        self.model.train()

    def _process_inputs(self, x):
        return torch.from_numpy(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])).to(self.device)

    def _process_outputs(self, y):
        return torch.from_numpy(utils.map_indices_to_genre(y, self.fma_meta_dir, self.fma_type)).to(self.device)

    def train_batch(self, x, y):
        x = self._process_inputs(x)
        y = self._process_outputs(y)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def begin_evaluation(self):
        self.model.eval()
        self.eval_cache = {
            'segment_indices': [],
            'genre_correct': [],
            'genre_pred': [],
            'accuracy_sum': 0,
            'num_batches': 0
        }

    def evaluate(self, x, y_indices):
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

    def post_evaluation(self, checkpoint):

        avg_accuracy = self.eval_cache['accuracy_sum'] / self.eval_cache['num_batches']
        checkpoint.model_specific['cv_accuracies'].append(avg_accuracy)
        print('Average accuracy per batch:', avg_accuracy)

        polled_accuracy = self._evaluate_polled()
        checkpoint.model_specific['polled_accuracies'].append(polled_accuracy)
        print('Polled accuracy:', polled_accuracy)

    def _evaluate_polled(self):
        segment_indices = np.array(self.eval_cache['segment_indices']).reshape(-1)
        genre_correct = np.array(self.eval_cache['genre_correct']).reshape(-1)
        genre_pred = np.array(self.eval_cache['genre_pred']).reshape(-1)

        idx_correct = {}
        idx_preds = {}
        for idx, correct, pred in zip(segment_indices, genre_correct, genre_pred):
            idx_correct[idx] = correct
            if idx not in idx_preds:
                idx_preds[idx] = []
            idx_preds[idx].append(pred)

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


## Convolutional Autoencoder

class _ConvAutoencoderModel(nn.Module):

    def __init__(self, input_dims, enc_len, num_init_filters=16, num_pools=2,
                 num_fc=2, fc_scale_down=8, kernel_size=5, padding=2,
                 shared_weights=False, skip_connections=False, enc_activation='sigmoid'):
        super(_ConvAutoencoderModel, self).__init__()
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

        # Cache
        self.pool_idx = {}
        self.conv_outputs = {}

    def _validate_hp(self):
        fc_len = self._get_fc_len()
        assert self.input_dims[0] % (2 ** self.num_pools) == 0 and self.input_dims[1] % (2 ** self.num_pools) == 0
        assert fc_len % (self.fc_scale_down ** (self.num_fc - 1)) == 0
        assert fc_len / (self.fc_scale_down ** (self.num_fc - 1)) >= self.enc_len

    def _get_fc_len(self):
        return (self.num_init_filters * (2 ** (self.num_pools - 1))) * int(
            self.input_dims[0] * self.input_dims[1] / (2 ** (self.num_pools * 2)))

    def _get_conv_dims(self):
        return ((2 ** (self.num_pools - 1)) * self.num_init_filters,
                int(self.input_dims[0] / (2 ** self.num_pools)),
                int(self.input_dims[1] / (2 ** self.num_pools)))

    def encode(self, x):
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
        # FC blocks
        for i in range(self.num_fc):
            i_fwd = self.num_fc - i - 1
            k_fc = 'fc{}'.format(i_fwd)
            k_dfc = 'dfc{}'.format(i)
            k_dfc_bias = 'dfc{}_bias'.format(i)
            if self.shared_weights:
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
                x = F.relu(x + self.conv_outputs[k_conv_b])
            if self.shared_weights:
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
        return self.decode(self.encode(x))


class ConvAutoencoder(ModelBase):

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
        self.model.train()

    def _process_inputs(self, x):
        return torch.from_numpy(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])).to(self.device)

    def train_batch(self, x, _):
        x = y = self._process_inputs(x)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def begin_evaluation(self):
        self.model.eval()

    def evaluate(self, x, _):
        x = y = self._process_inputs(x)
        with torch.no_grad():
            y_pred = self.model(x)
            return self.loss_fn(y_pred, y).item()

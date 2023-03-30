"""Provides a child class ConvNet of torch.nn.module to allow
   the construction of a convolutional neural network as well as
    a function to load a network and to initialize values of a tensor
"""

import os
import torch
import torch.nn as nn

from matplotlib import pylab as plt
from sklearn.metrics import balanced_accuracy_score

from utils import read_cfg_file

if torch.cuda.is_available():
    compute_device = torch.device("cuda:0")
# elif torch.backends.mps.is_available():
#    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")


def init_weights(m):
    """Initializes weights with values according to the method
       described in “Understanding the difficulty of training
       deep feedforward neural networks” - Glorot, alns_aa_counts.

    :param m: convolutional or linear layer
    """

    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=compute_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train_history = checkpoint['train_history']
    model.val_history = checkpoint['val_history']
    if 'opt_state_dict' in checkpoint.keys():
        model.opt_state = checkpoint['opt_state_dict']
    if 'scheduler_state_dict' in checkpoint.keys():
        model.scheduler_state = checkpoint['scheduler_state_dict']
    return model


def load_model(path, state='eval'):
    """Loads a model and sets it to evaluation or training mode

    :param path: <path/to/model folder> with <.pth> model files (string)
    :param state: 'eval' or 'train' to indicate desired model state (string)
    :return: the model (ConvNet object)
    """

    cfg = read_cfg_file(os.path.join(path, 'cfg.json'))
    models = []
    for fold in range(cfg['hyperparameters']['n_folds']):
        model_path = os.path.join(path, f'model-fold-{fold + 1}.pth')
        if os.path.exists(model_path):
            model = ConvNet(cfg['conv_net_parameters']).to(compute_device)
            model = load_checkpoint(model_path, model)
            if state == 'eval':
                model.eval()
            elif state == 'train':
                model.train()
            models.append(model)

    return models


class ConvNet(nn.Module):
    """Neural network with at least 1 linear layer,
       which can have multiple convolutional and linear layers
       for a binary classification task, where classes are empirical
       alignments and simulated alignments

        Attributes
        ----------
        conv_layers : nn.Sequential
            a sequence of conv1D, ReLu and maxpooling
        lin_layers : nn.Sequential
            one or multiple linear layers, the last one having 1 output node
        train_history : list of dictionaries keys: 'bacc', 'loss'
            accuracy and loss for each epoch on training dataset
        val_history : list of dictionaries keys: 'bacc', 'loss'
            accuracy and loss for each epoch on validaiton dataset

        Methods
        -------
        __init__(p)
            initializes the networks layers
        forward(x)
            performs feed forward pass
        training_step(batch)
        validation_step(batch)
            combines losses and accuracies for all batches
        plot(path=None)
            plots performance of the network
        save(path)
            save the model in a .pth file
    """

    def __init__(self, p):
        """Initializes network layers

        :param p: parameters that determine the network architecture (dict)
        """

        super(ConvNet, self).__init__()

        # number of filters/features per conv. layer
        n_features = p['channels']
        n_conv_layer = len(p['channels']) - 1

        # determine output size after conv. layers (input_plt_fct size for lin. layer)
        if p['do_maxpool'] == 1:  # local max pooling
            out_size = (int(p['input_size'] / 2 ** n_conv_layer) *
                        n_features[-1])
        elif p['do_maxpool'] == 2:  # global max pooling
            out_size = n_features[-1]
        else:  # no pooling
            out_size = p['input_size'] * n_features[-1]

        # convolutional layers
        if not isinstance(p['kernel_size'], list):
            p['kernel_size'] = [p['kernel_size']]
        self.conv_layers = []
        for i in range(n_conv_layer):
            conv1d = nn.Conv1d(n_features[i],
                               n_features[i + 1],
                               kernel_size=p['kernel_size'][i], stride=1,
                               padding=p['kernel_size'][i] // 2)

            self.conv_layers += [conv1d, nn.ReLU()]

            # if (p['do_maxpool'] == 1 or
            #        (p['do_maxpool'] == 2 and i < n_conv_layer - 1)):
            #    # local pooling
            #    self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if p['do_maxpool'] == 2 and i == n_conv_layer - 1:
                # global pooling after last conv layer
                ks = int(p['input_size'])
                self.conv_layers.append(nn.AvgPool1d(kernel_size=ks))

        if n_conv_layer == 0 and p['do_maxpool'] == 2:
            # global avg pooling -> global MSA AA frequencies
            ks = int(p['input_size'])
            self.conv_layers.append(nn.AvgPool1d(kernel_size=ks))

        self.conv_layers.append(nn.Dropout(0.2))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # fully connected layer(s)
        self.lin_layers = []
        n_lin_layers = p['n_lin_layer']
        for i in range(n_lin_layers):
            if i == n_lin_layers - 1:  # the last layer has a single output
                self.lin_layers.append(nn.Linear(out_size, 1))
            else:
                if n_lin_layers > 2 and i == 0:
                    # if at least 3 lin layers have a first "input_plt_fct layer" with
                    # same number of output nodes and dropout
                    self.lin_layers += [nn.Linear(out_size, out_size),
                                        nn.ReLU(),
                                        nn.Dropout(0.2)]
                elif i < n_lin_layers - 1:
                    self.lin_layers += [nn.Linear(out_size, out_size // 2),
                                        nn.ReLU()]
                    out_size = out_size // 2

        self.lin_layers = (nn.Sequential(*self.lin_layers)
                           if n_lin_layers > 0 else None)

        self.global_avgpool = (nn.AdaptiveAvgPool1d(1)
                               if n_lin_layers == 0 else None)

        self.train_history = {'loss': [], 'bacc': [], 'acc_emp': [],
                              'acc_sim': []}
        self.val_history = {'loss': [], 'bacc': [], 'acc_emp': [],
                            'acc_sim': []}
        self.opt_state = None  # state dict of optimizer
        self.scheduler_state = None  # state dict of scheduler

    def forward(self, x):
        if self.conv_layers is not None:
            out = self.conv_layers(x)
            out = out.reshape(out.size(0), -1)  # flattening
        else:
            out = x.view(x.shape[0], -1)

        if self.lin_layers is not None:
            out = self.lin_layers(out)
        else:  # global avg-pooling instead of a linear layer
            out = self.global_avgpool(out.unsqueeze(0))

        return out

    def feed(self, batch):
        alns, labels = batch

        alns = alns.to(compute_device)
        labels = labels.to(compute_device)

        out = self(alns).to(compute_device)  # generate predictions

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))

        return loss, out.squeeze(dim=1), labels

    def plot(self, path=None):
        """Generates a figure with 2 plots for loss and accuracy over epochs

        :param path: <path/to/> directory to save the plot to (string/None)
        """

        if len(self.train_history['loss']) > 0:
            train_col, val_col = '#1F77B4', '#FF7F0E'
            keys = ['bacc', 'acc_emp', 'acc_sim']
            line_styles = ['-', '--', ':']
            labels = ['', 'empirical', 'simulated']
            alphas = [1, 0.5, 0.5]

            fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(12., 6.))

            for key, ls, label, a in zip(keys, line_styles, labels, alphas):
                axs[0].plot(self.train_history[key], linestyle=ls, alpha=a,
                            label=f'{label} (train.)' if label != '' else '',
                            color=train_col)
                axs[0].plot(self.val_history[key], linestyle=ls, alpha=a,
                            label=f'{label} (val.)' if label != '' else '',
                            color=val_col)

            # axs[0].set_ylim([floor(plt.ylim()[0] * 100) / 100, 1.0])
            axs[0].set_xlabel('epoch')
            axs[0].set_ylabel('accuracy')
            axs[0].set_title(f'Accuracy vs. No. of epochs')
            axs[0].legend()

            line_train, = axs[1].plot(self.train_history['loss'],
                                      label='training',
                                      color=train_col)
            line_val, = axs[1].plot(self.val_history['loss'],
                                    label='validation',
                                    color=val_col)
            axs[1].set_xlabel('epoch')
            axs[1].set_ylabel('loss')
            axs[1].set_title(f'Loss vs. No. of epochs')

            plt.subplots_adjust(bottom=0.25)
            plt.legend([line_train, line_val], ['training', 'validation'],
                       bbox_to_anchor=[-0.1, -0.3], loc='lower center', ncol=2)

            if path is not None:
                plt.savefig(path)
            plt.close('all')

    def save(self, path):
        torch.save({
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_state_dict': self.state_dict(),
            'opt_state_dict': self.opt_state,
            'scheduler_state_dict': self.scheduler_state
        },
            path)


def accuracy(outputs, labels):
    """Calculates accuracy of network predictions

    :param outputs: network output for all examples (torch tensor)
    :param labels: 0 and 1 labels (0: empirircal, 1:simulated) (torch tensor)
    :return: accuracy values (between 0 and 1) (torch tensor)
    """
    accs = {}
    preds = torch.round(activation(outputs)).to(compute_device)

    accs['bacc'] = torch.FloatTensor([balanced_accuracy_score(
        labels.detach().cpu().numpy(), preds.detach().cpu().numpy())])

    for label, key in enumerate(['acc_emp', 'acc_sim']):
        class_mask = (labels == label).clone().detach()
        n = torch.sum(class_mask)
        n_correct = torch.sum(preds[class_mask] == labels[class_mask])
        accs[key] = n_correct / n if n.item() > 0 else torch.FloatTensor([-1])
        accs[key].to(compute_device)

    # acc = torch.tensor((torch.sum(preds == labels).item() / len(preds)))
    return accs


def activation(outputs):
    return torch.flatten(torch.sigmoid(outputs))

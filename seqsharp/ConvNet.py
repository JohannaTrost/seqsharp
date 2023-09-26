"""Provides a child class ConvNet of torch.nn.module

Allows the construction of a convolutional neural network. Includes functions
to initialize network weights, load network state and calculate accuracy next to
ConvNet class.
"""

import os
import torch
import torch.nn as nn

from matplotlib import pylab as plt
from sklearn.metrics import balanced_accuracy_score

from .utils import read_cfg_file

if torch.cuda.is_available():
    compute_device = torch.device("cuda:0")
else:
    compute_device = torch.device("cpu")


def init_weights(m):
    """Initializes weights with values according to the method
       described in “Understanding the difficulty of training
       deep feedforward neural networks” - Glorot.

    :param m: convolutional or linear layer
    """

    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def load_checkpoint(path, model):
    """Load checkpoint of pre-trained model

    Checkpoint includes the state dictionary (i.e. network weights), optimizer
    and eventually scheduler state, training and validation history.

    :param path: <path/to/checkpoint>
    :param model: pre-trained seq#-model
    :return: model at checkpoint
    """

    checkpoint = torch.load(path, map_location=compute_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train_history = checkpoint['train_history']
    model.val_history = checkpoint['val_history']

    # old models have key 'acc' instead of 'bacc'
    if 'acc' in model.val_history.keys():
        model.val_history['bacc'] = model.val_history['acc']
        del model.val_history['acc']
    if 'acc' in model.train_history.keys():
        model.train_history['bacc'] = model.train_history['acc']
        del model.train_history['acc']

    if 'opt_state_dict' in checkpoint.keys():
        model.opt_state = checkpoint['opt_state_dict']
    if 'scheduler_state_dict' in checkpoint.keys():
        model.scheduler_state = checkpoint['scheduler_state_dict']
    return model


def load_model(path, state='eval'):
    """Loads models of all folds and sets them to evaluation or training mode

    :param path: <path/to/model folder> with <.pth> model files (string)
    :param state: 'eval' or 'train' to indicate desired model state (string)
    :return: models for all folds (list of ConvNet objects)
    """

    cfg = read_cfg_file(os.path.join(path, 'cfg.json'))
    models = []
    for fold in range(cfg['training']['n_folds']):
        model_path = os.path.join(path, f'model_fold_{fold + 1}.pth')
        if os.path.exists(model_path):
            model = ConvNet(cfg['model']).to(compute_device)
            model = load_checkpoint(model_path, model)
            if state == 'eval':
                model.eval()
            elif state == 'train':
                model.train()
            models.append(model)
        else:
            models.append(None)

    return models


class ConvNet(nn.Module):
    """Seq# (convolution) neural network

       Neural network with at least 1 linear layer,
       which can have multiple convolutional and linear layers
       for a binary classification task, where classes are empirical
       alignments (0) and simulated alignments (1)

        Attributes
        ----------
        conv_layers : nn.Sequential a sequence of conv1D, ReLu and pooling
        lin_layers : nn.Sequential one or multiple linear layers, ReLU succeeds
                     inner nodes
        train_history : dictionary with keys: 'bacc', 'loss', acc_emp, acc_sim
                        accuracy and loss for each epoch on training dataset
        val_history : dictionary with keys: 'bacc', 'loss', acc_emp, acc_sim
                      accuracy and loss for each epoch on validaiton dataset
        opt_state : state dictionary of optimizer (initially None)
        scheduler_state : state dictionary of scheduler for cyclic lr
                          (initially None)

        Methods
        -------
        __init__(p)
            initializes the networks layers
        forward(x)
            performs feed forward pass
        feed(batch)
            process batch, compute loss and predictions
        plot(path=None)
            plot performance of the network
        save(path)
            save ConvNet object
    """

    def __init__(self, p):
        """Initializes network layers

        :param p: parameters for network architecture (dict)
        """

        super(ConvNet, self).__init__()

        # number of filters/features per conv. layer
        n_features = p['channels']
        n_conv_layer = len(p['channels']) - 1

        # ----- determine output size after conv. layers
        if p['pooling'] == 1:
            # local max pooling
            out_size = int(p['input_size'] / 2 ** n_conv_layer)
            out_size *= n_features[-1]
        elif p['pooling'] == 2:
            # global max pooling
            out_size = n_features[-1]
        else:
            # no pooling
            out_size = p['input_size'] * n_features[-1]

        # ----- convolutional layers
        self.conv_layers = []
        if n_conv_layer > 0:

            if not isinstance(p['kernel_size'], list):
                p['kernel_size'] = [p['kernel_size']]

            for i in range(n_conv_layer):

                conv1d = nn.Conv1d(n_features[i],
                                   n_features[i + 1],
                                   kernel_size=p['kernel_size'][i], stride=1,
                                   padding=p['kernel_size'][i] // 2)
                self.conv_layers += [conv1d, nn.ReLU()]

                if p['pooling'] == 1:
                    # local pooling
                    self.conv_layers.append(nn.MaxPool1d(kernel_size=2,
                                                         stride=2))
                if p['pooling'] == 2 and i == n_conv_layer - 1:
                    # global pooling after last conv. layer
                    ks = int(p['input_size'])
                    self.conv_layers.append(nn.AvgPool1d(kernel_size=ks))

            self.conv_layers.append(nn.Dropout(0.2))

        if n_conv_layer == 0 and p['pooling'] == 2:
            # global avg pooling -> global MSA compopsitions
            ks = int(p['input_size'])
            self.conv_layers.append(nn.AvgPool1d(kernel_size=ks))

        # ----- fully connected layer(s)
        self.lin_layers = []
        n_lin_layers = p['n_lin_layer']
        for i in range(n_lin_layers):

            if i == n_lin_layers - 1:
                # the last layer has a single output
                self.lin_layers.append(nn.Linear(out_size, 1))
            else:
                self.lin_layers += [nn.Linear(out_size, out_size // 2),
                                    nn.ReLU()]
                out_size = out_size // 2

        self.conv_layers = (nn.Sequential(*self.conv_layers)
                            if len(self.conv_layers) > 0 else None)
        self.lin_layers = (nn.Sequential(*self.lin_layers)
                           if n_lin_layers > 0 else None)

        self.train_history = {'loss': [], 'bacc': [], 'acc_emp': [],
                              'acc_sim': []}
        self.val_history = {'loss': [], 'bacc': [], 'acc_emp': [],
                            'acc_sim': []}
        self.opt_state = None  # state dict of optimizer
        self.scheduler_state = None  # state dict of scheduler

    def forward(self, x):
        """Performs feed forward pass on input"""

        if self.conv_layers is not None:
            out = self.conv_layers(x)
            out = out.reshape(out.size(0), -1)  # flattening
        else:
            out = x.view(x.shape[0], -1)

        if self.lin_layers is not None:
            out = self.lin_layers(out)

        return out

    def feed(self, batch):
        """Evaluate batch by ConvNet model

        :param batch: MSA representations and labels
        :return: loss, prediction, label for examples in batch
        """

        alns, labels = batch

        alns = alns.to(compute_device)
        labels = labels.to(compute_device)

        out = self(alns).to(compute_device)  # generate predictions

        criterion = nn.BCEWithLogitsLoss()  # includes sigmoid activation
        loss = criterion(out, torch.reshape(labels, out.shape))

        return loss, out.squeeze(dim=1), labels

    def plot(self, path=None):
        """Plot loss and BACC vs. epochs

        :param path: <path/to/> directory to save the plot to (string)
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
                plt.savefig(path, dpi=300)
            plt.close('all')

    def save(self, path):
        """Save model state (checkpoint)

        :param path: <path/to/> save model
        :return: None
        """

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
    :return: BACC and class accuracies (dictionary with torch tensors)
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

    return accs


def activation(outputs):
    """ Sigmoid activation for binary classification """
    return torch.flatten(torch.sigmoid(outputs))

"""Provides a child class ConvNet of torch.nn.module to allow
   the construction of a convolutional neural network as well as
    a function to load a network and to initialize values of a tensor
"""

from numpy import floor
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt

compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    """Initializes weights with values according to the method
       described in “Understanding the difficulty of training
       deep feedforward neural networks” - Glorot, X.

    :param m: convolutional or linear layer
    """

    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def load_net(path, params, state='eval'):
    """Loads a model and sets it to evaluation or training mode

    :param path: <path/to/*.pth> model to be loaded (string)
    :param params: model parameters (input size etc.) (dict)
    :param state: 'eval' or 'train' to indicate desired model state (string)
    :return: the model (ConvNet object)
    """

    model = ConvNet(params)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train_history = checkpoint['train_history']
    model.val_history = checkpoint['val_history']

    if state == 'eval':
        model.eval()
    elif state == 'train':
        model.train()

    return model.to(compute_device)


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
        sound : str
            the sound that the animal makes
        num_legs : int
            the number of legs the animal has (default 4)
        train_history : list of dictionaries keys: 'acc', 'loss'
            accuracy and loss for each epoch on training dataset
        val_history : list of dictionaries keys: 'acc', 'loss'
            accuracy and loss for each epoch on validaiton dataset

        Methods
        -------
        __init__(p)
            initializes the networks layers
        forward(x)
            performs feed forward pass
        training_step(batch)
        validation_step(batch)
        validation_epoch_end(outputs)
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

        out_size = (int(p['input_size'] / 2**p['nb_conv_layer']) *
                    p['nb_chnls'] * 2**p['nb_conv_layer'])

        # convolutional layers
        self.conv_layers = []
        for i in range(p['nb_conv_layer']):
            conv1d = nn.Conv1d(p['nb_chnls'] * (2**i),
                               p['nb_chnls'] * (2**(i+1)),
                               kernel_size=p['kernel_size'], stride=1,
                               padding=p['kernel_size'] // 2)

            self.conv_layers += [conv1d, nn.ReLU()]

            if p['do_maxpool'] and p['kernel_size'] > 1:  # down sampling
                self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.conv_layers = (nn.Sequential(*self.conv_layers)
                            if p['nb_conv_layer'] > 0 else None)

        self.drop_out = nn.Dropout(p=0.25) if p['nb_conv_layer'] > 0 else None

        # fully connected layer(s)
        self.lin_layers = []
        for i in range(max(p['nb_lin_layer'] - 1, 0)):
            self.lin_layers.append(nn.Linear(out_size, out_size // 2))
            out_size = out_size // 2
        self.lin_layers.append(nn.Linear(out_size, 1))

        self.lin_layers = nn.Sequential(*self.lin_layers)

        self.train_history = []
        self.val_history = []

    def forward(self, x):
        # out = self.inpt(x)
        if self.conv_layers is not None: out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)  # flattening
        if self.drop_out is not None: out = self.drop_out(out)
        out = self.lin_layers(out)
        return out

    def training_step(self, batch):
        alns, labels = batch

        alns = alns.to(compute_device)
        labels = labels.to(compute_device)

        out = self(alns)  # generate predictions, forward pass
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))

        return loss

    def validation_step(self, batch):
        alns, labels = batch

        alns, labels = alns.to(compute_device), labels.to(
            compute_device)

        out = self(alns)  # generate predictions

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))
        acc = accuracy(out, labels)

        return {'loss': loss.detach(), 'acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses

        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def plot(self, path=None):
        """Generates a figure with 2 plots for loss and accuracy over epochs

        :param path: <path/to/> directory to save the plot to (string/None)
        """

        if len(self.train_history) > 0 and len(self.val_history) > 0:
            accuracies_train = [result['acc'] for result in self.train_history]
            losses_train = [result['loss'] for result in self.train_history]

            accuracies_val = [result['acc'] for result in self.val_history]
            losses_val = [result['loss'] for result in self.val_history]

            fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(12., 6.))

            train_col = (0.518, 0.753, 0.776)
            val_col = (0.576, 1.0, 0.588)

            axs[0].plot(accuracies_train, '-x', label='training', color=train_col)
            axs[0].plot(accuracies_val, '-x', label='validation', color=val_col)
            axs[0].set_ylim([floor(plt.ylim()[0] * 100) / 100, 1.0])
            axs[0].set_xlabel('epoch')
            axs[0].set_ylabel('accuracy')
            axs[0].set_title(f'Accuracy vs. No. of epochs')

            line_train, = axs[1].plot(losses_train, '-x', label='training',
                                      color=train_col)
            line_val, = axs[1].plot(losses_val, '-x', label='validation',
                                    color=val_col)
            axs[1].set_xlabel('epoch')
            axs[1].set_ylabel('loss')
            axs[1].set_title(f'Loss vs. No. of epochs')

            plt.subplots_adjust(bottom=0.25)
            plt.legend([line_train, line_val], ['training', 'validation'],
                       bbox_to_anchor=[-0.1, -0.3], loc='lower center', ncol=2)

            if path is not None:
                plt.savefig(path)

    def save(self, path):
        torch.save({
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_state_dict': self.state_dict()
            },
            path)


def accuracy(outputs, labels):
    """Calculates accuracy of network predictions

    :param outputs: network output for all examples (torch tensor)
    :param labels: 0 and 1 labels (0: empirircal, 1:simulated) (torch tensor)
    :return: accuracy values (between 0 and 1) (torch tensor)
    """

    preds = torch.round(torch.flatten(torch.sigmoid(outputs))).to(compute_device)
    return torch.tensor((torch.sum(preds == labels).item() / len(preds)))
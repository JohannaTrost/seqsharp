from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from data_preprocessing import alns_from_fastas, TensorDataset
from sklearn.model_selection import KFold
from datetime import datetime


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Model(nn.Module):
    def __init__(self, input_size, nb_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, nb_classes)
        self.linear.reset_parameters()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)

    def training_step(self, batch):
        seq_pairs, labels = batch
        out = self(seq_pairs.float())  # generate predicitons
        loss = nn.functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        seq_pairs, labels = batch
        out = self(seq_pairs.float())  # generate predicitons
        loss = nn.functional.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=1)
    for epoch in range(0, epochs):
        # training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def main(args):
    try:
        fasta_dir = args[0]
        model_dir = None
        if len(args) > 1:
            model_dir = args[1]
    except IndexError:
        print('At least 1 argument is required: path to the directory containing the fasta files\nOptional second '
              'argument: path to the directory where results will be stored')

    batch_size = 128
    nb_seqs_per_align = 50
    nb_classes = 40  # number of multiple aligns
    epochs = 15
    lr = 0.001
    nb_folds = 5

    torch.manual_seed(42)
    random.seed(42)

    # k-fold validator
    kfold = KFold(nb_folds, shuffle=True)

    # preprocessing of data
    raw_aligns = alns_from_fastas(fasta_dir, nb_seqs_per_align, nb_classes)
    seq_len = 200

    # create unique subdir for the models
    if model_dir is not None:
        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        model_dir = model_dir + '/models-' + str(timestamp)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_history = []
    fold_eval = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(raw_aligns[0])): # [0] because the seqs within one class shall be split not the classes
        # train_aligns, val_aligns = np.split(np.asarray(raw_aligns), [int(nb_seqs_per_align*0.9)], axis=1)
        print(f'FOLD {fold}')
        print('----------------------------------------------------------------')

        # splitting dataset by splitting within each class
        train_ds = TensorDataset(np.take(raw_aligns, train_ids, axis=1), seq_len)
        val_ds = TensorDataset(np.take(raw_aligns, val_ids, axis=1), seq_len)
        # val_ds, train_ds = random_split(ds, [round(len(ds) * 0.1), len(ds) - round(len(ds) * 0.1)])

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        model_input_size = train_ds.data.shape[1] * train_ds.data.shape[2]
        model = Model(model_input_size, nb_classes)

        # train and validate model
        train_history.append(fit(epochs, lr, model, train_loader, val_loader))
        fold_eval.append(train_history[fold][-1]['val_acc'])

        # Saving the model
        print('Training process has finished. Saving trained model.\n')
        if model_dir is not None:
            torch.save(model.state_dict(), f'{model_dir}/model-fold-{fold}.pth')


    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
    print('----------------------------------------------------------------')
    for i, acc in enumerate(fold_eval):
        print(f'Fold {(i+1)}: {acc} %')
    print(f'Average: {np.sum(fold_eval) / len(fold_eval)} %')

    # plot the model evaluation per fold
    fig, axs = plt.subplots(1*nb_folds, 2, constrained_layout=True)
    for i, fold_history in enumerate(train_history):
        accuracies = [result['val_acc'] for result in fold_history]
        losses = [result['val_loss'] for result in fold_history]

        axs[i][0].plot(accuracies, '-x')
        axs[i][0].set_xlabel('epoch')
        axs[i][0].set_ylabel('accuracy')
        axs[i][0].set_title(f'Fold {(i+1)}: Accuracy vs. No. of epochs')

        axs[i][1].plot(losses, '-x')
        axs[i][1].set_xlabel('epoch')
        axs[i][1].set_ylabel('loss')
        axs[i][1].set_title(f'Fold {i+1}: Loss vs. No. of epochs')

    if model_dir is not None:
        plt.savefig(model_dir + '/fig.png')


if __name__ == '__main__':
    main(sys.argv[1:])

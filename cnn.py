import errno
import sys, random, os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
from data_preprocessing import aligns_from_fastas, TensorDataset

compute_device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ConvNet(nn.Module):
    def __init__(self, seq_len, nb_classes):
        super(ConvNet, self).__init__()

        # self.inpt = nn.Linear(seq_len, seq_len)
        # convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(46, 92, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))  # down-sampling
        self.layer2 = nn.Sequential(
            nn.Conv1d(92, 184, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))  # down-sampling

        self.drop_out = nn.Dropout(p=0.25)  # adds noise to prevent overfitting

        # fully connected layer
        self.fc = nn.Linear(int(seq_len / 4) * 184, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # name is obligatory
        # out = self.inpt(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),
                          -1)  # flattening from (seqlen/4)x46 to Nx1
        out = self.drop_out(out)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    def training_step(self, batch):
        seq_pairs, labels = batch

        seq_pairs = seq_pairs
        seq_pairs = seq_pairs.to(compute_device)
        labels = labels.to(compute_device)

        out = self(seq_pairs)  # generate predictions, forward pass
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))
        print('training acc: {}'.format(accuracy(out, labels).detach()))
        return loss

    def validation_step(self, batch):
        seq_pairs, labels = batch

        seq_pairs, labels = seq_pairs.to(compute_device), labels.to(
            compute_device)

        out = self(seq_pairs)  # generate predictions

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))
        acc = accuracy(out, labels)

        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # combine losses

        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                     result[
                                                                         'val_loss'],
                                                                     result[
                                                                         'val_acc']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)  # loss/acc mean for epoch


def fit(epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.Adagrad):
    optimizer = opt_func(model.parameters(), lr)
    history = []

    for epoch in range(0, epochs):

        # training Phase
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()  # calcul of gradients
            optimizer.step()

        # validation phase
        model.eval()
        with torch.no_grad():
            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

    return history


def main(args):

    # -------------------- handling arguments -------------------- #

    if len(args) >= 2:
        real_fasta_path = args[0]
        sim_fasta_path = args[1]
        model_path = None

        if len(args) > 2:
            model_path = args[2]
            # create unique subdir for the model
            timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
            model_path = model_path + '/cnn-' + str(timestamp)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        if not os.path.exists(real_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    real_fasta_path)
        if not os.path.exists(sim_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    sim_fasta_path)
    else:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
               'At least 2 arguments are required: path to the directory '
               'containing the hogenom fasta files\npath to the directory '
               'containing the simulated fasta files\nOptional second argument: '
               'path to the directory where results will be stored')

    # -------------------- setting parameters -------------------- #

    # data specific parameters
    nb_protein_families = 63  # number of multiple aligns
    min_seqs_per_align, max_seqs_per_align = 4, 200
    seq_len = 200

    # hyperparameters
    batch_size = 128
    epochs = 15
    lr = 0.0001
    nb_folds = 8

    torch.manual_seed(42)
    random.seed(42)

    # k-fold validator
    kfold = KFold(nb_folds, shuffle=True)

    # -------------------- loading data ------------------------------- #

    # load sets of multiple aligned sequences
    real_aligns = aligns_from_fastas(real_fasta_path, min_seqs_per_align,
                                     max_seqs_per_align, nb_protein_families)
    sim_aligns = aligns_from_fastas(sim_fasta_path, min_seqs_per_align,
                                    max_seqs_per_align, nb_protein_families)

    # -------------------- k-fold cross validation -------------------- #

    # init for evaluation
    train_history = []
    fold_eval = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(real_aligns)):

        print(f'FOLD {fold + 1}')
        print(
            '----------------------------------------------------------------')

        # splitting dataset by protein families

        train_ds = TensorDataset([real_aligns[i] for i in train_ids],
                                 [sim_aligns[i] for i in train_ids],
                                 seq_len)
        val_ds = TensorDataset([real_aligns[i] for i in val_ids],
                               [sim_aligns[i] for i in val_ids],
                               seq_len)

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        input_size = train_ds.data.shape[2]  # seq len
        model = ConvNet(input_size, nb_protein_families)
        model = model.to(compute_device)

        # train and validate model
        train_history.append(fit(epochs, lr, model, train_loader, val_loader))
        fold_eval.append(train_history[fold][-1]['val_acc'])
        break
        # saving the model
        print('Training process has finished.\n')
        #  if model_path is not None:
        #  torch.save(model.state_dict(), f'{model_path}/model-fold-{fold}.pth')

    # print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
    print('----------------------------------------------------------------')

    for i, acc in enumerate(fold_eval):
        print(f'Fold {(i + 1)}: {acc} %')

    print(f'Average: {np.sum(fold_eval) / len(fold_eval)} %')

    # plot the model evaluation per fold
    fig, axs = plt.subplots(1 * nb_folds, 2, constrained_layout=True)

    for i, fold_history in enumerate(train_history):
        accuracies = [result['val_acc'] for result in fold_history]
        losses = [result['val_loss'] for result in fold_history]

        axs[i][0].plot(accuracies, '-x')
        axs[i][0].set_xlabel('epoch')
        axs[i][0].set_ylabel('accuracy')
        axs[i][0].set_title(f'Fold {(i + 1)}: Accuracy vs. No. of epochs')

        axs[i][1].plot(losses, '-x')
        axs[i][1].set_xlabel('epoch')
        axs[i][1].set_ylabel('loss')
        axs[i][1].set_title(f'Fold {i + 1}: Loss vs. No. of epochs')

    if model_path is not None:
        print(model_path)
        # plt.savefig(model_path + '/fig.png')


if __name__ == '__main__':
    main(sys.argv[1:])

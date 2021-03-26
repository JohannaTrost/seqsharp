import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
from data_preprocessing import alignments_from_fastas, build_dataset, TensorDataset

compute_device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ConvNet(nn.Module):
    def __init__(self, seq_len, nb_classes):
        super(ConvNet, self).__init__()
        self.inpt = nn.Linear(seq_len, seq_len)
        self.layer1 = nn.Sequential(
            nn.Conv1d(46, 92, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)) # down-sampling
        self.layer2 = nn.Sequential(
            nn.Conv1d(92, 184, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)) # down-sampling

        self.drop_out = nn.Dropout(p=0.25) # layer that prevents overfitting
        # fully connected layers
        self.fc = nn.Linear(int(seq_len / 4) * 184, nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): # name is obligatory
        #out = self.inpt(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # flattens data from (seqlen/4)x46 to Nx1
        out = self.drop_out(out)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    def training_step(self, batch):
        seq_pairs, labels = batch
        seq_pairs = seq_pairs.type(torch.FloatTensor)
        seq_pairs, labels = seq_pairs.to(compute_device), labels.to(
            compute_device)

        out = self(seq_pairs)  # generate predicitons, forward pass
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        return loss

    def validation_step(self, batch):
        seq_pairs, labels = batch
        seq_pairs = seq_pairs.type(torch.FloatTensor)
        seq_pairs, labels = seq_pairs.to(compute_device), labels.to(
            compute_device)
        out = self(seq_pairs)  # generate predicitons
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
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


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adagrad):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(0, epochs):
        # training Phase
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward() # calcul of gradients
            optimizer.step()
        # validation phase
        model.eval()
        with torch.no_grad():
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

    # for testing
    # model_dir = '/home/jtrost/PycharmProjects'
    model_dir = None
    if compute_device == "cuda":
        fasta_dir = '/mnt/Clusterdata/fasta'
    else:
        fasta_dir = '/home/jtrost/Clusterdata/fasta'

    # create unique subdir for the models
    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    if model_dir is not None:
        model_dir = model_dir + '/cnn-' + str(timestamp)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    batch_size = 512
    nb_seqs_per_align = 75
    nb_classes = 40  # number of multiple alignments
    seq_len = 300
    epochs = 15
    lr = 0.0001
    nb_folds = 5

    torch.manual_seed(42)
    random.seed(42)

    # k-fold validator
    kfold = KFold(nb_folds, shuffle=True)

    # preprocessing of data
    raw_alignments = alignments_from_fastas(fasta_dir, nb_seqs_per_align, nb_classes)

    train_history = []
    fold_eval = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(raw_alignments[0])): # [0] because the seqs within one class shall be split not the classes
        # train_alignments, val_alignments = np.split(np.asarray(raw_alignments), [int(nb_seqs_per_align*0.9)], axis=1)
        print(f'FOLD {fold+1}')
        print('----------------------------------------------------------------')

        # splitting dataset by splitting within each class
        train_ds = TensorDataset(np.take(raw_alignments, train_ids, axis=1), seq_len)
        val_ds = TensorDataset(np.take(raw_alignments, val_ids, axis=1), seq_len)
        # val_ds, train_ds = random_split(ds, [round(len(ds) * 0.1), len(ds) - round(len(ds) * 0.1)])

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        input_size = train_ds.data.shape[2] # seq len
        model = ConvNet(input_size, nb_classes)
        model = model.to(compute_device)

        # train and validate model
        train_history.append(fit(epochs, lr, model, train_loader, val_loader))
        fold_eval.append(train_history[fold][-1]['val_acc'])

        # saving the model
        print('Training process has finished. Saving trained model.\n')
        #  if model_dir is not None:
            #  torch.save(model.state_dict(), f'{model_dir}/model-fold-{fold}.pth')


    # print fold results
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
        print(model_dir)
        # plt.savefig(model_dir + '/fig.png')


if __name__ == '__main__':
    main(sys.argv[1:])

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from data_preprocessing import alignments_from_fastas, build_dataset


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
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation phase
        result = evaluate(model, val_loader)
        # model.epoch_end(epoch, result)
        history.append(result)
    return history


def main():
    for i in range(50):
        seed = int((i % 2 + 14) * i*20)
        print(seed)
        random.seed(seed)
        torch.manual_seed(10)

        batch_size = 128
        nb_seqs_per_align = 50
        nb_classes = 40  # number of multiple alignments
        epochs = 30
        lr = 0.001
	
        # preprocessing of data
        raw_alignments = alignments_from_fastas('/home/jtrost/Clusterdata/fasta', nb_seqs_per_align, nb_classes)
        min_seq_len = min(len(min([seq for seqs in raw_alignments for seq in seqs], key=len)), 200)

        train_alignments, val_alignments = np.split(np.asarray(raw_alignments), [int(nb_seqs_per_align*0.9)], axis=1)

        train_ds = build_dataset(train_alignments, min_seq_len)
        val_ds = build_dataset(val_alignments, min_seq_len)

        model_input_size = train_ds.data.shape[1] * train_ds.data.shape[2] * train_ds.data.shape[3]
        # val_ds, train_ds = random_split(ds, [round(len(ds) * 0.1), len(ds) - round(len(ds) * 0.1)])

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        model = Model(model_input_size, nb_classes)

        # train and validate model
        history = fit(epochs, lr, model, train_loader, val_loader)

        # plot the model evaluation
        accuracies = [result['val_acc'] for result in history]
        losses = [result['val_loss'] for result in history]
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(accuracies, '-x')
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('accuracy')
        axs[0].set_title('Accuracy vs. No. of epochs')

        axs[1].plot(losses, '-x')
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('loss')
        axs[1].set_title('Loss vs. No. of epochs')

        plt.savefig('/home/jtrost/PycharmProjects/figs/fig'+str(seed)+'.png')


if __name__ == '__main__':
    main()

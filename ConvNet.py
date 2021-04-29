import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import floor

compute_device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(outputs, labels):
    preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
    return torch.tensor((torch.sum(preds == labels).item() / len(preds)))


class ConvNet(nn.Module):
    def __init__(self, seq_len):
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
        self.fc = nn.Linear(int(seq_len / 2) * 92,
                            1)  # for 2 layers: int(seq_len / 4) * 184 !
        self.softmax = nn.Softmax(dim=1)

        self.train_history = []
        self.val_history = []

    def forward(self, x):  # name is obligatory
        # out = self.inpt(x)
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0),
                          -1)  # flattening from (seqlen/4)x46 to Nx1
        out = self.drop_out(out)
        out = self.fc(out)
        # out = self.softmax(out)
        return out

    def training_step(self, batch):
        seq_pairs, labels = batch

        seq_pairs = seq_pairs.to(compute_device)
        labels = labels.to(compute_device)

        out = self(seq_pairs)  # generate predictions, forward pass
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels, out.shape))
        # print('training acc: {}'.format(accuracy(out, labels).detach()))
        return loss

    def validation_step(self, batch):
        seq_pairs, labels = batch

        seq_pairs, labels = seq_pairs.to(compute_device), labels.to(
            compute_device)

        out = self(seq_pairs)  # generate predictions

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

    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch,
                                                                     result[
                                                                         'loss'],
                                                                     result[
                                                                         'acc']))

    def plot(self, path=None):
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

            line_train, = axs[1].plot(losses_train, '-x', label='training', color=tuple([c/1.3 for c in train_col]))
            line_val, = axs[1].plot(losses_val, '-x', label='validation', color=tuple([c/1.3 for c in val_col]))
            axs[1].set_xlabel('epoch')
            axs[1].set_ylabel('loss')
            axs[1].set_title(f'Loss vs. No. of epochs')

            plt.subplots_adjust(bottom=0.25)
            plt.legend([line_train, line_val], ['training', 'validation'], bbox_to_anchor=[-0.1, -0.3],
                       loc='lower center', ncol=2)

            if path is not None:
                plt.savefig(path)

    def save(self, path):
        torch.save({
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_state_dict': self.state_dict()
            },
            path)

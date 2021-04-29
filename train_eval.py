import errno
import sys, random, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
from data_preprocessing import aligns_from_fastas, TensorDataset, encode_align, \
    make_seq_pairs
from utils import write_config_file
import ConvNet

compute_device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)  # loss/acc mean for epoch


def fit(epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.Adagrad):

    optimizer = opt_func(model.parameters(), lr)

    # validation phase with initialized weights (untrained network)
    model.eval()
    with torch.no_grad():
        # eval for training dataset
        train_result = evaluate(model, train_loader)
        model.epoch_end(0, train_result)
        model.train_history.append(train_result)
        # eval for validataion dataset
        val_result = evaluate(model, val_loader)
        model.epoch_end(0, val_result)
        model.val_history.append(val_result)

    for epoch in range(1, epochs+1):

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
            # eval for training dataset
            train_result = evaluate(model, train_loader)
            model.epoch_end(epoch, train_result)
            model.train_history.append(train_result)
            # eval for validataion dataset
            val_result = evaluate(model, val_loader)
            model.epoch_end(epoch, val_result)
            model.val_history.append(val_result)


def load_net(self, path, seq_len, state='eval'):
    model = ConvNet(seq_len)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train_history = checkpoint['train_history']
    model.val_history = checkpoint['val_history']
    if state == 'eval':
        model.eval()
    elif state == 'train':
        model.train()


def plot_folds(train_history_folds, val_history_folds, path=None):
    accs_train_folds = []
    losses_train_folds = []
    accs_val_folds = []
    losses_val_folds = []
    for fold in range(len(train_history_folds)):
        accs_train_folds.append([result['acc'] for result in train_history_folds[fold]])
        losses_train_folds.append([result['loss'] for result in train_history_folds[fold]])
        accs_val_folds.append([result['acc'] for result in val_history_folds[fold]])
        losses_val_folds.append([result['loss'] for result in val_history_folds[fold]])

    # calculate mean over folds per epoch
    avg_accs_train = np.mean(accs_train_folds, axis=0)
    avg_losses_train = np.mean(losses_train_folds, axis=0)
    avg_accs_val = np.mean(accs_val_folds, axis=0)
    avg_losses_val = np.mean(losses_val_folds, axis=0)

    # calculate standard deviation over folds per epoch
    std_accs_train = np.std(accs_train_folds, axis=0)
    std_losses_train = np.std(losses_train_folds, axis=0)
    std_accs_val = np.std(accs_val_folds, axis=0)
    std_losses_val = np.std(losses_val_folds, axis=0)

    # plot the model evaluation
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(12., 6.))

    train_col, train_col_a = (0.518, 0.753, 0.776), (0.518, 0.753, 0.776, 0.2)
    val_col, val_col_a = (0.576, 1.0, 0.588), (0.576, 1.0, 0.588, 0.3)

    axs[0].plot(avg_accs_train, '-x', color=train_col)
    axs[0].fill_between(range(len(avg_accs_train)), avg_accs_train - std_accs_train,
                        avg_accs_train + std_accs_train, color=train_col_a)
    axs[0].plot(avg_accs_val, '-x', color=val_col)
    axs[0].fill_between(range(len(avg_accs_val)), avg_accs_val - std_accs_val,
                        avg_accs_val + std_accs_val, color=val_col_a)
    axs[0].set_ylim([0.5, 1.0])
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')
    axs[0].set_title(f'Fold {(fold + 1)}: Accuracy vs. No. of epochs')

    line_train, = axs[1].plot(avg_losses_train, '-x', color=tuple([c/1.3 for c in train_col]))
    axs[1].fill_between(range(len(avg_losses_train)), avg_losses_train - std_losses_train,
                        avg_losses_train + std_losses_train, color=tuple([c/1.3 for c in train_col_a]))
    line_val, = axs[1].plot(avg_losses_val, '-x', color=tuple([c/1.3 for c in val_col]))
    axs[1].fill_between(range(len(avg_losses_val)), avg_losses_val - std_losses_val,
                        avg_losses_val + std_losses_val, color=tuple([c/1.3 for c in val_col_a]))
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title(f'Fold {fold + 1}: Loss vs. No. of epochs')

    plt.subplots_adjust(bottom=0.25)
    plt.legend([line_train, line_val], ['training', 'validation'], bbox_to_anchor=[-0.1, -0.3],
               loc='lower center', ncol=2)

    if path is not None:
        plt.savefig(path)


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
    min_seqs_per_align, max_seqs_per_align = 4, 300
    seq_len = 300

    # hyperparameters
    batch_size = 1024
    epochs = 2
    lr = 0.001
    optimizer = 'Adagrad'
    nb_folds = 2

    if model_path is not None:
        write_config_file(nb_protein_families,
                          min_seqs_per_align,
                          max_seqs_per_align,
                          seq_len, batch_size,
                          epochs, lr, optimizer,
                          nb_folds, model_path)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # k-fold validator
    kfold = KFold(nb_folds, shuffle=True, random_state=42)

    # -------------------- data preparation ------------------------------- #
    print("Loading alignments ...")
    # load sets of multiple aligned sequences
    real_aligns = aligns_from_fastas(real_fasta_path, min_seqs_per_align,
                                     max_seqs_per_align, nb_protein_families)
    sim_aligns = aligns_from_fastas(sim_fasta_path, min_seqs_per_align,
                                    max_seqs_per_align, nb_protein_families)

    print("Encoding alignments ...")
    # one-hot encode sequences shape: (nb_aligns, nb_seqs, amino acids, seq_length)
    real_aligns = [encode_align(align, seq_len, padding='data') for align in
                   real_aligns]
    sim_aligns = [encode_align(align, seq_len, padding='data') for align in
                  sim_aligns]

    print("Pairing sequences ...")
    start = time.time()
    # make pairs !additional dim for each multiple alingment needs to be flattened before passed to CNN!
    real_pairs_per_align = [make_seq_pairs(align) for align in real_aligns]
    sim_pairs_per_align = [make_seq_pairs(align) for align in sim_aligns]
    print(f'Finished pairing after {round(time.time() - start, 2)}s\n')


    # -------------------- k-fold cross validation -------------------- #

    # init for evaluation
    train_history_folds = []
    val_history_folds = []
    fold_eval = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(real_aligns)):

        print(f'FOLD {fold + 1}')
        print(
            '----------------------------------------------------------------')

        # splitting dataset by alignments
        print("Building training and validation dataset ...")
        start = time.time()
        train_ds = TensorDataset([real_pairs_per_align[i] for i in train_ids],
                                 [sim_pairs_per_align[i] for i in train_ids], shuffle=True)
        val_ds = TensorDataset([real_pairs_per_align[i] for i in val_ids],
                               [sim_pairs_per_align[i] for i in val_ids], shuffle=True)
        print(f'Finished after {round(time.time() - start, 2)}s\n')

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        input_size = train_ds.data.shape[2]  # seq len
        model = ConvNet(input_size)
        model = model.to(compute_device)

        # train and validate model
        fit(epochs, lr, model, train_loader, val_loader)
        train_history_folds.append(model.train_history)
        val_history_folds.append(model.val_history)
        fold_eval.append(val_history_folds[fold][-1]['acc'])

        # saving the model
        print('\nTraining process has finished.')
        if model_path is not None:
            torch.save({
                        'train_history': model.train_history,
                        'val_history': model.val_history,
                        'model_state_dict': model.state_dict(),
                       },
                       f'{model_path}/model-fold-{fold + 1}.pth')

        if model_path is not None:
            plt.savefig(f'{model_path}/fig-fold-{fold + 1}.png')
            print(f'Saved model and evaluation plot to {model_path} ...\n')

    # print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
    print('----------------------------------------------------------------')

    for i, acc in enumerate(fold_eval):
        print(f'Fold {(i + 1)}: {acc} %')

    print(f'Average: {np.sum(fold_eval) / len(fold_eval)} %')


if __name__ == '__main__':
    main(sys.argv[1:])

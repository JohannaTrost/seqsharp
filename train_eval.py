import errno
import sys, random, os

import pandas as pd
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datetime import datetime
from data_preprocessing import TensorDataset, DatasetAln, data_prepro, padding, \
    nb_seqs_per_alns, seq_len_per_alns, get_aa_freqs, mmse, distance_stats, alns_from_fastas
from plots import plot_folds
from utils import write_config_file
from ConvNet import ConvNet, evaluate, load_net, accuracy

compute_device = "cuda" if torch.cuda.is_available() else "cpu"
compute_device = "cpu"

def fit(epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.Adagrad):
    optimizer = opt_func(model.parameters(), lr)

    # validation phase with initialized weights (untrained network)
    model.eval()
    with torch.no_grad():
        # eval for training dataset
        train_result = evaluate(model, train_loader)
        # model.epoch_end(0, train_result)
        model.train_history.append(train_result)
        # eval for validataion dataset
        val_result = evaluate(model, val_loader)
        model.epoch_end(0, val_result)
        model.val_history.append(val_result)

    for epoch in range(1, epochs + 1):

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


def eval_per_align(conv_net, real_alns, sim_alns,
                   fastas_real, fastas_sim, indices, pairs=False):
    # evaluate val data per alignment
    acc_real = {'acc': [], 'aln': np.asarray(fastas_real)[indices]}
    acc_sim = {'acc': [], 'aln': np.asarray(fastas_sim)[indices]}
    conv_net.eval()
    if pairs:
        for i in indices:
            loader_real = DataLoader(DatasetAln(real_alns[i], True),
                                     len(real_alns[i]))
            loader_sim = DataLoader(DatasetAln(sim_alns[i], False),
                                    len(sim_alns[i]))
            with torch.no_grad():
                    # eval for real align
                    result = evaluate(conv_net, loader_real)
                    conv_net.epoch_end('-', result)
                    acc_real['acc'].append(result['acc'])
                    # eval for sim align
                    result = evaluate(conv_net, loader_sim)
                    conv_net.epoch_end('-', result)
                    acc_sim['acc'].append(result['acc'])
    else:
        with torch.no_grad():
            # prediction for empirical alns
            alns = torch.from_numpy(np.asarray(real_alns)[indices]).float()
            labels = torch.FloatTensor([0]*len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)
            conv_net.to(compute_device)
            outputs = conv_net(alns)
            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_real['acc'] = list(map(int, (preds == labels)))

            # prediction for simulated alns
            alns = torch.from_numpy(np.asarray(sim_alns)[indices]).float()
            labels = torch.FloatTensor([1]*len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)
            conv_net.to(compute_device)
            outputs = conv_net(alns)
            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_sim['acc'] = list(map(int, (preds == labels)))

    return acc_real, acc_sim


def kfold_eval_per_aln(k, real_alns, sim_alns,
                       fastas_real, fastas_sim, seq_len, model_path):

    # k-fold validator
    kfold = KFold(k, shuffle=True, random_state=42)

    val_real_folds = []
    train_real_folds = []
    val_sim_folds = []
    train_sim_folds = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(real_alns)):
        print(f'FOLD {fold + 1}')
        print(
            '----------------------------------------------------------------')
        # splitting dataset by alignments
        print("Building training and validation dataset ...")

        # loading model
        convNet = load_net(f'{model_path}/model-fold-{fold + 1}.pth', seq_len, state='eval')
        convNet.to(compute_device)

        # evaluate training data per alignment
        train_acc_real, train_acc_sim = eval_per_align(convNet, real_alns, sim_alns,
                                                       fastas_real, fastas_sim, train_ids)

        # evaluate val data per alignment
        val_acc_real, val_acc_sim = eval_per_align(convNet, real_alns, sim_alns,
                                                   fastas_real, fastas_sim, val_ids)

        val_real_folds.append(val_acc_real)
        train_real_folds.append(train_acc_real)
        val_sim_folds.append(val_acc_sim)
        train_sim_folds.append(train_acc_sim)

    return train_real_folds, train_sim_folds, val_real_folds, val_sim_folds


def generate_eval_dict(fold, train_real, train_sim, val_real, val_sim,
                       real_alns, sim_alns, save=None):
    accs = train_real['acc'] + val_real['acc'] + train_sim['acc'] + \
           val_sim['acc']
    ids_real = np.concatenate((train_real['aln'], val_real['aln']))
    ids_sim = np.concatenate((train_sim['aln'], val_sim['aln']))
    ids = np.concatenate((ids_real, ids_sim))

    # save info about alignments
    is_val = [0] * len(train_real['acc']) + [1] * len(val_real['acc']) + \
             [0] * len(train_sim['acc']) + [1] * len(val_sim['acc'])
    # distances real aln to all real training data seperated from simulated
    dists_real_sim = np.asarray([
        [mmse(real_alns[id1], real_alns[id2]) for id2 in train_real['aln']]
        if id1 in ids_real
        else
        [mmse(sim_alns[id1], sim_alns[id2]) for id2 in train_sim['aln']]
        for id1 in ids])
    # distances of alignment to simulated and real training data
    all_alns = dict(real_alns, **sim_alns)
    dists_both = np.asarray([
        [mmse(all_alns[id1], all_alns[id2])
         for id2 in np.concatenate((train_real['aln'], train_sim['aln']))]
        for id1 in ids])

    mse_stats_real_sim = distance_stats(dists_real_sim)
    mse_stats_both = distance_stats(dists_both)

    dat_dict = {'fold': [fold] * len(ids),
                'id': ids,
                'accuracy': accs,
                'is_val': is_val,
                'mean_mse_sep': mse_stats_real_sim['mean'],
                'max_mse_sep': mse_stats_real_sim['max'],
                'min_mse_sep': mse_stats_real_sim['min'],
                'mean_mse_both': mse_stats_both['mean'],
                'max_mse_both': mse_stats_both['max'],
                'min_mse_both': mse_stats_both['min']
                }

    df = pd.DataFrame(dat_dict)

    if save is not None:
        csv_string = df.to_csv(index=False)
        with open(save, 'a') as file:
            file.write(csv_string)

    return df


def main(args):
    # -------------------- handling arguments -------------------- #

    if len(args) >= 3:
        real_fasta_path = args[0]
        sim_fasta_path = args[1]
        shuffle = True if args[2] == 'y' else False
        model_path = None

        if len(args) > 3:
            model_path = args[3]
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
    min_seqs_per_align, max_seqs_per_align = 4, 3000
    seq_len = 3000
    raw_real_alns, _ = alns_from_fastas(real_fasta_path,
                                        min_seqs_per_align,
                                        max_seqs_per_align,
                                        nb_protein_families)

    seq_len = int(min(seq_len, np.max(seq_len_per_alns(raw_real_alns))))

    # hyperparameters
    batch_size = 32
    epochs = 50
    lr = 0.01
    optimizer = 'Adagrad'
    nb_folds = 10

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

    # ----------------------------------------------- data preparation ----------------------------------------------- #
    real_alns, sim_alns, fastas_real, fastas_sim = data_prepro(
        real_fasta_path, sim_fasta_path, nb_protein_families,
        min_seqs_per_align, max_seqs_per_align, seq_len,
        csv_path=(f'{model_path}/alns_stats.csv'
                  if model_path is not None else None))

    real_alns_dict = {fastas_real[i]: real_alns[i] for i in
                      range(len(real_alns))}
    sim_alns_dict = {fastas_sim[i]: sim_alns[i] for i in range(len(sim_alns))}

    # -------------------- k-fold cross validation -------------------- #

    # init for evaluation
    train_history_folds = []
    val_history_folds = []
    fold_eval = []
    dfs = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(real_alns)):

        print(f'FOLD {fold + 1}')
        print(
            '----------------------------------------------------------------')

        # splitting dataset by alignments
        print("Building training and validation dataset ...")
        start = time.time()

        train_ds = TensorDataset([real_alns[i] for i in train_ids],
                                 [sim_alns[i] for i in train_ids], shuffle)
        val_ds = TensorDataset([real_alns[i] for i in val_ids],
                               [sim_alns[i] for i in val_ids], shuffle)
        print(f'Finished after {round(time.time() - start, 2)}s\n')

        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size)

        # generate model
        input_size = train_ds.data.shape[2]  # seq len
        nb_chnls = train_ds.data.shape[1]  # 1 channel per amino acid
        model = ConvNet(input_size, nb_chnls)
        model = model.to(compute_device)

        # train and validate model
        fit(epochs, lr, model, train_loader, val_loader)
        train_history_folds.append(model.train_history)
        val_history_folds.append(model.val_history)
        fold_eval.append(val_history_folds[fold][-1]['acc'])

        # saving the model and results
        print('\nTraining process has finished.')
        if model_path is not None:
            model.save(f'{model_path}/model-fold-{fold + 1}.pth')
            model.plot(f'{model_path}/fig-fold-{fold + 1}.png')

            real_pred_tr, sim_pred_tr = eval_per_align(model, real_alns,
                                                       sim_alns, fastas_real,
                                                       fastas_sim, train_ids)
            real_pred_va, sim_pred_va = eval_per_align(model, real_alns,
                                                       sim_alns, fastas_real,
                                                       fastas_sim, val_ids)
            df = generate_eval_dict(fold, real_pred_tr, sim_pred_tr, real_pred_va,
                                    sim_pred_va, real_alns_dict, sim_alns_dict)
            dfs.append(df)

            print(f'Saved model and evaluation plot to {model_path} ...\n')
        else:
            model.plot()

    # print/save fold results
    plot_folds(train_history_folds, val_history_folds,
               f'{model_path}/fig-fold-eval.png'
               if model_path is not None else None)
    if len(dfs) > 0:
        df_c = pd.concat(dfs)
        csv_string = df_c.to_csv(index=False)
        with open(model_path + '/aln_train_eval.csv', 'w') as file:
            file.write(csv_string)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
    print('----------------------------------------------------------------')

    for i, acc in enumerate(fold_eval):
        print(f'Fold {(i + 1)}: {acc} %')

    print(f'Average: {np.sum(fold_eval) / len(fold_eval)} %')


if __name__ == '__main__':
    main(sys.argv[1:])

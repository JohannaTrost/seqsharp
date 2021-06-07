"""Functions to train and evaluate a neural network"""

import os
import gc

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import KFold
from torch import nn as nn
from torch.utils.data import DataLoader

from ConvNet import load_net, compute_device
from preprocessing import DatasetAln
from stats import mse, distance_stats

process = psutil.Process(os.getpid())

torch.cuda.empty_cache()

gc.collect()

compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'compute device : {compute_device}')


def accuracy(outputs, labels):
    """Calculates accuracy of network predictions

    :param outputs: network output for all examples (torch tensor)
    :param labels: 0 and 1 labels (0: empirircal, 1:simulated) (torch tensor)
    :return: accuracy values (between 0 and 1) (torch tensor)
    """

    preds = torch.round(torch.flatten(torch.sigmoid(outputs))).to(compute_device)
    return torch.tensor((torch.sum(preds == labels).item() / len(preds)))


def evaluate(model, val_loader):
    """Feeds the model with data and gets its performance

    :param model: ConvNet object
    :param val_loader: networks input data (DataLoader object)
    :return: losses and accuracies for each data batch (dict)
    """

    outputs, losses, labels = [], [], []

    for batch in val_loader:
        alns, labels_batch = batch

        alns = alns.to(compute_device)
        labels_batch = labels_batch.to(compute_device)

        out = model(alns)  # generate predictions

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.reshape(labels_batch, out.shape))

        losses.append(loss.detach())
        outputs = (out if len(outputs) == 0
                   else torch.cat((outputs, out)).to(compute_device))
        labels = (labels_batch if len(labels) == 0
                  else torch.cat((labels, labels_batch))).to(compute_device)

    epoch_acc = accuracy(outputs, labels)
    epoch_loss = torch.stack(losses).mean()

    return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}


def fit(epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.Adagrad):
    """
    Training a model to learn a function to distinguish between simulated and
    empirical alignments (with validation step at the end of each epoch)
    :param epochs: number of repitions of training (integer)
    :param lr: learning rate (float)
    :param model: ConvNet object
    :param train_loader: training dataset (DataLoader)
    :param val_loader: validation dataset (DataLoader)
    :param opt_func: optimizer (torch.optim)
    """

    optimizer = opt_func(model.parameters(), lr)

    # validation phase with initialized weights (untrained network)
    model.eval()
    with torch.no_grad():
        # eval for training dataset
        train_result = evaluate(model, train_loader)
        model.train_history.append(train_result)
        # eval for validataion dataset
        val_result = evaluate(model, val_loader)
        print(f"Epoch [0], loss: {val_result['loss']}, "
              f"acc: {val_result['acc']}")
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
            print(f"Epoch [{epoch}], loss: {train_result['loss']}, "
                  f"acc: {train_result['acc']}")
            model.train_history.append(train_result)
            # eval for validataion dataset
            val_result = evaluate(model, val_loader)
            print(f"Epoch [{epoch}], loss: {val_result['loss']}, "
                  f"acc: {val_result['acc']}")
            model.val_history.append(val_result)


def eval_per_align(conv_net, real_alns, sim_alns,
                   fastas_real, fastas_sim, indices, pairs=False):
    """
    Predicting whether given alignments are simulated or empirical using a
    trained network
    :param conv_net: ConvNet object
    :param real_alns: list of floats of shape (alignments, aa channels, sites)
    :param sim_alns: list of floats of shape (alignments, aa channels, sites)
    :param fastas_real: list of alignment identifiers (string list)
    :param fastas_sim: list of alignment identifiers (string list)
    :param indices: list of integers to pick alignments (e.g. from training set)
    :param pairs: whether or not the alignments are represented as pairs (bool)
    (which would mean an additional (pair-)dimension in real_alns, sim_alns)
    :return: acc_real, acc_sim : dictionaries with accuracies and alignment ids
    """
    # evaluate val data per alignment
    acc_real = {'acc': [], 'aln': np.asarray(fastas_real)[indices]}
    acc_sim = {'acc': [], 'aln': np.asarray(fastas_sim)[indices]}

    conv_net.eval()
    conv_net = conv_net.to(compute_device)

    if pairs:
        for i in indices:
            loader_real = DataLoader(DatasetAln(real_alns[i], True),
                                     len(real_alns[i]))
            loader_sim = DataLoader(DatasetAln(sim_alns[i], False),
                                    len(sim_alns[i]))
            with torch.no_grad():
                    # eval for real align
                    result = evaluate(conv_net, loader_real)
                    print(f"loss: {result['loss']}, acc: {result['acc']}")
                    acc_real['acc'].append(result['acc'])
                    # eval for sim align
                    result = evaluate(conv_net, loader_sim)
                    print(f"loss: {result['loss']}, acc: {result['acc']}")
                    acc_sim['acc'].append(result['acc'])
                    del result
    else:
        with torch.no_grad():
            # prediction for empirical alns
            alns = torch.from_numpy(np.asarray(real_alns)[indices]).float()
            labels = torch.FloatTensor([0]*len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)

            outputs = conv_net(alns)

            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_real['acc'] = list(map(int, (preds == labels)))

            # prediction for simulated alns
            alns = torch.from_numpy(np.asarray(sim_alns)[indices]).float()
            labels = torch.FloatTensor([1]*len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)

            outputs = conv_net(alns)
            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_sim['acc'] = list(map(int, (preds == labels)))

            del alns, labels, outputs, preds

    return acc_real, acc_sim


def kfold_eval_per_aln(k, real_alns, sim_alns, fastas_real, fastas_sim,
                       seq_len, nb_chnls, nb_conv_layer, nb_lin_layer,
                       model_path):
    """
    Loading a trained model and computing predictions on given data
    :param k: number of folds
    :param real_alns: list of floats of shape (alignments, aa channels, sites)
    :param sim_alns: list of floats of shape (alignments, aa channels, sites)
    :param fastas_real: list of empirical alignment identifiers (string list)
    :param fastas_sim: list of simulated alignment identifiers (string list)
    :param seq_len: number of sites (integer)
    :param nb_chnls: number of channels (integer) typically 1 channel per aa
    :param nb_conv_layer: number of convoltional layers (integer)
    :param nb_lin_layer: number of linear layers (integer)
    :param model_path: <path/to/> trained model
    :return: accuracies and alignment ids for training and validation sets
    """

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
        convNet = load_net(f'{model_path}/model-fold-{fold + 1}.pth',
                           seq_len, nb_chnls, nb_conv_layer, nb_lin_layer,
                           state='eval')
        convNet = convNet.to(compute_device)

        # evaluate training data per alignment
        train_acc_real, train_acc_sim = eval_per_align(convNet, real_alns,
                                                       sim_alns, fastas_real,
                                                       fastas_sim, train_ids)

        # evaluate val data per alignment
        val_acc_real, val_acc_sim = eval_per_align(convNet, real_alns, sim_alns,
                                                   fastas_real, fastas_sim,
                                                   val_ids)

        val_real_folds.append(val_acc_real)
        train_real_folds.append(train_acc_real)
        val_sim_folds.append(val_acc_sim)
        train_sim_folds.append(train_acc_sim)

    return train_real_folds, train_sim_folds, val_real_folds, val_sim_folds


def generate_eval_dict(fold, train_real, train_sim, val_real, val_sim,
                       real_alns, sim_alns, save=None):
    """
    Generate data frame containing training specific information about the
    alignment e.g. the distance between an alignment and the training data set
    :param fold: number of folds (int)
    :param train_real: empirical training data (dict with accuracies and ids)
    :param train_sim: simulated training data (dict with accuracies and ids)
    :param val_real: empirical validation data (dict with accuracies and ids)
    :param val_sim: simulated validation data (dict with accuracies and ids)
    :param real_alns: dict: keys: alignment ids values: alignment representation
    :param sim_alns: dict: keys: alignment ids values: alignment representation
    :param save: <path/to/> save the data frame of alignment infos as a csv file
    :return: data frame with training related infos about each alignment
    """
    accs = train_real['acc'] + val_real['acc']
    accs += train_sim['acc'] + val_sim['acc']
    ids_real = np.concatenate((train_real['aln'], val_real['aln']))
    ids_sim = np.concatenate((train_sim['aln'], val_sim['aln']))
    ids = np.concatenate((ids_real, ids_sim))

    # save info about alignments
    is_val = [0] * len(train_real['acc']) + [1] * len(val_real['acc']) + \
             [0] * len(train_sim['acc']) + [1] * len(val_sim['acc'])
    # distances real aln to all real training data seperated from simulated
    dists_real_sim = np.asarray([
        [mse(real_alns[id1], real_alns[id2]) for id2 in train_real['aln']]
        if id1 in ids_real
        else
        [mse(sim_alns[id1], sim_alns[id2]) for id2 in train_sim['aln']]
        for id1 in ids])
    # distances of alignment to simulated and real training data
    all_alns = dict(real_alns, **sim_alns)
    dists_both = np.asarray([
        [mse(all_alns[id1], all_alns[id2])
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

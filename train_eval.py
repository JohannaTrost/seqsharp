"""Functions to train and evaluate a neural network"""

import gc

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ConvNet import compute_device, accuracy
from preprocessing import DatasetAln
from stats import mse, distance_stats

torch.cuda.empty_cache()

gc.collect()


def evaluate(model, val_loader):
    """Feeds the model with data and returns its performance

    :param model: ConvNet object
    :param val_loader: networks input data (DataLoader object)
    :return: losses and accuracies for each data batch (dict)
    """

    losses = []
    preds = torch.tensor([]).to(compute_device)
    labels = torch.tensor([]).to(compute_device)

    for batch in val_loader:
        batch_loss, batch_y_pred, batch_y_true = model.feed(batch)
        losses.append(batch_loss)
        preds = torch.cat((preds, batch_y_pred))
        labels = torch.cat((labels, batch_y_true))

    # combine losses and accuracies
    epoch_acc_loss = accuracy(preds, labels)
    epoch_acc_loss['loss'] = torch.stack(losses).mean()

    for key, val in epoch_acc_loss.items():
        epoch_acc_loss[key] = val.item()

    return epoch_acc_loss

def fit(epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.Adagrad, start_epoch=0):
    """
    Training a model to learn a function to distinguish between simulated and
    empirical alignments (with validation step at the end of each epoch)
    :param start_epoch: possibility to continue training from this epoch
    :param epochs: number of repetitions of training (integer)
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
        for key, val in train_result.items():
            model.train_history[key].append(val)
        # eval for validataion dataset
        val_result = evaluate(model, val_loader)
        for key, val in val_result.items():
            model.val_history[key].append(val)

        print(f"Epoch [0], loss: {val_result['loss']}, "
              f"acc: {val_result['acc']}, emp. acc: {val_result['acc_emp']}, "
              f"sim. acc: {val_result['acc_sim']}")

    for epoch in range(start_epoch+1, epochs + 1):

        print(f'Epoch [{epoch}]')

        # training Phase
        model.train()
        for i, batch in enumerate(train_loader):
            loss, _, _ = model.feed(batch)
            optimizer.zero_grad()
            loss.backward()  # calcul of gradients
            optimizer.step()

        # validation phase
        model.eval()
        with torch.no_grad():
            # eval for training dataset
            train_result = evaluate(model, train_loader)
            for key, val in train_result.items():
                model.train_history[key].append(val)
            print(f"Training: Loss: {np.round(train_result['loss'], 4)}, "
                  f"Acc.: {np.round(train_result['acc'], 4)}")
            # eval for validataion dataset
            val_result = evaluate(model, val_loader)
            for key, val in val_result.items():
                model.val_history[key].append(val)
            print(f"Validation: Loss: {np.round(val_result['loss'], 4)}, "
                  f"Acc.: {np.round(val_result['acc'], 4)}, "
                  f"Emp. acc.: {np.round(val_result['acc_emp'], 4)}, "
                  f"Sim. acc.: {np.round(val_result['acc_sim'], 4)}")


def eval_per_align(conv_net, real_alns, sim_alns,
                   fastas_real, fastas_sim, indices, pairs=False):
    """
    Predicting whether given alignments are simulated or empirical using a
    given (trained) network
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
            labels = torch.FloatTensor([0] * len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)

            outputs = conv_net(alns)

            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_real['acc'] = list(map(int, (preds == labels)))

            # prediction for simulated alns
            alns = torch.from_numpy(np.asarray(sim_alns)[indices]).float()
            labels = torch.FloatTensor([1] * len(indices))
            alns, labels = alns.to(compute_device), labels.to(compute_device)

            outputs = conv_net(alns)
            preds = torch.round(torch.flatten(torch.sigmoid(outputs)))
            acc_sim['acc'] = list(map(int, (preds == labels)))

            del alns, labels, outputs, preds

    return acc_real, acc_sim


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

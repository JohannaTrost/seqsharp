"""Functions to train and evaluate a neural network"""

import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from matplotlib import pylab as plt

from ConvNet import compute_device, accuracy
from preprocessing import DatasetAln
from stats import mse, distance_stats
from tompy import median_smooth

torch.cuda.empty_cache()

gc.collect()


def find_lr_bounds(model, train_loader, opt_func, save, lr_range=None,
                   lr_find_epochs=3, prefix=''):
    start = time.time()
    start_lr, end_lr = (1e-07, 0.1) if lr_range == '' or lr_range else lr_range

    lr_lambda = lambda x: np.exp(x * np.log(end_lr / start_lr) / (
            lr_find_epochs * len(train_loader)))
    optimizer = opt_func(model.parameters(), start_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    lr_find_loss = []
    lr_find_lr = []

    iter = 0
    smoothing = 0.05

    print('Starting determination of suitable lr bounds')

    for epoch in range(lr_find_epochs):
        print(f'Epoch [{epoch + 1}]')
        model.train()
        for i, batch in enumerate(train_loader):
            loss, _, _ = model.feed(batch)
            optimizer.zero_grad()
            loss.backward()  # calcul of gradients
            optimizer.step()
            # Update LR
            scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            # smooth the loss
            loss = loss.detach().item()
            if iter == 0:
                lr_find_loss.append(loss)
            else:
                loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss)

            iter += 1

    print(f'LR finder finished after {(time.time() - start) / 60} min')

    high_bound_lr = lr_find_lr[np.argmin(lr_find_loss)] / 10
    low_bound_lr = high_bound_lr / 6

    print(f'lr = [{low_bound_lr}, {high_bound_lr}]')

    if save is not None and save != '':
        save = f'{save}/lrfinder'
        if not os.path.exists(save):
            os.mkdir(save)
        fig, ax = plt.subplots()
        ax.plot(lr_find_lr, lr_find_loss)
        ylims = ax.get_ylim()
        ax.vlines(high_bound_lr, *ylims, color='r')
        plt.xscale('log')
        plt.savefig(f'{save}/{prefix}find_lr_loss_{start_lr}_{end_lr}_'
                    f'{lr_find_epochs}.png')
        plt.close('all')

        plt.plot(lr_find_lr)
        plt.savefig(f'{save}/{prefix}find_lr_{start_lr}_{end_lr}_'
                    f'{lr_find_epochs}.png')
        plt.close('all')

    return high_bound_lr, low_bound_lr


def validation(model, train_loader, val_loader):
    model.eval()
    with torch.no_grad():
        # eval for training dataset
        train_result = evaluate(model, train_loader)
        # eval for validataion dataset
        val_result = evaluate(model, val_loader)
    for key, val in train_result.items():
        model.train_history[key].append(val)
    print(f"Training: Loss: {np.round(train_result['loss'], 4)}, "
          f"Acc.: {np.round(train_result['acc'], 4)}")
    for key, val in val_result.items():
        model.val_history[key].append(val)
    print(f"Validation: Loss: {np.round(val_result['loss'], 4)}, "
          f"Acc.: {np.round(val_result['acc'], 4)}, "
          f"Emp. acc.: {np.round(val_result['acc_emp'], 4)}, "
          f"Sim. acc.: {np.round(val_result['acc_sim'], 4)}")
    return model


def evaluate_folds(val_hist_folds, nb_folds, which='best'):
    """Return acc., emp./sim. acc. and loss of best epoch for each fold

    :param val_hist_folds: acc., emp./sim. acc. and loss for each epoch and fold
    :param nb_folds: number of folds
    :return: dict with acc., emp./sim. acc. and loss of epoch with min.
    loss for each fold
    """

    val_folds = {'loss': [], 'acc': [], 'acc_emp': [], 'acc_sim': []}
    for fold in range(nb_folds):
        if which == 'best':
            epoch = np.argmax(val_hist_folds['acc'][fold])
        elif which == 'last':
            epoch = val_hist_folds['acc'][fold][-1]
        for key in val_folds.keys():
            val_folds[key].append(val_hist_folds[key][fold][epoch])
    return val_folds


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


def fit(lr, model, train_loader, val_loader, opt_func=torch.optim.Adagrad,
        start_epoch=0, max_epochs=1000, min_epochs=100, patience=2,
        step_size=50, min_delta=1e-04):
    """
    Training a model to learn a function to distinguish between simulated and
    empirical alignments (with validation step at the end of each epoch)
    :param min_delta: threshold for early stopping (min. loss difference)
    :param patience: nb. of epochs without change that triggers early stopping
    :param max_epochs: max. number of repetitions of training (integer)
    :param start_epoch: possibility to continue training from this epoch
    :param lr: learning rate (float)
    :param model: ConvNet object
    :param train_loader: training dataset (DataLoader)
    :param val_loader: validation dataset (DataLoader)
    :param opt_func: optimizer (torch.optim)
    """
    # init optimizer and scheduler (if used)
    if isinstance(lr, list) or isinstance(lr, tuple):
        optimizer = opt_func(model.parameters(), lr[0])
        # a range of LRs is given indicating usage of CLR scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, lr[0],
                                                      lr[1],
                                                      cycle_momentum=False)
    else:
        optimizer = opt_func(model.parameters(), lr)
        scheduler = None

    # load state dict of optimizer/scheduler to resume training
    if model.opt_state is not None:
        optimizer.load_state_dict(model.opt_state)
    if model.scheduler_state is not None:
        scheduler.load_state_dict(model.scheduler_state)

    print('Epoch [0]')
    # validation phase with initialized weights (untrained network)
    model = validation(model, train_loader, val_loader)

    no_imporv_cnt = 0
    time_per_step = []
    for epoch in range(start_epoch + 1, max_epochs + 1):

        print(f'Epoch [{epoch}]')

        # training Phase
        model.train()
        for i, batch in enumerate(train_loader):
            start_step = time.time()
            loss, _, _ = model.feed(batch)
            time_per_step.append(time.time() - start_step)

            optimizer.zero_grad()
            loss.backward()  # calcul of gradients
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # validation phase
        model = validation(model, train_loader, val_loader)

        if (epoch + 1) % step_size == 0 and epoch > min_epochs - 1:
            # do eval for early stopping every 50th epoch
            # after reaching min num epochs

            # smooth_val_loss = median_smooth(model.val_history['loss'], 25)
            curr_val_loss = np.min(model.val_history['loss'][-step_size:])

            if prev_val_loss - curr_val_loss < min_delta:
                no_imporv_cnt += 1
                if no_imporv_cnt >= patience:
                    print(f'\nEarly stopping at epoch {epoch}\n')
                    break
            else:
                no_imporv_cnt = 0
            prev_val_loss = curr_val_loss
        elif epoch == min_epochs - (step_size + 1):
            prev_val_loss = np.min(model.val_history['loss'][-step_size:])

    model.opt_state = optimizer.state_dict()
    if model.scheduler_state is not None:  # because CLR is optional
        model.scheduler_state = scheduler.state_dict()

    return np.mean(time_per_step)




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
    :param pairs: whether the alignments are represented as pairs (bool)
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

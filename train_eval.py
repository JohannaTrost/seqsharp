"""Functions to train and evaluate a neural network"""

import gc
import os
import time

import numpy as np
import pandas as pd
import torch

from matplotlib import pylab as plt

from ConvNet import compute_device, accuracy


torch.cuda.empty_cache()

gc.collect()


def find_lr_bounds(model, train_loader, opt_func, save, lr_range=None,
                   lr_find_epochs=5, prefix=''):
    start = time.time()
    start_lr, end_lr = (1e-07, 0.1) if lr_range == '' else lr_range

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
    """Return acc., emp./sim. acc. and loss of best/last epoch for each fold

    :param val_hist_folds: acc., emp./sim. acc. and loss for each epoch and fold
    :param nb_folds: number of folds
    :return: dict with acc., emp./sim. acc. and loss of epoch with min.
    loss for each fold
    """

    val_folds = {'loss': [], 'acc': [], 'acc_emp': [], 'acc_sim': []}
    best_epochs = []
    for fold in range(nb_folds):
        if which == 'best':
            epoch = np.argmax(val_hist_folds[fold]['acc'])
            best_epochs.append((int(epoch)))
        elif which == 'last':
            epoch = -1
        for key in val_folds.keys():
            val_folds[key].append(val_hist_folds[fold][key][epoch])

    return val_folds, best_epochs


def get_close_baccs(baccs, selected_epochs):
    # select 4 epochs preceding and succeeding best epoch
    bacc_around_sel = []
    for fold, epoch in enumerate(selected_epochs):
        n_epochs = len(baccs[fold])
        close_epochs = np.zeros(n_epochs, dtype=bool)
        close_epochs[epoch - 6:epoch + 7] = True
        close_epochs[epoch] = False
        bacc_around_sel.append(np.asarray(baccs[fold])[close_epochs])
    # adjust number of selected epochs when n_epochs < epoch + 7 in a fold
    n_sel_epochs = np.min([len(b) for b in bacc_around_sel])
    bacc_around_sel = [b[:n_sel_epochs] for b in bacc_around_sel]
    bacc_around_sel = np.asarray(bacc_around_sel)
    avg_bacc = np.mean(bacc_around_sel, axis=0)  # mean over folds

    return avg_bacc


def print_model_performance(models, save=''):
    n_folds = len(models)
    val_hist_folds = [m.val_history for m in models]

    val_folds, best_epochs = evaluate_folds(val_hist_folds, n_folds)
    df = pd.DataFrame(val_folds)
    df.rename({'acc': 'bacc'}, axis=1, inplace=True)
    df_folds = df.copy()
    df_folds.index = [f'Fold {i}' for i in range(1, len(models) + 1)]
    df_folds.insert(0, 'best_epoch', best_epochs)
    print('---- Performance on validation data\n')
    print(df_folds)

    df['abs(emp-sim)'] = np.abs(df['acc_emp'] - df['acc_sim'])
    df_summarize = df.describe().loc[['mean', 'std', 'min', 'max']]
    close_epochs = pd.DataFrame(get_close_baccs([x['acc'] for x in val_hist_folds],
                                best_epochs))
    close_epochs = close_epochs.describe().loc[['mean', 'std', 'min', 'max']]
    df_summarize.insert(2, 'bacc_close_epochs', close_epochs)
    print('\n---- Summary\n')
    print(df_summarize)

    if save != '':
        df_summarize.to_csv(save)

    return df_summarize


def evaluate(model, val_loader):
    """Feeds the model with data and returns its performance

    :param model: ConvNet object
    :param val_loader: networks input_plt_fct data (DataLoader object)
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


def save_checkpoint(model, optimizer, scheduler, fold, save):
    # save state dict
    model.opt_state = optimizer.state_dict()
    if model.scheduler_state is not None:  # because CLR is optional
        model.scheduler_state = scheduler.state_dict()
    # save each fold directly if there is only one bs
    # otherwise only the best performing model is saved
    model.save(f'{save}/model-fold-{fold + 1}.pth')
    model.plot(f'{save}/fig-fold-{fold + 1}.png')
    print('\nSave checkpoint\n')


def fit(lr, model, train_loader, val_loader, opt_func=torch.optim.Adagrad,
        start_epoch=0, max_epochs=1000, min_epochs=100, patience=2,
        step_size=50, min_delta=1e-04, save='', fold=None):
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

            optimizer.zero_grad()
            loss.backward()  # calcul of gradients
            optimizer.step()
            time_per_step.append(time.time() - start_step)

        if scheduler is not None:
            scheduler.step()

        # validation phase
        model = validation(model, train_loader, val_loader)

        save_checkpoint(model, optimizer, scheduler, fold, save)
        if epoch % 10 == 0 and save != '':
            # save checkpoint every 10 epochs
            if (np.min(model.val_history['loss'][:-10]) >
                    np.min(model.val_history['loss'][-10:])):
                save_checkpoint(model, optimizer, scheduler, fold, save)

        if epoch % step_size == 0 and epoch > min_epochs - 1:
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
                save_checkpoint(model, optimizer, scheduler, fold, save)

            prev_val_loss = curr_val_loss
        elif epoch == min_epochs - (step_size + 1):
            prev_val_loss = np.min(model.val_history['loss'][-step_size:])

    return np.mean(time_per_step)

import numpy as np
import matplotlib.pylab as plt
import matplotlib

from utils import extract_accuary_loss

matplotlib.use("Agg")


def plot_folds(train_history_folds, val_history_folds, std=True,
               plot=None, plotname='', path=None):
    """Plots validation and training history over folds

    Left plot with mean and standard diviation of accuracies, right plot with
    mean of loss. Each plot contains both validation and training results.
    x-axis showing training epochs, y-axis showing values [0, 1] for accracies.
    If a plot is specified, given histories will be added to that plot.

    :param train_history_folds: dicts list with training acc. and loss of folds
    :param val_history_folds: dicts list with validation acc. and loss of folds
    :param std: if True standard dev. plotted for acc., not plotted otherwise
    :param plot: fig and axses of a matplotlib plot (None)
    :param plotname: (short) word that describes the data (string)
    :param path: directory where plot will be saved as png (string)
    :return: figure, axes of matplotlib plot
    """

    nb_folds = len(train_history_folds)

    results = extract_accuary_loss(train_history_folds, val_history_folds)
    accs_folds_t, losses_folds_t, accs_folds_v, losses_folds_v = results

    # calculate mean over folds per epoch
    avg_accs_t = np.mean(accs_folds_t, axis=0)
    avg_losses_t = np.mean(losses_folds_t, axis=0)
    avg_accs_v = np.mean(accs_folds_v, axis=0)
    avg_losses_v = np.mean(losses_folds_v, axis=0)

    # calculate standard deviation over folds per epoch
    std_accs_train = np.std(accs_folds_t, axis=0)
    std_accs_val = np.std(accs_folds_v, axis=0)

    del accs_folds_t, losses_folds_t, accs_folds_v, losses_folds_v

    # plot the model evaluation
    fig, axs = (plt.subplots(ncols=2, sharex=True, figsize=(12., 6.))
                if plot is None else plot[:2])

    # 1. subplot: accuracy
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')
    axs[0].set_title(f'Accuracy vs. No. of epochs over {nb_folds} folds')

    line_train, = axs[0].plot(avg_accs_t, '-',
                              label=f'training {plotname}(mean)')
    line_val, = axs[0].plot(avg_accs_v, '-',
                            label=f'validation {plotname}(mean)')
    if std:
        axs[0].fill_between(range(len(avg_accs_t)),
                            avg_accs_t - std_accs_train,
                            avg_accs_t + std_accs_train,
                            color=line_train.get_color(), alpha=0.3,
                            label=f'training {plotname}(standard deviation)')
        axs[0].fill_between(range(len(avg_accs_v)),
                            avg_accs_v - std_accs_val,
                            avg_accs_v + std_accs_val,
                            color=line_val.get_color(),
                            alpha=0.3,
                            label=f'validation {plotname}(standard deviation)')
    axs[0].set_ylim([np.floor(plt.ylim()[0] * 100) / 100, 1.0])

    # 2. subplot: loss
    axs[1].plot(avg_losses_t, '-')
    axs[1].plot(avg_losses_v, '-')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title(f'Loss vs. No. of epochs over {nb_folds} folds')

    plt.subplots_adjust(bottom=0.25)

    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels,
               bbox_to_anchor=[-0.1, -0.3], loc='lower center', ncol=2)

    if path is not None:
        plt.savefig(path)

    return fig, axs


def plot_hist_quantiles(datasets, labels=None, xlabels=None, ylabels=None,
                        path=None):
    """Plotting histograms of given data sets and their 0.25, 0.5 and 0.75
    quantiles

    :param datasets: list of array of list with numeric data
    :param labels: label per data set to indicate its type
                   e.g. number of sequences (list string)
    :param xlabels: x-axis labels for the data sets (list string)
    :param ylabels: y-axis labels for the data sets (list string)
    :param path: directory where plot will be saved as png (string)
    """

    if ylabels is None:
        ylabels = []
    if xlabels is None:
        xlabels = []
    if labels is None:
        labels = []

    if len(datasets) > 1:
        fig, axs = plt.subplots(ncols=2, nrows=np.ceil(len(datasets) / 2),
                                sharex=True, figsize=(12., 6.))
    else:
        fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True,
                                figsize=(12., 6.))

    for i, data in enumerate(datasets):

        label = labels[i] if i < len(labels) else ''
        xlabel = xlabels[i] if i < len(xlabels) else ''
        ylabel = ylabels[i] if i < len(ylabels) else ''

        q1 = np.quantile(data, 0.25)
        q2 = np.quantile(data, 0.5)
        q3 = np.quantile(data, 0.75)

        axs[i].hist(data, bins=200, label=label)
        axs[i].axvline(x=q1, label=f"0.05q = {q1}", c='#6400e4')
        axs[i].axvline(x=q2, label=f"0.5q = {q2}", c='#fd4d3f')
        axs[i].axvline(x=q3, label=f"0.95q = {q3}", c='#4fe0b0')
        axs[i].xlabel(xlabel)
        axs[i].ylabel(ylabel)
        axs[i].legend()

    if path is not None:
        plt.savefig(path)

import os

import numpy as np
from matplotlib import pylab as plt
# import matplotlib

from utils import extract_accuary_loss

# matplotlib.use("Agg")


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

    cmap = matplotlib.cm.get_cmap('viridis')

    if len(datasets) > 1:
        fig, axs = plt.subplots(ncols=2, nrows=int(np.ceil(len(datasets) / 2)),
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

        if len(datasets) > 1:
            axs[i].hist(data, bins=200, label=label, color=cmap(1 / (i + 1)))
            axs[i].axvline(x=q1, label=f"0.25q = {q1}", c='#6400e4')
            axs[i].axvline(x=q2, label=f"0.5q = {q2}", c='#fd4d3f')
            axs[i].axvline(x=q3, label=f"0.75q = {q3}", c='#4fe0b0')
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel(ylabel)
            axs[i].legend()
        else:
            axs.hist(data, bins=200, label=label, color=cmap(1 / (i + 1)))

            axs.axvline(x=q1, label=f"0.25q = {q1}", c='#6400e4')
            axs.axvline(x=q2, label=f"0.5q = {q2}", c='#fd4d3f')
            axs.axvline(x=q3, label=f"0.75q = {q3}", c='#4fe0b0')
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            axs.legend()

    if path is not None:
        plt.savefig(path)


def plot_weights(model_path, config_path):
    model_path = '/home/jtrost/beegfs/mlaa/results/ocaml-lin/cnn-30-Jun-2021-03:38:54.258092'
    # config_path = f'{model_path}/config.json'
    config_path = '/home/jtrost/beegfs/mlaa/configs/config-30-Jun-2021-03:38:54.258092.json'

    config = read_config_file(config_path)

    model_params = config['conv_net_parameters']
    model_params['input_size'] = config['data']['nb_sites']
    model_params['nb_chnls'] = 23
    nb_sites = config['data']['nb_sites']

    files = os.listdir(model_path)
    model_paths = [f'{model_path}/{file}' for file in files
                   if file.endswith('.pth')]

    # collect all the weights
    aas = 'OARNDCQEGHILKMFPSTWYVX-'
    weights_all_folds = {}
    for i, path in enumerate(model_paths):
        # generate model
        model = load_net(path, model_params, state='eval')
        weights = model.state_dict()[
            'lin_layers.0.weight'].data.cpu().numpy()[0]

        for j, aa in enumerate(aas):
            w_norm = ((weights[nb_sites * j:nb_sites * (j + 1)] -
                       np.min(weights)) / (np.max(weights) - np.min(weights)))

            if aa in weights_all_folds.keys():
                weights_all_folds[aa] += [w_norm]
            else:
                weights_all_folds[aa] = [w_norm]

    # calculate mean and std over folds
    avg_w = {}
    std_w = {}
    for key, val in weights_all_folds.items():
        avg_w[key] = np.mean(val, axis=0)
        std_w[key] = np.std(val, axis=0)

    del weights_all_folds

    avg_w_arr = np.asarray([val for lst in avg_w.values() for val in lst])
    std_w_arr = np.asarray([val for lst in std_w.values() for val in lst])

    fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True,
                            figsize=(12., 6.))

    line, = axs.plot(avg_w_arr, '.', markersize=1)
    axs.margins(x=0)
    axs.fill_between(range(len(avg_w_arr)),
                        avg_w_arr - std_w_arr,
                        avg_w_arr + std_w_arr,
                        color=line.get_color(), alpha=0.3)
    ylim = np.max(avg_w_arr + std_w_arr)
    for j, aa in enumerate(aas):
        axs.axvline(x=nb_sites * (j + 1), color='grey', linewidth=0.5)
        axs.annotate(aa, ((nb_sites * (j + 1)) - (nb_sites / 2), ylim))

    if not os.path.exists(f'{model_path}/weights'):
        os.mkdir(f'{model_path}/weights')

    fig.savefig(f'{model_path}/weights/weights-overview.png')

    # separate plots for each aa
    for aa, w in avg_w.items():
        sub_fig, sub_axs = plt.subplots(ncols=1, nrows=1, sharex=True,
                                        figsize=(12., 6.))
        line, = sub_axs.plot(w, '.', markersize=2)
        sub_axs.margins(x=0)
        sub_axs.fill_between(range(len(w)), w - std_w[aa], w + std_w[aa],
                         color=line.get_color(), alpha=0.3)
        sub_axs.set_ylim(np.min(avg_w_arr), np.max(avg_w_arr))
        sub_axs.set_title(aa)
        sub_axs.set_xlabel('Number of sites')
        sub_axs.set_ylabel('Weight (scaled [0, 1])')

        if not os.path.exists(f'{model_path}/weights'):
            os.mkdir(f'{model_path}/weights')
        sub_fig.savefig(f'{model_path}/weights/{aa}.png')

"""
real_fasta_path = '/mnt/Clusterdata/fasta_no_gaps'
sim_fasta_path = '../../data/ocaml_fasta_263hog_w1p'

config_path = '/mnt/Clusterdata/mlaa/configs/config.json'

config = read_config_file(config_path)

alns,_,_ = raw_alns_prepro([real_fasta_path,sim_fasta_path], config['data'])

real_alns, sim_alns = alns

#real_aa = np.array(get_aa_freqs(real_alns, dict=False, gaps=False))
#sim_aa = np.array(get_aa_freqs(sim_alns, dict=False, gaps=False))

aas = list('ARNDCQEGHILKMFPSTWYVX') + ['other']
#i = 0
#for r, s in zip(real_aa, sim_aa):
#    plot_hist_quantiles([r, s], labels = [f'real({aa[i]})', f'sim({aa[i]})'], path=f'freqs_comp/freqs-{aa[i]}.png')
#    i += 1
    
#real_aa_avg = np.mean(real_aa, axis=1)
#sim_aa_avg = np.mean(sim_aa, axis=1)

# get total aa frequencies
real_sim_freqs = []
for aligns in alns:
    freqs = np.zeros(22)
    for aln in aligns:
        for seq in aln:
            for i, aa in enumerate(aas):
                freqs[i] += seq.count(aa)
            freqs[-1] += (seq.count('B') + seq.count('Z') + seq.count('J') +
                          seq.count('U') + seq.count('O'))

    freqs /= np.sum(freqs)
    # limit to 6 digits after the comma
    freqs = np.floor(np.asarray(freqs) * 10 ** 6) / 10 ** 6
    real_sim_freqs.append(freqs)

real_aa, sim_aa = real_sim_freqs

# aa freqs from weighted sum
profiles = np.genfromtxt('profiles_weights/263-hogenom-profiles.tsv', delimiter='\t')
weights = np.genfromtxt('profiles_weights/263-hogenom-weights.csv', delimiter=',')
wp = np.dot(profiles, weights)
wp = np.concatenate((wp, [0., 0.]))

x = np.arange(len(aas))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, real_freqs, width, label='real')
#rects3 = ax.bar(x, freqs, width, label='sim')
rects2 = ax.bar(x, bloom_aa, width, label='bloom\nprofiles*weights')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('aa frequency')
ax.set_xticks(x)
ax.set_xticklabels(aas)
ax.legend()

fig.tight_layout()

plt.savefig('freqs_distr_ocaml_vs_real/bar_bloom_real.png')
"""

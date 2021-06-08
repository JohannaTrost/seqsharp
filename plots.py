import numpy as np
import matplotlib.pylab as plt
import matplotlib
import pandas as pd

matplotlib.use("Agg")

def plot_folds(train_history_folds, val_history_folds, std=True,
               plot=None, plotname='', path=None):

    accs_train_folds = []
    losses_train_folds = []
    accs_val_folds = []
    losses_val_folds = []

    for fold in range(len(train_history_folds)):
        accs_train_fold = []
        losses_train_fold = []
        accs_val_fold = []
        losses_val_fold = []

        for i in range(len(train_history_folds[fold])):
            accs_train_fold.append(train_history_folds[fold][i]['acc'])
            losses_train_fold.append(train_history_folds[fold][i]['loss'])
            accs_val_fold.append(val_history_folds[fold][i]['acc'])
            losses_val_fold.append(val_history_folds[fold][i]['loss'])

        accs_train_folds.append(accs_train_fold)
        losses_train_folds.append(losses_train_fold)
        accs_val_folds.append(accs_val_fold)
        losses_val_folds.append(losses_val_fold)

    del accs_train_fold, losses_train_fold, accs_val_fold, losses_val_fold

    # calculate mean over folds per epoch
    avg_accs_train = np.mean(accs_train_folds, axis=0)
    avg_losses_train = np.mean(losses_train_folds, axis=0)
    avg_accs_val = np.mean(accs_val_folds, axis=0)
    avg_losses_val = np.mean(losses_val_folds, axis=0)

    # calculate standard deviation over folds per epoch
    std_accs_train = np.std(accs_train_folds, axis=0)
    std_accs_val = np.std(accs_val_folds, axis=0)

    del accs_train_folds, losses_train_folds, accs_val_folds, losses_val_folds

    # plot the model evaluation
    fig, axs = (plt.subplots(ncols=2, sharex=True, figsize=(12., 6.))
                if plot is None else plot[:2])

    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')
    axs[0].set_title(f'Accuracy vs. No. of epochs over {fold + 1} folds')

    line_train, = axs[0].plot(avg_accs_train, '-', label=f'training {plotname}(mean)')
    line_val, = axs[0].plot(avg_accs_val, '-', label=f'validation {plotname}(mean)')
    if std:
        axs[0].fill_between(range(len(avg_accs_train)),
                            avg_accs_train - std_accs_train,
                            avg_accs_train + std_accs_train,
                            color=line_train.get_color(), alpha=0.3,
                            label=f'training {plotname}(standard deviation)')
        axs[0].fill_between(range(len(avg_accs_val)),
                            avg_accs_val - std_accs_val,
                            avg_accs_val + std_accs_val,
                            color=line_val.get_color(),
                            alpha=0.3,
                            label=f'validation {plotname}(standard deviation)')
    axs[0].set_ylim([np.floor(plt.ylim()[0] * 100) / 100, 1.0])

    # 2. subplot: loss
    axs[1].plot(avg_losses_train, '-')
    axs[1].plot(avg_losses_val, '-')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title(f'Loss vs. No. of epochs over {fold + 1} folds')

    plt.subplots_adjust(bottom=0.25)

    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels,
               bbox_to_anchor=[-0.1, -0.3], loc='lower center', ncol=2)

    if path is not None:
        plt.savefig(path)

    return fig, axs


def plot_eval_per_aln(fold, aln_stats_df, train_eval_df):
    df = pd.merge(aln_stats_df,
                  train_eval_df[train_eval_df['fold'] == fold],
                  on=['id'])
    plt.scatter(df[df['simulated'] == 0]['mean_mse_sep'],
                df[df['simulated'] == 0]['accuracy'])
    plt.show()


def plot_acc_per_aln(df, save_path=None):
    df = df.sort_values(by=['accuracy'], ascending=False)

    ticks = df.id
    # ticks = df.index.map(str)

    reference = np.mean(df['accuracy'][df['is_val'] == 0])

    fig, ax1 = plt.subplots(figsize=(12., 6.))

    ax1.plot(ticks, df['accuracy'], '-', color='grey')
    ax1.grid(axis='x')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('alignment')
    ax1.tick_params(axis='x', labelsize=7)
    ax1.margins(x=0.005)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(-70)
    for x in range(len(df['accuracy'])):
        if df['is_val'][x] == 1:
            an_val = ax1.annotate('o', (ticks[x], df['accuracy'][x]),
                                  horizontalalignment='center',
                                  verticalalignment='center')
        if df['padding'][x] > 0:
            an_pad = ax1.annotate('x', (ticks[x], df['accuracy'][x]),
                                  horizontalalignment='center',
                                  verticalalignment='top')
        if df['simulated'][x] == 1:
            ax1.plot(ticks[x], df['accuracy'][x], 'm.',
                     label='simulated alignment')
        else:
            ax1.plot(ticks[x], df['accuracy'][x], 'b.',
                     label='empirical alignment')

    ax1.hlines(reference, *ax1.get_xlim(), color='grey', linestyles='dashed',
               label='mean training accuracy')

    ax2 = ax1.twinx()
    color = 'tab:olive'
    ax2.set_ylabel('mean squared error', color=color)
    ax2.plot(ticks, df['mean_dists'], color=color,
             label='mean mse to training data')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xticklabels(df.index.map(str))

    fig.tight_layout()

    # avoid repetition of labels in legend
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels_ax1 + labels_ax2, handles_ax1 + handles_ax2))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left', ncol=4)
    # legend for annotations
    ax1.text(len(df.id) * 0.75, 0, f'{an_pad.get_text()} - padded\n'
                                   f'{an_val.get_text()} - validation data',
             color='black',
             bbox=dict(boxstyle="round", alpha=0.25, facecolor="white",
                       edgecolor="grey"))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    print('train  > avg {} | val > avg {}'.format(
        np.mean(df['accuracy'][df['is_val'] == False] > reference),
        np.mean(df['accuracy'][df['is_val']] > reference)
    ))


def plot_dists(alns_dists, fastas_real):
    fig, ax = plt.subplots(1, 1, figsize=(12., 6.))

    inds = np.argsort(alns_dists)[::-1]

    ticks = list(map(str, range(len(fastas_real))))

    ax.plot(np.asarray(ticks)[inds], np.asarray(alns_dists)[inds])
    ax.set_xlabel('alignment')
    ax.set_ylabel('mean distance')
    for tick in ax.get_xticklabels():
        tick.set_rotation(70)

    # ax[0].scatter(combs.flatten(), alns_dists.flatten())
    # for tick in ax[0].get_xticklabels():
    #    tick.set_rotation(45)

    # cax = ax[0].matshow(alns_dists)
    # fig.colorbar(cax)

    plt.show()
    plt.savefig('results/dists_aln_p.png')


def count_low_acc_alns_per_fold(datasets):
    bad_preds = {}
    for dataset in datasets:
        for fold in range(len(dataset)):
            for i in range(len(dataset[fold]['acc'])):
                if dataset[fold]['acc'][i] < 0.6:
                    if dataset[fold]['aln'][i][1] in bad_preds.keys():
                        bad_preds[dataset[fold]['aln'][i][1]] += 1
                    else:
                        bad_preds[dataset[fold]['aln'][i][1]] = 1
    return bad_preds


def plot_nb_low_acc_alns(bad_preds, df, save_path=None):
    s_bad_preds = {k: v for k, v in
                   sorted(bad_preds.items(), key=lambda item: item[1],
                          reverse=True)}

    fig, ax = plt.subplots(1, 1, figsize=(12., 6.))
    ax.barh(list(s_bad_preds.keys()), s_bad_preds.values())
    ax.set_title('Number of folds for alignments with accuracy <0.6')
    ax.set_xlabel('number of folds')
    ax.set_ylabel('alignment')
    ax.set_xticks(range(1, 11))

    for i, (k, v) in enumerate(s_bad_preds.items()):
        if v > 1 and k in list(df['name']):
            bar_str = 'sl : {}(avg {}), ns : {}(avg {})'. \
                format(list(df[df['name'] == k].seq_length)[0],
                       np.median(df['seq_length']),
                       list(df[df['name'] == k].number_seqs)[0],
                       np.median(df['number_seqs']))
            ax.text(v, i - 0.3, bar_str, color='grey')
    ax.margins(y=0.004)
    plt.box(False)
    plt.figtext(0.8, 0.6, 'sl - sequence length\nns - number of sequences',
                color='grey', bbox=dict(facecolor='none', edgecolor='grey',
                                        boxstyle='round,pad=1'))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)


def plot_hist_quantiles(datasets, labels=[], xlabels=[], ylabels=[], path=None):

    if len(datasets) > 1:
        fig, axs = plt.subplots(ncols=2, nrows=np.ceil(len(datasets) / 2),
                                sharex=True, figsize=(12., 6.))
    else:
        fig, axs = plt.subplots(ncols=1, nrows=1, sharex = True,
                                figsize = (12., 6.))


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


"""
# data specific parameters
real_fasta_path = 'data/fasta_no_gaps'
sim_fasta_path = 'data/sim_fasta_seqs'
model_path = 'results/cnn-29-Apr-2021-20:38:57.134481-real-sim'

nb_protein_families = 63  # number of multiple aligns
min_seqs_per_align, max_seqs_per_align = 4, 300
seq_len = 300

nb_folds = 10

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ------------------------------------------------- data preparation ------------------------------------------------- #
real_alns, sim_alns, fastas_real, fastas_sim = data_prepro(real_fasta_path,
                                                           sim_fasta_path,
                                                           nb_protein_families,
                                                           min_seqs_per_align,
                                                           max_seqs_per_align,
                                                           seq_len,
                                                           csv_path=
                                                           f'{model_path}/'
                                                           f'alns_stats.csv')
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

real_pairs_per_align, sim_pairs_per_align, fastas_real_p, fastas_sim_p = \
    data_prepro(real_fasta_path, sim_fasta_path, nb_protein_families,
                min_seqs_per_align, max_seqs_per_align, seq_len, pairs=True)

train_real_folds, train_sim_folds, val_real_folds, val_sim_folds = kfold_eval_per_aln(nb_folds, real_pairs_per_align,
                                                                                      sim_pairs_per_align, fastas_real_p,
                                                                                      fastas_sim_p, seq_len, model_path)

bad_preds = count_low_acc_alns_per_fold([train_real_folds, train_sim_folds, val_real_folds, val_sim_folds])

dists_p = {}
for i in [0, 1, 2]:
    with open(f'data/{i}-aln-dists-p.txt', 'r') as file:
        for line in file:
            s_line = line.split(', ')
            aln_id = s_line[-1].split(',')[1][:-1]
            dists_p[aln_id] = s_line[:-1] + [s_line[-1].split(',')[0]]

aln_ids = dists_p.keys()
alns_dists_p = np.mean(np.asarray(list(dists_p.values())).astype(float), axis=0)

real_alns_dict = {fastas_real[i]: real_alns[i] for i in range(len(real_alns))}
sim_alns_dict = {fastas_sim[i]: sim_alns[i] for i in range(len(sim_alns))}
dfs = []
for i, (train_real, train_sim, val_real, val_sim) in enumerate(zip(train_real_folds,
                                                                   train_sim_folds,
                                                                   val_real_folds,
                                                                   val_sim_folds)):
    df = generate_eval_dict(i, train_real, train_sim, val_real, val_sim,
                            real_alns_dict, sim_alns_dict)
    dfs.append(df)
    # plot_acc_per_aln(df, f'{model_path}/acc_per_aln_fold_{i}.png')

df_c = pd.concat(dfs)
csv_string = df_c.to_csv(index=False)
with open(model_path + '/aln_train_eval.csv', 'w') as file:
    file.write(csv_string)

plot_nb_low_acc_alns(bad_preds, df)
plot_dists(alns_dists_p, aln_ids)

"""

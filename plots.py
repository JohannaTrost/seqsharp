import os

import numpy as np
import seaborn as sns
# import matplotlib as mpl
# mpl.use('TkAgg')
import pandas as pd

from matplotlib import pylab as plt
from sklearn.decomposition import PCA

from ConvNet import load_model
from train_eval import evaluate_folds
from utils import confidence_ellipse, pred_runtime, \
    get_divisor_min_diff_quotient


# matplotlib.use("Agg")


def plot_groups_folds(bs_lst, time_per_step, save=''):
    n_bs = time_per_step.shape[0]
    n_folds = time_per_step.shape[1]
    fig, ax = plt.subplots(figsize=(16, 9))
    for i in range(n_bs):
        ax.scatter(np.repeat(i, n_folds), time_per_step[i], alpha=0.4,
                   marker='.')
    ax.plot(np.arange(n_bs), np.mean(time_per_step, axis=1),
            label='Mean across folds', color='grey', linewidth=1, marker='o')
    ax.set_xticks(np.arange(n_bs))
    ax.set_xticklabels([str(bs) for bs in bs_lst])
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Time per step')
    plt.tight_layout()
    if save != '' and save is not None:
        plt.savefig(save)
    plt.close('all')


def plot_corr_pred_sl(sl, scores, save=''):
    fig, ax = plt.subplots(ncols=len(sl.keys()), nrows=3, figsize=(16, 9))
    # scatter
    for i, key in enumerate(scores.keys()):
        ax[0, i].scatter(sl[key], scores[key])
        ax[0, i].set_ylabel('Prediction score (0 - emp, 1 - sim)')
        ax[0, i].set_xlabel('Number of sites')
        ax[0, i].set_title(key)
        sl_sort = np.argsort(sl[key])
        ax[1, i].plot(scores[key][sl_sort], label='scores (by sl)')
        ax[2, i].plot(sl[key], label='sl (by scores)')
        ax[1, i].legend()
        ax[2, i].legend()
    plt.tight_layout()
    if save != '':
        plt.savefig(save)


def plot_compare_msa_stats(stats, fastas, group_labels, sample_size=42,
        save=None, figsize=None):
    sample_inds = [np.random.randint(len(fastas[0]), size=sample_size)]
    s_fastas = np.asarray(fastas[0])[sample_inds[0]]
    sample_inds += [[np.where(fs == rf.replace('.fasta', '.fa'))[0][0]
                     for rf in s_fastas]
                    for fs in fastas[1:]]
    sample_inds = np.asarray(sample_inds)

    plots_shape = (get_divisor_min_diff_quotient(sample_size),
                   sample_size // get_divisor_min_diff_quotient(sample_size))
    with sns.axes_style("whitegrid"):
        if figsize is not None:
            fig, ax = plt.subplots(*plots_shape, figsize=figsize)
        else:
            fig, ax = plt.subplots(*plots_shape)
        for i in range(sample_size):
            row, col = np.unravel_index(i, plots_shape)
            for j, group in enumerate(group_labels):
                sns.kdeplot(stats[j][sample_inds[j][i]], fill=True, alpha=0.5,
                            linewidth=0, ax=ax[row, col], label=group)
            if i == 0:
                ax[row, col].legend()
        plt.tight_layout()
        if save is None:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.savefig(save)
            plt.close('all')


def plot_model_emp_sim_accs(model_paths, model_names, ax=None, cols=None):
    models = [load_model(mp) for mp in model_paths]
    val_res = [evaluate_folds([mf.val_history for mf in m], len(m))[0]
               for m in models]

    n_folds = len(val_res[0]['loss'])
    n_models = len(val_res)

    m_acc = np.asarray([mres['acc'] for mres in val_res])
    m_acc_emp = np.asarray([mres['acc_emp'] for mres in val_res])
    m_acc_sim = np.asarray([mres['acc_sim'] for mres in val_res])

    if ax is None:
        ax = plt.gca()

    # plot folds acc. (emp. and sim.)
    x = np.arange(1, n_models + 1).repeat(n_folds)
    ax.scatter(x - 0.1, m_acc_emp.flatten(), color='coral', alpha=0.5,
               label='emp', marker='.')
    ax.scatter(x + 0.1, m_acc_sim.flatten(), color='c', alpha=0.5, label='sim',
               marker='.')
    # mean per class
    x = np.arange(1, n_models + 1)
    ax.scatter(x - 0.1, m_acc_emp.mean(axis=1), color='grey', alpha=0.5,
               marker='D')
    ax.scatter(x + 0.1, m_acc_sim.mean(axis=1), color='grey', alpha=0.5,
               label='class mean', marker='D')

    # plot mean of BACC over folds
    ax.plot(x, m_acc.mean(axis=1), 'o', color='grey',
            label='BACC (folds mean)')

    # ylims = (np.true_divide(np.floor(ax.get_ylim()[0] * 10**2), 10**2), 1)
    ylims = ax.get_ylim()
    ax.set_ylim((0.9 - 0.005 * (ylims[1] - ylims[0]),
                 1 + 0.04 * (ylims[1] - ylims[0])))
    ax.set_yticks(np.arange(ylims[0], ylims[1] + 0.05, 0.05))
    ax.set_xticks(np.arange(1, n_models + 1))
    ax.set_xticklabels(model_names, rotation=0)
    ax.set(frame_on=False)
    ax.set_xticks(np.arange(0.5, n_models + 0.5, 0.5), minor=True)
    ax.set_yticks(np.arange(ylims[0], ylims[1] + 0.025, 0.025))
    ax.set_yticks(np.arange(ylims[0] + 0.0125, ylims[1], 0.0125), minor=True)
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5,
            alpha=0.4)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5,
            alpha=0.2)
    ax.legend()

    return ax


def plot_model_folds_accs(model_paths, model_names, ax=None, cols=None):
    models = [load_model(mp) for mp in model_paths]
    val_res = [evaluate_folds([mf.val_history for mf in m], len(m))[0]
               for m in models]
    m_acc = np.asarray([mres['acc'] for mres in val_res])

    n_folds = m_acc.shape[1]
    n_models = m_acc.shape[0]

    if ax is None:
        ax = plt.gca()

    # plot all folds acc. (BACC)
    x = np.arange(1, n_models + 1).repeat(n_folds)
    if cols is None:
        ax.scatter(x, m_acc.flatten(), color='grey', alpha=0.4,
                   label='BACC per fold', marker='.')
    else:  # use given colors for different models
        labels = [''] * (n_models - 1) + ['BACC per fold']
        for i in range(n_models):
            ax.scatter(x[x == i + 1], m_acc[i], color=cols[i], alpha=0.4,
                       label=labels[i], marker='.')
    ax.plot(np.arange(1, n_models + 1), m_acc.mean(axis=1), color='grey',
            label='BACC (folds mean)', linewidth=1, marker='o')

    # ylims = (np.true_divide(np.floor(ax.get_ylim()[0] * 10**2), 10**2), 1)
    ylims = ax.get_ylim()
    ax.set_ylim((0.9 - 0.005 * (ylims[1] - ylims[0]),
                 1 + 0.04 * (ylims[1] - ylims[0])))
    ax.set_yticks(np.arange(ylims[0], ylims[1] + 0.05, 0.05))
    ax.set_xticks(np.arange(1, n_models + 1))
    ax.set_xticklabels(model_names, rotation=0)
    ax.set(frame_on=False)
    ax.set_xticks(np.arange(0.5, n_models + 0.5, 0.5), minor=True)
    ax.set_yticks(np.arange(ylims[0], ylims[1] + 0.025, 0.025))
    ax.set_yticks(np.arange(ylims[0] + 0.0125, ylims[1], 0.0125), minor=True)
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5,
            alpha=0.4)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5,
            alpha=0.2)
    # ax.legend()

    return ax


def plot_comp_model_accs(accs, model_names, save=''):
    """TODO

    :param accs: n_models x n_folds array
    :return:
    """
    x = np.arange(accs.shape[0]).repeat(accs.shape[1])
    fig, ax = plt.subplots()
    ax.scatter(x, accs.flatten(), alpha=0.5)
    ax.set_xticks(np.arange(accs.shape[0]))
    ax.set_xticklabels(model_names, rotation=45)
    ax.plot(accs.mean(axis=1))
    ax.set_ylabel('accuracy')
    plt.tight_layout()
    if save != '':
        plt.savefig(save, format='svg')
    plt.close('all')


def plot_aa_dens(real, sim, save, aas='ARNDCQEGHILKMFPSTWYV'):
    n_rows, n_cols = 4, 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))
    for i in range(n_rows * n_cols):
        r, c = np.unravel_index(i, (n_rows, n_cols))
        print(f'{i}: {r},{c}')
        sns.kdeplot(sim[i], ax=axs[r, c],
                    color='c', linewidth=0, fill=True, alpha=0.5)
        sns.kdeplot(real[i], ax=axs[r, c],
                    color='coral', linewidth=0, fill=True, alpha=0.5)
        axs[r, c].set_aspect('auto')
        axs[r, c].set_ylabel('')
        axs[r, c].text(0.9, 0.9, aas[i], ha='center', va='center', size=15,
                       color='Grey', transform=axs[r, c].transAxes)
        # axs[r, c].axis('off')

    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    plt.savefig(save)
    plt.pause(1)
    plt.close('all')


def plot_em_learning_curves(n_runs, n_iter, vlbs_dips, vlbs, debug, test,
        lk_type, optimal_lk=None, save_path=''):
    # highlight dips with red edge color of marker
    edgecol = np.repeat('b', n_runs * n_iter * 2).reshape((n_runs, n_iter * 2))
    edgecol[:, ::2] = 'g'  # e-step
    edgecol[vlbs_dips > 0] = 'r'  # loglk dip

    # move annotation to avoid overlaps
    show_annot = (vlbs_dips > 1e-05)  # annotate all dips > 1e-05
    move_annot_inds = np.where(show_annot)
    move_by = np.zeros_like(vlbs)
    move_by[show_annot] = 0.1  # adding 5% of (ylim max- ylim min) to position
    # don't move every other annotation to avoid overlap
    move_by[move_annot_inds[0][1::2], move_annot_inds[1][1::2]] = 0.01

    n_rows, n_cols = 2, max(int(np.ceil(n_runs / 2)), 1)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                            sharey=True, figsize=(16, 9))
    for col in range(n_cols):
        for row in range(n_rows):
            run = (row * n_cols) + col
            curr_axis = axs[row, col] if n_cols > 1 else axs[row]
            if run < n_runs:
                if debug:  # plot e-step vlb in green
                    curr_axis.plot(np.arange(0, n_iter, 0.5), vlbs[run])
                    curr_axis.scatter(np.arange(0, n_iter),
                                      vlbs[run, ::2], c='g',
                                      edgecolors=edgecol[run, ::2],
                                      label='E-Step')
                else:
                    curr_axis.plot(np.arange(0.5, n_iter),
                                   vlbs[run, 1::2])

                # plot M-step vlb in blue
                curr_axis.scatter(np.arange(0.5, n_iter),
                                  vlbs[run, 1::2], c='blue',
                                  edgecolors=edgecol[run, 1::2],
                                  label='M-Step')
                if optimal_lk is not None:
                    # show optimal loglk. on true parameters
                    curr_axis.hlines(y=optimal_lk, color='red',
                                     xmin=0,
                                     xmax=n_iter - 0.5)  # lk with given

                # -------- annotate dips
                ylim = curr_axis.get_ylim()
                yax_size = np.abs(ylim[0] - ylim[1])
                for x, y, dip, move in zip(np.arange(0, n_iter, 0.5), vlbs[run],
                                           vlbs_dips[run], move_by[run]):
                    if move > 0:  # dip > 1e-05
                        annot_pos = (x, y + yax_size * move)
                        annot = np.round(dip, 5) if dip < 1 else int(dip)
                        curr_axis.annotate(annot, (x, y),
                                           xytext=annot_pos,
                                           arrowprops=dict(arrowstyle="->"))
                if test:
                    # titles indicating particular inits
                    if run == 0:
                        curr_axis.set_title(
                            'EM initialized with uniform params.')
                    elif run == 1:
                        curr_axis.set_title(
                            'EM initialized with true params.')
                    else:
                        curr_axis.set_title(f'Run {run + 1}')
                else:
                    curr_axis.set_title(f'Run {run + 1}')

                curr_axis.set_xlabel('Iterations')
                curr_axis.set_ylabel(lk_type)
                curr_axis.set_xticks(np.arange(0, n_iter))

                if debug:  # show legend to distinguish m- and e-step
                    curr_axis.legend()

    fig.tight_layout()
    if save_path != '':
        fig.savefig(f'{save_path}/lk/em_optimizaton_history_{lk_type}.png')
    plt.close(fig)


def make_fig(plt_fct, input_plt_fct, subplt_shape, figsize=(16, 9), save=''):
    fig, axs = plt.subplots(*subplt_shape, figsize=figsize)
    plt_fct(*input_plt_fct, axs=axs)
    if save != '' and save is not None:
        plt.savefig(save)
    return fig, axs


def plot_folds(models, axs):
    """Plots validation and training history over folds

    TODO
    """
    folds_val = [m.val_history for m in models]
    folds_train = [m.train_history for m in models]
    n_folds = len(folds_train)
    train_col, val_col = '#1F77B4', '#FF7F0E'

    for i in range(len(axs)):
        if axs[i] is None:
            axs[i] = plt.gca()

    # subplots: BACC and Loss
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('BACC')
    axs[0].set_title(f'BACC vs. No. of epochs (for {n_folds} folds)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'Loss vs. No. of epochs (for {n_folds} folds)')
    for fold in range(n_folds):
        axs[0].plot(folds_val[fold]['acc'], '-', color=val_col,
                    label=f'validation' if fold == 0 else '', alpha=0.3)
        axs[0].plot(folds_train[fold]['acc'], '-', color=train_col,
                    label=f'training' if fold == 0 else '', alpha=0.3)
        axs[1].plot(folds_val[fold]['loss'], '-', color=val_col,
                    label=f'validation' if fold == 0 else '', alpha=0.3)
        axs[1].plot(folds_train[fold]['loss'], '-', color=train_col,
                    label=f'training' if fold == 0 else '', alpha=0.3)
    plt.subplots_adjust(bottom=0.25)
    handles, labels = axs[0].get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=[-0.1, -0.3],
               loc='lower center', ncol=2)

    return axs[0], axs[1]


def freq_real_sim_violins(real_msas, sim_msas, aas=None, save_path='',
        scale='area'):
    """
    Violin plots per AA frequencies over real and simulated data as well
    as their 20 PCs from PCA

    :param real_msas: ndarray of frequencies N x N_AA
    :param sim_msas: ndarray of frequencies N x N_AA
    :param aas: str ndarray or list of amino acid names
    :param save_path: path to directory to save plot
    :return: -
    """
    if aas is None:
        aas = np.asarray(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
                          'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

    n_alns_real = real_msas.shape[0]
    n_alns_sim = sim_msas.shape[0]
    n_aas = len(aas)

    pca = PCA(n_components=20)
    pca_real = pca.fit_transform(real_msas[:, :20])
    pca_sim = pca.transform(sim_msas[:, :20])

    plot_titles = ['AA frequencies', 'PCs']
    x_axis_ticks = [aas, np.arange(1, n_aas + 1).astype(str)]
    x_axis_labels = ['AA', 'PCs']
    y_axis_labels = ['frequency', '']

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(2 * 11.7, 10))

    for i, ziped in enumerate(zip([real_msas, pca_real], [sim_msas, pca_sim])):
        real, sim = list(ziped)
        real_sim_freqs = np.concatenate((real[:, :20], sim[:, :20]))
        flat_freqs = np.reshape(real_sim_freqs[:, :n_aas], n_aas *
                                (n_alns_real + n_alns_sim))

        aas_rep = np.repeat(x_axis_ticks[i][np.newaxis],
                            (n_alns_real + n_alns_sim),
                            axis=0).reshape(n_aas * (n_alns_real + n_alns_sim))
        msa_id = np.arange(1, (n_alns_real + n_alns_sim) + 1).repeat(n_aas)
        is_sim = np.repeat([0, 1], [n_alns_real * n_aas, n_alns_sim * n_aas])

        df = pd.DataFrame({'msa_id': msa_id, 'is_sim': is_sim, 'AA': aas_rep,
                           'freq': flat_freqs})
        df['id'] = df.index

        sns.set_theme(style="whitegrid")

        sns.violinplot(x="AA", y="freq", hue="is_sim", data=df, palette="Set2",
                       ax=ax[i], scale=scale)
        ax[i].set_xlabel(x_axis_labels[i])
        ax[i].set_ylabel(y_axis_labels[i])

        ax[i].set_title(plot_titles[i])

    # ax.set_ylim([0, 0.25])
    # df.loc[df['AA'].isin(aas[:4])]
    plt.tight_layout()
    plt.savefig(f'{save_path}', dpi=75)
    plt.close('all')


def freq_compare_n_cl_violins(real_msas, sim_msas_sets, aas=None, save_path=''):
    """
    Violin plots per AA frequencies over real and simulated data as well
    as their 20 PCs from PCA

    :param real_msas: ndarray of frequencies N x N_AA
    :param sim_msas: ndarray of frequencies N x N_AA
    :param aas: str ndarray or list of amino acid names
    :param save_path: path to directory to save plot
    :return: -
    """
    if aas is None:
        aas = np.asarray(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
                          'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])

    n_alns_real = real_msas.shape[0]
    n_alns_sim = sim_msas_sets.shape[1]
    n_aas = len(aas)

    # x_axis_ticks = [aas, np.arange(1, n_aas + 1).astype(str)]
    if n_aas == 20:  # TODO
        split = 4
    else:
        split = 1
    aas_split = np.split(np.arange(n_aas), split)

    fig, ax = plt.subplots(split, 1, figsize=(16., 9.))
    for i, aa_inds in enumerate(aas_split):
        real_sim_freqs = np.concatenate(sim_msas_sets[:, :, aa_inds], axis=0)
        real_sim_freqs = np.concatenate((real_sim_freqs, real_msas[:, aa_inds]))

        n_cls = np.repeat(np.asarray(['sim', 'th'], dtype=str),
                          np.repeat(n_alns_sim, sim_msas_sets.shape[0]))
        n_cls = np.concatenate((n_cls, np.repeat('real', n_alns_real)))
        n_cls = np.repeat(n_cls[np.newaxis], len(aa_inds), axis=0).flatten()

        flat_freqs = np.reshape(real_sim_freqs[:, :len(aa_inds)], len(aa_inds) *
                                real_sim_freqs.shape[0])

        aas_rep = np.repeat(aas[aa_inds][np.newaxis], real_sim_freqs.shape[0],
                            axis=0).flatten()
        msa_id = np.arange(1, real_sim_freqs.shape[0] + 1).repeat(len(aa_inds))

        df = pd.DataFrame({'msa_id': msa_id, '': n_cls, 'AA': aas_rep,
                           'freq': flat_freqs})
        df['id'] = df.index

        sns.set_theme(style="whitegrid")
        if split > 1:
            sns.violinplot(x="AA", y="freq", hue='', data=df,
                           palette="Set2", ax=ax[i], scale='area')
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

    if split == 1:
        sns.violinplot(x="AA", y="freq", hue='', data=df, palette="Set2",
                       ax=ax, scale='area')
        ax.set_xlabel('AA')
        ax.set_ylabel('')

    # ax.set_ylim([0, 0.25])
    # df.loc[df['AA'].isin(aas[:4])]
    plt.tight_layout()
    plt.savefig(f'{save_path}', dpi=100)
    plt.close('all')


def pca_plot():  # TODO
    em_name = 'sim_5cls_6000alns_1000_22606'
    n_cl = 5
    n_pro = 64
    sim_files = [file for file in os.listdir('data/freq_samples')
                 if em_name in file]
    real_msas = np.genfromtxt('data/freq_samples/fasta_no_gaps_alns.csv',
                              skip_header=True, delimiter=',')
    sim_msas = np.asarray(
        [np.genfromtxt(f'data/freq_samples/{file}', delimiter=',',
                       skip_header=True) for file in sim_files])
    init_msa_freqs = np.genfromtxt(
        'results/profiles_weights/sim_5cls_6000alns_1000_22606/init_weights/msa_freqs.csv',
        delimiter=',', skip_header=True)
    init_weight = np.genfromtxt(
        'results/profiles_weights/sim_5cls_6000alns_1000_22606/init_weights/init_weights.csv',
        delimiter=',', skip_header=True)
    res_w = np.asarray([np.genfromtxt(
        f'results/profiles_weights/sim_5cls_6000alns_1000_22606/cl{i + 1}_pro_weights_1.csv')
        for i in range(n_cl)])

    init_w_freq = np.asarray([init_weight[i, :n_pro] @ profiles.T
                              for i in range(n_cl)])
    res_w_freq = np.asarray([res_w[i] @ profiles.T for i in range(n_cl)])

    run = 2  # is first run

    save_path = 'results/pca/test'

    # exclude outliers
    mean_msa = real_msas[:, :20].mean(axis=0)
    dists = np.sum((real_msas[:, :20] - mean_msa) ** 2, axis=1) ** 0.5
    real_msas = real_msas[dists < np.quantile(dists, 0.99)]

    pca = PCA(n_components=20)
    pca_msa_freqs = pca.fit_transform(real_msas[:, :20])
    # predict
    pca_sim = pca.transform(sim_msas[run, :, :20])
    pca_init_msa = pca.transform(init_msa_freqs[:, :20])
    pca_init_w = pca.transform(init_w_freq)
    pca_res_w = pca.transform(res_w_freq)

    sample_real = np.random.randint(len(pca_msa_freqs),
                                    size=int(len(pca_msa_freqs) * 0.1))
    sample_sim = np.arange(0, len(pca_sim), 10).astype(int)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(pca_msa_freqs[sample_real, 0], pca_msa_freqs[sample_real, 1],
               color='coral', s=1, alpha=0.2)
    sc = ax.scatter(pca_sim[sample_sim, 0], pca_sim[sample_sim, 1],
                    c=sim_msas[run, sample_sim, 20], s=1, alpha=0.2)

    confidence_ellipse(pca_msa_freqs[:, 0], pca_msa_freqs[:, 1], ax, n_std=2,
                       edgecolor='coral')

    for cl in sim_msas[run, sample_sim, 20]:
        col = list(sc.to_rgba(cl))
        col[3] = 0.5
        confidence_ellipse(pca_sim[np.where(sim_msas[run, :, 20] == cl), 0],
                           pca_sim[np.where(sim_msas[run, :, 20] == cl), 1], ax,
                           n_std=2, edgecolor=tuple(col))

    ax.scatter(pca_init_msa[:, 0], pca_init_msa[:, 1], c=init_msa_freqs[:, 21],
               s=7, label='initial MSA')
    ax.scatter(pca_init_w[:, 0], pca_init_w[:, 1], c=init_msa_freqs[:, 21],
               s=7, label='initial weights@profiles', marker='^')
    ax.scatter(pca_res_w[:, 0], pca_res_w[:, 1], c=init_msa_freqs[:, 21],
               s=7, label='final weights@profiles', marker='s')

    plt.legend()

    plt.savefig(f'{save_path}/pca.png', dpi=200)
    plt.close('all')


def plot_pred_runtime(n_alns=None, n_cl=None, save=None):
    if n_cl is None:
        n_cl = [1, 2, 4, 6, 8]
    if n_alns is None:
        n_alns = [20, 40, 60, 80, 100]

    X, Y = np.meshgrid(n_alns, n_cl)
    Z = pred_runtime(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('number of MSAs')
    ax.set_ylabel('number of clusters')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, label='runtime in s')
    for ii in [180, 255, 270, 315]:
        ax.view_init(elev=10., azim=ii)
        if save is None:
            plt.savefig(f'../results/runtime_cls_alns{ii}.png')
        else:
            plt.savefig(f'{save}{ii}.png')
    plt.close('all')


def get_ylim(data, factor=1.1):
    """Get y-axis limits of plotted data

    :param data: array
    :return: tuple with low and high y-axis limit
    """

    d_min = np.min(data)
    d_max = np.max(data)
    d_diff = d_max - d_min
    d_avg = d_min / 2 + d_max / 2

    return [d_avg - d_diff / 2 * factor, d_avg + d_diff / 2 * factor]


"""
# load data
from preprocessing import raw_alns_prepro, aa_freq_samples
fasta_paths = ['../data/hogenom_fasta', '../data/alisim_poisson_gapless_trees',
               '../data/alisim_lg_gc_gapless']
#alns, fastas, _  = raw_alns_prepro(fasta_paths)
freqs = aa_freq_samples('../data', [d.split('../data/')[1] for d in fasta_paths],
                        sample_prop=1.0, n_alns=None, levels=['sites', 'msa'])

# get theoretical msa dots - weights@profiles
#w_files = [f for f in os.listdir("../results/profiles_weights/sim_edcl64_1cl_1aln")
#           if 'best' in f and 'pro' in f]
#weights = [np.genfromtxt(f'../results/profiles_weights/sim_edcl64_1cl_1aln/{f}', delimiter=',')
#           for f in w_files]
#weights = np.asarray(weights)
#profiles = np.genfromtxt('../results/profiles_weights/profiles/64-edcluster-profiles.tsv',
#                         delimiter='\t')
#th_msa = np.matmul(profiles, weights.T).T

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pylab as plt

pca = make_pipeline(StandardScaler(), PCA(n_components=2))
pca_real_site_freqs = pca.fit_transform(freqs['hogenom_fasta_sites'][:20].T)
pca_sim_freqs = [pca.transform(freqs[d.split('../data/')[1] + '_sites'][:20].T)
                 for d in fasta_paths[1:]]
#pca_msa_th_freqs = pca.transform(th_msa)

#tsne = TSNE()
#tsne_msa_freqs = tsne.fit_transform(msa_freqs)
#tsne_msa_sim_freqs = tsne.fit_transform(msa_sim_freqs)
#tsne_msa_th_freqs = tsne.fit_transform(th_msa)

s = 25

def make_heatmap(xx, yy, s=25):

    plot_this = np.zeros((s, s))

    o_min = np.min([xx.min(), yy.min()])
    xx -= o_min
    yy -= o_min

    o_max = np.max([xx.max(), yy.max()])
    xx /= o_max
    xx *= s-1

    yy /= o_max
    yy *= s-1

    for x_pp, y_pp in zip(xx, yy):
        plot_this[int(y_pp), int(x_pp)] += 1

    plot_this[plot_this == 0] = np.NaN

    return plot_this

def plot_pca(data, ps=3, alpha=0.6, clim=60, save=None, titles=None):

    xlim = (np.min([d[:, 0].min() for d in data]),
            np.max([d[:, 0].max() for d in data]))
    ylim = (np.min([d[:, 1].min() for d in data]),
            np.max([d[:, 1].max() for d in data]))

    fig, axs = plt.subplots(ncols=len(data), nrows=2, figsize=(12., 4.5))
    for i in range(len(data)):
        #axs[i, 1].scatter(data[i][:, 0], data[i][:, 1], s=ps, color='coral',
        #                  alpha=alpha)
        #axs[i, 1].set_xlim(xlim)
        #axs[i, 1].set_ylim(ylim)

        x, y = data[i][:, 0].copy(), data[i][:, 1].copy()
        plot_this = make_heatmap(x, y, s)
        hm = axs[i].imshow(plot_this, clim=[0, clim],
                              cmap='turbo', interpolation='bilinear')
        axs[i].invert_yaxis()
        if titles is not None:
            axs[i].set_title(titles[i])
        axs[i].axis('off')

        #confidence_ellipse(data[0][:, 0], data[0][:, 1], axs[i, 1], n_std=2,
        #               edgecolor='coral')
    #fig.delaxes(axs[2][1])
    fig.colorbar(hm, ax=axs[i])
    plt.tight_layout()
    plt.savefig(save)
    plt.close('all')

plot_names = ['Empirical sites',
              'Poisson', 'LG+GC']
plot_pca([pca_real_site_freqs] + pca_sim_freqs,
         save='../figs/emp_sim_pca_site_freqs.pdf',
         titles=plot_names,
         clim=7000)

# violin plots
# explained variance per variable: Var X PC
# loading_mat = pca.components_.T * np.sqrt(pca.explained_variance_)
aas = np.asarray(list('ARNDCQEGHILKMFPSTWYV'))
vars_sort_by_importance_pc1 = np.argsort(-1*np.abs(pca.components_[0,:]))
vars_sort_by_importance_pc2 = np.argsort(-1*np.abs(pca.components_[1,:]))

freq_compare_n_cl_violins(msa_freqs[:, vars_sort_by_importance_pc1[:5]],
                          np.asarray([msa_sim_freqs[:, vars_sort_by_importance_pc1[:5]],
                           th_msa[:, vars_sort_by_importance_pc1[:5]]]),
                          aas=aas[vars_sort_by_importance_pc1[:5]],
                          save_path='../results/violin_test_freqs_1aln_1cl_pc1order.png')
freq_compare_n_cl_violins(msa_freqs[:, vars_sort_by_importance_pc2[:5]],
                          np.asarray([msa_sim_freqs[:, vars_sort_by_importance_pc2[:5]],
                           th_msa[:, vars_sort_by_importance_pc2[:5]]]),
                          aas=aas[vars_sort_by_importance_pc2[:5]],
                          save_path='../results/violin_test_freqs_1aln_1cl_pc2order.png')
"""

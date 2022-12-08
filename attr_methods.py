import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import StratifiedKFold

from stats import get_n_sites_per_msa
from utils import read_cfg_file, get_model_performance
from preprocessing import raw_alns_prepro, make_msa_reprs
from ConvNet import load_net
from preprocessing import TensorDataset
from captum.attr import Saliency, DeepLift, IntegratedGradients


def plot_summary_attr(map, fig, ax, alphabet, summary_meth='max',
                      sum_ax='channels'):
    im = ax.imshow(map, cmap=plt.cm.viridis, interpolation='none',
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    if sum_ax == 'channels':
        ax.set_xticks(range(len(alphabet)))
        ax.set_xticklabels(list(alphabet))
    if sum_ax == 'sites':
        ax.set_xlabel(sum_ax)
    # ax.set_ylabel(f'MSA attributions\n({summary_meth}. over {sum_ax})')
    ax.set(frame_on=False)


def get_attr(msa_repr, attr_meth, multiply_by_inputs=False):
    input = msa_repr.unsqueeze(0)
    input.requires_grad = True
    # attributions
    if attr_meth == 'saliency':
        saliency = Saliency(net)
        return saliency.attribute(input).squeeze().cpu().detach().numpy().T
    elif attr_meth == 'deeplift':
        dl = DeepLift(net)
        return dl.attribute(input).squeeze().cpu().detach().numpy().T
    elif attr_meth == 'integratedgradients':
        ig = IntegratedGradients(net, multiply_by_inputs=multiply_by_inputs)
        return ig.attribute(input).squeeze().cpu().detach().numpy().T


attr_method = 'integratedgradients'
xinput = False

seq_type = 'DNA'
data_dir = '../../data/simulations/dna_sim' if seq_type == 'DNA' \
    else '../../data'
real_path = f'{data_dir}/treebase_fasta' if seq_type == 'DNA' \
    else f'{data_dir}/hogenom_fasta'

# sims = ['new_bonk', 'alisim', 'modelteller_complex', 'SpartaABC',
#        'Sparta_modified']
sims = ['new_bonk']

# first is k1 then k5
res_path = '../results/dna' if seq_type == 'DNA' else '../results'
sim_m_paths = [[f'{res_path}/new_bonk/cnn-14-Nov-2022-14:09:25.151821',
                f'{res_path}/new_bonk/cnn-14-Nov-2022-14:10:08.922490']]
'''
sim_m_paths = [[f'{res_path}/alisim/cnn-14-Nov-2022-14:08:46.048967',
                f'{res_path}/alisim/cnn-14-Nov-2022-14:10:08.738292'],
               [f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:08:46.045950',
                f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:10:08.848031']]
[f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:09:21.422234',
f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:10:08.685692'],
[f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:09:24.567552', 
f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:10:54.429464']]'''

#sim_m_paths = [
#    [f'{res_path}/cnn_alisim_lg_c60_gapless_k1_02-Dec-2022-17:14:46.840664',
#     f'{res_path}/cnn_alisim_lg_c60_gapless_k5_02-Dec-2022-17:29:42.178949']]

n_folds = 10
kfold = StratifiedKFold(n_folds, shuffle=True, random_state=42)

# load emp data
n_alns = None

real_alns, real_fastas, _ = raw_alns_prepro([real_path], n_alns=n_alns,
                                            seq_len=10000 if seq_type == 'DNA'
                                            else 1479, molecule_type=seq_type)
real_fastas, real_alns = real_fastas[0], real_alns[0]

for sim, model_paths in zip(sims, sim_m_paths):
    print(sim)
    sim_path = f'{data_dir}/' \
               f'{"simulations/" if seq_type == "protein" else ""}{sim}'
    cfg = read_cfg_file(f'{model_paths[0]}/cfg.json')
    n_real, n_sim = cfg['data']['nb_alignments']
    cnn_seq_len = cfg['conv_net_parameters']['input_size']

    # load simulations
    sim_alns, sim_fastas, _ = raw_alns_prepro([sim_path], n_alns=n_alns,
                                              seq_len=cnn_seq_len,
                                              molecule_type=seq_type)
    sim_fastas, sim_alns = sim_fastas[0], sim_alns[0]
    seq_lens = get_n_sites_per_msa([real_alns, sim_alns])

    # get MSA embeddings
    aln_reprs_real = make_msa_reprs([real_alns], [real_fastas], cnn_seq_len,
                                    molecule_type=seq_type)[0]
    aln_reprs_sim = make_msa_reprs([sim_alns], [sim_fastas], cnn_seq_len,
                                   molecule_type=seq_type)[0]
    data = np.concatenate([aln_reprs_real.copy(), aln_reprs_sim.copy()]).astype(
        np.float32)
    del aln_reprs_real
    del aln_reprs_sim
    labels = np.concatenate((np.zeros(len(real_alns)), np.ones(len(sim_alns))))

    attrs_models, inds_models, val_inds_models = {}, {}, {}
    preds = {}
    pad = {}
    for model_path in model_paths:
        cfg = read_cfg_file(f'{model_path}/cfg.json')
        ks = 'k' + str(cfg['conv_net_parameters']['kernel_size'])
        print(ks)

        # choose best fold
        val_folds, header = get_model_performance(model_path)
        fold = np.argmin(val_folds.T[header == 'loss'])
        # get validation data for that fold
        val_ids = [val_ids for i, (_, val_ids) in enumerate(kfold.split(data,
                                                                        labels))
                   if i == fold][0]
        val_ds = TensorDataset(data[val_ids].copy(), labels[val_ids])

        # get network-related params
        model_params = cfg['conv_net_parameters']
        cnn_seq_len = model_params['input_size']
        batch_size, epochs, lr, opt, nb_folds = cfg['hyperparameters'].values()

        # load net
        net = load_net(f'{model_path}/model-fold-{fold + 1}.pth', model_params)

        # get acc. and sort alns by acc.
        preds_real = torch.flatten(
            torch.sigmoid(
                net(val_ds.data[val_ds.labels == 0]))).detach().cpu().numpy()
        inds_sort_real = np.argsort(preds_real)  # best to worst
        preds_sim = torch.flatten(
            torch.sigmoid(
                net(val_ds.data[val_ds.labels == 1]))).detach().cpu().numpy()
        inds_sort_sim = np.argsort(preds_sim * -1)  # best to worst

        # get indices of best, worst, mode MSAs
        inds = {'real': inds_sort_real,
                'sim': inds_sort_sim}

        # save padding
        padding = {}
        for key, val in inds.items():
            classe = 0 if 'real' in key else 1
            cl_sl = np.asarray(seq_lens[classe])[val]
            pad_inds = np.zeros((len(val), 2)).astype(int)

            pad_before = (np.repeat(cnn_seq_len, len(val)) - cl_sl) // 2
            pad_before[pad_before < 0] = 0

            for i, p in enumerate(pad_before):
                pad_inds[i][0] = p
                pad_inds[i][1] = p + cl_sl[i]

            padding[key] = pad_inds

        # get attributions
        attrs = {}
        for key, i in inds.items():
            label = 0 if 'real' in key else 1
            attrs[key] = np.asarray([get_attr(msa, attr_method, xinput)
                                     for msa in
                                     val_ds.data[val_ds.labels == label][i]])

        attrs_models[ks] = attrs
        preds[ks] = {'real': 1 - preds_real[inds_sort_real],
                     'sim': preds_sim[inds_sort_sim]}
        pad[ks] = padding
        inds_models[ks] = inds
        val_inds_models[ks] = val_ids

        # plot accs.
        base_name = f'{sim}_k{str(cfg["conv_net_parameters"]["kernel_size"])}'

        fig, ax = plt.subplots()
        ax.plot(1 - preds_real[inds_sort_real], label='empirical')
        ax.plot(preds_sim[inds_sort_sim], label='simulated')
        ax.set_ylabel('accuracy')
        ax.set_xlabel('msa')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../figs/{base_name}acc_per_aln_sorted.pdf')
        plt.close('all')

    mol_ax, site_ax = 0, 1
    alphabet = 'ACGT-' if seq_type == 'DNA' else 'ARNDCQEGHILKMFPSTWYV-'

    # plot summary maps
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    # plot channel importance
    for i, (key, attrs) in enumerate(attrs_models.items()):
        chanl_attr_max = {}
        for k, val in attrs.items():
            chanl_attr_max[k] = np.asarray([np.max(np.abs(attr), axis=mol_ax)
                                            for attr in val])
        # max
        plot_summary_attr(chanl_attr_max['real'], fig, ax[i, 0], alphabet)
        ax[i, 0].set_title(f'{key} empirical')
        plot_summary_attr(chanl_attr_max['sim'], fig, ax[i, 1], alphabet)
        ax[i, 1].set_title(f'{key} simulated')

    plt.tight_layout()
    plt.savefig(f'../figs/{sim}_k1k5_{attr_method}_max.pdf')
    plt.close('all')

    # plot site importance
    # p1_real, p1_sim = int(len(real_alns) * 0.01), int(len(sim_alns) * 0.01)
    for i, (key, attrs) in enumerate(attrs_models.items()):
        site_attr_max, site_attr_avg = {}, {}
        for k, val in attrs.items():
            site_attr_max[k] = np.zeros((len(val), cnn_seq_len))
            site_attr_avg[k] = np.zeros((len(val), cnn_seq_len))
            for j, attr in enumerate(val):
                start, end = pad[key][k][j]
                max_chnls = np.max(np.abs(attr[start:end]), axis=site_ax)
                mean_chnls = np.mean(np.abs(attr[start:end]), axis=site_ax)
                site_attr_max[k][j, :len(max_chnls)] = max_chnls
                site_attr_avg[k][j, :len(mean_chnls)] = mean_chnls

        fig, ax = plt.subplots(ncols=2, figsize=(16, 9))
        # empirical
        # Adding Twin Axes
        ax2 = ax[0].twinx()
        ax2.set_ylabel('MSA accuracy')
        ax2.tick_params(axis='y')
        yticks = [int(i * len(preds[key]['sim']) / 5)
                  for i in range(5)] + [len(preds[key]['sim']) - 1]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(np.round(preds[key]['sim'][yticks], 4))
        plot_summary_attr(site_attr_max['real'], fig, ax[0], alphabet,
                          sum_ax='sites')
        ax[0].set_title('Empirical')
        # simulated
        # Adding Twin Axes
        ax2 = ax[1].twinx()
        ax2.set_ylabel('MSA accuracy')
        ax2.tick_params(axis='y')
        yticks = [int(i * len(preds[key]['sim']) / 5)
                  for i in range(5)] + [len(preds[key]['sim']) - 1]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(np.round(preds[key]['sim'][yticks], 4))
        plot_summary_attr(site_attr_max['sim'], fig, ax[1], alphabet,
                          sum_ax='sites')
        ax[1].set_title('Simulated')
        # plt.tight_layout()
        plt.savefig(f'../figs/{key}_{sim}_{attr_method}_sites_max.pdf')
        plt.close('all')

    # plot individual attribution maps
    n = 10
    for key, attrs in attrs_models.items():
        t_ks = f'{sim}_{key}_{attr_method}'
        for k, val in attrs.items():
            title = f'{t_ks}_{k}'
            label = 0 if 'real' in k else 1
            fig, ax = plt.subplots(nrows=n * 2, figsize=(20, 20))
            for i, (attr, msa) in enumerate(zip(val[:n],
                                                ds.data[ds.labels == label][
                                                    inds_models[key][k]])):
                # if not np.all(attr == 0):
                if 'best' in k:
                    classe, pred_ind = k.replace('best_', ''), i
                else:
                    classe, pred_ind = k.replace('wrst_', ''), -n + i
                # title += f', {k}({str(np.round(preds[key][classe][pred_ind], 4))})'

                start, end = pad[key][k][i]
                # attribution map
                sm_im = ax[i * 2].imshow(np.abs(attr[start:end]).T,
                                         cmap=plt.cm.viridis,
                                         aspect="auto", interpolation="none")
                cbar = fig.colorbar(sm_im, ax=ax[i * 2], orientation="vertical")
                # msa
                sm_im = ax[i * 2 + 1].imshow(
                    msa[:, start:end].detach().cpu().numpy(),
                    cmap=plt.cm.viridis,
                    aspect="auto", interpolation="none")
                cbar_msa = fig.colorbar(sm_im, ax=ax[i * 2 + 1],
                                        orientation="vertical")
                for axi in [i * 2, i * 2 + 1]:
                    ax[axi].set_yticks(range(len(alphabet)))
                    ax[axi].set_yticklabels(list(alphabet))
                for tick in ax[i * 2 + 1].yaxis.get_major_ticks():
                    tick.label.set_fontsize('x-small')
                ax[i * 2].set_xticklabels([])
                ax[i * 2].get_xaxis().set_visible(False)
                # ax[i].set_title(title)
            ax[-1].set_xlabel('sites')
            plt.subplots_adjust(wspace=0.0, hspace=0.025)
            plt.tight_layout()
            save = title.replace(' ', '_').replace(']', '')
            save = save.replace(',', '').replace(':', '').replace('[', '')
            plt.savefig(f'../figs/{save}.pdf')
            plt.close('all')

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


def plot_msa_attr(saliency_map, ig_xinput_map, msa, molecule_type='protein',
        save=''):
    alphabet = 'ACGT-' if molecule_type == 'DNA' else 'ARNDCQEGHILKMFPSTWYV-'
    fig, ax = plt.subplots(nrows=3, figsize=(16, 9))
    # saliency map, IG x input, MSA site frequencies
    for i, map in enumerate([np.abs(saliency_map).T, np.abs(ig_xinput_map).T,
                             msa]):
        im = ax[i].imshow(map, cmap=plt.cm.viridis, aspect="auto",
                          interpolation="none")
        fig.colorbar(im, ax=ax[i], orientation="vertical")

        ax[i].set_yticks(range(len(alphabet)))
        ax[i].set_yticklabels(list(alphabet))
        if i < 2:  # remove x ticks for first 2 "rows" -> min space between plts
            ax[i].set_xticklabels([])
            ax[i].get_xaxis().set_visible(False)

    ax[2].set_xlabel('sites')
    plt.subplots_adjust(wspace=0.0, hspace=0.025)

    plt.savefig(save)
    plt.close('all')


def plot_summary(attrs, pad_mask, sum_ax, preds=None,
        molecule_type='protein', save=''):
    mol_ax, site_ax = 0, 1
    alphabet = 'ACGT-' if molecule_type == 'DNA' else 'ARNDCQEGHILKMFPSTWYV-'
    map_height = (attrs['emp'][0].shape[0] if sum_ax == 'sites'
                  else attrs['emp'][0].shape[1])

    # plot channel importance
    fig, ax = plt.subplots(2, 2, figsize=(16, 16 if sum_ax == 'sites' else 9),
                           sharex=True)
    site_attr = {}
    for i, op in enumerate(['max', 'avg']):
        site_attr[op] = {}
        for j, (cl, attr_cl) in enumerate(attrs.items()):
            site_attr[op][cl] = np.zeros((len(attr_cl), map_height))
            for a, attr in enumerate(attr_cl):
                sl = np.sum(pad_mask[cl][a]) if sum_ax == 'sites' else None
                if op == 'max':
                    site_attr[op][cl][a, :sl] = np.max(
                        np.abs(attr[pad_mask[cl][a]]),
                        axis=site_ax if sum_ax == 'sites' else mol_ax)
                if op == 'avg':
                    site_attr[op][cl][a, :sl] = np.mean(
                        np.abs(attr[pad_mask[cl][a]]),
                        axis=site_ax if sum_ax == 'sites' else mol_ax)

            ax_summary_attr(site_attr[op][cl], fig, ax[i, j],
                            alphabet, sum_ax,
                            preds=preds[cl] if preds is not None else None)
            ax[i, j].set_title(f'{cl} {op}')

    plt.savefig(save)
    plt.close('all')


def plot_pred_scores(preds, save=''):
    fig, ax = plt.subplots()
    for cl, scores in preds.items():
        ax.plot(scores, label=cl)
    ax.set_ylabel('accuracy')
    ax.set_xlabel('msa')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close('all')


def get_sorted_pred_scores(model, val_ds, classes=None):
    if classes is None:
        classes = ['emp', 'sim']
    sort_by_pred, preds = {}, {}
    for l, cl in enumerate(classes):
        preds[cl] = torch.sigmoid(model(val_ds.data[val_ds.labels == l]))
        preds[cl] = torch.flatten(preds[cl]).detach().cpu().numpy()
        if l == 0:
            sort_by_pred[cl] = np.argsort(preds[cl])  # best to worst
        else:
            sort_by_pred[cl] = np.argsort(preds[cl] * -1)  # best to worst
        # sort prediction scores
        preds[cl] = preds[cl][sort_by_pred[cl]]
    return preds, sort_by_pred


def ax_summary_attr(map, fig, ax, alphabet, sum_ax='channels', preds=None):
    im = ax.imshow(map.T, cmap=plt.cm.viridis, interpolation='none',
                   aspect="auto")
    fig.colorbar(im, ax=ax, location='left', aspect=50)
    if sum_ax == 'channels':
        ax.set_yticks(range(len(alphabet)))
        ax.set_yticklabels(list(alphabet))
    if sum_ax == 'sites':
        ax.set_ylabel(sum_ax)
    if preds is not None:
        twinx_col = 'coral'
        ax2 = ax.twinx()
        ax2.plot(preds, color=twinx_col, linewidth=1)
        ax2.tick_params(axis="y", labelcolor=twinx_col)
        ax2.set_ylabel('Prediction score\n(best - left to worst - right)',
                       color=twinx_col)
        ax2.set_ylim(0, 1)
    ax.set_xlabel(f'MSA')
    ax.set(frame_on=False)


def get_attr(msa_repr, model, attr_meth, multiply_by_inputs=False):
    input = msa_repr.unsqueeze(0)
    input.requires_grad = True
    # attributions
    if attr_meth == 'saliency':
        saliency = Saliency(model)
        return saliency.attribute(input).squeeze().cpu().detach().numpy().T
    elif attr_meth == 'deeplift':
        dl = DeepLift(model)
        return dl.attribute(input).squeeze().cpu().detach().numpy().T
    elif attr_meth == 'integratedgradients':
        ig = IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs)
        return ig.attribute(input).squeeze().cpu().detach().numpy().T

#TODO below
"""
attr_method = 'saliency'
xinput = False

seq_type = 'protein'
data_dir = '../../data/simulations/dna_sim' if seq_type == 'DNA' \
    else '../../data'
emp_path = f'{data_dir}/treebase_fasta' if seq_type == 'DNA' \
    else f'{data_dir}/hogenom_fasta'

# sims = ['new_bonk', 'alisim', 'modelteller_complex', 'SpartaABC',
#        'Sparta_modified']
sims = ['alisim_lg_gapless']

# first is k1 then k5
res_path = '../results/dna' if seq_type == 'DNA' else '../results'
sim_m_paths = [
    [f'{res_path}/cnn_alisim_lg_gapless_k1_30-Nov-2022-14:59:01.990550',
     f'{res_path}/cnn_alisim_lg_gapless_k5_30-Nov-2022-14:59:28.119344']]
'''
sim_m_paths = [[f'{res_path}/alisim/cnn-14-Nov-2022-14:08:46.048967',
                f'{res_path}/alisim/cnn-14-Nov-2022-14:10:08.738292'],
              [f'{res_path}/new_bonk/cnn-14-Nov-2022-14:09:25.151821',
               f'{res_path}/new_bonk/cnn-14-Nov-2022-14:10:08.922490']
               [f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:08:46.045950',
                f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:10:08.848031']]
[f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:09:21.422234',
f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:10:08.685692'],
[f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:09:24.567552', 
f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:10:54.429464']]'''

# sim_m_paths = [
#    [f'{res_path}/cnn_alisim_lg_c60_gapless_k1_02-Dec-2022-17:14:46.840664',
#     f'{res_path}/cnn_alisim_lg_c60_gapless_k5_02-Dec-2022-17:29:42.178949']]

n_folds = 10
kfold = StratifiedKFold(n_folds, shuffle=True, random_state=42)

# load emp data
n_alns = None

emp_alns, emp_fastas, _ = raw_alns_prepro([emp_path], n_alns=n_alns,
                                            seq_len=10000 if seq_type == 'DNA'
                                            else 1479, molecule_type=seq_type)
emp_fastas, emp_alns = emp_fastas[0], emp_alns[0]

for sim, model_paths in zip(sims, sim_m_paths):
    print(sim)
    sim_path = f'{data_dir}/' \
               f'{"simulations/" if seq_type == "protein" else ""}{sim}'
    cfg = read_cfg_file(f'{model_paths[0]}/cfg.json')
    n_emp, n_sim = cfg['data']['nb_alignments']
    cnn_seq_len = cfg['conv_net_parameters']['input_size']

    # load simulations
    sim_alns, sim_fastas, _ = raw_alns_prepro([sim_path], n_alns=n_alns,
                                              seq_len=cnn_seq_len,
                                              molecule_type=seq_type)
    sim_fastas, sim_alns = sim_fastas[0], sim_alns[0]
    seq_lens = np.concatenate(get_n_sites_per_msa([emp_alns, sim_alns]))

    # get MSA embeddings
    aln_reprs_emp = make_msa_reprs([emp_alns], [emp_fastas], cnn_seq_len,
                                    molecule_type=seq_type)[0]
    aln_reprs_sim = make_msa_reprs([sim_alns], [sim_fastas], cnn_seq_len,
                                   molecule_type=seq_type)[0]
    data = np.concatenate([aln_reprs_emp.copy(), aln_reprs_sim.copy()]).astype(
        np.float32)
    del aln_reprs_emp
    del aln_reprs_sim
    labels = np.concatenate((np.zeros(len(emp_alns)), np.ones(len(sim_alns))))

    xin_models, attrs_models, inds_models, val_inds_models = {}, {}, {}, {}
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
        preds_emp = torch.flatten(
            torch.sigmoid(
                net(val_ds.data[val_ds.labels == 0]))).detach().cpu().numpy()
        inds_sort_emp = np.argsort(preds_emp)  # best to worst
        preds_sim = torch.flatten(
            torch.sigmoid(
                net(val_ds.data[val_ds.labels == 1]))).detach().cpu().numpy()
        inds_sort_sim = np.argsort(preds_sim * -1)  # best to worst

        # get indices of best, worst, mode MSAs
        inds = {'emp': inds_sort_emp,
                'sim': inds_sort_sim}

        # save padding
        padding = {}
        for key, val in inds.items():
            cl_inds = (val_ids[labels[val_ids] == 0] if 'emp' in key
                       else val_ids[labels[val_ids] == 1])
            cl_sl = np.asarray(seq_lens[cl_inds])[val]
            pad_inds = np.zeros((len(val), 2)).astype(int)

            pad_before = (np.repeat(cnn_seq_len, len(val)) - cl_sl) // 2
            pad_before[pad_before < 0] = 0

            for i, p in enumerate(pad_before):
                pad_inds[i][0] = p
                pad_inds[i][1] = p + cl_sl[i]

            padding[key] = pad_inds

        # get attributions
        attrs, xins = {}, {}
        for key, i in inds.items():
            label = 0 if 'emp' in key else 1
            attrs[key] = np.asarray([get_attr(msa, attr_method, xinput)
                                     for msa in
                                     val_ds.data[val_ds.labels == label][i]])
            xins[key] = np.asarray(
                [get_attr(msa, 'integratedgradients', multiply_by_inputs=True)
                 for msa in
                 val_ds.data[val_ds.labels == label][i]])

        attrs_models[ks] = attrs
        xin_models[ks] = xins
        preds[ks] = {'emp': preds_emp[inds_sort_emp],
                     'sim': preds_sim[inds_sort_sim]}
        pad[ks] = padding
        inds_models[ks] = inds
        val_inds_models[ks] = val_ids

        # plot accs.
        base_name = f'{sim}_k{str(cfg["conv_net_parameters"]["kernel_size"])}'

        fig, ax = plt.subplots()
        ax.plot(1 - preds_emp[inds_sort_emp], label='empirical')
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
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), sharex=True)

    # plot channel importance
    for i, (key, attrs) in enumerate(attrs_models.items()):
        chanl_attr_max = {}
        for k, val in attrs.items():
            chanl_attr_max[k] = np.asarray([np.max(np.abs(attr), axis=mol_ax)
                                            for attr in val])
        # max
        plot_summary_attr(chanl_attr_max['emp'], fig, ax[i, 0], alphabet,
                          preds=preds[key]['emp'])
        ax[i, 0].set_title(f'{key} empirical')
        plot_summary_attr(chanl_attr_max['sim'], fig, ax[i, 1], alphabet,
                          preds=preds[key]['sim'])
        ax[i, 1].set_title(f'{key} simulated')

    plt.savefig(f'../figs/{sim}_k1k5_{attr_method}_max.pdf')
    plt.close('all')

    # plot site importance
    # p1_emp, p1_sim = int(len(emp_alns) * 0.01), int(len(sim_alns) * 0.01)
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
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

        # empirical
        seq_len_sort = np.argsort(pad[key]['emp'][:, 1])
        # sort msas by sequence length plot 50% of max seq. len.
        d = site_attr_max['emp'][seq_len_sort][:, :int(cnn_seq_len * 0.5)]
        plot_summary_attr(d, fig, ax[i, 0], alphabet, sum_ax='sites')
        ax[i, 0].set_title(f'{key} empirical')
        # simulated
        seq_len_sort = np.argsort(pad[key]['sim'][:, 1])
        d = site_attr_max['sim'][seq_len_sort][:, :int(cnn_seq_len * 0.5)]
        plot_summary_attr(d, fig, ax[i, 1], alphabet, sum_ax='sites')
        ax[i, 1].set_title(f'{key} simulated')
    # plt.tight_layout()
    plt.savefig(f'../figs/{sim}_k1k5_{attr_method}_sites_max.pdf')
    plt.close('all')

    # plot individual attribution maps
    n = 10
    for key, attrs in attrs_models.items():
        t_ks = f'{sim}_{key}_{attr_method}'
        for k, val in attrs.items():
            title = f'{t_ks}_{k}'
            label = 0 if 'emp' in k else 1
            msas = data[val_inds_models[key]]  # validation msas
            msas = msas[labels[val_inds_models[key]] == label]
            msas = msas[
                inds_models[key][k]]  # sorted accordng to attribution maps
            select = np.concatenate((np.arange(n), np.arange(-n, 0)))
            for i, sel in enumerate(select):
                fig, ax = plt.subplots(nrows=3, figsize=(16, 9))

                start, end = pad[key][k][sel]
                # attribution map
                sm_im = ax[0].imshow(np.abs(val[sel][start:end]).T,
                                     cmap=plt.cm.viridis,
                                     aspect="auto", interpolation="none")
                cbar = fig.colorbar(sm_im, ax=ax[0], orientation="vertical")
                # attr x input
                sm_im = ax[1].imshow(
                    np.abs(xin_models[key][k][sel][start:end]).T,
                    cmap=plt.cm.viridis,
                    aspect="auto", interpolation="none")
                cbar = fig.colorbar(sm_im, ax=ax[1], orientation="vertical")
                # msa
                sm_im = ax[2].imshow(msas[sel][:, start:end],
                                     cmap=plt.cm.viridis,
                                     aspect="auto", interpolation="none")
                cbar_msa = fig.colorbar(sm_im, ax=ax[2],
                                        orientation="vertical")
                for axi in [0, 1]:
                    ax[axi].set_yticks(range(len(alphabet)))
                    ax[axi].set_yticklabels(list(alphabet))
                ax[0].set_xticklabels([])
                ax[0].get_xaxis().set_visible(False)
                ax[1].set_xlabel('sites')
                plt.subplots_adjust(wspace=0.0, hspace=0.025)
                score = '%.2f' % np.round(preds[key][k][sel], 2)
                plt.savefig(f'../figs/{i + 1}_{score}_{title}.pdf')
                plt.close('all')
"""

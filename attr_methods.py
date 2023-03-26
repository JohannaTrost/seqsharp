import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import Saliency, DeepLift, IntegratedGradients


def plot_msa_attr(saliency_map, ig_xinput_map, msa, molecule_type='protein',
                  save=''):
    alphabet = 'ACGT-' if molecule_type == 'DNA' else 'ARNDCQEGHILKMFPSTWYV-'
    fig, ax = plt.subplots(nrows=3, figsize=(16, 9))
    # saliency map, IG x input_plt_fct, MSA site frequencies
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
                 molecule_type='protein', save='', max_sl=None):
    mol_ax, site_ax = 0, 1
    alphabet = 'ACGT-' if molecule_type == 'DNA' else 'ARNDCQEGHILKMFPSTWYV-'
    map_y_len = (attrs['emp'][0].shape[0] if sum_ax == 'sites'
                 else attrs['emp'][0].shape[1])

    # plot channel/site importance
    fig, ax = plt.subplots(2, 2, figsize=(16, 16 if sum_ax == 'sites' else 9),
                           sharex=True)
    summary_maps = {}
    for i, op in enumerate(['max', 'avg']):
        summary_maps[op] = {}
        for j, (cl, attr_cl) in enumerate(attrs.items()):
            summary_maps[op][cl] = np.zeros((len(attr_cl),
                                             map_y_len if max_sl is None
                                             else max_sl))
            for a, attr in enumerate(attr_cl):
                sl = np.sum(pad_mask[cl][a]) if sum_ax == 'sites' else None
                if max_sl is not None:
                    sl = min(sl, max_sl)
                map = np.abs(attr[pad_mask[cl][a]])[:sl]
                if len(map) == 0:
                    break
                if op == 'max':
                    summary_maps[op][cl][a, :sl] = np.max(
                        map, axis=site_ax if sum_ax == 'sites' else mol_ax)
                if op == 'avg':
                    summary_maps[op][cl][a, :sl] = np.mean(
                        map, axis=site_ax if sum_ax == 'sites' else mol_ax)

            ax_summary_attr(summary_maps[op][cl], fig, ax[i, j],
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
    # fig.colorbar(im, ax=ax, location='left', aspect=50)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("left", size="5%", pad=0.1)
    fig.colorbar(im, ax=ax)  # , cax=cax)
    # cax.yaxis.tick_left()
    # cax.yaxis.set_label_position('left')
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


# TODO below

'''
molecule_type = 'protein'
data_dir = '../../data/simulations/dna_sim' if molecule_type == 'DNA' \
    else '../../data'
emp_path = f'{data_dir}/treebase_fasta' if molecule_type == 'DNA' \
    else f'{data_dir}/hogenom_fasta'

# sims = ['new_bonk', 'alisim', 'modelteller_complex', 'SpartaABC',
#        'Sparta_modified']
sims = ['alisim_poisson_gapless_trees']

# first is k1 then k5
res_path = '../results/dna' if molecule_type == 'DNA' \
    else '../results/all_model_res_06-Jan-2023'
sim_m_paths = [
    [f'{res_path}/cnn_alisim_poisson_gapless_k1_21-Dec-2022-16:58:04.728293',
     f'{res_path}/cnn_alisim_poisson_gapless_k5_29-Nov-2022-21:02:57.107130'],
]

"""
sim_m_paths = [[f'{res_path}/alisim/cnn-14-Nov-2022-14:08:46.048967',
                f'{res_path}/alisim/cnn-14-Nov-2022-14:10:08.738292'],
              [f'{res_path}/new_bonk/cnn-14-Nov-2022-14:09:25.151821',
               f'{res_path}/new_bonk/cnn-14-Nov-2022-14:10:08.922490']
               [f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:08:46.045950',
                f'{res_path}/modelteller_complex/cnn-14-Nov-2022-14:10:08.848031']]
[f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:09:21.422234',
f'{res_path}/SpartaABC/cnn-14-Nov-2022-14:10:08.685692'],
[f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:09:24.567552', 
f'{res_path}/Sparta_modified/cnn-14-Nov-2022-14:10:54.429464']]"""

# sim_m_paths = [
#    [f'{res_path}/cnn_alisim_lg_c60_gapless_k1_02-Dec-2022-17:14:46.840664',
#     f'{res_path}/cnn_alisim_lg_c60_gapless_k5_02-Dec-2022-17:29:42.178949']]

n_folds = 10
kfold = StratifiedKFold(n_folds, shuffle=True, random_state=42)

# load emp data
n_alns = None

emp_alns, emp_fastas, _ = raw_alns_prepro([emp_path], n_alns=n_alns,
                                          seq_len=10000 if molecule_type == 'DNA'
                                          else 1479,
                                          molecule_type=molecule_type)
if molecule_type == 'protein':
    # remove first 2 sites
    emp_alns = [[seq[2:] for seq in aln] for aln in emp_alns]
emp_fastas, emp_alns = emp_fastas[0], emp_alns[0]

for sim, model_paths in zip(sims, sim_m_paths):
    print(sim)
    sim_path = f'{data_dir}/' \
               f'{"simulations/" if molecule_type == "protein" else ""}{sim}'
    cfg = read_cfg_file(f'{model_paths[0]}/cfg.json')
    n_emp, n_sim = cfg['data']['nb_alignments']
    cnn_seq_len = cfg['conv_net_parameters']['input_size']

    # load simulations
    sim_alns, sim_fastas, _ = raw_alns_prepro([sim_path], n_alns=n_alns,
                                              seq_len=cnn_seq_len,
                                              molecule_type=molecule_type)

    if molecule_type == 'protein':
        # remove first 2 sites
        sim_alns = [[seq[2:] for seq in aln] for aln in sim_alns]

    sim_fastas, sim_alns = sim_fastas[0], sim_alns[0]
    fastas = np.concatenate((emp_fastas, sim_fastas))
    seq_lens = np.concatenate(get_n_sites_per_msa([emp_alns, sim_alns]))

    # get MSA embeddings
    aln_reprs_emp = make_msa_reprs([emp_alns], [emp_fastas], cnn_seq_len,
                                   molecule_type=molecule_type)[0]
    aln_reprs_sim = make_msa_reprs([sim_alns], [sim_fastas], cnn_seq_len,
                                   molecule_type=molecule_type)[0]
    data = np.concatenate([aln_reprs_emp.copy(), aln_reprs_sim.copy()]).astype(
        np.float32)
    del aln_reprs_emp
    del aln_reprs_sim
    labels = np.concatenate((np.zeros(len(emp_alns)), np.ones(len(sim_alns))))

    for model_path in model_paths:
        cfg = read_cfg_file(f'{model_path}/cfg.json')
        ks = 'k' + str(cfg['conv_net_parameters']['kernel_size'])
        print(ks)

        # -------------------- attribution study -------------------- #

        # choose best fold
        val_folds, header = get_model_performance(model_path)
        best_fold = np.argmax(val_folds.T[header == 'acc'])
        if ks == 'k1':
            n_folds = 1
        else:
            n_folds = 10

        for fold in range(n_folds):
            if ks == 'k1':
                fold = 9
            print(f'Compute/plot attribution scores from fold-{fold + 1}-model')

            attr_path = f'{model_path}/attribution_fold{fold + 1}'
            if fold == best_fold:
                attr_path += '_best'
            if not os.path.exists(attr_path):
                os.mkdir(attr_path)

            # get validation data for that fold
            val_ids = [val_ids
                       for i, (_, val_ids) in enumerate(kfold.split(data,
                                                                    labels))
                       if i == fold][0]
            
            val_ds = TensorDataset(data[val_ids].copy(), labels[val_ids])

            # get net
            # get network-related params
            model_params = cfg['conv_net_parameters']
            cnn_seq_len = model_params['input_size']
            batch_size, epochs, lr, opt, nb_folds = cfg[
                'hyperparameters'].values()

            # load net
            model = load_net(f'{model_path}/model-fold-{fold + 1}.pth',
                             model_params)

            # get predition scores and order to sort MSAs by scores
            preds, sort_by_pred = get_sorted_pred_scores(model, val_ds)
            plot_pred_scores(preds, save=f'{attr_path}/val_pred_scores.pdf')

            test_res = accuracy(model(val_ds.data), val_ds.labels)
            print(test_res)
            print('Saved history:')
            print(model.val_history['acc'][-1])
            print(model.val_history['acc_emp'][-1])
            print(model.val_history['acc_sim'][-1])
            print(model.val_history['loss'][-1])

            # get masks to remove padding
            pad_mask = {}  # sorted by pred. score
            for l, (cl, sort_inds) in enumerate(sort_by_pred.items()):
                # filter emp/sim msas and sort by prediction score
                pad_mask[cl] = val_ds.data[val_ds.labels == l][sort_inds]
                # sum over channels: 1 = no padding, 0 = padding
                pad_mask[cl] = pad_mask[cl].sum(axis=1).detach().cpu().numpy()
                pad_mask[cl] = pad_mask[cl].astype(bool)

            # get attributions
            attrs, xins = {}, {}
            for cl, i in sort_by_pred.items():
                label = 0 if 'emp' in cl else 1
                attrs[cl] = np.asarray([get_attr(msa, model, 'saliency')
                                        for msa in
                                        val_ds.data[val_ds.labels == label][i]])
                xins[cl] = np.asarray(
                    [get_attr(msa, model, 'integratedgradients',
                              multiply_by_inputs=True)
                     for msa in val_ds.data[val_ds.labels == label][i]])

            # plot site/channel importance
            # plot_summary(attrs, pad_mask, 'channels', preds, molecule_type,
            #              save=f'{attr_path}/channel_attr_preds.pdf')
            plot_summary(attrs, pad_mask, 'channels', None, molecule_type,
                         save=f'{attr_path}/channel_attr.pdf')
            # plot_summary(attrs, pad_mask, 'sites', preds, molecule_type,
            #              save=f'{attr_path}/site_attr_preds.pdf')
            plot_summary(attrs, pad_mask, 'sites', None, molecule_type,
                         save=f'{attr_path}/site_attr.pdf')
            plot_summary(attrs, pad_mask, 'sites', None, molecule_type,
                         save=f'{attr_path}/site_attr_200.pdf', max_sl=200)
            plot_summary(attrs, pad_mask, 'sites', None, molecule_type,
                         save=f'{attr_path}/site_attr_50.pdf', max_sl=50)

            # plot individual saliency maps

            # indices of n best predicted (decreasing) and n worst predicted
            # (increasing) MSAs
            n = 10
            select = np.concatenate((np.arange(n), np.arange(-n, 0)))

            for l, cl in enumerate(attrs.keys()):
                # validation data, sorted according to attribution maps
                msas = val_ds.data[val_ds.labels == l][sort_by_pred[cl]]
                fnames = fastas[val_ids][val_ds.labels == l][sort_by_pred[cl]]
                for i, sel in enumerate(select):
                    score = '%.2f' % np.round(preds[cl][sel], 2)

                    sal_map = attrs[cl][sel][pad_mask[cl][sel]]
                    ig_map = xins[cl][sel][pad_mask[cl][sel]]
                    msa = msas[sel][:, pad_mask[cl][sel]].detach().cpu().numpy()

                    plot_msa_attr(sal_map, ig_map, msa, molecule_type,
                                  save=f'{attr_path}/'
                                       f'[{i}]_{fnames[sel].split(".")[0]}'
                                       f'_{score}_{cl}.pdf')
'''
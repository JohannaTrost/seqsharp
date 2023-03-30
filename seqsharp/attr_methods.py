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
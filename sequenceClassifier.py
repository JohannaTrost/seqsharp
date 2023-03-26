"""Program to classify empirical and simulated sequence alignments

Allows to train and evaluate a (convolutional) neural network or use a
trained network. Plots can be generated to provide information about a
dataset with alignments or the performance of a network.

Please execute 'python sequenceClassifier.py --help' to view all the options
"""
import argparse
import errno
import os
import random
import time
import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from minepy import MINE
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from ConvNet import ConvNet, load_model, compute_device
from attr_methods import get_attr, get_sorted_pred_scores, plot_pred_scores, \
    plot_summary, plot_msa_attr
from preprocessing import TensorDataset, raw_alns_prepro, make_msa_reprs, \
    load_msa_reprs
from plots import plot_folds, plot_corr_pred_sl, plot_groups_folds, make_fig
from stats import get_n_sites_per_msa
from utils import write_cfg_file, read_cfg_file
from train_eval import fit, evaluate,  find_lr_bounds, print_model_performance

torch.cuda.empty_cache()

gc.collect()


def main():
    sep_line = '-------------------------------------------------------' \
               '---------'

    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', nargs='*', type=str, required=True,
                        help='Specify the <path/to/> directory(s) containing '
                             'simulated alignments (in fasta format)')
    parser.add_argument('--emp', type=str,
                        help='Specify the <path/to/> directory(s) containing '
                             'empirical alignments (in fasta format)')
    parser.add_argument('-t', '--training', action='store_true',
                        help='Datasets will be used to train the neural '
                             'network (specified with --datasets option). '
                             'Requires --cfg and --datasets.')
    parser.add_argument('--test', action='store_true',
                        help='Alignments will be passed to (a) trained '
                             'model(s). Requires --models, --datasets and '
                             '--cfg')
    parser.add_argument('--attr', action='store_true',
                        help='Generates attribution maps using validation data.'
                             ' Requires --models (only if not --training) and '
                             '--datasets')
    parser.add_argument('--clr', action='store_true',
                        help='Indicate to use cyclic learning rates '
                             '(in this case lr given in config is ignored). '
                             'Requires --training.')
    parser.add_argument('-m', '--models', type=str,
                        help='<path/to> directory with trained model(s). '
                             'These models will then be tested on a given data '
                             'set. --cfg, --datasets and --test are '
                             'required for this option.')
    parser.add_argument('-c', '--cfg', type=str,
                        help='<path/to> cfg file (.json) or directory '
                             'containing: hyperparameters, data specific '
                             'parameters and parameters determinin the '
                             'structure of the Network. If a directory is '
                             'given, the latest modified json file will be '
                             'used')
    parser.add_argument('-s', '--save', type=str,
                        help='<path/to> directory where trained models and '
                             'result plots will be saved')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the sites of alignments/pairs in the '
                             'first directory specified')
    parser.add_argument('--molecule_type', choices=['DNA', 'protein'],
                        default='protein',
                        help='Specify if you use DNA or protein MSAs')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume training, starting from last epoch in '
                             'each fold.')

    args = parser.parse_args()

    # ---------------------------- get arguments ---------------------------- #

    test, train, resume = args.test, args.training, args.resume
    emp_fasta_path, sim_fasta_path = args.emp, args.sim
    result_path = args.save if args.save else None
    model_path = args.models
    cfg_path = args.cfg
    molecule_type = args.molecule_type
    shuffle = args.shuffle
    attribution = args.attr
    clr = args.clr

    # -------------------- verify argument usage -------------------- #

    if not (train ^ test ^ resume) or (train and test and resume):
        parser.error('Use either --training, --test or --resume.')

    if test and (not model_path or not sim_fasta_path):
        parser.error('--test requires --modles and --sim')

    if (train or resume) and not emp_fasta_path:
        parser.error('Training requires exactly 2 datasets: one directory '
                     'containing empirical alignments and one simulated '
                     'alignments.')

    # ------------------ verify existence of files/folders ------------------ #

    for sim_path in sim_fasta_path:
        if not os.path.exists(sim_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    sim_path)
    if emp_fasta_path and not os.path.exists(emp_fasta_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                emp_fasta_path)
    if result_path and not os.path.exists(result_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                result_path)
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                model_path)
    if cfg_path and not os.path.exists(cfg_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                cfg_path)

    # determine cfg path from model path
    if model_path and not cfg_path:
        if os.path.isdir(model_path):
            cfg_path = model_path + '/cfg.json'
        elif os.path.isfile(model_path):
            cfg_path = os.path.dirname(model_path) + '/cfg.json'

    # laod config
    cfg = read_cfg_file(cfg_path)

    # get and create output folder
    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    if result_path is not None and not model_path:
        if not result_path.split('/')[-1].startswith('cnn-'):
            # create unique subdir for the model(s)
            result_path += '/cnn_'
            result_path += sim_fasta_path[0].split('/')[-1]
            result_path += '_' + str(timestamp)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    elif model_path:
        result_path = model_path

    # save original cfg with time stamp
    write_cfg_file(cfg, cfg_path=cfg_path, timestamp=timestamp)

    # -------------------- configure parameters -------------------- #

    # parameters for loading data
    n_sites = cfg['data']['nb_sites']
    n_alns = cfg['data']['nb_alignments']
    n_alns = n_alns if isinstance(n_alns, list) else [n_alns]
    cnt_datasets = len(sim_fasta_path)
    cnt_datasets += 1 if emp_fasta_path else 0
    if len(n_alns) == cnt_datasets:
        n_alns = [int(x) for x in n_alns]
    elif not model_path:
        n_alns = int(n_alns[0])
    else:
        n_alns = None

    # parameters of network architecture
    model_params = cfg['conv_net_parameters']

    # hyperparameters
    batch_size, epochs, lr, lr_range, opt, nb_folds = cfg[
        'hyperparameters'].values()

    if opt == 'Adagrad':
        optimizer = torch.optim.Adagrad
    elif opt == 'SGD':
        optimizer = torch.optim.SGD
    elif opt == 'Adam':
        optimizer = torch.optim.Adam
    else:
        raise ValueError(
            'Please specify either Adagrad, Adam or SGD as optimizer')

    # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
    kfold = StratifiedKFold(nb_folds, shuffle=True, random_state=42)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'\nRandom seed: {seed}\n')

    # ------------------------- load/preprocess data ------------------------- #

    first_input_file = os.listdir(emp_fasta_path)[0]
    fmts = ['.fa', '.fasta', '.phy']
    correct_fmt = np.any([first_input_file.endswith(f) for f in fmts])
    if correct_fmt:
        alns, fastas, data_dict = raw_alns_prepro(
            [emp_fasta_path] + sim_fasta_path,
            n_alns, n_sites, shuffle=shuffle, molecule_type=molecule_type)

        # update config with info from loaded data
        for key, val in data_dict.items():
            cfg['data'][key] = val

        if molecule_type == 'protein' and '-' not in ''.join(alns[1][0]):
            # for protein data without gaps
            for i in range(len(alns)):
                # remove first 2 sites
                alns[i] = [[seq[2:] for seq in aln] for aln in alns[i]]

        seq_lens = np.concatenate(get_n_sites_per_msa(alns))

        print(molecule_type)
        emp_alns, sim_alns = make_msa_reprs(alns,
                                            cfg['data']['nb_sites'],
                                            cfg['data']['padding'],
                                            molecule_type=molecule_type)
        fastas = np.concatenate(fastas)
        del alns
    elif first_input_file.endswith('.csv'):  # msa representations is given
        emp_alns, fastas_emp = load_msa_reprs(emp_fasta_path, n_alns[0])
        sim_alns, fastas_sim = load_msa_reprs(emp_fasta_path, n_alns[1])

    n_emp_alns, n_sim_alns = len(emp_alns), len(sim_alns)
    data = np.asarray(emp_alns + sim_alns, dtype='float32')
    labels = np.concatenate((np.zeros(n_emp_alns), np.ones(n_sim_alns)))

    if model_path:
        state = 'train' if resume else 'eval'
        loaded_models = load_model(model_path, state)

    # -------------------- k-fold cross validation -------------------- #

    print(f'\nCompute device: {compute_device}\n')

    bs_lst = batch_size if isinstance(batch_size, list) else [batch_size]
    epochs_lst = epochs if isinstance(epochs, list) else [epochs]
    train_throughput = np.zeros((len(bs_lst), nb_folds))

    for i, (bs, epcs) in enumerate(zip(bs_lst, epochs_lst)):
        # usually one iteration except for bs determination

        print(sep_line)
        print(f"\tBatch size: {bs}\n"
              f"\tLearning rate: {lr_range if lr_range != '' else lr}")
        print(sep_line + '\n')

        models = []
        lrs = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(data,
                                                                labels)):
            print(f'FOLD {fold + 1}')
            print(sep_line)

            # splitting dataset by alignments
            print("Building training and validation dataset ...")
            start = time.time()
            train_ds = TensorDataset(data[train_ids], labels[train_ids])
            val_ds = TensorDataset(data[val_ids], labels[val_ids])
            print(f'Finished after {round(time.time() - start, 2)}s\n')

            bs = int(bs)
            train_loader = DataLoader(train_ds, bs, shuffle=True,
                                      num_workers=4)
            val_loader = DataLoader(val_ds, bs, num_workers=4)

            model_params['input_size'] = train_ds.data.shape[2]  # seq len
            if model_path:
                # use existing model for testing or continuing training
                model = loaded_models[fold]
            else:
                model = ConvNet(model_params).to(compute_device)
            if train:
                # determine lr (either lr from cfg or lr finder)
                if isinstance(lr, list) and len(lr) == nb_folds:
                    # cfg lr is list of lrs per fold
                    fold_lr = lr[fold]
                elif clr or lr_range != '':  # use lr finder
                    min_lr, max_lr = find_lr_bounds(model, train_loader,
                                                    optimizer,
                                                    result_path if fold == 0
                                                    else '', lr_range,
                                                    prefix=f'{fold + 1}_'
                                                    if fold == 0 else '')
                    if opt == 'Adagrad' and clr:
                        max_lr = max(0.1, max_lr)
                    if clr:
                        lrs.append((min_lr, max_lr))
                    else:  # lr finder determined lr for non-clr
                        lrs.append(max_lr)
                    fold_lr = lrs[-1]
                else:
                    fold_lr = lr  # single lr given in cfg

                time_per_step = fit(fold_lr,
                                    model,
                                    train_loader,
                                    val_loader,
                                    optimizer,
                                    max_epochs=epcs,
                                    save=result_path
                                    if len(bs_lst) == 1 else '',
                                    fold=fold)
                train_throughput[i, fold] = time_per_step
                models.append(model.to('cpu'))
            elif test:
                with torch.no_grad():
                    train_result = evaluate(model, train_loader)
                    for key, val in train_result.items():
                        model.train_history[key].append(val)
                    # eval for validataion dataset
                    val_result = evaluate(model, val_loader)
                    for key, val in val_result.items():
                        model.val_history[key].append(val)

                    print(f"loss: {val_result['loss']}, "
                          f"acc: {val_result['acc']}, "
                          f"emp. acc: {val_result['acc_emp']}, "
                          f"sim. acc: {val_result['acc_sim']}")


        # print avg across folds time per step
        print(f'Avg. time-per-step across folds: '
              f'{np.round(train_throughput[i], 2)}s')

        save_sum = os.path.join(result_path,
                                f'val_folds_{timestamp}_{str(bs)}.csv')
        print_model_performance(models, save=save_sum if result_path else '')

        # save config
        if result_path is not None:
            # save cfg
            cfg['hyperparameters']['lr'] = lrs if len(lrs) > 0 else lr
            write_cfg_file(cfg, cfg_path if cfg_path is not None else '',
                           result_path if result_path is not None else '',
                           timestamp)

            # save plot of learning curve
            if train or resume:
                fig_file = 'fig-fold-eval'
                fig_file += f'-{bs}' if len(bs_lst) > 1 else ""
                fig_file += '.png'
                make_fig(plot_folds, [models], (1, 2),
                         save=os.path.join(result_path, fig_file))
            print(f'\nSaved results to {result_path}\n')
        else:
            print(
                f'\nNot saving models and evaluation plots. Please use '
                f'--save and specify a directory if you want to save '
                f'your results!\n')

        # -------------------- attribution study -------------------- #
        if attribution:
            # choose best fold
            fold_eval = [np.min(model.val_history['loss']) for model in
                         models]
            best_fold = np.argmin(fold_eval)
            worst_fold = np.argmax(fold_eval)
            for fold, (_, val_ids) in enumerate(kfold.split(data,
                                                            labels)):
                if fold == best_fold or fold == worst_fold:
                    print(f'Compute/plot attribution scores from fold-'
                          f'{fold + 1}-model')
                    if result_path != '':
                        attr_path = f'{result_path}/attribution_fold' \
                                    f'{fold + 1}'
                        if fold == best_fold:
                            attr_path += '_min_loss'
                        elif fold == worst_fold:
                            attr_path += '_max_loss'
                        if not os.path.exists(attr_path):
                            os.mkdir(attr_path)
                    else:
                        attr_path = ''

                    # get validation data for that fold
                    val_ds = TensorDataset(data[val_ids].copy(),
                                           labels[val_ids])

                    # get net
                    model = models[fold]

                    # get predition scores and order to sort MSAs by scores
                    preds, sort_by_pred = get_sorted_pred_scores(model,
                                                                 val_ds)
                    plot_pred_scores(preds,
                                     save=f'{attr_path}/val_pred_scores.pdf')

                    # correlation of seq len and predictions
                    val_sl = {}  # sorted by pred. score
                    for l, (cl, sort_inds) in enumerate(
                            sort_by_pred.items()):
                        # filter emp/sim msas and sort by prediction score
                        val_sl[cl] = seq_lens[val_ids][val_ds.labels == l][
                            sort_inds]
                    print('Sequence length vs prediction scores')
                    for key in val_sl.keys():
                        mine = MINE()
                        mine.compute_score(preds[key], val_sl[key])
                        corr = np.corrcoef([preds[key], val_sl[key]])

                        print(f'{key}\n\tMIC={mine.mic()}')
                        print(f'\tPearson={corr[0][1]}\n')
                    plot_corr_pred_sl(val_sl, preds,
                                      f'{attr_path}/score_sl.pdf')

                    # get masks to remove padding
                    pad_mask = {}  # sorted by pred. score
                    for l, (cl, sort_inds) in enumerate(
                            sort_by_pred.items()):
                        # filter emp/sim msas and sort by prediction score
                        pad_mask[cl] = val_ds.data[val_ds.labels == l][
                            sort_inds]

                        # sum over channels: 1 = no padding, 0 = padding
                        pad_mask[cl] = pad_mask[cl].sum(
                            axis=1).detach().cpu().numpy()
                        pad_mask[cl] = pad_mask[cl].astype(bool)

                    # get attributions
                    attrs, xins = {}, {}
                    for cl, i in sort_by_pred.items():
                        label = 0 if 'emp' in cl else 1
                        attrs[cl] = np.asarray(
                            [get_attr(msa, model, 'saliency')
                             for msa in
                             val_ds.data[val_ds.labels == label][i]])
                        xins[cl] = np.asarray(
                            [get_attr(msa, model, 'integratedgradients',
                                      multiply_by_inputs=True)
                             for msa in
                             val_ds.data[val_ds.labels == label][i]])

                    # plot site/channel importance
                    # plot_summary(attrs, pad_mask, 'channels', preds,
                    # molecule_type,
                    #              save=f'{attr_path}/channel_attr_preds.pdf')
                    plot_summary(attrs, pad_mask, 'channels', None,
                                 molecule_type,
                                 save=f'{attr_path}/channel_attr.pdf')
                    # plot_summary(attrs, pad_mask, 'sites', preds,
                    # molecule_type,
                    #              save=f'{attr_path}/site_attr_preds.pdf')
                    plot_summary(attrs, pad_mask, 'sites', None,
                                 molecule_type,
                                 save=f'{attr_path}/site_attr.pdf')
                    plot_summary(attrs, pad_mask, 'sites', None,
                                 molecule_type,
                                 save=f'{attr_path}/site_attr_200.pdf',
                                 max_sl=200)
                    plot_summary(attrs, pad_mask, 'sites', None,
                                 molecule_type,
                                 save=f'{attr_path}/site_attr_50.pdf',
                                 max_sl=50)

                    # plot individual saliency maps

                    # indices of n best predicted (decreasing) and
                    # n worst predicted
                    # (increasing) MSAs
                    n = 5
                    select = np.concatenate(
                        (np.arange(n), np.arange(-n, 0)))

                    for l, cl in enumerate(attrs.keys()):
                        # validation data, sorted according to attr. maps
                        msas = val_ds.data[val_ds.labels == l][
                            sort_by_pred[cl]]
                        fnames = fastas[val_ids][val_ds.labels == l][
                            sort_by_pred[cl]]

                        for i, sel in enumerate(select):
                            score = '%.2f' % np.round(preds[cl][sel], 2)

                            sal_map = attrs[cl][sel][pad_mask[cl][sel]]
                            ig_map = xins[cl][sel][pad_mask[cl][sel]]
                            msa = msas[sel][:,
                                  pad_mask[cl][sel]].detach().cpu().numpy()
                            msa_id = fnames[sel].split(".")[0]
                            plot_msa_attr(sal_map, ig_map, msa,
                                          molecule_type,
                                          save=f'{attr_path}/'
                                               f'[{i}]_{msa_id}_{score}_'
                                               f'{cl}.pdf')
    if len(bs_lst) > 1 and result_path is not None:
        np.savetxt(f'{result_path}/step_time.csv',
                   train_throughput, delimiter=',')
        plot_groups_folds(bs_lst, train_throughput,
                          f'{result_path}/step_time.pdf')


if __name__ == '__main__':
    main()

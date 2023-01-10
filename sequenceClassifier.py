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
import sys
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from ConvNet import ConvNet, load_net, compute_device
from attr_methods import get_attr, get_sorted_pred_scores, plot_pred_scores, \
    plot_summary, plot_msa_attr
from preprocessing import TensorDataset, alns_from_fastas, raw_alns_prepro, \
    make_msa_reprs, load_msa_reprs
from plots import plot_folds, plot_hist_quantiles
from stats import get_n_sites_per_msa, get_n_seqs_per_msa
from utils import write_cfg_file, read_cfg_file, merge_fold_hist_dicts, \
    fold_val_dict2csv
from train_eval import fit, eval_per_align, generate_eval_dict, evaluate, \
    evaluate_folds, find_lr_bounds

torch.cuda.empty_cache()

gc.collect()


def main():
    sep_line = '-------------------------------------------------------' \
               '---------'

    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', nargs='*', type=str, required=True,
                        help='Specify the <path/to/> directory(s) containing '
                             'alignments (in fasta format)')
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
    parser.add_argument('-m', '--models', type=str,
                        help='<path/to> directory with trained model(s). '
                             'These models will then be tested on a given data '
                             'set. --cfg, --datasets and --test are '
                             'required for this option.')
    parser.add_argument('--emp', nargs='*', type=int,
                        help='Indicates that given data set has empirical '
                             'alignments for --models option. '
                             'Otherwise they are assumed to be simulated')  # TODO
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
    parser.add_argument('--track_stats', action='store_true',
                        help='Generate a csv file with information about the '
                             'data e.g. the number of sites and the training')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the sites of alignments/pairs in the '
                             'first directory specified')
    parser.add_argument('--plot_stats', action='store_true',
                        help='Generates histograms of number of sites and '
                             'number of sequences of given alignments '
                             '(specified with --datasets)')
    parser.add_argument('--pairs', action='store_true',
                        help='A representation for each pair of sequences in '
                             'an alignment will be used')
    parser.add_argument('--molecule_type', choices=['DNA', 'protein'],
                        help='Specify if you use DNA or protein MSAs')
    parser.add_argument('--continu', action='store_true',
                        help='Continue from given lr-bs-combi within '
                             'default bs-lr-grid.')

    args = parser.parse_args()

    # -------------------- verify arguments usage -------------------- #

    # exclusive usage of -m
    if args.models and any(arg for arg in (args.track_stats, args.shuffle)):
        parser.error('--models cannot be used in combination with '
                     '--shuffle or --track_stats')

    if args.test and (not args.models or not args.datasets):
        parser.error('--test requires --modles and --datasets')

    if args.training and len(args.datasets) != 2:
        parser.error('Training requires exactly 2 datasets: one directory '
                     'containing empirical alignments and one with simulated '
                     'alignments.')

    if args.plot_stats and not all(arg for arg in (args.datasets, args.cfg,
                                                   args.save)):
        parser.error('--plot_stats requires --datasets, --cfg and --save')

    if args.molecule_type:
        molecule_type = args.molecule_type
    else:  # by default protein MSAs are expected
        molecule_type = 'protein'

    if args.plot_stats:
        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        cfg = read_cfg_file(args.cfg)
        path = args.save

        if not os.path.exists(path):
            os.makedirs(path)

        if len(args.datasets) == 2:
            emp_fasta_path, sim_fasta_path = args.datasets

            emp_alns = alns_from_fastas(emp_fasta_path, False,
                                        cfg['data']['nb_alignments'],
                                        molecule_type=molecule_type)[0]
            sim_alns = alns_from_fastas(sim_fasta_path, False,
                                        cfg['data']['nb_alignments'],
                                        molecule_type=molecule_type)[0]

            plot_hist_quantiles((get_n_sites_per_msa(emp_alns),
                                 get_n_sites_per_msa(sim_alns)),
                                ('empirical', 'simulated'),
                                ['Number of sites'] * 2,
                                path=f'{path}/hist-sites-{timestamp}.png')
            plot_hist_quantiles((get_n_seqs_per_msa(emp_alns),
                                 get_n_seqs_per_msa(sim_alns)),
                                ('empirical', 'simulated'),
                                ['Number of sequences'] * 2,
                                path=f'{path}/hist_nb_seqs-{timestamp}.png')
        else:
            data_dirs = args.datasets
            for i, dir in enumerate(data_dirs):
                cfg = read_cfg_file(args.cfg)

                alns = alns_from_fastas(dir, False,
                                        cfg['data']['nb_alignments'],
                                        molecule_type=molecule_type)[0]

                plot_hist_quantiles([get_n_sites_per_msa(alns)],
                                    xlabels=['Number of sites'],
                                    path=f'{path}/hist-sites-{timestamp}'
                                         f'-{i}.png')
                plot_hist_quantiles([get_n_seqs_per_msa(alns)],
                                    xlabels=['Number of sequences'],
                                    path=f'{path}/hist_nb_seqs-{timestamp}'
                                         f'-{i}.png')

    if args.training or args.test or args.attr:
        emp_fasta_path, sim_fasta_path = args.datasets
        shuffle = args.shuffle
        pairs = args.pairs
        result_path = args.save if args.save else None

        if args.models and not args.cfg:
            model_path = args.models
            if os.path.isdir(model_path):
                cfg_path = model_path + '/cfg.json'
            elif os.path.isfile(model_path):
                cfg_path = os.path.dirname(model_path) + '/cfg.json'
        else:
            cfg_path = args.cfg

        cfg = read_cfg_file(cfg_path)

        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        if result_path is not None and not args.models:
            if not result_path.split('/')[-1].startswith('cnn-'):
                # create unique subdir for the model(s)
                result_path += '/cnn_'
                result_path += sim_fasta_path.split('/')[-1] + '_k'
                result_path += str(cfg['conv_net_parameters']['kernel_size'])
                result_path += '_' + str(timestamp)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
        elif args.models:
            result_path = model_path

        if not os.path.exists(emp_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    emp_fasta_path)
        if not os.path.exists(sim_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    sim_fasta_path)

        if not os.path.exists(sim_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    cfg_path)

        # -------------------- cfgure parameters -------------------- #
        if args.test or args.models:
            cfg['data']['nb_alignments'] = None
        else:
            if cfg['data']['nb_alignments'] != '':
                cfg['data']['nb_alignments'] = int(cfg['data']['nb_alignments'])
        model_params = cfg['conv_net_parameters']

        # hyperparameters
        batch_size, epochs, lr, opt, nb_folds = cfg[
            'hyperparameters'].values()

        if opt == 'Adagrad':
            optimizer = torch.optim.Adagrad
        elif opt == 'SGD':
            optimizer = torch.optim.SGD
        elif opt == 'Adam':
            optimizer = torch.optim.Adam
        else:
            raise ValueError(
                'Please specify either Adagrad or SGD as optimizer')

        # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
        kfold = StratifiedKFold(nb_folds, shuffle=True, random_state=42)

        seed = 42  # + np.random.randint(100) if args.continu else 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f'\nRandom seed: {seed}\n')

        # ------------------------- data preparation ------------------------- #

        first_input_file = os.listdir(emp_fasta_path)[0]
        fmts = ['.fa', '.fasta', '.phy']
        if np.any([first_input_file.endswith(f) for f in fmts]):
            alns, fastas, data_dict = raw_alns_prepro(
                [emp_fasta_path, sim_fasta_path],
                cfg['data']['nb_alignments'],
                cfg['data']['nb_sites'],
                shuffle=shuffle,
                molecule_type=molecule_type)
            fastas_emp, fastas_sim = fastas.copy()

            # update config with info from loaded data
            for key, val in data_dict.items():
                cfg['data'][key] = val

            if molecule_type == 'protein':
                for i in range(len(alns)):
                    # remove first 2 sites
                    alns[i] = [[seq[2:] for seq in aln] for aln in alns[i]]

            emp_alns, sim_alns = make_msa_reprs(alns, fastas,
                                                cfg['data']['nb_sites'],
                                                cfg['data']['padding'],
                                                pairs,
                                                csv_path=(
                                                    f'{result_path}/'
                                                    f'alns_stats.csv'
                                                    if args.track_stats
                                                    else None),
                                                molecule_type=molecule_type
                                                )
            del alns, fastas
        elif first_input_file.endswith('.csv'):  # msa representations given
            emp_alns, fastas_emp = load_msa_reprs(emp_fasta_path, pairs,
                                                  cfg['data'][
                                                      'nb_alignments'])
            sim_alns, fastas_sim = load_msa_reprs(emp_fasta_path, pairs,
                                                  cfg['data']['nb_alignments'])

        data_size = sys.getsizeof(emp_alns) + sys.getsizeof(sim_alns)
        data_size += sys.getsizeof(fastas_emp) + sys.getsizeof(fastas_sim)
        print(f'Preloaded data uses : '
              f'{data_size / 10 ** 9}GB\n')

        n_emp_alns, n_sim_alns = len(emp_alns), len(sim_alns)
        data = np.asarray(emp_alns + sim_alns, dtype='float32')
        labels = np.concatenate((np.zeros(n_emp_alns), np.ones(n_sim_alns)))

        if args.track_stats:
            emp_alns_dict = {fastas_emp[i]: emp_alns[i] for i in
                             range(len(emp_alns))}
            sim_alns_dict = {fastas_sim[i]: sim_alns[i] for i in
                             range(len(sim_alns))}

        if args.models:
            if os.path.isdir(model_path):
                files = os.listdir(model_path)
                model_paths = [f'{model_path}/{file}' for file in files
                               if file.endswith('.pth')]
            elif os.path.isfile(model_path):
                model_paths = [model_path]
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        model_path)

        if (batch_size == '' and lr == '') or (
                args.continu and not args.models):
            param_space = {
                'batch_size': [32, 64, 128],
                'lr': [0.1, 0.01, 0.001, 0.0001]
            }
        else:
            param_space = {
                'batch_size': batch_size
                if isinstance(batch_size, list)
                else [batch_size],
                'lr': lr if isinstance(lr, list) else [lr],
            }

        lrs, bss = np.meshgrid(param_space['lr'], param_space['batch_size'])
        lrs, bss = lrs.flatten(), bss.flatten()
        if args.continu and not args.models:
            continu_ind = [i for i, (lr_c, bs_c) in enumerate(zip(lrs, bss))
                           if lr_c == lr and bs_c == batch_size][0] + 1
            lrs, bss = lrs[continu_ind:], bss[continu_ind:]
            print(lrs)
            print(bss)

        # -------------------- k-fold cross validation -------------------- #

        print(f'\nCompute device: {compute_device}\n')

        if 'val_acc' in cfg.keys():
            best = {'val_acc': cfg['val_acc']}
            print(best)
        else:
            best = {'val_acc': -np.inf}

        # explore hyper-parameter-space
        for bs, lr in zip(bss, lrs):

            print(sep_line)
            print(f"Current Parameters:\n\tBatch size: "
                  f"{bs}\n\tLearning rate: {lr}")
            print(sep_line + '\n')

            models = []
            dfs = []

            for fold, (train_ids, val_ids) in enumerate(kfold.split(data,
                                                                    labels)):
                print(f'FOLD {fold + 1}')
                print(sep_line)

                # splitting dataset by alignments
                print("Building training and validation dataset ...")
                start = time.time()
                train_ds = TensorDataset(data[train_ids], labels[train_ids],
                                         pairs)
                val_ds = TensorDataset(data[val_ids], labels[val_ids], pairs)
                print(f'Finished after {round(time.time() - start, 2)}s\n')

                bs = int(bs)
                train_loader = DataLoader(train_ds, bs, shuffle=True,
                                          num_workers=4)
                val_loader = DataLoader(val_ds, bs, num_workers=4)

                # generate model
                model_params['input_size'] = train_ds.data.shape[2]  # seq len
                model_params['nb_chnls'] = train_ds.data.shape[
                    1]  # 1 channel per amino acid

                if args.models:
                    state = 'train' if args.continu else 'eval'
                    # load model if testing or if continuing training
                    model = load_net(model_paths[fold]
                                     if len(model_paths) == nb_folds
                                     else model_paths[0],
                                     model_params,
                                     state=state).to(compute_device)
                else:
                    model = ConvNet(model_params).to(compute_device)

                if args.training:
                    min_lr, max_lr = find_lr_bounds(model, train_loader,
                                                    optimizer, result_path,
                                                    prefix=f'{fold + 1}_')
                    fit((min_lr, max_lr), model, train_loader, val_loader,
                        optimizer, max_epochs=epochs)
                elif args.test:
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

                models.append(model.to('cpu'))

                if args.track_stats:
                    emp_train_ids = train_ids[:(len(train_ds) // 2)]
                    sim_train_ids = train_ids[(len(train_ds) // 2):]
                    sim_train_ids -= n_emp_alns
                    emp_pred_tr, sim_pred_tr = eval_per_align(model, emp_alns,
                                                              sim_alns,
                                                              fastas_emp,
                                                              fastas_sim,
                                                              emp_train_ids)
                    emp_pred_va, sim_pred_va = eval_per_align(model, emp_alns,
                                                              sim_alns,
                                                              fastas_emp,
                                                              fastas_sim,
                                                              sim_train_ids)

                    df = generate_eval_dict(fold, emp_pred_tr, sim_pred_tr,
                                            emp_pred_va, sim_pred_va,
                                            emp_alns_dict, sim_alns_dict)
                    dfs.append(df)

            train_hist_folds, val_hist_folds = merge_fold_hist_dicts(
                [model.train_history for model in models],
                [model.val_history for model in models])
            # get acc., emp. acc., sim. acc, and loss of epoch with max. acc.
            val_folds = evaluate_folds(val_hist_folds, nb_folds)
            # evaluation for hyper-parameter selection
            val_acc = np.mean(val_folds['acc'])

            # save validation acc./loss for each lr-bs-config
            if result_path is not None:
                fold_val_dict2csv(val_folds,
                                  f'{result_path}/'
                                  f'val_folds_{timestamp}_{str(bs)}_'
                                  f'{str(lr)}.csv')
            if val_acc >= best['val_acc']:
                best['val_acc'] = val_acc
                best['models'] = models
                best['dfs'] = dfs
                best['batch_size'] = bs
                best['lr'] = lr
                print(sep_line)
                print(f'New best lr: {lr} and bs: {bs}')

                # ------------------------- results ------------------------- #

                # plot learning curve
                for fold, model in enumerate(models):
                    if result_path is not None:
                        model.save(f'{result_path}/model-fold-{fold + 1}.pth')
                        model.plot(f'{result_path}/fig-fold-{fold + 1}.png')
                    else:
                        model.plot()

                # save cfg
                cfg['hyperparameters']['batch_size'] = best['batch_size']
                cfg['hyperparameters']['lr'] = best['lr']
                cfg['val_acc'] = best['val_acc']
                write_cfg_file(cfg,
                               result_path if result_path is not None else '',
                               cfg_path if cfg_path is not None else '',
                               timestamp)

                # print/save overall fold results
                if result_path is not None:
                    # save plot of learning curve
                    if args.training:
                        plot_folds(train_hist_folds, val_hist_folds,
                                   path=f'{result_path}/fig-fold-eval.png')
                    print(f'\nSaved models and evaluation plots to '
                          f'{result_path}\n')
                else:
                    print(
                        f'\nNot saving models and evaluation plots. Please use '
                        f'--save and specify a directory if you want to save '
                        f'your results!\n')

                # save statistics of training process
                if len(best['dfs']) > 0 and result_path is not None:
                    df_c = pd.concat(best['dfs'])
                    csv_string = df_c.to_csv(index=False)
                    with open(result_path + '/aln_train_eval.csv', 'w') as file:
                        file.write(csv_string)

                    print(f'Saved statistics to {result_path}'
                          f'/aln_train_eval.csv ...\n')

                # print k-fold cross-validation evaluation
                print(sep_line)
                print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
                fold_eval = [np.max(model.val_history['acc']) for model in
                             best['models']]
                for i, acc in enumerate(fold_eval):
                    print(f'\tFold {(i + 1)}: {acc} %')
                print(f'Average: {np.mean(fold_eval)} %')
                print(sep_line)
        print(f"\nBest hyper-parameters:\n\tBatch size: {best['batch_size']}\n"
              f"\tLearning rate: {best['lr']}")

        if args.attr:

            # -------------------- attribution study -------------------- #
            # choose best fold
            best_fold = np.argmax(fold_eval)
            for fold in range(nb_folds):
                print(f'Compute/plot attribution scores from fold-'
                      f'{fold + 1}-model')
                if result_path != '':
                    attr_path = f'{result_path}/attribution_fold{fold + 1}'
                    if fold == best_fold:
                        attr_path += '_best'
                    if not os.path.exists(attr_path):
                        os.mkdir(attr_path)
                else:
                    attr_path = ''

                # get validation data for that fold
                val_ids = [val_ids
                           for i, (_, val_ids) in enumerate(kfold.split(data,
                                                                        labels))
                           if i == fold][0]

                val_ds = TensorDataset(data[val_ids].copy(), labels[val_ids])

                # get net
                model = best['models'][fold]

                # get predition scores and order to sort MSAs by scores
                preds, sort_by_pred = get_sorted_pred_scores(model, val_ds)
                plot_pred_scores(preds, save=f'{attr_path}/val_pred_scores.pdf')

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

                    for i, sel in enumerate(select):
                        score = '%.2f' % np.round(preds[cl][sel], 2)

                        sal_map = attrs[cl][sel][pad_mask[cl][sel]]
                        ig_map = xins[cl][sel][pad_mask[cl][sel]]
                        msa = msas[sel][:, pad_mask[cl][sel]].detach().cpu().numpy()

                        plot_msa_attr(sal_map, ig_map, msa, molecule_type,
                                      save=f'{attr_path}/[{i}]_{score}_{cl}.pdf')


if __name__ == '__main__':
    main()

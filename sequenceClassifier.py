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
from preprocessing import TensorDataset, alns_from_fastas, raw_alns_prepro, \
    make_msa_reprs, load_msa_reprs
from plots import plot_folds, plot_hist_quantiles
from stats import get_n_sites_per_msa, get_n_seqs_per_msa
from utils import write_cfg_file, read_cfg_file, merge_fold_hist_dicts
from train_eval import fit, eval_per_align, generate_eval_dict, evaluate, \
    evaluate_folds

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
    parser.add_argument('-m', '--models', type=str,
                        help='<path/to> directory with trained model(s). '
                             'These models will then be tested on a given data '
                             'set. --cfg, --datasets and --test are '
                             'required for this option.')
    parser.add_argument('--real', nargs='*', type=int,
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
            real_fasta_path, sim_fasta_path = args.datasets

            real_alns = alns_from_fastas(real_fasta_path, False,
                                         cfg['data']['nb_alignments'],
                                         molecule_type=molecule_type)[0]
            sim_alns = alns_from_fastas(sim_fasta_path, False,
                                        cfg['data']['nb_alignments'],
                                        molecule_type=molecule_type)[0]

            plot_hist_quantiles((get_n_sites_per_msa(real_alns),
                                 get_n_sites_per_msa(sim_alns)),
                                ('empirical', 'simulated'),
                                ['Number of sites'] * 2,
                                path=f'{path}/hist-sites-{timestamp}.png')
            plot_hist_quantiles((get_n_seqs_per_msa(real_alns),
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

    if args.training or args.test:
        real_fasta_path, sim_fasta_path = args.datasets
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

        if not os.path.exists(real_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    real_fasta_path)
        if not os.path.exists(sim_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    sim_fasta_path)

        if not os.path.exists(sim_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    cfg_path)

        # -------------------- cfgure parameters -------------------- #

        if args.test or args.models:
            cfg['data']['nb_alignments'] = None

        model_params = cfg['conv_net_parameters']

        # hyperparameters
        batch_size, epochs, lr, opt, nb_folds = cfg[
            'hyperparameters'].values()

        if opt == 'Adagrad':
            optimizer = torch.optim.Adagrad
        elif opt == 'SGD':
            optimizer = torch.optim.SGD
        else:
            raise ValueError(
                'Please specify either Adagrad or SGD as optimizer')

        # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
        kfold = StratifiedKFold(nb_folds, shuffle=True, random_state=42)

        seed = 42 + np.random.randint(100) if args.continu else 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f'\nRandom seed: {seed}\n')

        # ------------------------- data preparation ------------------------- #

        first_input_file = os.listdir(real_fasta_path)[0]
        fmts = ['.fa', '.fasta', '.phy']
        if np.any([first_input_file.endswith(f) for f in fmts]):
            alns, fastas, data_dict = raw_alns_prepro(
                [real_fasta_path, sim_fasta_path],
                cfg['data']['nb_alignments'],
                cfg['data']['nb_sites'],
                shuffle=shuffle,
                molecule_type=molecule_type)
            fastas_real, fastas_sim = fastas.copy()

            # update config with info from loaded data
            for key, val in data_dict.items():
                cfg['data'][key] = val

            if molecule_type == 'protein':
                for i in range(len(alns)):
                    # remove first 2 sites
                    alns[i] = [[seq[2:] for seq in aln] for aln in alns[i]]

            real_alns, sim_alns = make_msa_reprs(alns, fastas,
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
            real_alns, fastas_real = load_msa_reprs(real_fasta_path, pairs,
                                                    cfg['data'][
                                                        'nb_alignments'])
            sim_alns, fastas_sim = load_msa_reprs(real_fasta_path, pairs,
                                                  cfg['data']['nb_alignments'])

        data_size = sys.getsizeof(real_alns) + sys.getsizeof(sim_alns)
        data_size += sys.getsizeof(fastas_real) + sys.getsizeof(fastas_sim)
        print(f'Preloaded data uses : '
              f'{data_size / 10 ** 9}GB\n')

        n_real_alns, n_sim_alns = len(real_alns), len(sim_alns)
        data = np.asarray(real_alns + sim_alns, dtype='float32')
        labels = np.concatenate((np.zeros(n_real_alns), np.ones(n_sim_alns)))

        if args.track_stats:
            real_alns_dict = {fastas_real[i]: real_alns[i] for i in
                              range(len(real_alns))}
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

        if (batch_size == '' and lr == '') or (args.continu and not args.models):
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
                                          num_workers=8)
                val_loader = DataLoader(val_ds, bs, num_workers=8)

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
                        fit(lr, model, train_loader, val_loader, optimizer,
                            max_epochs=epochs)
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
                    real_train_ids = train_ids[:(len(train_ds) // 2)]
                    sim_train_ids = train_ids[(len(train_ds) // 2):]
                    sim_train_ids -= n_real_alns
                    real_pred_tr, sim_pred_tr = eval_per_align(model, real_alns,
                                                               sim_alns,
                                                               fastas_real,
                                                               fastas_sim,
                                                               real_train_ids)
                    real_pred_va, sim_pred_va = eval_per_align(model, real_alns,
                                                               sim_alns,
                                                               fastas_real,
                                                               fastas_sim,
                                                               sim_train_ids)

                    df = generate_eval_dict(fold, real_pred_tr, sim_pred_tr,
                                            real_pred_va, sim_pred_va,
                                            real_alns_dict, sim_alns_dict)
                    dfs.append(df)

            train_hist_folds, val_hist_folds = merge_fold_hist_dicts(
                [model.train_history for model in models],
                [model.val_history for model in models])
            # get acc., emp. acc., sim. acc, and loss of epoch with min. loss
            val_folds = evaluate_folds(val_hist_folds, nb_folds)
            # evaluation for hyper-parameter selection
            val_acc = np.mean(val_folds['acc'])

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
                    np.savetxt(f'{result_path}/fold-validation-{timestamp}.csv',
                               np.asarray(list(val_folds.values())).T,
                               header=','.join(list(val_folds.keys())),
                               comments='', delimiter=',')
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
                fold_eval = [model.val_history['acc'][-1] for model in
                             best['models']]
                for i, acc in enumerate(fold_eval):
                    print(f'\tFold {(i + 1)}: {acc} %')
                print(f'Average: {np.mean(fold_eval)} %')
                print(sep_line)

        print(f"\nBest hyper-parameters:\n\tBatch size: {best['batch_size']}\n"
              f"\tLearning rate: {best['lr']}")


if __name__ == '__main__':
    main()

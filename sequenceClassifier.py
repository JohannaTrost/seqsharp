"""Program to classify empirical and simulated sequence alignments

Allows to train and evaluate a (convolutional) neural network or use a
trained network. Plots can be generated to provide information about a
dataset with alignments or the performance of a network.

Please execute 'python sequenceClassifier.py --help' to view all the options
"""

import argparse
import errno
import itertools
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
from utils import write_cfg_file, read_cfg_file
from train_eval import fit, eval_per_align, generate_eval_dict, evaluate

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
    parser.add_argument('--real', action='store_true',
                        help='Indicates that given data set has empirical '
                             'alignments for --models option. '
                             'Otherwise they are assumed to be simulated')
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

    args = parser.parse_args()

    # -------------------- verify arguments usage -------------------- #

    # exclusive usage of -m
    if (args.models and any(arg for arg in (args.training, args.track_stats,
                                            args.shuffle))):
        parser.error('--models cannot be used in combination with --training, '
                     '--shuffle or --track_stats')

    if (args.datasets and not args.cfg) or (
            args.cfg and not args.datasets):
        parser.error('--datasets and --cfg have to be used together')

    if args.test and (not args.models or not args.datasets):
        parser.error('--test requires --modles and --datasets')

    if args.test and len(args.datasets) > 1:
        parser.error('--test only takes one set of alignments '
                     '(one directory specified with --datasets)')

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

    if args.training:
        real_fasta_path, sim_fasta_path = args.datasets
        shuffle = args.shuffle
        pairs = args.pairs
        result_path = args.save if args.save else None
        cfg_path = args.cfg

        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

        if result_path is not None:
            # create unique subdir for the model(s)
            result_path = result_path + '/cnn-' + str(timestamp)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

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

        cfg = read_cfg_file(cfg_path)

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

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # k-fold validator
        kfold = StratifiedKFold(nb_folds, shuffle=True, random_state=42)

        # ------------------------- data preparation ------------------------- #

        first_input_file = os.listdir(real_fasta_path)[0]
        if first_input_file.endswith('.fasta'):
            alns, fastas, cfg['data'] = raw_alns_prepro([real_fasta_path,
                                                         sim_fasta_path],
                                                        cfg['data'],
                                                        shuffle=shuffle,
                                                        molecule_type=molecule_type)
            fastas_real, fastas_sim = fastas.copy()

            real_alns, sim_alns = make_msa_reprs(alns, fastas, cfg['data'],
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
                                                    cfg['data']['nb_alignments'])
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

        if batch_size == '' and lr == '':
            param_space = {
                'batch_size': [32, 64, 128, 256, 512],
                'lr': [0.1, 0.01, 0.001, 0.0001]
            }
        else:
            param_space = {
                'batch_size': batch_size
                if isinstance(batch_size, list)
                else [batch_size],
                'lr': lr if isinstance(lr, list) else [lr],
            }

        param_combis = list(itertools.product(
            *[np.arange(len(param_space['batch_size'])),
              np.arange(len(param_space['lr']))]))

        # -------------------- k-fold cross validation -------------------- #

        print(f'\nCompute device: {compute_device}\n')

        best = {'param_eval': {
            'val_acc': -np.inf,
            'train_acc': -np.inf,
            'val_std': -np.inf,
            'train_std': -np.inf,
        }}

        # explore hyper-parameter-space
        for bs_ind, lr_ind in param_combis:

            print(sep_line)
            print(f"Current Parameters:\n\tBatch size: "
                  f"{param_space['batch_size'][bs_ind]}\n"
                  f"\tLearning rate: {param_space['lr'][lr_ind]}")
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

                train_loader = DataLoader(train_ds,
                                          param_space['batch_size'][bs_ind],
                                          shuffle=True)
                val_loader = DataLoader(val_ds,
                                        param_space['batch_size'][bs_ind])

                # generate model
                model_params['input_size'] = train_ds.data.shape[2]  # seq len
                model_params['nb_chnls'] = train_ds.data.shape[
                    1]  # 1 channel per amino acid

                model = ConvNet(model_params)
                model = model.to(compute_device)

                # train and validate model
                fit(epochs, param_space['lr'][lr_ind],
                    model, train_loader, val_loader, optimizer)

                models.append(model.to('cpu'))

                if args.track_stats:
                    real_pred_tr, sim_pred_tr = eval_per_align(model, real_alns,
                                                               sim_alns,
                                                               fastas_real,
                                                               fastas_sim,
                                                               train_ids)
                    real_pred_va, sim_pred_va = eval_per_align(model, real_alns,
                                                               sim_alns,
                                                               fastas_real,
                                                               fastas_sim,
                                                               val_ids)

                    df = generate_eval_dict(fold, real_pred_tr, sim_pred_tr,
                                            real_pred_va, sim_pred_va,
                                            real_alns_dict, sim_alns_dict)
                    dfs.append(df)

            # evaluation index summing accuracies and
            # substracting standard deviation
            param_eval = {}
            param_eval['val_acc'] = np.mean([model.val_history[-1]['acc']
                                             for model in models])
            param_eval['train_acc'] = np.mean([model.train_history[-1]['acc']
                                               for model in models])
            param_eval['val_std'] = np.std([model.val_history[-1]['acc']
                                            for model in models])
            param_eval['train_std'] = np.std([model.train_history[-1]['acc']
                                              for model in models])

            if param_eval['val_acc'] > best['param_eval']['val_acc']:
                best['param_eval'] = param_eval
                best['models'] = models
                best['dfs'] = dfs
                best['batch_size'] = param_space['batch_size'][bs_ind]
                best['lr'] = param_space['lr'][lr_ind]

        # -------------------------- treat results -------------------------- #

        # save cfg
        cfg['hyperparameters']['batch_size'] = best['batch_size']
        cfg['hyperparameters']['lr'] = best['lr']

        write_cfg_file(cfg,
                       result_path if result_path is not None else '',
                       cfg_path if cfg_path is not None else '',
                       timestamp)

        print(f"\nBest hyper-parameters:\n\tBatch size: {best['batch_size']}\n"
              f"\tLearning rate: {best['lr']}")

        # print/save overall fold results
        plot_folds([model.train_history for model in best['models']],
                   [model.val_history for model in best['models']],
                   path=f'{result_path}/fig-fold-eval.png'
                   if result_path is not None else None)

        # save plots for each fold
        for fold, model in enumerate(best['models']):
            if result_path is not None:
                model.save(f'{result_path}/model-fold-{fold + 1}.pth')
                model.plot(f'{result_path}/fig-fold-{fold + 1}.png')
            else:
                model.plot()

        # save statistics of training process
        if len(best['dfs']) > 0 and result_path is not None:
            df_c = pd.concat(best['dfs'])
            csv_string = df_c.to_csv(index=False)
            with open(result_path + '/aln_train_eval.csv', 'w') as file:
                file.write(csv_string)

            print(f'Saved statistics to {result_path}/aln_train_eval.csv ...\n')

        # print k-fold cross-validation evaluation
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
        print(
            '----------------------------------------------------------------')

        fold_eval = [model.val_history[-1]['acc'] for model in best['models']]

        for i, acc in enumerate(fold_eval):
            print(f'\tFold {(i + 1)}: {acc} %')

        print(f'Average: {np.mean(fold_eval)} %')

        if result_path is not None:
            print(f'\nSaved models and evaluation plots to {result_path}\n')
        else:
            print(f'\nNot saving models and evaluation plots. Please use '
                  f'--save and specify a directory if you want to save your '
                  f'results!\n')

    if args.test:
        model_path = args.models
        fasta_path = args.datasets[0]
        pairs = args.pairs
        cfg_path = args.cfg
        is_real = args.real

        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    fasta_path)

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    cfg_path)

        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            model_paths = [f'{model_path}/{file}' for file in files
                           if file.endswith('.pth')]
        elif os.path.isfile(model_path):
            model_paths = [model_path]
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    model_path)

        # -------------------- cfgure parameters -------------------- #

        cfg = read_cfg_file(cfg_path)

        model_params = cfg['conv_net_parameters']

        batch_size = cfg['hyperparameters']['batch_size']

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # ------------------------- data preparation ------------------------- #

        first_input_file = os.listdir(fasta_path)[0]
        if first_input_file.endswith('.fasta'):
            alns, fastas, cfg['data'] = raw_alns_prepro([fasta_path],
                                                        cfg['data'],
                                                        quantiles=[
                                                            False])

            alns = make_msa_reprs(alns, fastas, cfg['data'], pairs,
                                  molecule_type=molecule_type)[0]

        elif first_input_file.endswith('.csv'):  # msa representations given
            alns, fastas = load_msa_reprs(fasta_path, pairs)

        # ------------------ load and evaluate model(s) ------------------- #

        print("Building validation dataset ...\n")

        if is_real:
            ds = TensorDataset(alns, [])
        else:
            ds = TensorDataset([], alns)

        loader = DataLoader(ds, batch_size)

        model_params['input_size'] = ds.data.shape[2]  # seq len
        model_params['nb_chnls'] = ds.data.shape[1]  # 1 channel per aa

        accs = []
        accs_after_train = []

        for i, path in enumerate(model_paths):
            # generate model
            model = load_net(path, model_params, state='eval')
            with torch.no_grad():
                result = evaluate(model, loader)

                accs.append(result['acc'])
                accs_after_train.append(model.val_history[-1]['acc'])

                print(f"model {i + 1}, accuracy: {np.round(result['acc'], 4)}"
                      f"(val. of trained model {np.round(accs_after_train[i], 4)})")

        if len(accs) > 1:
            print(f'\nAverage: {np.round(np.mean(accs), 4)}, '
                  f'Standard deviation: {np.round(np.std(accs), 4)}\n'
                  f'Average acc. after training: '
                  f'{np.round(np.mean(accs_after_train), 4)}, '
                  f'Standard deviation: '
                  f'{np.round(np.std(accs_after_train), 4)}')


if __name__ == '__main__':
    main()

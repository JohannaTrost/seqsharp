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
import psutil
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from ConvNet import ConvNet
from preprocessing import TensorDataset, data_prepro, alns_from_fastas
from plots import plot_folds, plot_hist_quantiles
from stats import get_nb_sites, nb_seqs_per_alns
from utils import write_config_file, read_config_file
from train_eval import fit, eval_per_align, generate_eval_dict

process = psutil.Process(os.getpid())

torch.cuda.empty_cache()

gc.collect()

compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
compute_device = torch.device("cpu")


def main():
    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', type=str, nargs=2,
                        help='Specify the <path/to/> the empirical alignments '
                             'directory and the <path/to/> the simulated '
                             'alignments directory (the files need to be in '
                             'fasta format). A network will be trainined to '
                             'classify emirical and simulated alignments')
    parser.add_argument('-c', '--config', type=str,
                        help='<path/to> config file (.json) or directory '
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
                             'data e.g. the number of sites')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the sites of alignments in the first '
                             'directory specified')
    parser.add_argument('-m', '--models', type=str, nargs='*',
                        help='<path/to> directory with trained model(s). '
                             'Plot the performance of the model(s). Multiple '
                             'directories can be specified to compare '
                             'different trainings. Use <-s> option to save '
                             'the plots')
    parser.add_argument('--plot_stats', action='store_true',
                        help='')
    parser.add_argument('--pairs', action='store_true',
                        help='A representation for each pair of sequences in '
                             'an alignment will be used')

    args = parser.parse_args()

    # exclusive usage of -m (works only with -s)
    if (args.models and any(arg is not None for arg in
                           (args.training, args.config, args.track_stats))):
        parser.error('--models cannot be used in combination with --training '
                     '--config or --track_stats')
    if (args.training and not args.config) or (args.config and not args.training):
        parser.error('--training and --config have to be used together')

    if args.plot_stats:
        real_fasta_path, sim_fasta_path = args.training
        config = read_config_file(args.config)

        real_alns = alns_from_fastas(real_fasta_path,
                                     config['data']['min_seqs_per_align'],
                                     config['data']['max_seqs_per_align'],
                                     config['data']['nb_alignments'])[0]
        sim_alns = alns_from_fastas(sim_fasta_path,
                                     config['data']['min_seqs_per_align'],
                                     config['data']['max_seqs_per_align'],
                                     config['data']['nb_alignments'])[0]

        path = args.save
        timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

        plot_hist_quantiles((get_nb_sites(real_alns), get_nb_sites(sim_alns)),
                            ('empirical', 'simulated'),
                            ['Number of sites'] * 2,
                            path=f'{path}/hist-sites-{timestamp}.png')
        plot_hist_quantiles((nb_seqs_per_alns(real_alns),
                             nb_seqs_per_alns(sim_alns)),
                            ('empirical', 'simulated'),
                            ['Number of sequences'] * 2,
                            path=f'{path}/hist_nb_seqs-{timestamp}.png')


    elif args.training:
        real_fasta_path, sim_fasta_path = args.training
        shuffle = args.shuffle
        pairs = args.pairs
        result_path = args.save if args.save else None
        config_path = args.config

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
                                    config_path)



        # -------------------- configure parameters -------------------- #

        config = read_config_file(config_path)

        model_params = config['conv_net_parameters']

        # hyperparameters
        batch_size, epochs, lr, opt, nb_folds = config['hyperparameters'].values()

        if opt == 'Adagrad':
            optimizer = torch.optim.Adagrad
        elif opt == 'SGD':
            optimizer = torch.optim.SGD
        else:
            raise ValueError('Please specify either Adagrad or SGD as optimizer')

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # k-fold validator
        kfold = KFold(nb_folds, shuffle=True, random_state=42)

        # ------------------------- data preparation ------------------------- #

        data = data_prepro(real_fasta_path, sim_fasta_path,
                           config['data'], pairs,
                           csv_path=(f'{result_path}/alns_stats.csv'
                                     if args.track_stats else None))

        real_alns, sim_alns, fastas_real, fastas_sim, config['data'] = data

        print(f'preloaded data uses : '
              f'{sys.getsizeof(data) / 10**9}GB')

        if args.track_stats:
            real_alns_dict = {fastas_real[i]: real_alns[i] for i in
                              range(len(real_alns))}
            sim_alns_dict = {fastas_sim[i]: sim_alns[i] for i in
                             range(len(sim_alns))}

        # ------------------------- save config ------------------------- #

        write_config_file(config,
                          result_path if result_path is not None else '',
                          config_path,
                          timestamp)

        # -------------------- k-fold cross validation -------------------- #

        # init for evaluation
        train_history_folds = []
        val_history_folds = []
        fold_eval = []
        dfs = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(real_alns)):

            print(f'FOLD {fold + 1}')
            print(
                '----------------------------------------------------------------')

            # splitting dataset by alignments
            print("Building training and validation dataset ...")
            start = time.time()
            train_ds = TensorDataset([real_alns[i] for i in train_ids],
                                     [sim_alns[i] for i in train_ids],
                                     shuffle, pairs)
            val_ds = TensorDataset([real_alns[i] for i in val_ids],
                                   [sim_alns[i] for i in val_ids],
                                   shuffle, pairs)
            print(f'Finished after {round(time.time() - start, 2)}s\n')

            train_loader = DataLoader(train_ds, batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size)

            print(f'datasets and loaders : '
                  f'{(sys.getsizeof(train_ds) + sys.getsizeof(val_ds) + sys.getsizeof(train_loader) + sys.getsizeof(val_loader)) / 10**9}GB')

            # generate model
            model_params['input_size'] = train_ds.data.shape[2]  # seq len
            model_params['nb_chnls'] = train_ds.data.shape[1]  # 1 channel per amino acid

            model = ConvNet(model_params)
            model = model.to(compute_device)

            # train and validate model
            fit(epochs, lr, model, train_loader, val_loader, optimizer)

            train_history_folds.append(model.train_history)
            val_history_folds.append(model.val_history)
            fold_eval.append(val_history_folds[fold][-1]['acc'])

            # saving the model and results
            print('\nTraining process has finished.')

            if result_path is not None:
                model.save(f'{result_path}/model-fold-{fold + 1}.pth')
                model.plot(f'{result_path}/fig-fold-{fold + 1}.png')

                print(f'Saved model and evaluation plot to {result_path} ...\n')
            else:
                model.plot()

            if args.track_stats:
                real_pred_tr, sim_pred_tr = eval_per_align(model, real_alns,
                                                           sim_alns,
                                                           fastas_real,
                                                           fastas_sim,
                                                           train_ids)
                real_pred_va, sim_pred_va = eval_per_align(model, real_alns,
                                                           sim_alns, fastas_real,
                                                           fastas_sim, val_ids)
                df = generate_eval_dict(fold, real_pred_tr, sim_pred_tr,
                                        real_pred_va, sim_pred_va,
                                        real_alns_dict, sim_alns_dict)
                dfs.append(df)

        # print/save fold results
        plot_folds(train_history_folds, val_history_folds,
                   path=f'{result_path}/fig-fold-eval.png'
                   if result_path is not None else None)

        if len(dfs) > 0:
            df_c = pd.concat(dfs)
            csv_string = df_c.to_csv(index=False)
            with open(result_path + '/aln_train_eval.csv', 'w') as file:
                file.write(csv_string)

            print(f'Saved statistics to {result_path}/aln_train_eval.csv ...\n')

        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {nb_folds} FOLDS')
        print('----------------------------------------------------------------')

        for i, acc in enumerate(fold_eval):
            print(f'Fold {(i + 1)}: {acc} %')

        print(f'Average: {np.sum(fold_eval) / len(fold_eval)} %')


if __name__ == '__main__':
    main()

"""Useful helper functions of all kinds"""

import json
import os
import errno
import math

from ConvNet import load_net


def write_config_file(config, model_path, config_path, timestamp):
    """Save parameters in a json file

    If a recent config contains the same parameters it will be put into
    this json file by putting together the result paths and comments

    :param config: data-,model-specific and hyper-parameters (dictionary)
    :param model_path: directory where model/results are stored (string)
    :param config_path: directory to configs or config file (string)
    :param timestamp: format %d-%b-%Y-%H:%M:%S.%f (string)
    """

    if config_path == '':
        config_dir = ''
    elif os.path.isfile(config_path):
        config_dir = config_path.rpartition('/')[0]
    elif os.path.isdir(config_path):
        config_dir = config_path
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                config_path)

    if config_dir != '':
        out_path = f'{config_dir}/config-{timestamp}.json'

        # get second and third latest config files
        files = [f'{config_dir}/{file}' for file in os.listdir(config_dir)
                 if f'{config_dir}/{file}' != config_path]
        files = sorted(files, key=lambda t: -os.stat(t).st_mtime)[1:3]

        same_config = {}
        for file in files:
            older_config = read_config_file(file)
            if (older_config['data'] == config['data'] and
                    older_config['hyperparameters'] == config['hyperparameters'] and
                    older_config['conv_net_parameters'] == config[
                        'conv_net_parameters']):
                same_config = older_config
                same_config_file = file
                break

        # saving model path(s) and comments
        if same_config != {}:
            if type(same_config['results_path']) is list:
                res_paths = same_config['results_path'] + [model_path]
            else:
                res_paths = [same_config['results_path'], model_path]
            config['results_path'] = list(filter(None, res_paths))

            if type(same_config['comments']) is list:
                if type(config['comments']) is not list:
                    comments = same_config['comments'] + [config['comments']]
                else:
                    comments = same_config['comments'] + config['comments']
            else:
                comments = [same_config['comments'], config['comments']]
            config['comments'] = list(filter(None, comments))
        else:
            config['results_path'] = model_path

        # save config to file
        with open(out_path, "w") as outfile:
            json.dump(config, outfile)

        # delete old config
        if same_config != {}:
            os.remove(same_config_file)

    else:
        config['results_path'] = model_path

    if model_path != '':
        with open(f'{model_path}/config.json', "w") as outfile:
            json.dump(config, outfile)


def read_config_file(path):
    """Read parameters from a json file

    If directory is given, config will be taken fro latest modified file

    :param path: <path/to> config file or directory
    :return: data-,model-specific and hyper-parameters (dictionary)
    """

    if os.path.exists(path):
        if os.path.isdir(path):
            files = [f'{path}/{file}' for file in os.listdir(path)]
            path = max(files, key=os.path.getctime)

        with open(path) as json_file:
            return json.load(json_file)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def split_lst(lst, nb_parts):
    """Split list in *nb_parts* equal parts"""
    n = math.ceil(len(lst) / nb_parts)
    for i in range(0, len(lst), n):
        yield lst[i: n + i]


def flatten_lst(lst):
    """Remove 1 dimension from list"""
    flattened_lst = []
    for _lst in lst:
        flattened_lst += _lst
    return flattened_lst


def dim(lst):
    """Return number of dimensions of a list"""
    if not type(lst) == list:
        return 0
    return 1 + dim(lst[0])


def collect_histories_folds(path, nb_folds, model_params):
    """Gets validation and training history from models in *path* directory

    :param path: <path/to> directory containing .pth file(s) (string)
    :param nb_folds: number of folds/models (integer)
    :param seq_len: number of sites (integer)
    :param nb_chnls: number of channels (typically 1 per amino acid) (integer)
    :return: 2 lists of training/validation history dictionaries
    """

    train_history_folds = []
    val_history_folds = []

    for fold in range(1, nb_folds + 1):
        model_path = f'{path}/model-fold-{fold}.pth'
        model = load_net(model_path, model_params)
        train_history_folds.append(model.train_history)
        val_history_folds.append(model.val_history)

    return train_history_folds, val_history_folds


def extract_accuary_loss(train_history_folds, val_history_folds):
    """Extracts losses and accuracies from training and validation history"""

    accs_train_folds = []
    losses_train_folds = []
    accs_val_folds = []
    losses_val_folds = []

    for fold in range(len(train_history_folds)):
        accs_train_fold = []
        losses_train_fold = []
        accs_val_fold = []
        losses_val_fold = []

        for i in range(len(train_history_folds[fold])):
            accs_train_fold.append(train_history_folds[fold][i]['acc'])
            losses_train_fold.append(train_history_folds[fold][i]['loss'])
            accs_val_fold.append(val_history_folds[fold][i]['acc'])
            losses_val_fold.append(val_history_folds[fold][i]['loss'])

        accs_train_folds.append(accs_train_fold)
        losses_train_folds.append(losses_train_fold)
        accs_val_folds.append(accs_val_fold)
        losses_val_folds.append(losses_val_fold)

    return (accs_train_folds, losses_train_folds,
            accs_val_folds, losses_val_folds)

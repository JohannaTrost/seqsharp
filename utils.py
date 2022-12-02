"""Useful helper functions of all kinds"""

import json
import os
import errno
import pickle
import warnings

import numpy as np
import matplotlib.transforms as transforms
from datetime import datetime
from tompy import CustomPDF

from ConvNet import load_net
from matplotlib import pylab as plt
from matplotlib.patches import Ellipse


def get_model_performance(model_path):
    """Get array of (most recent) BACC, class acc. and loss of model for all
    folds

    :param model_path: <path/to> model containing 'fold-validation'-file
    :return: array of BACC, class acc. and loss for each fold and csv header
    """
    val_files = [f for f in os.listdir(model_path) if
                 f.startswith('fold-validation')]
    # get most recent val. results
    if len(val_files) > 1:
        file_age = []
        for f in val_files:
            if f.startswith('fold-validation-'):  # timestamp given
                time = datetime.strptime(f.strip('fold-validation-.csv'),
                                         "%d-%b-%Y-%H:%M:%S.%f")
            else:  # if timestamp is not given then this is the oldest result
                time = datetime.strptime('01-Nov-1000-00:00:00.000000',
                                         "%d-%b-%Y-%H:%M:%S.%f")
            file_age.append(time)
        ind_most_recent = np.argmin(file_age)
        val_file = val_files[ind_most_recent]
    else:
        val_file = val_files[0]

    return fold_val_from_csv(f'{model_path}/{val_file}')


def fold_val_dict2csv(dict, path):
    """Save dictionary with BACC, class acc. and loss for each fold to csv

    :param dict: dictionary with BACC, class acc. and loss for each fold
    :param path: <path/to> csv file
    :return: -
    """
    tab = np.asarray(list(dict.values())).T
    header = ','.join(list(dict.keys()))
    np.savetxt(path, tab, header=header, comments='', delimiter=',')


def fold_val_from_csv(path):
    """Get array of BACC, class acc. and loss for each fold and csv header

    :param path: <path/to> csv file
    :return: array of BACC, class acc. and loss for each fold and csv header
    """
    header = np.asarray(np.genfromtxt(path, names=True,
                                      delimiter=',').dtype.names)
    val_folds = np.genfromtxt(path, skip_header=True, delimiter=',')
    return val_folds, header


def write_cfg_file(cfg, model_path, cfg_path, timestamp=None):
    """Save parameters in a json file

    If a recent cfg contains the same parameters it will be put into
    this json file by putting together the result paths and comments

    :param cfg: data-,model-specific and hyper-parameters (dictionary)
    :param model_path: directory where model/results are stored (string)
    :param cfg_path: directory to cfgs or cfg file (string)
    :param timestamp: format %d-%b-%Y-%H:%M:%S.%f (string)
    """

    if cfg_path == '':
        cfg_dir = ''
    elif os.path.isfile(cfg_path):
        cfg_dir = cfg_path.rpartition('/')[0]
    elif os.path.isdir(cfg_path):
        cfg_dir = cfg_path
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                cfg_path)

    if cfg_dir != '':
        out_path = f'{cfg_dir}/cfg-{timestamp}.json'

        # get second and third latest cfg files
        files = [f'{cfg_dir}/{file}' for file in os.listdir(cfg_dir)
                 if f'{cfg_dir}/{file}' != cfg_path and
                 file.split('.')[-1] == 'json']
        files = sorted(files, key=lambda t: -os.stat(t).st_mtime)[1:3]

        print(files)

        same_cfg = {}
        for file in files:
            older_cfg = read_cfg_file(file)
            if (older_cfg['data'] == cfg['data'] and
                    older_cfg['hyperparameters'] == cfg[
                        'hyperparameters'] and
                    older_cfg['conv_net_parameters'] == cfg[
                        'conv_net_parameters']):
                same_cfg = older_cfg
                same_cfg_file = file
                break

        # saving model path(s) and comments
        if same_cfg != {}:
            if type(same_cfg['results_path']) is list:
                res_paths = same_cfg['results_path'] + [model_path]
            else:
                res_paths = [same_cfg['results_path'], model_path]
            cfg['results_path'] = list(filter(None, res_paths))

            if type(same_cfg['comments']) is list:
                if type(cfg['comments']) is not list:
                    comments = same_cfg['comments'] + [cfg['comments']]
                else:
                    comments = same_cfg['comments'] + cfg['comments']
            else:
                comments = [same_cfg['comments'], cfg['comments']]
            cfg['comments'] = list(filter(None, comments))
        else:
            cfg['results_path'] = model_path

        # save cfg to file
        with open(out_path, "w") as outfile:
            json.dump(cfg, outfile)

        # delete old cfg
        if same_cfg != {}:
            os.remove(same_cfg_file)

    else:
        cfg['results_path'] = model_path

    if model_path != '':
        with open(f'{model_path}/cfg.json', "w") as outfile:
            json.dump(cfg, outfile)


def read_cfg_file(path):
    """Read parameters from a json file

    If directory is given, cfg will be taken from latest modified file

    :param path: <path/to> cfg file or directory
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
    n = int(np.ceil(len(lst) / nb_parts))
    return [lst[i: n + i] for i in range(0, len(lst), n)]


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


def get_histories_folds(path, nb_folds, model_params):
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

    return merge_fold_hist_dicts(train_history_folds, val_history_folds)


def merge_fold_hist_dicts(train_history_folds, val_history_folds):
    """Extracts losses and accuracies from training and validation history"""
    # init dicts
    train_folds, val_folds = {}, {}
    for key in train_history_folds[0].keys():
        train_folds[key] = []
        val_folds[key] = []
    # populate dicts merging dicts of multiple folds
    for fold in range(len(train_history_folds)):
        for key, hist in train_history_folds[fold].items():
            train_folds[key].append(hist)
        for key, hist in val_history_folds[fold].items():
            val_folds[key].append(hist)

    return train_folds, val_folds


def largest_remainder_method(fractions, total):
    portions = fractions * total
    ints = np.floor(portions)
    decimals = portions - ints
    ints[np.argsort(-1 * decimals)[:int(total - ints.sum())]] += 1
    return ints.astype(np.int32)


def argsort_like(sort_this, like_this):
    """ by Tommy Clausner (2022)"""

    inds_a = np.argsort(sort_this)
    inds_b = np.argsort(like_this)

    return inds_a[np.argsort(inds_b)]


def pol2cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def pred_runtime(n_alns, n_cl):
    """Factors stem from least-square regression on
    number of MSAs: [20, 40, 60, 80, 100]
    number of clusters: [1, 2, 4, 6, 8]
    average number of sites: [517.25, 498.0, 392.1333333333333, 445.1625, 443.66]
    given the runtime for all combinations of the above

    :param n_alns: array or single values
    :param n_cl: array or single value
    :return: predicted runtimes
    """
    return 0.24943235 * n_alns + 1.20771259 * n_cl # 0.03476806 * n_avg_sites


def growth_rate(a, b):
    return np.abs(a - b) / min(np.abs(a), np.abs(b)) * 100


def load_weights(weights_path, n_clusters, n_profiles, n_runs, best=True):
    cl_w_runs = np.zeros(n_clusters) if best else np.zeros((n_runs, n_clusters))
    pro_w_runs = (np.zeros((n_clusters, n_profiles)) if best
                  else np.zeros((n_runs, n_clusters, n_profiles)))

    for run in range(n_runs):
        if (os.path.exists(f'{weights_path}/cl_weights_{run+1}.csv')
                and not best):
            cl_w_runs[run] = np.genfromtxt(f'{weights_path}/cl_weights_{run+1}'
                                           f'.csv', delimiter=',')
        elif os.path.exists(f'{weights_path}/cl_weights_best{run+1}.csv'):
            if best:
                cl_w_runs = np.genfromtxt(f'{weights_path}/cl_weights_best'
                                          f'{run + 1}.csv', delimiter=',')
            else:
                cl_w_runs[run] = np.genfromtxt(f'{weights_path}/cl_weights_best'
                                               f'{run+1}.csv', delimiter=',')
        else:
            if not best:
                warnings.warn(f'No cluster weights for run {run+1}')

        for cl in range(n_clusters):
            if (not best and os.path.exists(
                    f'{weights_path}/cl{cl+1}_pro_weights_{run+1}.csv')):

                pro_w_runs[run, cl] = np.genfromtxt(
                    f'{weights_path}/cl{cl+1}_pro_weights_{run+1}'
                    f'.csv', delimiter=',')

            elif os.path.exists(f'{weights_path}/cl{cl+1}_pro_weights_best{run+1}'
                                f'.csv'):
                if best:
                    pro_w_runs[cl] = np.genfromtxt(
                        f'{weights_path}/cl{cl + 1}'
                        f'_pro_weights_best'
                        f'{run + 1}.csv', delimiter=',')
                else:
                    pro_w_runs[run, cl] = np.genfromtxt(
                        f'{weights_path}/cl{cl+1}_pro_weights_best{run+1}.csv',
                        delimiter=',')
            else:
                if not best:
                    warnings.warn(f'No profile weights for run {run+1}')

    return cl_w_runs, pro_w_runs


def save_custom_distr(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(CustomPDF(data), file)


def load_custom_distr(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


n_sites_pdf = load_custom_distr('data/n_sites_hogenom_6971.CustomPDF')
gamma_shape_pdf = load_custom_distr('data/gamma_shape.CustomPDF')
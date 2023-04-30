"""Useful helper functions of all kinds"""

import json
import os
import errno
import pickle
import warnings

import numpy as np
import matplotlib.transforms as transforms
from tompy import CustomPDF

from matplotlib.patches import Ellipse


def pickle_read(path):
    with open(path, "rb") as file:
        out = pickle.load(file)
    return out


def index_neighbours(index, arr_len, n=4):
    inds_template = np.arange(-n, n + 1)
    # indices relative to "index" in center
    inds = inds_template + index
    inds -= np.min([inds[0], 0])  # shift to right
    inds -= np.max([inds[-1] - arr_len + 1, 0])  # shift to left
    inds = inds[inds != index]
    return inds


def min_diff_divisor(divident):
    divisors = np.arange(2, divident // 2, dtype=int)
    divisors = divisors[np.repeat(divident, len(divisors)) % divisors == 0]
    quotients = np.repeat(divident, len(divisors)) / divisors
    return divisors[np.argmin(np.abs(divisors - quotients))]


def write_cfg_file(cfg, save, timestamp=None):
    """Save parameters in a json file

    If a recent cfg contains the same parameters it will be put into
    this json file by putting together the result paths and comments

    :param cfg: data-,model-specific and hyper-parameters (dictionary)
    :param save: directory/file to save the config to (string)
    :param timestamp: timestamp (string)
    """

    if os.path.isfile(save):
        with open(save, "w") as outfile:
            json.dump(cfg, outfile)
    elif os.path.isdir(save) and timestamp is None:
        with open(f'{save}/cfg.json', "w") as outfile:
            json.dump(cfg, outfile)
    elif os.path.isdir(save) and timestamp is not None:
        with open(f'{save}/cfg_{timestamp}.json', "w") as outfile:
            json.dump(cfg, outfile)
    else:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'{save} is neither a file nor a directory')


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


def split_lst(lst, n_parts):
    """Split list in *n_parts* equal parts"""
    n = int(np.ceil(len(lst) / n_parts))
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
        Input emp_pdfs.

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


def save_kde(kde_obj, pca, scaler, dir_path):
    with open(f'{dir_path}/kde.KDE', 'wb') as file:
        pickle.dump(kde_obj, file)
    with open(f'{dir_path}/pca.KDE', 'wb') as file:
        pickle.dump(pca, file)
    with open(f'{dir_path}/scaler.KDE', 'wb') as file:
        pickle.dump(scaler, file)


def load_kde(dir_path):
    with open(f'{dir_path}/kde.KDE', 'rb') as file:
        kde_obj = pickle.load(file)
    with open(f'{dir_path}/pca.KDE', 'rb') as file:
        pca = pickle.load(file)
    with open(f'{dir_path}/scaler.KDE', 'rb') as file:
        scaler = pickle.load(file)
    return kde_obj, pca, scaler
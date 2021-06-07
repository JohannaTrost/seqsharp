import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats._continuous_distns import _distn_names


def mse(aln1, aln2):
    """Mean square error of 2 alignment representations"""

    return np.mean(np.sum((aln1 - aln2) ** 2, axis=0))


def padding(alns, max_seq_len=300):
    """Generated list with padding size per alignment"""
    paddings = []
    for aln in alns:
        seq_len = len(aln[0])
        paddings.append(max(max_seq_len - seq_len, 0))
    return paddings


def nb_seqs_per_alns(alns):
    return [len(aln) for aln in alns]


def get_nb_sites(alns):
    return [len(aln[0]) for aln in alns]


def get_aa_freqs(alns, gaps=True, dict=True):
    """Returns amino acid frequencies for given alignments

    :param alns: list of multiple alignments (list of string list)
    :param gaps: there are gaps in alignments if true (boolean)
    :param dict: a dictionary shall be returned if true (boolean)
    :return: list of aa frequencies
    """

    aas = 'ARNDCQEGHILKMFPSTWYVX-' if gaps else 'ARNDCQEGHILKMFPSTWYV'
    aa_freqs_alns = []

    for aln in alns:
        freqs = np.zeros(22) if gaps else np.zeros(20)

        for seq in aln:
            for i, aa in enumerate(aas):
                freqs[i] += seq.count(aa)

        freqs /= (len(aln) * len(aln[0]))  # get proportions

        if gaps:
            # distribute the gap portion over all frequencies
            freqs += ((1 - sum(freqs)) / 20)

        # limit to 6 digits after the comma
        freqs = np.floor(np.asarray(freqs) * 10 ** 6) / 10 ** 6

        if dict:
            aa_freqs_alns.append({aas[i]: freqs[i] for i in range(len(aas))})
        else:
            aa_freqs_alns.append(freqs)

    return aa_freqs_alns


def distance_stats(dists):
    masked_dists = np.ma.masked_equal(dists, 0.0, copy=False)
    mean_mse = masked_dists.mean(axis=1).data
    max_mse = masked_dists.max(axis=1).data
    min_mse = masked_dists.min(axis=1).data

    return {'mean': mean_mse, 'max': max_mse, 'min': min_mse}


def generate_aln_stats_df(real_fastas, sim_fastas, real_alns, sim_alns,
                          max_seq_len, real_alns_repr, sim_alns_repr,
                          csv_path=None):
    """Returns a dataframe with information about empirical and simulated
       alignments with option to save the table as a csv file
    """

    dists_real = np.asarray([[mse(aln1, aln2) for aln2 in real_alns_repr]
                             for aln1 in real_alns_repr])
    dists_sim = np.asarray([[mse(aln1, aln2) for aln2 in sim_alns_repr]
                            for aln1 in sim_alns_repr])

    mse_stats_real = distance_stats(dists_real)
    mse_stats_sim = distance_stats(dists_sim)

    dat_dict = {'id': real_fastas + sim_fastas,
                'aa_freqs': get_aa_freqs(real_alns + sim_alns),
                'padding': padding(real_alns + sim_alns, max_seq_len),
                'number_seqs': nb_seqs_per_alns(real_alns + sim_alns),
                'seq_length': get_nb_sites(real_alns + sim_alns),
                'mean_mse_all': list(mse_stats_real['mean']) +
                                list(mse_stats_sim['mean']),
                'max_mse_all': list(mse_stats_real['max']) +
                               list(mse_stats_sim['max']),
                'min_mse_all': list(mse_stats_real['min']) +
                               list(mse_stats_sim['min']),
                'simulated': [0] * len(real_alns) + [1] * len(sim_alns)
                }

    df = pd.DataFrame(dat_dict)

    if csv_path is not None:
        csv_string = df.to_csv(index=False)
        with open(csv_path, 'w') as file:
            file.write(csv_string)

    return df


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    distributions = [getattr(st, distname) for distname in _distn_names
                     if distname != 'levy_stable']

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in distributions:
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            print(f'Could not use {distribution.name}')
            pass

    return best_distribution.name, best_params


def generate_data_from_dist(data):
    """Generate data based on the distribution of a given dataset"""

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200)
    best_dist = getattr(st, best_fit_name)

    arg = best_fit_params[:-2]
    loc = best_fit_params[-2]
    scale = best_fit_params[-1]

    new_data = best_dist.rvs(*arg, loc, scale,
                             np.asarray(data).shape).astype(int)
    new_data = [max(x, np.min(data)) for x in new_data]  # set lower limit

    return new_data

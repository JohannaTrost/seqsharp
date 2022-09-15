import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats._continuous_distns import _distn_names
from sklearn.decomposition import PCA

from utils import dim


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
    # multiple datasets of 'raw' MSAs (e.g. real and simulated)
    if dim(alns) == 3 and type(alns[0][0][0]) == str:
        return [[len(aln) for aln in dataset] for dataset in alns]

    return [len(aln) for aln in alns]


def get_nb_sites(alns):
    # multiple datasets of 'raw' MSAs (e.g. real and simulated)
    if dim(alns) == 3 and type(alns[0][0][0]) == str:
        return [[len(aln[0]) for aln in dataset] for dataset in alns]

    return [len(aln[0]) for aln in alns]


def get_aa_freqs(alns, gaps=True, dict=True):
    """Returns amino acid frequencies for given alignments

    :param alns: list of multiple alignments (list of string list)
    :param gaps: there are gaps in alignments if true (boolean)
    :param dict: a dictionary shall be returned if true (boolean)
    :return: list of aa frequencies
    """

    aas = 'ARNDCQEGHILKMFPSTWYVX-' if gaps else 'ARNDCQEGHILKMFPSTWYVX'
    aa_freqs_alns = []

    for aln in alns:
        freqs = np.zeros(23) if gaps else np.zeros(22)

        for seq in aln:
            for i, aa in enumerate(aas):
                freqs[i] += seq.count(aa)
            freqs[-1] += (seq.count('B') + seq.count('Z') + seq.count('J') +
                          seq.count('U') + seq.count('O'))

        freqs /= len(aln) * len(aln[0])  # get proportions

        # limit to 6 digits after the comma
        freqs = np.floor(np.asarray(freqs) * 10 ** 6) / 10 ** 6

        if dict:
            freq_dict = {aas[i]: freqs[i] for i in range(len(aas))}
            freq_dict['other'] = freqs[-1]

            aa_freqs_alns.append(freq_dict)
        else:
            aa_freqs_alns.append(freqs)

    return aa_freqs_alns


def distance_stats(dists):
    masked_dists = np.ma.masked_equal(dists, 0.0, copy=False)
    mean_mse = masked_dists.mean(axis=1).data
    max_mse = masked_dists.max(axis=1).data
    min_mse = masked_dists.min(axis=1).data

    return {'mean': mean_mse, 'max': max_mse, 'min': min_mse}


def generate_aln_stats_df(fastas, alns, max_seq_len, alns_repr, is_sim=[],
                          csv_path=None):
    """Returns a dataframe with information about input
       alignments with option to save the table as a csv file

    :param fastas: list of lists with fasta filenames (2D string list)
    :param alns: list of lists with MSAs (3D string list)
    :param max_seq_len: max. number of sites (integer)
    :param alns_repr: list of lists with MSA representations
    :param is_sim: list of 0s and 1s (0: real MSAs, 1: simulated MSAs)
    :param csv_path: <path/to> file to store dataframe
    :return: dataframe with information about input alignments
    """

    ids, aa_freqs, paddings, number_seqs, seq_length = [], [], [], [], []
    mean_mse_all, max_mse_all, min_mse_all = [], [], []

    for i in range(len(fastas)):
        ids += fastas[i]
        aa_freqs += get_aa_freqs(alns[i])
        paddings += padding(alns[i], max_seq_len)
        number_seqs += nb_seqs_per_alns(alns[i])
        seq_length += get_nb_sites(alns[i])

        dists = np.asarray([[mse(aln1, aln2) for aln2 in alns_repr[i]]
                                 for aln1 in alns_repr[i]])

        mean_mse_all += list(distance_stats(dists)['mean'])
        max_mse_all += list(distance_stats(dists)['max'])
        min_mse_all += list(distance_stats(dists)['min'])

    simulated = []
    if len(is_sim) > 0:
        for is_sim_, msa in zip(is_sim, alns):
            simulated += len(msa) * [is_sim_]
    elif len(alns) == 2:
        simulated = [0] * len(alns[0]) + [1] * len(alns[1])
    else:
        simulated = [-1] * (len(alns))

    dat_dict = {'id': ids,
                'aa_freqs': aa_freqs,
                'padding': paddings,
                'number_seqs': number_seqs,
                'seq_length': seq_length,
                'mean_mse_all': mean_mse_all,
                'max_mse_all': max_mse_all,
                'min_mse_all': min_mse_all,
                'simulated': simulated
                }

    df = pd.DataFrame(dat_dict)

    if csv_path is not None:
        csv_string = df.to_csv(index=False)
        with open(csv_path, 'w') as file:
            file.write(csv_string)

    return df


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data

        src: https://stackoverflow.com/questions/6620471/fitting-empirical
             -distribution-to-theoretical-ones-with-scipy-python
    """

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


def count_aas(data, level='msa', save=''):
    # pid = os.getpid()
    # print(f'starting process {pid}')

    aas = 'ARNDCQEGHILKMFPSTWYV'
    aa_counts_alns = []
    nb_sites = 0

    for aln in data:
        nb_seqs = len(aln)
        seq_len = len(aln[0])

        if level == 'sites':
            # transform alignment into array to make sites accessible
            aln_arr = np.empty((nb_seqs, seq_len), dtype='<U1')
            for j in range(nb_seqs):
                aln_arr[j, :] = np.asarray([aa for aa in aln[j]])

            aa_counts = np.zeros((len(aas), seq_len))
            # count aa at each site
            for site_ind in range(seq_len):
                site = ''.join([aa for aa in aln_arr[:, site_ind]])
                for i, aa in enumerate(aas):
                    aa_counts[i, site_ind] = site.count(aa)
            nb_sites += seq_len
        elif level == 'genes':
            aa_counts = np.zeros((len(aas), nb_seqs))
            # count aa for each gene
            for gene_ind in range(nb_seqs):
                for i, aa in enumerate(aas):
                    aa_counts[i, gene_ind] = aln[gene_ind].count(aa)
        elif level == 'msa':
            aa_counts = np.zeros((1, len(aas)))
            # count aa for each gene
            for gene_ind in range(nb_seqs):
                for i, aa in enumerate(aas):
                    aa_counts[0, i] += aln[gene_ind].count(aa)

        if len(aa_counts_alns) == 0:
            aa_counts_alns = aa_counts
        else:
            aa_counts_alns = np.concatenate((aa_counts_alns, aa_counts),
                                            axis=(0 if level == 'msa' else 1))

    if save != '':
        np.savetxt(save,
                   np.asarray(aa_counts),
                   delimiter=',',
                   fmt='%1.1f')
        print(f'Successfully saved {nb_sites} sites.\n')
    else:
        return aa_counts_alns


def freq_pca_from_raw_alns(data, n_components=2):
    """Legacy function to perform PCA on average MSA AA frequencies
    had been used for past experiments and test with EM and to evaluate
    simulatons

    :param data: list of lists with MSAs (3D string list)
    :param n_components: number of principal components
    :return: (n_alns x n_components) principal components, pca object
    """

    # get avg. MSA AA frequencies
    msa_freqs = count_aas(data, level='msa')
    msa_freqs /= np.repeat(msa_freqs.sum(axis=1)[:, np.newaxis], 20, axis=1)
    msa_freqs = np.round(msa_freqs, 8)
    # perform PCA and center resulting PCs
    pca = PCA(n_components=n_components)
    pca_msa_freqs = pca.fit_transform(msa_freqs)
    pca_msa_freqs_c = pca_msa_freqs - pca_msa_freqs.mean(axis=0)

    return pca_msa_freqs_c, pca

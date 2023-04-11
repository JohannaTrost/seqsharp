import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats._continuous_distns import _distn_names
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import dim


def sample_indel_params(kde_obj, pca, scaler, sample_size=1, min_rl=50,
                        max_rl=30000):
    new_data = []
    while len(new_data) < sample_size:
        tmp = kde_obj.resample(1)
        tmp = pca.inverse_transform(tmp.T)
        tmp = scaler.inverse_transform(tmp)[0]

        # apply SpartaABC prior limits on indel length
        cond = 1.001 <= tmp[0] <= 2
        cond &= 1.001 <= tmp[1] <= 2
        cond &= min_rl <= tmp[2] <= max_rl
        cond &= 0 <= tmp[3] <= 0.05
        cond &= 0 <= tmp[4] <= 0.05

        if cond:
            new_data.append(tmp)

    param_names = ['RIM A_D', 'RIM A_I', 'RIM RL', 'RIM R_D', 'RIM R_I']
    new_data_dict ={}
    for key, val in zip(param_names, np.asarray(new_data).T):
        new_data_dict[key] = val

    return new_data_dict


def kde(data, n_components=None):
    ndim = data.shape[1] if n_components is None else n_components

    # project the n-dimensional data to a lower dimension
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    pca = PCA(n_components=n_components, whiten=False)
    pcs = pca.fit_transform(data_scaled)

    var = np.sum(pca.explained_variance_ratio_[:n_components])
    print(f'Explained variance: {var}')

    kde_obj = st.gaussian_kde(pcs.T)

    return kde_obj, pca, scaler


def n_unique_mol_per_site(msa_reprs):
    msa_reprs = np.asarray(msa_reprs)
    # mask: 1 if frequency > 0, then sum over AA/nucleotides per site
    n_unique_mol = np.sum(msa_reprs > 0, axis=1)
    # Remove padding: if 0 then there is no AA/nucleotide at that side
    n_unique_mol = [msa[msa != 0] for msa in n_unique_mol]

    return n_unique_mol


def effect_size(group1, group2):
    # cohen's D
    # group1 > group2

    n1, n2 = len(group1), len(group2)
    pooled_sd = (n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)
    pooled_sd /= n1 + n2 - 2
    pooled_sd = pooled_sd ** 0.5

    d = (np.mean(group1) - np.mean(group2)) / pooled_sd

    if ((n1 + n2) / 2) < 50:  # correction for small sample size
        d *= (((n1 + n2) / 2) - 3) / (((n1 + n2) / 2) - 2.25)

    return d, pooled_sd


def interpret_cohens_d(cohens_d):
    """Determines text interpretation of effect size given Cohen's d value

    param cohens_d: float of Cohen's d value
    :returns: effect_size_interpretation: adjective to describe magnitude of
    effect size
    """
    if 0 <= cohens_d < 0.1:
        effect_size_interpretation = "Very Small"
    elif 0.1 <= cohens_d < 0.35:
        effect_size_interpretation = "Small"
    elif 0.35 <= cohens_d < 0.65:
        effect_size_interpretation = "Medium"
    elif 0.65 <= cohens_d < 0.9:
        effect_size_interpretation = "Large"
    elif cohens_d >= 0.9:
        effect_size_interpretation = "Very Large"
    return effect_size_interpretation


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


def get_n_seqs_per_msa(alns):
    # multiple datasets of 'raw' MSAs (e.g. real and simulated)
    if dim(alns) == 3 and type(alns[0][0][0]) == str:
        return [[len(aln) for aln in dataset] for dataset in alns]

    return [len(aln) for aln in alns]


def get_n_sites_per_msa(alns):
    # multiple datasets of 'raw' MSAs (e.g. real and simulated)
    if dim(alns) == 3 and type(alns[0][0][0]) == str:
        return [[len(aln[0]) if len(aln) > 0 else 0 for aln in dataset]
                for dataset in alns]

    return [len(aln[0]) if len(aln) > 0 else 0 for aln in alns]


def get_frac_sites_with(chars, aln):
    """TODO

    :param chars:
    :param aln:
    :return:
    """
    aln_arr = np.asarray([list(seq) for seq in aln])
    n_sites = aln_arr.shape[1]
    n_sites_with_chars = np.sum(
        [any([c in ''.join(aln_arr[:, j]) for c in chars])
         for j in range(n_sites)])
    return n_sites_with_chars / n_sites if n_sites_with_chars > 0 else 0


def distance_stats(dists):
    masked_dists = np.ma.masked_equal(dists, 0.0, copy=False)
    mean_mse = masked_dists.mean(axis=1).data
    max_mse = masked_dists.max(axis=1).data
    min_mse = masked_dists.min(axis=1).data

    return {'mean': mean_mse, 'max': max_mse, 'min': min_mse}


def generate_aln_stats_df(fastas, alns, max_seq_len, alns_repr, is_sim=[],
                          csv_path=None):
    """Returns a dataframe with information about input_plt_fct
       alignments with option to save the table as a csv file

    :param fastas: list of lists with fasta filenames (2D string list)
    :param alns: list of lists with MSAs (3D string list)
    :param max_seq_len: max. number of sites (integer)
    :param alns_repr: list of lists with MSA representations
    :param is_sim: list of 0s and 1s (0: real MSAs, 1: simulated MSAs)
    :param csv_path: <path/to> file to store dataframe
    :return: dataframe with information about input_plt_fct alignments
    """

    ids, aa_freqs, paddings, number_seqs, seq_length = [], [], [], [], []
    mean_mse_all, max_mse_all, min_mse_all = [], [], []

    for i in range(len(fastas)):
        ids += fastas[i]
        aa_freqs += get_aa_freqs(alns[i])
        paddings += padding(alns[i], max_seq_len)
        number_seqs += get_n_seqs_per_msa(alns[i])
        seq_length += get_n_sites_per_msa(alns[i])

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
    best_distr = []

    # Estimate distribution parameters from data
    for distribution in tqdm(distributions):
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
                best_distr.append((distribution, params, sse))

        except Exception:
            print(f'Could not use {distribution.name}')
            pass

    return sorted(best_distr, key=lambda x: x[2])


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


def count_mols(data, level='msa', molecule_type='protein', save=''):
    # pid = os.getpid()
    # print(f'starting process {pid}')

    alphabet = 'ARNDCQEGHILKMFPSTWYV' if molecule_type == 'protein' else 'ACGT'
    counts_alns = []
    n_sites = 0

    for aln in data:
        nb_seqs = len(aln)
        seq_len = len(aln[0])

        if level == 'sites':
            # transform alignment into array to make sites accessible
            aln_arr = np.empty((nb_seqs, seq_len), dtype='<U1')
            for j in range(nb_seqs):
                aln_arr[j, :] = np.asarray([mol for mol in aln[j]])

            mol_counts = np.zeros((len(alphabet), seq_len))
            # count mol at each site
            for site_ind in range(seq_len):
                site = ''.join([mol for mol in aln_arr[:, site_ind]])
                for i, mol in enumerate(alphabet):
                    mol_counts[i, site_ind] = site.count(mol)
            n_sites += seq_len
        elif level == 'genes':
            mol_counts = np.zeros((len(alphabet), nb_seqs))
            # count mol for each gene
            for gene_ind in range(nb_seqs):
                for i, mol in enumerate(alphabet):
                    mol_counts[i, gene_ind] = aln[gene_ind].count(mol)
        elif level == 'msa':
            mol_counts = np.zeros((1, len(alphabet)))
            # count mol for each gene
            for gene_ind in range(nb_seqs):
                for i, mol in enumerate(alphabet):
                    mol_counts[0, i] += aln[gene_ind].count(mol)

        if len(counts_alns) == 0:
            counts_alns = mol_counts
        else:
            counts_alns = np.concatenate((counts_alns, mol_counts),
                                            axis=(0 if level == 'msa' else 1))

    return counts_alns


def freq_pca_from_raw_alns(data, n_components=2, level='sites'):
    """Legacy function to perform PCA on average MSA AA frequencies
    had been used for past experiments and test with EM and to evaluate
    simulatons

    :param data: list of lists with MSAs (3D string list)
    :param n_components: number of principal components
    :return: (n_alns x n_components) principal components, pca object
    """

    # get avg. MSA AA frequencies
    freqs = count_mols(data, level=level)
    freqs /= np.repeat(freqs.sum(axis=1)[:, np.newaxis], 20, axis=1)

    # perform PCA and center resulting PCs
    pca = PCA(n_components=n_components)
    pca_freqs = pca.fit_transform(freqs)
    pca_freqs_c = pca_freqs - pca_freqs.mean(axis=0)

    return pca_freqs_c, pca

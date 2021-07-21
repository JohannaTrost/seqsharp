import multiprocessing
import os
import random
import time

import matplotlib
import numpy as np
from scipy.stats import multinomial

matplotlib.use('Agg')


def generate_counts(profiles, profile_weights, nb_seqs=300, nb_sites=10000):

    nb_categories = profiles.shape[1]
    nb_profiles = profiles.shape[0]

    if profile_weights is None:
        profile_weights = [1 / nb_profiles] * nb_profiles

    profile_selections = np.random.choice(range(nb_profiles),
                                          p=profile_weights, size=nb_sites)

    counts = np.empty((nb_sites, nb_categories))

    for i, ind in enumerate(profile_selections):  # for each profile
        counts[i, :] = np.random.multinomial(nb_seqs, profiles[ind])

    return counts


def dice(rolls=300, trials=100):
    data = []
    prob_lst = [[0.028, 0.328, 0.016, 0.04, 0.32, 0.268]] * 4 + \
               [[0.26760563, 0.13028169, 0.16549296, 0.25352113, 0.06338028,
                 0.11971831]] * 2 + \
               [[0.19777159, 0.10584958, 0.01671309, 0.25626741, 0.15041783,
                 0.2729805]]

    for probs in prob_lst:
        base = 1
        for ip in set([1 / p for p in probs]):
            base *= ip
        vals = sum([[n + 1] * int(base * p) for n, p in enumerate(probs)], [])
        len_vals = len(vals)
        mult = 1 if rolls // len_vals == 0 else rolls // len_vals + 1
        vals *= mult
        random.shuffle(vals)
        vals = vals[:rolls]
        vals[np.random.randint(0, len(vals), 1)[0]] = 3
        noise = 0.1
        newdata = (np.atleast_2d(
            [[vals.count(val) for val in set(vals)]] * trials) *
                   (1 - (np.random.random((trials, 6)) * noise))).astype(int)

        if len(data) > 1:
            data = np.concatenate((data, newdata))
        else:
            data = newdata + 0
    return data


def count_aas(data, save=True):
    # pid = os.getpid()
    # print(f'starting process {pid}')

    aas = 'ARNDCQEGHILKMFPSTWYV'
    aa_counts_sites = []
    nb_sites = 0

    for aln in data:
        nb_seqs = len(aln)
        seq_len = len(aln[0])
        # transform alignment into array to make sites accessible
        aln_arr = np.empty((nb_seqs, seq_len), dtype='<U1')
        for j in range(nb_seqs):
            aln_arr[j, :] = np.asarray([aa for aa in aln[j]])

        # count aa at each site
        for site_ind in range(seq_len):
            site = ''.join([aa for aa in aln_arr[:, site_ind]])
            counts = np.empty(len(aas))
            for i, aa in enumerate(aas):
                counts[i] = site.count(aa)
            aa_counts_sites.append(counts)
        nb_sites += seq_len

    if save:
        np.savetxt(f'../counts/{os.getpid()}-counts-{nb_sites}sites.csv',
                   np.asarray(aa_counts_sites),
                   delimiter=',',
                   fmt='%1.1f')
        print(f'Successfully saved {nb_sites} sites.\n')
    else:
        return aa_counts_sites


def e_step(freqs, scores_profiles_sites):
    """Performs E-step on model

    :param freqs: (Z) mixture component weights
    :param scores_profiles_sites: score for each profile at each site
    :return: weights: (N x Z), posterior probabilities for objects clusters assignments
    """

    # Compute weighted likelihood
    weighted_multi_prob = np.zeros((len(scores_profiles_sites[0]), len(freqs)))
    for z in range(len(freqs)):
        weighted_multi_prob[:, z] = freqs[z] * scores_profiles_sites[z]

    # To avoid division by 0
    weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps

    denum = weighted_multi_prob.sum(axis=1)
    weights = weighted_multi_prob / denum.reshape(-1, 1)

    return weights


def m_step(weights):
    return weights.sum(axis=0) / weights.sum()


def _multinomial_prob(counts, profile, log=False):
    """
    Evaluates the multinomial probability for a given vector of counts
    counts: (N x C), matrix of counts
    profile: (C), vector of multinomial parameters for a specific cluster z
    Returns:
    p: (N), scalar values for the probabilities of observing each count vector given the profiles
    """
    n = counts.sum(axis=-1)
    if log:
        return multinomial.logpmf(counts, n, profile)
    return multinomial.pmf(counts, n, profile)


def compute_vlb(X, freqs, profiles, weights):
    """Computes the variational lower bound

    :param X: (N x C), data points
    :param freqs: frequencies of profiles
    :param profiles: (263 x 20) given profiles
    :param weights: relative weight of profile z for count vector X[j]
    :return: value of variational lower bound
    """

    loss = 0
    for z in range(len(profiles)):
        loss += np.sum(
            weights[:, z] * (np.log(freqs[z]) +
                             _multinomial_prob(X, profiles[z], log=True)))
        loss -= np.sum(weights[:, z] * np.log(weights[:, z]))
    return loss


# fit model
def fit(max_iter, rtol, scores_profiles_sites, profiles, aa_counts_sites,
        init_uniform=False):

    # init profile weights
    if isinstance(init_uniform, list):
        freqs = init_uniform
    elif init_uniform:
        freqs = [1 / len(profiles)] * len(profiles)
    else:
        w = np.random.randint(1, len(profiles), len(profiles))
        freqs = w / w.sum()

    print(f'Initial weights: {freqs}')

    weights = None
    losses = []

    # training cycles
    for it in range(max_iter):
        start = time.time()

        weights = e_step(freqs, scores_profiles_sites)
        freqs = m_step(weights)

        loss = compute_vlb(aa_counts_sites, freqs, profiles, weights)
        print(f'Loss: {loss} (training cycle took '
              f'{np.round(time.time() - start, 4)}s)')

        losses.append(loss)

        if it > 0 and (np.abs((losses[it - 1] - loss) / losses[it - 1]) < rtol):
            print(f'Finished after {it + 1} iterations.')
            break
        if it > 3 and loss == losses[it - 2] == losses[it - 4]:
            print(f'Loss pending between 2 values')
            break

    return losses, freqs


def main():
    '''
    Start with random initialization *repeats* times and calculation of
    maximum likelihood for each profile at each empirical site
    Run optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.

    :param observations: (number of obsevations N, amino acids C), matrix of
    observed aa frequencies
    :result: The best parameters found along with the associated loss
    '''

    nb_cores = 32  # psutil.cpu_count(logical=False)

    p_dir = '../results/profile_estimation'
    profiles = np.genfromtxt(f'../../ocamlSimulator/263SelectedProfiles.tsv',
                             delimiter='\t').T

    profiles = []
    for i in range(3):
        w = np.random.randint(1, 100, 6)
        profiles.append(w / w.sum())

    # counts for all sites
    real_fasta_path = '/home/jtrost/beegfs/fasta_no_gaps'
    config_path = '/home/jtrost/beegfs/mlaa/configs/config.json'
    
    config = read_config_file(config_path)
    
    nb_alns, min_nb_seqs, max_nb_seqs, seq_len, padding = config['data'].values()
    fasta_paths = os.listdir(real_fasta_path)
    
    print("Loading alignments ...")

    """Counts samples
    counts = []

    for i, path in enumerate(fasta_paths):
        print(i)
        seqs = aln_from_fasta(f'{real_fasta_path}/{path}', max_nb_seqs)
        if len(seqs) >= min_nb_seqs:
            counts_sites = count_aas([seqs], save=False)
            sample = np.random.randint(0,
                                       len(counts_sites),
                                       int(np.ceil(len(counts_sites) * 0.125)))
            if len(counts) == 0:
                counts = np.asarray(counts_sites)[sample]
            else:
                counts = np.concatenate((counts, np.asarray(counts_sites)[sample]))

    ind_chunks = np.array_split(np.asarray(range(len(counts))), nb_cores)

    process_pool = multiprocessing.Pool(nb_cores)
    alns_chunks = [tuple([f'../counts_all_samples/counts_{i}.csv', counts[indices], '%1.1f', ',']) for i, indices in enumerate(ind_chunks)]

    process_pool.starmap(np.savetxt, alns_chunks)
    """

    # load counts
    counts_files = os.listdir(f'../counts')

    start = time.time()

    process_pool = multiprocessing.Pool(nb_cores)
    result = process_pool.starmap(np.genfromtxt,
                                  [(f'../counts/{file}', float, '#', ',') for
                                   file in counts_files])

    print(
        f'Get counts per site in {np.round((time.time() - start) / 60, 4)} min.')

    aa_counts_sites = np.empty((np.sum([len(res) for res in result]), 20))

    i = 0
    nb_sites_nan = 0
    for j, chunk in enumerate(result):
        for site_freqs in chunk:
            if not np.any(np.isnan(site_freqs)):
                aa_counts_sites[i] = site_freqs
                i += 1
            else:
                nb_sites_nan += 1
                print(
                    f'Chunk {j + 1} {i}th site contains nan. Counts: {site_freqs}')
    if nb_sites_nan > 0:
        aa_counts_sites = aa_counts_sites[:-nb_sites_nan]


    aa_counts_sites = generate_counts(profiles, nb_sites=100000)

    # training parameters
    max_iter = 10
    rtol = 1e-10

    # compute multinomial likelihoods for all profiles at all sites
    scores_profiles_sites = [_multinomial_prob(aa_counts_sites, profile)
                             for profile in profiles]

    losses = []
    freqs_reps = []
    end_losses = []

    rep_losses, freqs = fit(max_iter, rtol, scores_profiles_sites, profiles,
                            aa_counts_sites, init_uniform=True)
    losses.append(rep_losses)
    end_losses.append(rep_losses[-1])
    freqs_reps.append(freqs)

    for rep in range(10):
        rep_losses, freqs = fit(max_iter, rtol, scores_profiles_sites, profiles,
                                aa_counts_sites)

        losses.append(rep_losses)
        end_losses.append(rep_losses[-1])
        freqs_reps.append(freqs)

    best_estim_ind = np.argmax(end_losses)

    for i in range(3):
        print(
            f'real: {real_weights[i]} vs. estim.: {freqs_reps[best_estim_ind][i]}')

    """
    timestamp = time.time()

    result_path = f'{p_dir}/{timestamp}-best-weight-profile.csv'
    np.savetxt(result_path,
               best_freqs,
               delimiter=',')

    plt.hist(best_freqs)
    plt.savefig(f'{p_dir}/{timestamp}-hist.png')

    plt.plot(np.cumsum(sorted(best_freqs.tolist())))
    plt.savefig(f'{p_dir}/{timestamp}-cumsum.png')
    
    print(f'Successfully saved estimated weights in {result_path}')
    """

if __name__ == '__main__':
    main()

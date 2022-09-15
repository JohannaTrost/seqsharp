import errno
import os
import time
import warnings
from itertools import permutations

import numpy as np

from scipy.stats import multinomial, dirichlet
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import pol2cart

MINPOSFLOAT = np.nextafter(0, 1)
MINNEGFLOAT = np.nextafter(-np.inf, 0)

AAS = 'ARNDCQEGHILKMFPSTWYV'

RTOL = 1e-5
STARTTIME = time.time()


def multi_dens(data, profs):
    """Computes the multinomial likelihood per site for a given a set of profiles
    for a set of MSAs

    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :return: list (n_alns) with (n_sites x n_profiles) containing multinomial
    probabilities for a site given a profile
    """

    n_alns = len(data)
    n_profiles = profs.shape[0]
    p_sites_profiles = []
    for i in range(n_alns):
        # -------- P(A_i | v_k) for all sites
        n_aas_site = data[i].sum(axis=-1)
        n_sites = data[i].shape[0]
        p_aln_sites_profiles = np.zeros((n_sites, n_profiles))
        for k in range(n_profiles):
            if profs[k].sum() > 1:  # condition for pmf function
                prof = profs[k] / profs[k].sum()
            else:
                prof = profs[k]

            p_aln_sites_profiles[:, k] = multinomial.pmf(data[i], n_aas_site,
                                                         prof)

        # EM requires probs >0
        p_aln_sites_profiles[p_aln_sites_profiles == 0] = MINPOSFLOAT
        p_sites_profiles.append(p_aln_sites_profiles)

    return p_sites_profiles


def log_probs_remaining_sites(p_aln_sites_profiles, pro_w):
    """Log probability for an MSA excluding site j per given profiles and profile
    weights for each cluster

    :param p_aln_sites_profiles: array of size n_sites x n_profiles with
    probability for a site given a profile
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    (profile weights)
    :return: (n_sites x n_clusters) probability for remaining sites
    (all sites except site j) given a cluster
    """

    n_sites = p_aln_sites_profiles.shape[0]
    n_clusters = pro_w.shape[0]

    log_rems_cl = np.zeros((n_sites, n_clusters))

    # (n_sites x n_sites x n_profiles)
    p_sites_profs_rep = np.repeat(p_aln_sites_profiles[np.newaxis, :, :],
                                  n_sites, axis=0)

    # set diagonal to 0 to exclude a site
    diag_inds = np.diag_indices(n_sites, 2)
    p_sites_profs_rep[diag_inds[0], diag_inds[1], :] = 0

    for cl in range(n_clusters):
        # prob. for remaining sites
        weighted_sites_notj = p_sites_profs_rep @ pro_w[cl]
        weighted_sites_notj[diag_inds] = 1  # set 0s to 1 such that log(1) = 0
        # sum over sites for log prob. for the entire MSA
        log_rems_cl[:, cl] = np.sum(np.log(weighted_sites_notj), axis=1)

    return log_rems_cl


def e_step(data, profs, pro_w, cl_w, p_sites_profs=None):
    """Expectation step
    Compute posterior distribution for a cluster c and a profile z to generate
    a site of an alignment and posterior distribution for a cluster c to
    generate an alignment.

    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    (profile weights)
    :param cl_w: (n_clusters) MSA cluster probability (cluster weights)
    :param p_sites_profs: (n_sites x n_profiles) probability for a site given
    a profile
    :return: posterior probabilities for profile and cluster assignments
    n_alns elements list with arrays of size n_sites x n_clusters xn_profiles,
    n_alns elements list with arrays of size n_clusters
    """

    n_alns = len(data)
    n_profiles = profs.shape[0]
    n_clusters = cl_w.shape[0]
    ax_p, ax_c, ax_s = 2, 1, 1  # profile-, cluster-, site-axis

    # pi : probability for profile z and cluster c at site j of alignment i
    #      given weights and profiles and aa counts
    log_site_pi = [np.zeros((data[aln].shape[0], n_clusters, n_profiles))
                   for aln in range(n_alns)]
    site_pi = [np.zeros((data[aln].shape[0], n_clusters, n_profiles))
               for aln in range(n_alns)]
    log_aln_pi = np.zeros((n_alns, n_clusters))

    if p_sites_profs is None:
        p_sites_profs = multi_dens(data, profs)

    for aln in range(n_alns):

        n_sites = data[aln].shape[0]

        log_rem_sites = log_probs_remaining_sites(p_sites_profs[aln], pro_w)

        # -------- lk on alignment level: prob. of a cluster i given profiles,
        #          profile weights and MSAs
        log_aln_pi[aln] = np.sum(np.log(np.dot(pro_w, p_sites_profs[aln].T)),
                                 axis=ax_s)
        log_aln_pi[aln] += np.log(cl_w)

        # part of the formula for log_aln_pi was replaced by dot product(
        # /matmul)
        # alter_dot = np.sum([p_sites_profiles[aln] * pro_w[c] for c
        # in range(n_clusters)], axis=2) dot = np.dot(pro_w,
        # p_sites_profiles[aln].T) np.all(alter_dot == dot)

        # -------- lk on site level : pi
        log_sites_profs = np.log(p_sites_profs[aln])  # n_sites x n_profiles

        # (n_sites x) n_cl x n_pro
        log_site_pi[aln] = np.repeat(np.log(pro_w)[np.newaxis, :, :], n_sites,
                                     axis=0)
        # Add log cluster weights -> (n_sites x) n_cl (x n_pro)
        log_site_pi[aln] += np.repeat(np.repeat(np.log(cl_w)[np.newaxis, :],
                                                n_sites,
                                                axis=0)[:, :, np.newaxis],
                                      n_profiles, axis=2)
        # Add probs of remaining sites -> n_sites x n_cl (x n_pro)
        log_site_pi[aln] += np.repeat(log_rem_sites[:, :, np.newaxis],
                                      n_profiles, axis=2)
        # Add site probs -> n_sites (x n_cl x) n_pro
        log_site_pi[aln] += np.repeat(log_sites_profs[:, np.newaxis, :],
                                      n_clusters,
                                      axis=1)

        if np.any(log_site_pi[aln] == -np.inf):
            warnings.warn(
                f"In MSA {aln + 1} {np.sum(log_site_pi[aln] == -np.inf)}/"
                f"{n_sites * n_clusters * n_profiles} log_pi's are "
                f"-infinity")

            # log_pi[aln][log_pi[aln] == -np.inf] = np.nextafter(-np.inf, 0)

        # -------- log to prob (n_sites x n_cl x n_pro)

        # max log-lk per site
        max_logs = np.max(np.max(log_site_pi[aln], axis=ax_p), axis=ax_c)

        # add cluster and profile axis to recover shape
        max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=ax_c)
        max_logs = np.repeat(max_logs[:, :, np.newaxis], n_profiles, axis=ax_p)

        site_pi[aln] = np.exp(log_site_pi[aln] + np.abs(max_logs))
        site_pi[aln][site_pi[aln] == 0] = MINPOSFLOAT

        # --- normalizing pi
        sum_over_pro_cl = site_pi[aln].sum(axis=ax_p).sum(axis=ax_c)

        # add cluster and profile axis to recover shape
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, np.newaxis], n_clusters,
                                    axis=1)
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, :, np.newaxis],
                                    n_profiles, axis=2)

        site_pi[aln] /= sum_over_pro_cl
        site_pi[aln][site_pi[aln] == 0] = MINPOSFLOAT

    # log to prob for alignment level lk
    max_logs = np.max(log_aln_pi, axis=ax_c)  # per MSA
    max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=ax_c)
    aln_pi = np.exp(log_aln_pi + np.abs(max_logs))
    # normalize
    aln_pi = aln_pi / np.repeat(np.sum(aln_pi, axis=1)[:, np.newaxis],
                                n_clusters, axis=1)

    return site_pi, aln_pi


def m_step(site_pi, aln_pi, data):
    """Maximization step
    Optimization of profiles, profile weights and cluster weights to maximize
    the lower bound of the log-likelihood under the equality contraint that
    the AA probabilities of a profile sum to 1, the profile weights for a
    cluster sum to 1 and the cluster weights sum to 1

    :param site_pi: list (n_alns) of (n_sites x n_clusters x n_profiles)
    posterior probabilities for profile assignments
    :param aln_pi: (n_alns x n_clusters) posterior probabilities for
    cluster assignments
    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :return: mixture and mixture proportions:
    (n_clusters) MSA cluster probability (cluster weights),
    (n_clusters x n_profiles) mixture component weights
    (profile weights),
    (n_profiles x 20) multinomial amino acid mixture weights
    """

    ax_c, ax_p = 1, 2  # cluster-, profile-axis
    ax_a, ax_s, ax_aa = 0, 0, 2  # alignment-, site-, AA-axis

    n_alns = len(site_pi)
    n_aas = data[0].shape[-1]
    n_profiles, n_clusters = site_pi[0].shape[ax_p], site_pi[0].shape[ax_c]

    # n_aln x n_cl (x n_pro x n_aa)
    aln_pi_rep = np.repeat(aln_pi[:, :, np.newaxis], n_profiles, axis=ax_p)
    aln_pi_aa_rep = np.repeat(aln_pi_rep[:, :, :, np.newaxis], n_aas, axis=3)

    # -------- update cluster weights
    sum_over_alns = np.sum(aln_pi, axis=ax_a)
    aln_pi_denum = np.sum(sum_over_alns, axis=ax_s)
    aln_pi_denum = np.repeat(aln_pi_denum, n_clusters)
    # usually division by 1
    estim_cluster_weights = sum_over_alns / aln_pi_denum

    # -------- update profile weights
    sum_over_sites = np.asarray([np.sum(site_pi[aln], axis=0)
                                 for aln in range(n_alns)])
    sum_over_sites_alns = np.sum(aln_pi_rep * sum_over_sites, axis=ax_a)

    sum_over_prof = np.sum(sum_over_sites_alns, axis=1)
    sum_over_prof = np.repeat(sum_over_prof[:, np.newaxis], n_profiles,
                              axis=1)

    # weight by cluster probabilities
    estim_profile_weights = sum_over_sites_alns / sum_over_prof

    # -------- update profiles
    weighted_counts = np.zeros((n_alns, n_clusters, n_profiles, n_aas))
    for aln in range(n_alns):
        aa_counts_pro = np.repeat(data[aln][:, np.newaxis, :], n_profiles,
                                  axis=1)  # n_sites (x n_pro x) n_aas
        for cl in range(n_clusters):
            site_pi_aas = np.repeat(site_pi[aln][:, cl, :, np.newaxis], n_aas,
                                    axis=ax_aa)  # n_sites x n_pro (x n_aas)
            # weights sites and sum over sites
            weighted_counts[aln, cl] = np.sum(site_pi_aas * aa_counts_pro,
                                              axis=ax_s)

    # weights MSAs and sum over MSAs
    cl_profiles = np.sum(aln_pi_aa_rep * weighted_counts, axis=ax_a)

    estim_profiles = cl_profiles.sum(axis=0)  # sum over clusters

    # normalize
    denum = estim_profiles.sum(axis=1)  # sum over aas
    denum = np.repeat(denum[:, np.newaxis], n_aas, axis=1)  # usually 1
    estim_profiles /= denum

    return estim_cluster_weights, estim_profile_weights, estim_profiles


def lk_lower_bound(data, profs, pro_w, cl_w, site_pi, aln_pi):
    """
    Compute lower bound of log-likelihood to track convergence of the EM
    This value is expected to increase in every EM iteration.

    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    (profile weights)
    :param cl_w: (n_clusters) MSA cluster probability (cluster weights)
    :param site_pi: list (n_alns) of (n_sites x n_clusters x n_profiles)
    posterior probabilities for profile assignments
    :param aln_pi: list (n_alns) of size n_clusters: posterior probabilities for
    cluster assignments
    :return: Variational lower bound value, (n_sites x n_profiles) probability
    for a site given a profile
    """

    n_alns = len(data)
    n_clusters = cl_w.shape[0]

    # avoid division by 0
    pro_w[pro_w == 0] = MINPOSFLOAT
    cl_w[cl_w == 0] = MINPOSFLOAT

    # compute lower bound of full lk
    lk = 0

    # -------- log lk for sites given profiles
    p_sites_profs = multi_dens(data, profs)
    p_sites_profile_zeros = p_sites_profs.copy()

    for aln in range(n_alns):
        n_sites = data[aln].shape[0]
        # avoid -inf (divide by zero encountered in log error)
        p_sites_profs[aln][p_sites_profs[aln] == 0] = MINPOSFLOAT

        # -------- log Mult(v_s, A_ij)
        log_sites = np.repeat(np.log(p_sites_profs[aln])[:, np.newaxis, :],
                              n_clusters, axis=1)

        # -------- log p_rs
        log_pro_w_rep = np.repeat(np.log(pro_w)[np.newaxis, :, :],
                                  n_sites, axis=0)

        # -------- computing lower bound log-lk
        lk_12 = (site_pi[aln] * log_sites) + (site_pi[aln] * log_pro_w_rep)
        lk_12[np.isnan(lk_12)] = 0  # such that 0 * -inf = 0

        lk += np.sum(lk_12)

    # -------- lk on alignment level
    lk += np.sum(aln_pi * np.log(cl_w))

    return lk, p_sites_profile_zeros


def lk_per_site(pro_w, profs=None, aln_counts=None, p_sites_profs=None):
    """Site-wise likelihood of an MSA given mixture (profiles) and
    mixture component weights (profile weights)

    :param aln_counts: site-wise amino acid counts of a MSA
    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    per MSA cluster (profile weights)
    :return: (n_sites) probabilities of sites given profiles and profile weights
    """

    if p_sites_profs is not None:
        return pro_w @ p_sites_profs.T
    elif profs is not None and aln_counts is not None:
        return pro_w @ multi_dens([aln_counts], profs)[0].T
    else:
        raise ValueError("Either p_sites_profs or profs and aln_counts need to "
                         "be provided to compute lks per site")


def lks_alns_given_cls(p_sites_profs, pro_w, log=False):
    """(Log-)likelihoods for a set of MSAs given profile weights of MSA clusters
    and profiles

    :param p_sites_profs: (n_sites x n_profiles) probability for a site given
    a profile
    :param pro_w: (n_profiles) mixture component weights per MSA cluster
    (profile weights)
    :param log: True to return log-lk False to return lk
    :return: (n_alns x n_clusters) scaled log-lks or lks and constants used
    to scale log-lks
    """

    n_alns, n_clusters = len(p_sites_profs), len(pro_w)
    lk_aln_cl = np.zeros((n_alns, n_clusters))
    log_lk_aln_cl = np.zeros((n_alns, n_clusters))
    consts = np.zeros(n_alns)  # to scale log lks to avoid 0s in lks
    for aln in range(n_alns):
        p_sites_cl = lk_per_site(pro_w, p_sites_profs=p_sites_profs[aln])
        log_lk_aln_cl[aln] = np.sum(np.log(p_sites_cl),
                                    axis=1)  # sum over sites
        consts[aln] = np.abs(np.max(log_lk_aln_cl[aln]))
        lk_aln_cl[aln] = np.exp(log_lk_aln_cl[aln] + consts[aln])

    if log:
        return log_lk_aln_cl + consts
    else:
        return lk_aln_cl, consts


def theoretical_cl_freqs(profs, pro_w):
    """Expected average AA frequencies of a site of a given MSA cluster

    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_profiles) mixture component weights per MSA cluster
    (profile weights)
    :return: (20) vector of AA frequencies
    """

    return pro_w @ profs


def expected_sim_freqs(profs, pro_w, data=None, p_sites_profs=None):
    """Given n_alns MSAs, profiles and profile weights, returns expected
    AA frequency vectors for n_alns simulated MSAs. That is, weighting
    (and averaging) theoretical average cluster AA frequencies by the
    likelihood of an MSA given that cluster.
    Originally implemented to compare iEM and hEM estimates, where the number
    of clusters is equal to the number of MSAs.

    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    per MSA cluster (profile weights)
    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param p_sites_profs: list (n_alns) of arrays (n_sites x n_profiles)
    with probabilities for sites given profiles
    :return: (n_alns x 20) expected AA frequencies per MSA
    """

    if p_sites_profs is not None:
        pass
    elif profs is not None and data is not None:
        p_sites_profs = multi_dens(data, profs)
    else:
        raise ValueError("Either p_sites_profs or profs and aln_counts need to "
                         "be provided to compute lks per site")

    lks_aln_cl, _ = lks_alns_given_cls(p_sites_profs, pro_w)

    th_freqs = theoretical_cl_freqs(profs, pro_w)

    expct_sim_freqs = lks_aln_cl @ th_freqs
    # normalize
    denum = lks_aln_cl.sum(axis=1)  # sum over cluster-dimension
    denum = np.repeat(denum[:, np.newaxis], 20, axis=1)
    expct_sim_freqs /= denum

    return expct_sim_freqs


def joint_log_lk(pro_w, cl_w=None, profs=None, data=None, p_sites_profs=None):
    """Log-likelihood for a set of MSAs given profiles, profile weights and
    cluster weights

    :param pro_w: (n_clusters x n_profiles) mixture component weights
    per MSA cluster (profile weights)
    :param cl_w: (n_clusters) MSA cluster probability (cluster weights)
    if None then there is either only one cluster or iEM parameters and data
    are given where each MSA belongs to a cluster in the given order
    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param p_sites_profs: list (n_alns) of arrays (n_sites x n_profiles)
    with probabilities for sites given profiles
    :return: Log-likelihood value
    """

    ax_s = 1

    if p_sites_profs is not None:
        n_alns = len(p_sites_profs)
    elif profs is not None and data is not None:
        n_alns = len(data)
        p_sites_profs = multi_dens(data, profs)
    else:
        raise ValueError("Either p_sites_profs or profs and aln_counts need to "
                         "be provided to compute lks per site")

    if cl_w is None:  # cluster belongs to MSA in given order
        lks_sites = [lk_per_site(pro_w, p_sites_profs=p_sites_profs[aln])
                     for aln in range(n_alns)]
        log_lks_alns = [np.sum(np.log(lks_sites[aln]), axis=ax_s)
                        for aln in range(len(data))]
        return np.sum(log_lks_alns)

    else:  # consider all cluster-MSA combinations
        lk_aln_cl, consts = lks_alns_given_cls(p_sites_profs, pro_w)
        # weight and sum over cluster-dimension
        log_lks_alns = np.log(np.sum(cl_w * lk_aln_cl, axis=1)) - consts
        log_lk = np.sum(log_lks_alns)

        return log_lk


def em(data, n_iter, init_params, fix_params=None, run=None, trace_estep=False,
       save_path=""):
    """Run iterations of E- and M-step to estimate mixture (profiles) and
    hierarchical proportions of mixture components (profile and MSA-cluster
    weighst)

    :param data: list (n_alns) of arrays with site-wise amino acid counts
    :param n_iter: max. number of EM iterations
    :param init_params: list with arrays that are initial profiles,
    profile weights and cluster weights
    :param fix_params: dictionary with keys that are a subset of 'profs',
    'pro_w' and 'cl_w' and specify parameters that shall be fixed throughout
    the EM and thus won't be estimated
    :param run: integer to identify this EM run
    :param trace_estep: flag to trace lk after E-step as well as lk dips from
    one M-step till the next M-step
    :param save_path: <path/to/dir> where estimated and lks will be saved
    :return: estimates as list of [profiles, profile weights, cluster weights]
    and joint lk and lower bound lk values per iteration
    (or after E-step and M-step)
    """

    try:
        if run is not None:
            print(f'Run {run}')

        # -------- initialization

        start = time.time()

        estim_profs, estim_pro_w, estim_cl_w = init_params

        # optionally fix parameters
        if fix_params is not None:
            if 'profs' in fix_params.keys():
                estim_profs = fix_params['profs']
            if 'pro_w' in fix_params.keys():
                estim_pro_w = fix_params['pro_w']
            if 'cl_w' in fix_params.keys():
                estim_cl_w = fix_params['cl_w']
            if {'profs', 'pro_w', 'cl_w'}.issubset(set(fix_params.keys())):
                raise ValueError('Profiles, profile weights and cluster weights'
                                 'are fixed. No parameters left for estimation '
                                 'by the EM-algorithm')

        p_sites_profs = None  # to pass from lk. computation to next e-step

        # trace joint log lks and vlb (variational lower bound)
        lks_iter = np.zeros(n_iter)
        vlbs_iter = np.zeros(n_iter * 2)

        # -------- run EM iterations

        print('Iterations:')
        for iter in tqdm(range(n_iter)):

            curr_mstep, prev_mstep = iter * 2 + 1, (iter - 1) * 2 + 1
            curr_estep = iter * 2

            sites_pi, alns_pi = e_step(data, estim_profs, estim_pro_w,
                                       estim_cl_w, p_sites_profs)

            if trace_estep:
                vlbs_iter[curr_estep], p_sites_profs = lk_lower_bound(
                    data, estim_profs, estim_pro_w, estim_cl_w, sites_pi,
                    alns_pi)

            estim_cl_w, estim_pro_w, estim_profs = m_step(sites_pi, alns_pi,
                                                          data)

            # check for NANs after estimation
            if np.any(np.isnan(estim_profs)):
                print(f'{np.sum(np.isnan(estim_profs))} NAN(s) in estimated '
                      f'profiles')
            if np.any(np.isnan(estim_profs)):
                print(f'{np.sum(np.isnan(estim_pro_w))} NAN(s) in estimated '
                      f'profile weights')
            if np.any(np.isnan(estim_profs)):
                print(f'{np.sum(np.isnan(estim_cl_w))} NAN(s) in estimated '
                      f'cluster weights')

            if save_path != "":
                # save profiles
                if fix_params is None or 'profs' not in fix_params.keys():
                    np.savetxt(f'{save_path}/profiles_{run + 1}.tsv',
                               estim_profs.T, delimiter='\t')
                # save profile weights
                if fix_params is None or 'pro_w' not in fix_params.keys():
                    if len(estim_cl_w) > 1:
                        for cl in range(len(estim_cl_w)):
                            np.savetxt(f'{save_path}/'
                                       f'cl{cl + 1}_pro_weights_{run + 1}.csv',
                                       estim_pro_w[cl], delimiter=',')
                    else:
                        np.savetxt(f'{save_path}/pro_weights_{run + 1}.csv',
                                   estim_pro_w.T, delimiter=',')
                # save cluster weights
                if fix_params is None or 'cl_w' not in fix_params.keys():
                    if len(estim_cl_w) > 1:
                        np.savetxt(f'{save_path}/cl_weights_{run + 1}.csv',
                                   estim_cl_w, delimiter=',')

            # fix parameters
            if fix_params is not None:
                if 'profs' in fix_params.keys():
                    estim_profs = fix_params['profs']
                if 'pro_w' in fix_params.keys():
                    estim_pro_w = fix_params['pro_w']
                if 'cl_w' in fix_params.keys():
                    estim_cl_w = fix_params['cl_w']

            # -------- trace lk maximization

            vlbs_iter[curr_mstep], p_sites_profs = lk_lower_bound(
                data, estim_profs, estim_pro_w,
                estim_cl_w, sites_pi, alns_pi)
            lks_iter[iter] = joint_log_lk(estim_pro_w, estim_cl_w,
                                          estim_profs,
                                          p_sites_profs=p_sites_profs)

            if not trace_estep:
                vlbs_iter[curr_estep] = vlbs_iter[curr_mstep]

            # -------- EM stop condition
            if iter > 0:
                vlb_change = vlbs_iter[prev_mstep] - vlbs_iter[curr_mstep]
                vlb_change = np.abs(vlb_change / vlbs_iter[prev_mstep])
                if vlb_change < RTOL:
                    if iter < n_iter - 1:
                        vlbs_iter[curr_mstep:] = vlbs_iter[curr_mstep]
                        lks_iter[iter:] = lks_iter[iter]
                    print(
                        f'Run {run} : finished after {iter + 1} (max. {n_iter}) '
                        f'iterations\n')
                    break

        print(f'EM {run}, {iter} : {time.time() - start} s\n')

        return [estim_profs, estim_pro_w, estim_cl_w], vlbs_iter, lks_iter

    except KeyboardInterrupt:
        print(f'Keyboard interrupt in process: run {run + 1}')
    finally:
        print(f'cleaning up thread (run {run + 1})')


def draw_rand_params(n_clusters, n_profiles, n_alns, n_aas=20):
    """Draw profiles and profile weights from a Dirichlet distribution and
    cluster weights uniformly

    :param n_clusters: number of clusters
    :param n_profiles: number of profiles
    :param n_alns: number of alignments
    :param n_aas: number of amino acids, default: 20
    :return: (n_profiles x n_aas) profiles, (n_clusters x n_profiles) profile
    weights, (n_clusters) cluster weights
    """

    # init profiles
    profs = dirichlet.rvs([2 * n_aas] * n_aas, n_profiles)

    # init profile probabilities per cluster
    pro_w = dirichlet.rvs([2 * n_profiles] * n_profiles, n_clusters)

    # init cluster probabilities
    weights = np.random.randint(1, n_alns, n_clusters)
    cl_w = weights / weights.sum()

    return profs, pro_w, cl_w


def init_estimates(n_runs, n_clusters, n_profiles, n_alns, n_aas=20,
                   equal_inits=False, true_params=None, init_pro_w_strat=None):
    """Initialize profiles, profile weights and cluster weights randomly for a
    given number of runs

    :param n_runs: number of runs
    :param n_clusters: number of clusters
    :param n_profiles: number of profiles
    :param n_alns: number of MSAs
    :param n_aas: number of amino acids, default: 20
    :param equal_inits: if True init. of first run will have equal parameters,
    i.e. 1/20 for all profile values, 1/n_profiles for all profile weights and
    1/n_clusters for all cluster weights, a small amount of noise is added to
    all values, otherwise all parameters are drawn randomly
    :param true_params: list of [profiles, profile weights, cluster weights]
    that is used either as init. of the 1. or 2. (if equal_inits is set) run.
    Initially used to test EM when starting from true parameters
    :param init_pro_w_strat:
    default: draw profile weights for all clusters of a run 'in one go' using
    scipy's dirichlet function with size=n_clusters (see function
    draw_rand_params) (referred to as 'Seed [72]')
    1: for profile weights init. call draw_rand_params for each cluster
    separately such that there are random operation inbetween each call of the
    dirichlet function which now has size=1 (referred to as 'Seed [72] with
    other random operation in between initialization of profile weights of two
    clusters')
    2: Like default, except that there is an explicit random seed set for each
    run before drawing profile weights (referred to as 'Seed per EM run
    [1,...,15]')
    :return: list (n_runs) of lists with [profiles, profile weights,
    cluster weights] containing initial parameters for each run
    """

    params = []
    # draw all profile weights for all clusters at once successively
    # for all runs
    for run in range(n_runs):  # different init. per run
        params.append(draw_rand_params(n_clusters, n_profiles, n_alns,
                                       n_aas))

    if init_pro_w_strat == 1:
        # meme strategy initially implemented for iEM init
        # which resulted in better EM results (needs further investigation)
        for cl in range(n_clusters):
            for run in range(n_runs):  # init. profile weights
                params[run][1][cl] = draw_rand_params(1, n_profiles, n_alns,
                                                      n_aas)[1][0]
    elif init_pro_w_strat == 2:
        # set different random seed for each run for profile
        # weights
        for run in range(n_runs):  # different init. per run
            np.random.seed(run + 1)
            params[run][1] = draw_rand_params(n_clusters, n_profiles, n_alns,
                                              n_aas)[1]

    if equal_inits:  # first run init. with equal params.
        cl_w = np.ones(n_clusters) / n_clusters
        pro_w = np.ones((n_clusters, n_profiles)) / n_profiles
        profs = np.ones((n_profiles, n_aas)) / n_aas

        # add arbitrary noise
        cl_w += np.random.uniform(-(1 / n_clusters) * 1e-5,
                                  (1 / n_clusters) * 1e-5,
                                  n_clusters)
        pro_w += np.random.uniform(-(1 / n_profiles) * 1e-5,
                                   (1 / n_profiles) * 1e-5,
                                   (n_clusters, n_profiles))
        profs += np.random.uniform(-1e-5, 1e-5, (n_profiles, n_aas))
        # normalize
        cl_w /= cl_w.sum()
        pro_w /= np.repeat(pro_w.sum(axis=1)[:, np.newaxis], n_profiles,
                           axis=1)
        profs /= np.repeat(profs.sum(axis=1)[:, np.newaxis], n_aas, axis=1)

        params[0] = [profs, pro_w, cl_w]

    if true_params is not None:  # 1. or 2. run init with correct params
        if not equal_inits or n_runs == 1:
            params[0] = true_params
        else:
            params[1] = true_params

    return params


def select_init_msas(emp_freq_pcs, n_runs, n_clusters, strat='circle'):
    """Legacy function with alternative initialization strategy, where for each
    cluster one MSA is selected based on coordinates of PC 1 and 2 (centered)
    of average AA frequencies of MSAs (1 data point = 1 MSA), to represent the
    center of a cluster or the average cluster AA frequencies.

    Strategy 'circle':
    Since the PCA is a circular shaped point cloud a circle with the radius of
    the 0.8 quantile of all distances to the center includes about 80% of the
    MSAs. The MSA seletion consists of the clostest MSA to the center and the
    clostest MSAs to equally spaced coordinates on the circle. For each run
    the coordinates are slightly rotated on the circle. In a second step (not
    in this function) the EM-algorithm is run on each selected MSA to estimated
    init. parameters.
    Strategy 'kmeans':
    Cluster with kmeans for MSA selection and/or to get initial cluster weights.

    :param strat: string to choose either 'kmeans' or 'circle' strategy
    :param emp_freq_pcs: (n_alns x 2) first two principal components of average
    MSA AA frequencies
    :param n_runs: number of EM runs
    :param n_clusters: number of Clusters
    :return: (n_runs x n_clusters) indices of selected MSAs
    """

    n_runs, n_alns = int(n_runs), len(emp_freq_pcs)
    msa_inds = np.zeros((n_runs, n_clusters))
    target_coords = np.zeros((n_runs, n_clusters, 2))

    if strat == 'kmeans':
        cl_w_runs = np.zeros((n_runs, n_clusters))
        # kmeans on PC1 and 2
        for run in range(int(n_runs)):
            kmeans_pcs = KMeans(n_clusters=n_clusters,
                                random_state=run + 1).fit(emp_freq_pcs)
            target_coords[run] = kmeans_pcs.cluster_centers_
            # set em init cluster weights
            for cl in range(n_clusters):
                cl_w_runs[run, cl] = sum(kmeans_pcs.labels_ == cl) / n_alns

    elif strat == 'circle':
        # euclidean distance to center
        dists = np.sum(emp_freq_pcs ** 2, axis=1) ** 0.5

        # first cluster is close to the center of PC 1 and 2
        msa_inds[:, 0] = np.argsort(dists)[0:n_runs]
        target_coords[:, 0] = emp_freq_pcs[msa_inds[:, 0].astype(np.int32)]

        if n_clusters > 1:
            # circle includes about 80% of the data
            # only works if PCA has circular and not too ellipse-like shape
            r = np.quantile(dists, 0.8)
            # cluster angles
            angles = (360 / (n_clusters - 1)) * np.arange(1, n_clusters)
            angles = np.repeat(angles[np.newaxis, :], n_runs,
                               axis=0)
            # get rotation degrees - each run has different points on pca-circle
            rot_angles = (360 / (n_clusters - 1) / n_runs)
            rot_angles *= np.arange(0, n_runs)
            # rotate angles and normalize
            angles = np.add(angles, rot_angles[:, None])
            angles = (angles % 360 + 360) % 360

            # transform into cartesian coordinates
            thetas = np.deg2rad(angles)
            target_coords[:, 1:] = pol2cart(r, thetas).reshape((n_runs,
                                                                n_clusters - 1,
                                                                2))
    else:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'strat parameter must be either "circle" or "kmeans" '
                         f'but is "{strat}"')

    # select MSA closest to determined target coordiantes
    for run in range(n_runs):
        for cl, target in enumerate(target_coords[run]):
            dists = np.sum((emp_freq_pcs - target) ** 2, axis=1) ** 0.5
            if not cl == 0 and not strat == 'circle':
                msa_inds[run, cl] = int(np.argmin(dists))
            # otherwise msa_inds are centers determined above

    if strat == 'kmeans':
        return msa_inds, cl_w_runs
    elif strat == 'circle':
        return msa_inds


def generate_alns(profs, pro_w, cl_w, n_alns, n_sites, n_seqs):
    """Function to generate AA counts for EM debugging.
    Data is not drawn randomly, but generated without noise so that it
    accurately reflects profiles and weights.

    :param profs: (n_profiles x 20) multinomial amino acid mixture weights
    :param pro_w: (n_clusters x n_profiles) mixture component weights
    (profile weights)
    :param cl_w: (n_clusters) MSA cluster probability (cluster weights)
    :param n_alns: number of MSAs to be generated
    :param n_sites: number of sites
    :param n_seqs: number of sequences
    :return: list (n_alns) of arrays with site-wise amino acid counts
    """

    n_aas = profs.shape[1]
    n_profiles = pro_w.shape[1]
    n_clusters = pro_w.shape[0]

    # assign clusters and profiles according to given proportions
    cluster_alns_asso = np.concatenate([int(cl_w[i] * n_alns) * [i]
                                        for i in range(n_clusters)], axis=0)
    profile_site_asso = np.asarray([np.concatenate([int(pro_w[cl, i] * n_sites)
                                                    * [i]
                                                    for i in range(n_profiles)],
                                                   axis=0)
                                    for cl in cluster_alns_asso])
    aa_counts = []
    for aln in range(n_alns):
        sites_counts = np.zeros((n_sites, n_aas))
        for site in range(n_sites):
            sites_counts[site] = (profs[profile_site_asso[aln, site]] *
                                  n_seqs).astype(int)
        aa_counts.append(sites_counts)

    return aa_counts


def estimated_param_sorting(true_profs, estim_profs, true_pro_w, estim_pro_w,
                            true_cl_w, estim_cl_w):
    """Compute mean absolute error of estimated and true parameters for all
    possible profile and cluster orders and thereby choose best order with
    minimal error

    :param true_profs: (n_profiles x 20) true multinomial amino acid
    mixture weights (profiles)
    :param estim_profs: (n_profiles x 20) estimated profiles
    :param true_pro_w: (n_clusters x n_profiles) true mixture component weights
    (profile weights)
    :param estim_pro_w: (n_clusters x n_profiles) estimated profile weights
    :param true_cl_w: (n_clusters) true MSA cluster probability
    (cluster weights)
    :param estim_cl_w: (n_clusters) estimated cluster weights
    :return: tuple containing list with sorted estimated profiles, profile
    weights and cluster weights and a list with corresponding mean absolute
    errors of true and estimated parameters
    """

    n_profiles, n_clusters = len(true_profs), len(true_pro_w)

    profiles_inds = list(range(n_profiles))
    profiles_permuts = np.asarray(list(set(permutations(profiles_inds))))

    mae_profiles = np.zeros((len(profiles_permuts)))
    for i, order in enumerate(profiles_permuts):
        # mean absolute error of true profiles and different orders of
        # estimated profiles
        mae_profiles[i] = np.mean(np.abs(true_profs - estim_profs[order, :]))

    ind_min_prof_mae = np.argmin(mae_profiles)  # choose order with lowest error
    best_profile_order = profiles_permuts[ind_min_prof_mae, :]
    sorted_estim_profs = estim_profs[best_profile_order, :]

    # sort profile weights given order obtained by profile maes
    sorted_estim_pro_w = np.take_along_axis(estim_pro_w,
                                            np.repeat(
                                                best_profile_order[np.newaxis],
                                                n_clusters, axis=0),
                                            axis=1)

    # get optimal cluster order
    clusters_inds = list(range(n_clusters))
    clusters_permuts = np.asarray(list(set(permutations(clusters_inds))))

    mae_profile_weights = np.zeros((len(clusters_permuts)))
    for i, order in enumerate(clusters_permuts):
        # cluster order according to profile weights
        mae_profile_weights[i] = np.mean(np.abs(true_pro_w -
                                                sorted_estim_pro_w[order]))
    ind_min_pro_w_mae = np.argmin(mae_profile_weights)
    best_cluster_order = clusters_permuts[ind_min_pro_w_mae, :]

    maes = [mae_profiles[ind_min_prof_mae],
            mae_profile_weights[ind_min_pro_w_mae],
            np.mean(np.abs(true_cl_w - estim_cl_w[best_cluster_order]))
            ]

    return [sorted_estim_profs, sorted_estim_pro_w[best_cluster_order],
            estim_cl_w[best_cluster_order]], maes


def get_dips(lks):
    """Detect dips of loglks from previous to current iteration
    :param lks: (n_runs, n_iters) (lower bound) loglks of data
    :return: (n_runs, n_iters) containing dip if lk dip and 0 otherwise
    """
    dips = np.zeros_like(lks)
    prev = lks[:, :-1]
    curr = lks[:, 1:]
    dips_inds = curr < prev
    dips_diff = np.abs(curr - prev)
    dips[:, 1:][dips_inds] = dips_diff[dips_inds]

    return dips

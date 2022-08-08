import multiprocessing
import os
import sys
import time
import warnings

from itertools import permutations
from scipy.stats import multinomial, dirichlet
from matplotlib import pylab as plt, rcParams
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

from preprocessing import raw_alns_prepro
from utils import read_config_file, split_lst, pol2cart
from stats import count_aas

np.random.seed(72)

MINPOSFLOAT = np.nextafter(0, 1)
MINNEGFLOAT = np.nextafter(-np.inf, 0)

AAS = 'ARNDCQEGHILKMFPSTWYV'

RTOL = 1e-5
STARTTIME = time.time()


def multi_dens(data, profiles):
    n_alns = len(data)
    n_profiles = profiles.shape[0]
    sites_profile_probs = []
    for i in range(n_alns):
        # -------- P(A_i | v_k) for all sites
        n_aas_site = data[i].sum(axis=-1)
        sites_profile_probs.append(
            np.asarray([multinomial.pmf(data[i], n_aas_site,
                                        profiles[k] / profiles[k].sum()
                                        if profiles[k].sum() > 1
                                        else profiles[k])
                        # np.asarray([np.prod(profiles[k]**data[i], axis=1)
                        for k in range(n_profiles)]).T)
        sites_profile_probs[i][sites_profile_probs[i] == 0] = MINPOSFLOAT
    return sites_profile_probs


def log_probs_rems(p_sites_profs, pro_w):
    n_sites = p_sites_profs.shape[0]
    n_clusters = pro_w.shape[0]
    n_profiles = pro_w.shape[1]

    # site_masks = ~np.eye(n_sites, dtype=bool)
    log_rems_cl = np.zeros((n_sites, n_clusters))

    p_sites_profs_rep = np.repeat(p_sites_profs[np.newaxis, :, :], n_sites,
                                  axis=0)

    # p_sites_profs_rep_masked = (p_sites_profs_rep *
    #                            np.repeat(site_masks[:, :, np.newaxis],
    #                                      n_profiles, axis=2))

    # set diagonal to 0 to exclude a site
    diag_inds = np.diag_indices(n_sites, 2)
    p_sites_profs_rep[diag_inds[0], diag_inds[1], :] = 0

    for cl in range(n_clusters):
        # prob. for remaining sites
        weighted_sites_notj = p_sites_profs_rep @ pro_w[cl]
        weighted_sites_notj[diag_inds] = 1  # from 0 to 1 such that log(1) = 0
        log_rems_cl[:, cl] = np.sum(np.log(weighted_sites_notj), axis=1)

    return log_rems_cl


def e_step(data, profs, pro_w, cl_w, probs_sites=None):
    n_alns = len(data)
    n_profiles = profs.shape[0]
    n_clusters = cl_w.shape[0]
    ax_p, ax_c, ax_s = 2, 1, 1

    # ******************************* E-STEP  ******************************* #
    # pi : probability for profile z and cluster c at site j of alignment i
    #      given weights and profiles and aa counts
    log_pi = [np.zeros((data[aln].shape[0], n_clusters, n_profiles))
              for aln in range(n_alns)]
    pi = [np.zeros((data[aln].shape[0], n_clusters, n_profiles))
          for aln in range(n_alns)]
    log_aln_cl = np.zeros((n_alns, n_clusters))

    if probs_sites is None:
        p_sites_profs = multi_dens(data, profs)
        log_rems_alns = [log_probs_rems(p_sites_profs[aln], pro_w)
                         for aln in range(n_alns)]
    else:
        log_rems_alns = probs_sites[1]
        p_sites_profs = probs_sites[0]

    for aln in range(n_alns):

        n_sites = data[aln].shape[0]

        # -------- lk on alignment level
        log_aln_cl[aln] = np.sum(np.log(np.dot(pro_w, p_sites_profs[aln].T)),
                                 axis=ax_s)
        log_aln_cl[aln] += np.log(cl_w)

        # part of the formula can be replaced by dot product(/matmul)
        # alter_dot = np.sum([p_sites_profs[aln] * pro_w[c]
        #                    for c in range(n_clusters)], axis=2)
        # dot = np.dot(pro_w, p_sites_profs[aln].T)
        # np.all(alter_dot == dot)

        # -------- lk on site level : pi
        log_sites_profs = np.log(p_sites_profs[aln])  # n_sites x n_profiles

        # (n_sites x) n_cl x n_pro
        log_pi[aln] = np.repeat(np.log(pro_w)[np.newaxis, :, :], n_sites,
                                axis=0)
        # (n_sites x) n_cl (x n_pro)
        log_pi[aln] += np.repeat(np.repeat(np.log(cl_w)[np.newaxis, :], n_sites,
                                           axis=0)[:, :, np.newaxis],
                                 n_profiles, axis=2)
        # n_sites x n_cl (x n_pro)
        log_pi[aln] += np.repeat(log_rems_alns[aln][:, :, np.newaxis],
                                 n_profiles, axis=2)
        # n_sites (x n_cl x) n_pro
        log_pi[aln] += np.repeat(log_sites_profs[:, np.newaxis, :], n_clusters,
                                 axis=1)

        if np.any(log_pi[aln] == -np.inf):
            warnings.warn(
                f"In MSA {aln + 1} {np.sum(log_pi[aln] == -np.inf)}/"
                f"{n_sites * n_clusters * n_profiles} log_pi's are "
                f"-infinity")

            # log_pi[aln][log_pi[aln] == -np.inf] = np.nextafter(-np.inf, 0)

        # -------- log to prob (n_sites x n_cl x n_pro)

        # max log-lk per site
        max_logs = np.max(np.max(log_pi[aln], axis=ax_p), axis=ax_c)

        # add cluster and profile axis to recover shape
        max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=ax_c)
        max_logs = np.repeat(max_logs[:, :, np.newaxis], n_profiles, axis=ax_p)

        pi[aln] = np.exp(log_pi[aln] + np.abs(max_logs))
        pi[aln][pi[aln] == 0] = MINPOSFLOAT

        # normalizing pi
        sum_over_pro_cl = pi[aln].sum(axis=ax_p).sum(axis=ax_c)

        # add cluster and profile axis to recover shape
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, np.newaxis], n_clusters,
                                    axis=1)
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, :, np.newaxis],
                                    n_profiles, axis=2)

        pi[aln] /= sum_over_pro_cl

        # if not np.all(pi[aln].sum(axis=ax_p).sum(axis=ax_c) == 1):
        # print(list(pi[aln].sum(axis=ax_p).sum(axis=ax_c)))

        pi[aln][pi[aln] == 0] = MINPOSFLOAT

    # log to prob for alignment level lk
    max_logs = np.max(log_aln_cl, axis=ax_c)  # per MSA
    max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=ax_c)
    aln_pi = np.exp(log_aln_cl + np.abs(max_logs))
    # normalize
    aln_pi = aln_pi / np.repeat(np.sum(aln_pi, axis=1)[:, np.newaxis],
                                n_clusters, axis=1)

    # if not np.all(aln_pi.sum(axis=ax_c) == 1):
    # print(list(aln_pi.sum(axis=ax_c)))

    """
    # verify pi & aln_pi
    if np.all(aln_pi.argmax(axis=ax_c) == cluster_alns_asso):
        # (n_aln, n_sites, n_cl, !n_pro!)
        pro_site_inds = np.argmax(np.asarray(pi), axis=3)

        for aln, cl in enumerate(cluster_alns_asso):
            if not np.all(pro_site_inds[aln, :, cl] == profile_site_asso[cl, :]):
                print(list(pro_site_inds[aln, :, cl]
                           == profile_site_asso[cl, :]))
    """
    return pi, aln_pi


def m_step(site_pi, aln_pi, data):
    ax_c, ax_p = 1, 2
    ax_s, ax_aa = 0, 2

    n_alns = len(site_pi)
    n_aas = data[0].shape[-1]
    n_profiles, n_clusters = site_pi[0].shape[ax_p], site_pi[0].shape[ax_c]

    # n_aln x n_cl (x n_pro x n_aa)
    aln_pi_rep = np.repeat(aln_pi[:, :, np.newaxis], n_profiles, axis=ax_p)
    aln_pi_aa_rep = np.repeat(aln_pi_rep[:, :, :, np.newaxis], n_aas, axis=3)

    # cluster weights
    sum_over_alns = np.sum(aln_pi, axis=0)
    aln_pi_denum = np.sum(sum_over_alns, axis=0)
    aln_pi_denum = np.repeat(aln_pi_denum, n_clusters)
    # usually division by 1
    estim_cluster_weights = sum_over_alns / aln_pi_denum

    # profile weights
    sum_over_sites_alns = np.zeros((n_clusters, n_profiles))
    for aln in range(n_alns):
        sum_over_sites = np.sum(site_pi[aln], axis=0)
        sum_over_sites_alns += aln_pi_rep[aln] * sum_over_sites

    sum_over_prof = np.sum(sum_over_sites_alns, axis=1)
    sum_over_prof = np.repeat(sum_over_prof[:, np.newaxis], n_profiles,
                              axis=1)

    # weight by cluster probabilities
    estim_profile_weights = sum_over_sites_alns / sum_over_prof

    # profiles
    cl_profiles = np.zeros((n_clusters, n_profiles, n_aas))
    for aln in range(n_alns):
        for cl in range(n_clusters):
            site_pi_aas = np.repeat(site_pi[aln][:, cl, :, np.newaxis], n_aas,
                                    axis=ax_aa)  # n_sites x n_pro (x n_aas)
            aa_counts_pro = np.repeat(data[aln][:, np.newaxis, :], n_profiles,
                                      axis=1)  # n_sites (x n_pro x) n_aas
            weighted_counts = site_pi_aas * aa_counts_pro

            # sum over alns and sites
            cl_profiles += aln_pi_aa_rep[aln] * np.sum(weighted_counts,
                                                       axis=ax_s)

    estim_profiles = cl_profiles.sum(axis=0)  # sum over clusters
    # normalize
    denum = estim_profiles.sum(axis=1)  # sum over aas
    denum = np.repeat(denum[:, np.newaxis], n_aas, axis=1)  # usually 1
    estim_profiles /= denum

    return estim_cluster_weights, estim_profile_weights, estim_profiles


def compute_vlb_fl(data, cl_w, pro_w, profs, site_pi, aln_pi):
    n_alns = len(data)
    n_profiles = profs.shape[0]
    n_clusters = cl_w.shape[0]

    # avoid division by 0
    pro_w[pro_w == 0] = MINPOSFLOAT
    cl_w[cl_w == 0] = MINPOSFLOAT

    # compute lower bound of full lk
    lk = 0

    # -------- log lk for sites given profiles
    sites_profs_probs = multi_dens(data, profs)
    sites_profile_probs_zeros = sites_profs_probs.copy()
    # avoid -inf (divide by zero encountered in log error)
    for aln in range(n_alns):
        sites_profs_probs[aln][sites_profs_probs[aln] == 0] = MINPOSFLOAT

    # -------- log lk for remaining sites
    log_other_sites_cl = [log_probs_rems(sites_profs_probs[aln],
                                         pro_w)
                          for aln in range(n_alns)]

    for aln in range(n_alns):
        n_sites = data[aln].shape[0]

        # -------- log Mult(v_s, A_ij)
        log_sites = np.repeat(np.log(sites_profs_probs[aln])[:, np.newaxis, :],
                              n_clusters, axis=1)
        # log_sites += np.repeat(log_other_sites_cl[aln][:, :, np.newaxis],
        #                       n_profiles, axis=2)

        # -------- log p_rs
        log_pro_w_rep = np.repeat(np.log(pro_w)[np.newaxis, :, :],
                                  n_sites, axis=0)

        # -------- computing lower bound log-lk
        lk_12 = (site_pi[aln] * log_sites) + (site_pi[aln] * log_pro_w_rep)
        lk_12[np.isnan(lk_12)] = 0  # such that 0 * -inf = 0

        lk += np.sum(lk_12)

    # -------- lk on alignment level
    lk += np.sum(aln_pi * np.log(cl_w))

    return lk, [sites_profile_probs_zeros, log_other_sites_cl]


def lk_per_site(aln_counts, profiles, weights):
    prob_site_prof = np.asarray([multinomial.pmf(aln_counts,
                                                 aln_counts.sum(axis=-1),
                                                 profile)
                                 for profile in profiles])
    return prob_site_prof.T @ weights


def lks_alns_given_cls(alns_counts, pro_w, p_sites_profs, log=False):
    n_alns, n_cl = len(alns_counts), len(pro_w)
    lk_aln_cl = np.zeros((n_alns, n_cl))
    log_lk_aln_cl = np.zeros((n_alns, n_cl))
    consts = np.zeros(n_alns)
    for aln in range(n_alns):
        p_sites_cl = pro_w @ p_sites_profs[aln].T
        log_lk_aln_cl[aln] = np.sum(np.log(p_sites_cl), axis=1)
        consts[aln] = np.abs(np.max(log_lk_aln_cl[aln]))
        lk_aln_cl[aln] = np.exp(log_lk_aln_cl[aln] + consts[aln])

    if log:
        return log_lk_aln_cl + consts
    else:
        return lk_aln_cl, consts


def theoretical_cl_freqs(profiles, weights):
    return weights @ profiles


def expected_sim_freqs(counts, profiles, pro_w):
    p_sites_profs = multi_dens(counts, profiles)
    lk_aln_cl, _ = lks_alns_given_cls(counts, pro_w, p_sites_profs)

    th_freqs = np.asarray(
        [theoretical_cl_freqs(profiles, weights) for weights in
         pro_w])

    expected_sim_freqs = np.asarray(
        [aln_i_lk_cls @ th_freqs for aln_i_lk_cls in lk_aln_cl])

    return expected_sim_freqs


def full_log_lk(data, profs, pro_w, cl_w=None):
    ax_s = 1
    p_sites_profs = multi_dens(data, profs)

    if cl_w is None:  # cluster belongs to MSA in given order
        return np.sum([np.sum(np.log(pro_w @ p_sites_profs[aln].T), axis=ax_s)
                       for aln in range(len(data))])
    else:  # consider all cluster-MSA combinations
        lk_aln_cl, consts = lks_alns_given_cls(data, pro_w, p_sites_profs)
        log_lk = np.sum(np.log(np.sum(cl_w * lk_aln_cl, axis=1)) - consts)

        return log_lk


def em(init_params, profiles, aa_counts, n_iter, run=None,
       test=False,
       save_path=""):
    try:
        if run is not None:
            print(f'Run {run}')

        start = time.time()

        estim_profiles, estim_profile_weights, estim_cluster_weights = init_params

        n_clusters = len(estim_cluster_weights)

        # fix parameters
        if not test:
            estim_profiles = profiles
            # estim_profile_weights = profile_weights
            # estim_cluster_weights = cluster_weights

        probs_sites = None  # will be passed from lk. computation to next e-step

        lks_iter = np.zeros(n_iter)
        vlbs_iter = np.zeros((n_iter * 2))
        vlbs_dips_iter = np.zeros((n_iter * 2, 2))

        for iter in tqdm(range(n_iter)):

            e_step_start = time.time()

            pi, p_aln_cl = e_step(aa_counts, estim_profiles,
                                  estim_profile_weights,
                                  estim_cluster_weights,
                                  probs_sites)

            # print(f'\t E-step finished {int(time.time() - STARTTIME)}s (took '
            #      f'{int(time.time() - e_step_start)}s)')

            # vlbs_iter[iter * 2], _ = compute_vlb_fl(aa_counts,
            #                                               estim_cluster_weights,
            #                                               estim_profile_weights,
            #                                               estim_profiles, pi,
            #                                               p_aln_cl)

            estim_cluster_weights, estim_profile_weights, estim_profiles = m_step(
                pi, p_aln_cl, aa_counts)

            if np.any(np.isnan(estim_profiles)):
                print(f'{np.sum(np.isnan(estim_profiles))} NAN')
            # print(f'\t M-step finished {int(time.time() - STARTTIME)}s')

            if save_path != "":
                """print(estim_profiles)
                np.savetxt(f'{save_path}/profiles_{run + 1}.tsv',
                           estim_profiles.T, delimiter='\t')"""
                if len(estim_cluster_weights) > 1:
                    np.savetxt(f'{save_path}/cl_weights_{run + 1}.csv',
                               estim_cluster_weights, delimiter=',')
                    for cl in range(len(estim_cluster_weights)):
                        np.savetxt(
                            f'{save_path}/cl{cl + 1}_pro_weights_{run + 1}.csv',
                            estim_profile_weights[cl], delimiter=',')
                else:
                    np.savetxt(f'{save_path}/pro_weights_{run + 1}.csv',
                               estim_profile_weights.T, delimiter=',')

            # fix parameters
            # estim_cluster_weights = cluster_weights
            # estim_profile_weights = profile_weights
            if not test:
                estim_profiles = profiles

            # *************************** lk *************************** #

            lks_iter[iter] = full_log_lk(aa_counts, estim_profiles,
                                         estim_profile_weights,
                                         estim_cluster_weights)

            vlbs_iter[iter * 2 + 1], probs_sites = compute_vlb_fl(
                aa_counts, estim_cluster_weights, estim_profile_weights,
                estim_profiles, pi, p_aln_cl)

            # this could be the vlb after the e-step
            vlbs_iter[iter * 2] = vlbs_iter[iter * 2 + 1]

            # print(f'\t Computed lk {int(time.time() - STARTTIME)}s')

            # get vlb drops
            if iter > 0:
                # only accounts for dips after e-step
                if vlbs_iter[iter * 2 + 1] < vlbs_iter[(iter - 1) * 2 + 1]:
                    curr_dip = np.abs(vlbs_iter[iter * 2 + 1] -
                                      vlbs_iter[(iter - 1) * 2 + 1])

                    vlbs_dips_iter[iter * 2] = np.asarray([curr_dip, iter])
                    vlbs_dips_iter[iter * 2 + 1] = np.asarray([curr_dip, iter])

            if iter > 0 and np.abs(
                    (vlbs_iter[(iter - 1) * 2 + 1] - vlbs_iter[iter * 2 + 1])
                    / vlbs_iter[(iter - 1) * 2 + 1]) < RTOL:
                if iter < n_iter - 1:
                    vlbs_iter[iter * 2 + 1:] = vlbs_iter[iter * 2 + 1]
                print(f'Run {run} : finished after {iter + 1} (max. {n_iter}) '
                      f'iterations')
                break

            """
            if vlbs_iter[iter * 2 + 1] < vlbs_iter[iter * 2]:
                curr_dip = np.abs(vlbs_iter[iter * 2 + 1] -
                                  vlbs_iter[iter * 2])

                dips_iter[iter * 2 + 1] = np.asarray([curr_dip, iter])

                if iter > 0:
                    print(f'{vlbs_iter[iter * 2 + 1]} < '
                          f'{vlbs_iter[iter * 2]}')
                    print(f'dip : {curr_dip}')
                    print('prev. profiles')
                    print(estimates_iter[iter - 1][0])
                    print('curr. profiles')
                    print(estim_profiles)
                    print('prev. profile w.')
                    print(estimates_iter[iter - 1][1])
                    print('curr. profile w.')
                    print(estim_profile_weights)
                    print('prev. cl. w.')
                    print(estimates_iter[iter - 1][2])
                    print('curr. cl. w.')
                    print(estim_cluster_weights)
            estimates_iter.append(
                [estim_profiles, estim_profile_weights, estim_cluster_weights])
            """

        print(f'EM {run}, {iter} : {time.time() - start} s')

        return ([estim_profiles, estim_profile_weights, estim_cluster_weights],
                vlbs_iter, vlbs_dips_iter, lks_iter)

    except KeyboardInterrupt:
        print("Keyboard interrupt in process: ", run)
    finally:
        print("cleaning up thread", run)


def init_estimates(n_runs, n_clusters, n_profiles, n_aas, n_alns, test,
                   true_params=None):
    params = []
    for run in range(n_runs):
        if run == 0 and test:  # uniform
            cl_w = np.ones(n_clusters) / n_clusters
            pro_w = np.ones(
                (n_clusters, n_profiles)) / n_profiles
            prof = np.ones((n_profiles, n_aas)) / n_aas
        elif run == 1 and test and len(true_params) == 3:  # correct params
            cl_w, pro_w, prof = true_params
        else:
            # init profiles
            prof = dirichlet.rvs([2 * n_aas] * n_aas, n_profiles)

            # init profile probabilities per cluster
            # pro_w = dirichlet.rvs([2 * n_profiles] * n_profiles, n_clusters)
            pro_w = np.zeros((n_clusters, n_profiles))
            for cl in range(n_clusters):
                alphas = np.random.gamma(np.random.uniform(0.5, 5),
                                         np.random.randint(1, 3), n_profiles)
                pro_w[cl] = dirichlet.rvs(alphas)[0]

            # init cluster probabilities
            weights = np.random.randint(1, n_alns, n_clusters)
            cl_w = weights / weights.sum()

        params.append([prof, pro_w, cl_w])

    return params


def select_init_msa(data, n, n_cl):
    msa_inds = np.zeros((n, n_cl))
    dists = np.sum(data ** 2, axis=1) ** 0.5
    # center cluster
    msa_inds[:, 0] = np.argsort(dists)[0:int(n)]

    if n_cl > 1:
        r = np.quantile(dists, 0.8)
        # cluster angles
        angles = (360 / (n_cl - 1)) * np.arange(1, n_cl)
        # per run (use this method for half of the runs)
        angles = np.repeat(angles[np.newaxis, :], int(n), axis=0)
        # get rotation degrees
        rot_angles = (360 / (n_cl - 1) / int(n))
        rot_angles *= np.arange(0, int(n))
        # rotate angles and normalize
        angles = np.add(angles, rot_angles[:, None])
        angles = (angles % 360 + 360) % 360

        thetas = np.deg2rad(angles)
    else:
        thetas = int(n) * [[]]

    for run in range(int(n)):
        for i, theta in enumerate(thetas[run]):
            target = pol2cart(r, theta)
            dists = np.sum((data - target) ** 2, axis=1) ** 0.5
            msa_inds[run, i + 1] = int(np.argmin(dists))

    return msa_inds


def main(args):
    # ****************************** PARAMETERS ****************************** #

    em_config_path = args[0]
    em_config = read_config_file(em_config_path)
    n_alns, n_iter, n_runs, test, n_profiles, n_clusters, n_proc, fasta_in_path, \
    config_path_msa, profile_path, save_path = em_config.values()

    if not test:
        # load alns
        if not os.path.exists(f'{save_path}'):
            os.mkdir(f'{save_path}')
        if not os.path.exists(f'{save_path}/lk'):
            os.mkdir(f'{save_path}/lk')  # for saving likelihoods
        if not os.path.exists(f'{save_path}/init_weights'):
            os.mkdir(f'{save_path}/init_weights')
        profiles = np.genfromtxt(profile_path, delimiter='\t').T

        config = read_config_file(config_path_msa)
        all_n_alns = config['data']['nb_alignments']
        raw_alns, raw_fastas, config['data'] = raw_alns_prepro([fasta_in_path],
                                                               config['data'])

        print(f'Alignments loaded : {int(time.time() - STARTTIME)}s')

        sample_inds = np.arange(0, all_n_alns,
                                np.round(all_n_alns /
                                         n_alns))[:n_alns].astype(int)
        if len(sample_inds) < n_alns:
            sample_inds = np.concatenate(
                (sample_inds, sample_inds[:(n_alns - len(sample_inds))] + 1))

        alns = [raw_alns[0][ind] for ind in sample_inds]
        fastas = [raw_fastas[0][ind] for ind in sample_inds]

        if save_path != '':
            np.savetxt(f'{save_path}/init_weights/real_fastanames4estim.txt',
                       fastas, delimiter=',', fmt='%s')

        print(config['data'])

        aa_counts = [count_aas([aln], 'sites').T for aln in alns]
        print(f'Count vectors generated : {int(time.time() - STARTTIME)}s')

        n_aas = aa_counts[0].shape[-1]

        print(f'\nEstimation on {len(alns)} MSAs\n')

    else:
        n_alns = 80
        n_sites = 40
        n_seqs = 40
        n_aas = 6

        n_profiles = 4
        n_clusters = 3

        cluster_weights = np.asarray([0.1, 0.6, 0.3])
        profile_weights = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                                      [0.05, 0.05, 0.8, 0.1],  # ])
                                      [0.2, 0.15, 0.3, 0.35]])
        profiles = np.asarray([[0., 0., 0.25, 0.25, 0.5, 0.],
                               [0.05, 0.05, 0.05, 0.05, 0.4, 0.4],
                               [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                               [0.05, 0.05, 0.7, 0.05, 0.05, 0.1]])

        # ***************************+ SIMULATION ***************************+ #

        cluster_alns_asso = np.concatenate(
            [int(cluster_weights[i] * n_alns) * [i]
             for i in range(n_clusters)], axis=0)
        profile_site_asso = np.asarray(
            [np.concatenate([int(profile_weights[cl, i] *
                                 n_sites) * [i]
                             for i in range(n_profiles)],
                            axis=0)
             for cl in cluster_alns_asso])
        aa_counts = []
        for aln in range(n_alns):
            sites_counts = np.zeros((n_sites, n_aas))
            for site in range(n_sites):
                sites_counts[site] = (profiles[profile_site_asso[aln, site]] *
                                      n_seqs).astype(int)
            aa_counts.append(sites_counts)

        # **************************** OPTIMAL lk ************************ #

        pi, p_aln_cl = e_step(aa_counts, profiles, profile_weights,
                              cluster_weights)
        optimal_lk, _ = compute_vlb_fl(aa_counts, cluster_weights,
                                       profile_weights,
                                       profiles, pi, p_aln_cl)

    # determine initial parameters
    np.random.seed(72)
    init_params = init_estimates(n_runs, n_clusters, n_profiles, n_aas, n_alns,
                                 test,
                                 true_params=[cluster_weights, profile_weights,
                                              profiles]
                                 if test else None)

    for run in range(n_runs):  # TODO remove
        #for cl_aln in range(n_clusters):
        #    init_params[run][1][cl_aln] = \
        #        init_estimates(1, 1, n_profiles, n_aas, n_alns, test,
        #                       true_params=None)[0][1][0]
        init_params[run][2] = np.ones(n_clusters) / n_clusters

    # set first two cluster weights with the help of kmean on PCs
    n_runs_select = 0
    msa_inds4pro_w_init = np.zeros((n_runs_select, n_clusters))
    msa_freqs = count_aas(alns, level='msa')
    msa_freqs /= np.repeat(msa_freqs.sum(axis=1)[:, np.newaxis], 20, axis=1)
    msa_freqs = np.round(msa_freqs, 8)

    pca = PCA(n_components=2)
    pca_msa_freqs = pca.fit_transform(msa_freqs)
    pca_msa_freqs_c = pca_msa_freqs - pca_msa_freqs.mean(axis=0)

    # kmeans on PC1 and 2
    seeds = [3874, 98037]
    for run in range(int(n_runs_select)):
        kmeans_pcs = KMeans(n_clusters=n_clusters, random_state=seeds[run]).fit(
            pca_msa_freqs_c)
        cl_coord = kmeans_pcs.cluster_centers_
        for i, target in enumerate(cl_coord):
            dists = np.sum((pca_msa_freqs_c - target) ** 2, axis=1) ** 0.5
            msa_inds4pro_w_init[run, i] = int(np.argmin(dists))
        # set em init cluster weights
        init_params[run][2] = np.asarray([sum(kmeans_pcs.labels_ == i) / n_alns
                                          for i in range(n_clusters)])
        # EM on msa for init profile weights
        init_pro_w = np.zeros((n_clusters, n_profiles))
        for cl, aln in enumerate(msa_inds4pro_w_init[run]):
            rand_params = [profiles, np.array([init_params[run][1][cl]]),
                           np.ones(1)]
            init_pro_w[cl] = em(rand_params, profiles, [aa_counts[int(aln)]],
                                n_iter)[0][1]

        init_params[run][1] = init_pro_w

    # TODO don't forget to delete this later
    # iem_path = '../results/profiles_weights/iEM_30cl_30aln_30runs'
    # for run in range(n_runs):
    #    init_params[run][1] = np.asarray(
    #        [np.genfromtxt(f'{iem_path}/cl{cl + 1}_pro_weights_{run + 1}.csv')
    #         for cl in range(n_clusters)])

    estimates_runs = []
    loglks = np.zeros((n_runs, n_iter))
    vlbs = np.zeros((n_runs, n_iter * 2))
    vlbs_dips = np.zeros((n_runs, n_iter * 2, 2))

    if not test:
        try:
            print(f'Start EM : {int(time.time() - STARTTIME)}s')
            # result = em(init_params[0], aa_counts, n_iter, 1)
            # estimates_runs.append(result[0])
            # lks[0] = result[1]
            # dips[0] = result[2]

            cl_w_runs = np.zeros((n_runs, n_clusters))
            pro_w_runs = np.zeros((n_runs, n_clusters, n_profiles))
            pro_runs = np.zeros((n_runs, n_profiles, n_aas))
            process_pool = multiprocessing.Pool(n_proc)

            for runs in np.reshape(np.arange(n_runs)[:, np.newaxis],
                                   (n_runs // n_proc, n_proc)):
                result = process_pool.starmap(em,
                                              zip(init_params[runs[0]:
                                                              runs[-1] + 1],
                                                  [profiles] * n_proc,
                                                  [aa_counts] * n_proc,
                                                  [n_iter] * n_proc, runs,
                                                  [test] * n_proc,
                                                  [save_path] * n_proc))
                for proc, run in zip(range(n_proc), runs):
                    cl_w_runs[run] = result[proc][0][2]
                    pro_w_runs[run] = result[proc][0][1]
                    pro_runs[run] = result[proc][0][0]
                    vlbs[run] = result[proc][1]
                    vlbs_dips[run] = result[proc][2]
                    loglks[run] = result[proc][3]

            best_lk_run = np.argmax(vlbs[:, -1], axis=0)

            # ------ rename parameter files to indicate best parameters by vlb

            # os.rename(f'{save_path}/profiles_{best_lk_run + 1}.tsv',
            #          f'{save_path}/profiles_best{best_lk_run + 1}.tsv')

            if n_clusters > 1:
                np.savetxt(f'{save_path}/cl_weights_best{best_lk_run + 1}.csv',
                           cl_w_runs[best_lk_run], delimiter=',')
                os.rename(f'{save_path}/cl_weights_{best_lk_run + 1}.csv',
                          f'{save_path}/cl_weights_best{best_lk_run + 1}.csv')
                for cl in range(n_clusters):
                    os.rename(f'{save_path}/cl{cl + 1}_pro_weights_'
                              f'{best_lk_run + 1}.csv',
                              f'{save_path}/cl{cl + 1}_pro_weights_best'
                              f'{best_lk_run + 1}.csv')
            elif n_clusters == 1:
                os.rename(f'{save_path}/pro_weights_{best_lk_run + 1}.csv',
                          f'{save_path}/pro_weights_best{best_lk_run + 1}.csv')

        except KeyboardInterrupt:
            print("Keyboard interrupt in main:")
        finally:
            print("cleaning up main")
    else:
        for run in range(n_runs):
            print(f'{run + 1}. Run')

            estimates_run, vlbs_run, dips_run, _ = em(init_params[run],
                                                      profiles,
                                                      aa_counts, n_iter,
                                                      test=test)

            estimates_runs.append(estimates_run)
            vlbs[run] = vlbs_run
            vlbs_dips[run] = dips_run

            estim_profiles, estim_profile_weights, estim_cluster_weights = \
                estimates_run

            # *************************** ERROR *************************** #
            """
            profiles_inds = list(range(n_profiles))
            profiles_permuts = np.asarray(list(set(permutations(profiles_inds))))
    
            mae_profiles = np.zeros((len(profiles_permuts)))
            mae_profile_weights = np.zeros((len(profiles_permuts)))
    
            for i, order in enumerate(profiles_permuts):
                # profile order according to profiles
                mae_profiles[i] = np.mean(
                    np.abs(profiles - estim_profiles[order, :]))
    
            ind_min_mae = np.argmin(mae_profiles)
            best_profile_order = profiles_permuts[ind_min_mae, :]
    
            # mae for profile weights given order obtained by profile maes
            sorted_alpha_profiles = np.take_along_axis(
                estim_profile_weights,
                np.repeat(best_profile_order[np.newaxis], n_clusters, axis=0),
                axis=1)
    
            # get optimal cluster order
            clusters_inds = list(range(n_clusters))
            clusters_permuts = np.asarray(list(set(permutations(clusters_inds))))
    
            mae_profile_weights = np.zeros((len(clusters_permuts)))
    
            for i, order in enumerate(clusters_permuts):
                # cluster order according to profile weights
                mae_profile_weights[i] = np.mean(np.abs(profile_weights -
                                                        sorted_alpha_profiles[
                                                            order]))
            ind_min_mae = np.argmin(mae_profile_weights)
            best_cluster_order = clusters_permuts[ind_min_mae, :]
    
            # compute errors for all iterations given best profile and cluster order
            maes_profiles, maes_profile_weights, maes_cluster_weights = [], [], []
    
            for iter, params in enumerate(estimates_run):
                pro, prow, clw = params
    
                # compute profiles error
                mae_profiles = np.mean(np.abs(profiles
                                              - pro[best_profile_order, :]))
                # compute profiles weights error
                reorderd_alpha_profiles = prow[best_cluster_order]
                reorderd_alpha_profiles = np.take_along_axis(
                    reorderd_alpha_profiles,
                    np.repeat(best_profile_order[np.newaxis], n_clusters, axis=0),
                    axis=1)
                mae_profile_weights = np.mean(np.abs(profile_weights
                                                     - reorderd_alpha_profiles))
                # compute cluster weights error
                mae_cluster_weights = np.mean(np.abs(cluster_weights
                                                     - clw[best_cluster_order]))
    
                maes_profiles.append(mae_profiles)
                maes_profile_weights.append(mae_profile_weights)
                maes_cluster_weights.append(mae_cluster_weights)
    
                # update saved parameters applying best parameter order
                estimates_run[iter] = [pro[best_profile_order, :],
                                       reorderd_alpha_profiles,
                                       clw[best_cluster_order]]
    
            maes_runs['cl.w.'].append(maes_cluster_weights)
            maes_runs['pro.w.'].append(maes_profile_weights)
            maes_runs['pro.'].append(maes_profiles)
            """

    # **************************** SAVE RESULTS ***************************** #

    np.savetxt(f'{save_path}/lk/vlbs_{n_runs}runs_{n_iter}iter.csv', vlbs,
               delimiter=',')
    np.savetxt(f'{save_path}/lk/loglks_{n_runs}runs_{n_iter}iter.csv', loglks,
               delimiter=',')

    vlbs[vlbs == -np.inf] = np.min(vlbs[vlbs > -np.inf])

    # **************************** PLOT RESULTS ***************************** #

    # dips[dips == np.inf] = 1  # TODO
    plt.plot(vlbs[:, -1])
    axes = plt.gca()
    y_lims = axes.get_ylim()
    plt.close()
    plt.ylim(y_lims)
    plt.bar(np.arange(1, n_runs + 1), vlbs[:, -1],
            tick_label=np.arange(1, n_runs + 1))
    plt.xlabel('EM')
    plt.ylabel('log-likelihood')
    plt.tight_layout()
    plt.savefig(f'{save_path}/lk/em_lk_bar.png')
    plt.close()

    dips_mask = vlbs_dips[:, :, 0].copy()
    dips_mask[dips_mask > 0] = 1
    # determine edge colors for markers : red edge at dips
    edgecol_e = np.repeat(
        np.asarray(['blue'] * n_iter, dtype=str)[np.newaxis, :],
        n_runs, axis=0)
    edgecol_m = np.repeat(np.asarray(['orange'] * n_iter)[np.newaxis, :],
                          n_runs, axis=0)
    edgecol_e[dips_mask[:, np.arange(0, n_iter * 2, 2)].astype(bool)] = 'red'
    edgecol_m[dips_mask[:, np.arange(1, n_iter * 2, 2)].astype(bool)] = 'red'
    # e-step and m-step mask for lks
    e_step_mask = np.zeros(n_iter * 2).astype(bool)
    m_step_mask = np.zeros(n_iter * 2).astype(bool)
    e_step_mask[np.arange(0, n_iter * 2, 2)] = True
    m_step_mask[np.arange(1, n_iter * 2, 2)] = True

    n_rows, n_cols = 3, 3  # 2, int(n_runs / 2)

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                            figsize=(n_cols * 6, n_rows * 6))
    for col in range(n_cols):
        for row in range(n_rows):
            run = (row * (n_cols)) + col
            if run < n_runs:
                # axs[row, col].plot(np.arange(0.5, n_iter + 0.5, 0.5),
                #                   lks[run])
                axs[row, col].plot(np.arange(n_iter), vlbs[run, m_step_mask])
                if test:
                    axs[row, col].hlines(y=optimal_lk, color='red',
                                         xmin=0.5,
                                         xmax=n_iter + 0.5)  # lk with given

                # axs[row, col].scatter(np.arange(0.5, n_iter),
                #                      lks[run, e_step_mask],
                #                      c='blue', edgecolors=edgecol_e[run],
                #                      label='E-Step')
                axs[row, col].scatter(np.arange(n_iter),
                                      vlbs[run, m_step_mask],
                                      c='orange', edgecolors=edgecol_m[run],
                                      label='M-Step')
                # annotate all dips > 1e-05
                move_annot = False
                # for x, y, dip in zip(np.arange(0.5, n_iter * 2 + 0.5, 0.5),
                #                     lks[run], dips[run, :, 0]):
                for x, y, dip in zip(np.arange(n_iter), vlbs[run, e_step_mask],
                                     vlbs_dips[run, e_step_mask, 0]):
                    if dip > 1e-05:
                        move_by = 0.05 if move_annot else 0.01
                        if dip < 1:
                            axs[row, col].annotate(np.round(dip, 5), (x, y),
                                                   xytext=(
                                                       x,
                                                       y + np.abs(y * move_by)),
                                                   arrowprops=dict(
                                                       arrowstyle="->"))
                        else:
                            axs[row, col].annotate(int(dip), (x, y),
                                                   xytext=(
                                                       x,
                                                       y + np.abs(y * move_by)),
                                                   arrowprops=dict(
                                                       arrowstyle="->"))
                        move_annot = not (move_annot)

                axs[row, col].set_xlabel('Iterations')
                axs[row, col].set_xticks(np.arange(0, n_iter))
                axs[row, col].set_ylabel('lk')

                # if row == 0 and col == 0:
                #    ylims = axs[row, col].get_ylim()
                # else:
                #    axs[row, col].set_ylim(ylims)

                axs[row, col].set_title(f'Run {run + 1}')
                # enable legend when showing e-step lks.
                # axs[row, col].legend()

        # titles
        if test:
            if col == 0:
                axs[0, col].set_title('Run with uniform initial params.')
            elif col == 1:
                axs[0, col].set_title('Run with correct params. as initial '
                                      'weights')

    # fig.suptitle(f'Test EM : {n_runs} runs')
    fig.tight_layout()
    fig.savefig(f'{save_path}/lk/sim_eval_em.png')
    plt.close(fig)

    print(f'Total runtime : {(time.time() - STARTTIME) / 60}min.')

    """recover distances of parameters
    cl_w_runs = np.zeros((n_proc, n_clusters))
    pro_w_runs = np.zeros((n_proc, n_clusters, n_profiles))
    for run in range(n_proc):
       pro_w_runs[run] = np.genfromtxt(f'../results/1646411196.2028363_prow{run+2}proc.csv', delimiter=',')
       cl_w_runs[run] = np.genfromtxt(f'../results/1646411196.2028363_clw{run+2}proc.csv', delimiter=',')
    """

    print(save_path)


if __name__ == '__main__':
    main(sys.argv[1:])

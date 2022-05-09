import multiprocessing
import os
import sys
import time
import warnings

from itertools import permutations
from scipy.stats import multinomial, dirichlet
from matplotlib import pylab as plt, rcParams
from sklearn.decomposition import PCA

import numpy as np
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
            np.asarray([multinomial.pmf(data[i], n_aas_site, profiles[k])
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

    # ********************************** E-STEP  ********************************** # 
    # pi : probability for profile z and cluster c at site j of alignment i given
    #      weights and profiles and aa counts
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

        # -------- lh on alignment level
        log_aln_cl[aln] = np.sum(np.log(np.dot(pro_w, p_sites_profs[aln].T)),
                                 axis=1)
        log_aln_cl[aln] += np.log(cl_w)

        # -------- lh on site level : pi
        log_sites_profs = np.log(p_sites_profs[aln])

        log_pi[aln] = np.repeat(np.log(pro_w)[np.newaxis, :, :], n_sites,
                                axis=0)
        log_pi[aln] += np.repeat(np.repeat(np.log(cl_w)[np.newaxis, :], n_sites,
                                           axis=0)[:, :, np.newaxis],
                                 n_profiles, axis=2)
        log_pi[aln] += np.repeat(log_rems_alns[aln][:, :, np.newaxis],
                                 n_profiles, axis=2)
        log_pi[aln] += np.repeat(log_sites_profs[:, np.newaxis, :], n_clusters,
                                 axis=1)

        if np.any(log_pi[aln] == -np.inf):
            warnings.warn(
                f"In MSA {aln + 1} {np.sum(log_pi[aln] == -np.inf)}/"
                f"{n_sites * n_clusters * n_profiles} log_pi's are "
                f"-infinity")

            # log_pi[aln][log_pi[aln] == -np.inf] = np.nextafter(-np.inf, 0)

        # log to prob
        max_logs = np.max(np.max(log_pi[aln], axis=2), axis=1)  # per site
        max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=1)
        max_logs = np.repeat(max_logs[:, :, np.newaxis], n_profiles, axis=2)

        pi[aln] = np.exp(log_pi[aln] + np.abs(max_logs))
        pi[aln][pi[aln] == 0] = MINPOSFLOAT

        # normalizing pi
        sum_over_pro_cl = pi[aln].sum(axis=2).sum(axis=1)
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, np.newaxis], n_clusters,
                                    axis=1)
        sum_over_pro_cl = np.repeat(sum_over_pro_cl[:, :, np.newaxis],
                                    n_profiles, axis=2)
        pi[aln] /= sum_over_pro_cl
        pi[aln][pi[aln] == 0] = MINPOSFLOAT

    # log to prob for alignment level lh
    p_aln_cl = np.exp(log_aln_cl + np.repeat(
        np.abs(np.max(log_aln_cl, axis=1))[:, np.newaxis], n_clusters, axis=1))
    p_aln_cl = p_aln_cl / np.repeat(np.sum(p_aln_cl, axis=1)[:, np.newaxis],
                                    n_clusters, axis=1)  # normalize

    return pi, p_aln_cl


def m_step(pi, cluster_probs_aln, aa_counts):
    n_alns = len(pi)
    n_aas = aa_counts[0].shape[-1]
    n_profiles = pi[0].shape[2]
    n_clusters = pi[0].shape[1]

    # cluster_probs_aln = np.asarray([np.mean(np.sum(pi[aln], axis=2), axis=0)
    #                                for aln in range(n_alns)])

    cluster_probs_aln_rep = np.repeat(cluster_probs_aln[:, :, np.newaxis],
                                      n_profiles, axis=2)
    cluster_probs_aln_aa_rep = np.repeat(
        cluster_probs_aln_rep[:, :, :, np.newaxis],
        n_aas, axis=3)

    # cluster weights
    estim_cluster_weights = np.mean(cluster_probs_aln /
                                    np.repeat(np.sum(cluster_probs_aln,
                                                     axis=1)[:, np.newaxis],
                                              n_clusters, axis=1),
                                    axis=0)  # usually division by 1

    # profile weights
    for aln in range(n_alns):
        sum_over_profs = np.sum(pi[aln], axis=2)
        sum_over_profs[sum_over_profs == 0] = 1  # avoid divison by 0

    estim_profile_weights = [pi[aln] /
                             np.repeat(np.sum(pi[aln],
                                              axis=2)[:, :, np.newaxis],
                                       n_profiles, axis=2)
                             for aln in range(n_alns)]

    estim_profile_weights = np.asarray([np.sum(estim_profile_weights[aln],
                                               axis=0)
                                        for aln in range(n_alns)])

    # weight by cluster probabilities
    estim_profile_weights = np.sum(cluster_probs_aln_rep *
                                   estim_profile_weights,
                                   axis=0)
    estim_profile_weights /= np.repeat(
        np.sum(estim_profile_weights, axis=1)[:, np.newaxis],
        n_profiles, axis=1)

    # profiles
    aln_cl_profiles = np.zeros((n_alns, n_clusters, n_profiles, n_aas))
    for aln in range(n_alns):
        for cl in range(n_clusters):
            aln_cl_profiles[aln, cl] = pi[aln][:, cl, :].T.dot(
                aa_counts[aln])
            aln_cl_profiles[aln, cl] /= np.repeat(
                aln_cl_profiles[aln, cl].sum(axis=1)[:, np.newaxis], n_aas,
                axis=1)  # sum over amino acids

    # weighted sum over alignments with cluster weights
    estim_profiles = np.sum(cluster_probs_aln_aa_rep *
                            aln_cl_profiles,
                            axis=0)
    estim_profiles = estim_profiles.sum(axis=0)  # sum over clusters
    # ensure that profil sum is <= 1
    estim_profiles /= np.repeat(estim_profiles.sum(axis=1)[:, np.newaxis],
                                n_aas, axis=1)

    return estim_cluster_weights, estim_profile_weights, estim_profiles


def compute_vlb_fl(data, alpha_clusters, alpha_profiles,
                   beta, gamma, p_aln_cl):
    n_alns = len(data)
    n_profiles = beta.shape[0]
    n_cluster = alpha_clusters.shape[0]

    # avoid division by 0
    alpha_profiles[alpha_profiles == 0] = np.nextafter(0, 1)
    alpha_clusters[alpha_clusters == 0] = np.nextafter(0, 1)

    # compute lower bound of full lh
    lh = 0

    # -------- log lh for sites given profiles
    sites_profs_probs = multi_dens(data, beta)
    sites_profile_probs_zeros = sites_profs_probs.copy()
    # avoid division by zero
    for aln in range(n_alns):
        sites_profs_probs[aln][sites_profs_probs[aln] == 0] = np.nextafter(0, 1)

    # -------- log lh for remaining sites
    log_other_sites_cl = [log_probs_rems(sites_profs_probs[aln],
                                         alpha_profiles)
                          for aln in range(n_alns)]

    for aln in range(n_alns):
        n_sites = data[aln].shape[0]

        # -------- log-lhs per sites (sum of the above)
        log_sites = np.repeat(np.log(sites_profs_probs[aln])[:, np.newaxis, :],
                              n_cluster, axis=1)
        log_sites += np.repeat(log_other_sites_cl[aln][:, :, np.newaxis],
                               n_profiles, axis=2)

        # -------- computing full average log-lh
        log_alpha_pro_rep = np.repeat(np.log(alpha_profiles)[np.newaxis, :, :],
                                      n_sites, axis=0)

        gamma[aln][gamma[aln] == 0] = np.nextafter(0, 1)
        weighted_lhs = gamma[aln] * (log_sites + log_alpha_pro_rep)
        weighted_lhs[np.isnan(weighted_lhs)] = 0  # such that 0 * -inf = 0
        lh += np.sum(weighted_lhs)

    # -------- lh on alignment level
    lh += np.sum(p_aln_cl * np.log(alpha_clusters))

    return lh, [sites_profile_probs_zeros, log_other_sites_cl]


def lh_per_site(aln_counts, profiles, weights):
    prob_site_prof = np.asarray([multinomial.pmf(aln_counts,
                                                 aln_counts.sum(axis=-1),
                                                 profile)
                                 for profile in profiles])
    return prob_site_prof.T @ weights


def em(init_params, profiles, aa_counts, n_iter, run=None, test=False,
       save_path=""):
    try:
        if run is not None:
            print(f'Run {run}')

        start = time.time()

        estim_profiles, estim_profile_weights, estim_cluster_weights = init_params

        # fix parameters
        if not test:
            estim_profiles = profiles
        # estim_profile_weights = profile_weights
        # estim_cluster_weights = cluster_weights
        probs_sites = None  # will be passed from lh. computation to next e-step

        # estimates_iter = []
        lhs_iter = np.zeros((n_iter * 2))
        dips_iter = np.zeros((n_iter * 2, 2))

        for iter in range(n_iter):

            # print(f'\tIteration : {iter + 1} {int(time.time() - STARTTIME)}s')

            e_step_start = time.time()

            pi, p_aln_cl = e_step(aa_counts, estim_profiles,
                                  estim_profile_weights,
                                  estim_cluster_weights,
                                  probs_sites)

            # print(f'\t E-step finished {int(time.time() - STARTTIME)}s (took '
            #      f'{int(time.time() - e_step_start)}s)')

            # lhs_iter[iter * 2], _ = compute_vlb_fl(aa_counts,
            #                                               estim_cluster_weights,
            #                                               estim_profile_weights,
            #                                               estim_profiles, pi,
            #                                               p_aln_cl)

            estim_cluster_weights, estim_profile_weights, estim_profiles = m_step(
                pi, p_aln_cl, aa_counts)

            # print(f'\t M-step finished {int(time.time() - STARTTIME)}s')

            if save_path != "":
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

            # *************************** lh *************************** #

            lhs_iter[iter * 2 + 1], probs_sites = compute_vlb_fl(
                aa_counts, estim_cluster_weights, estim_profile_weights,
                estim_profiles, pi, p_aln_cl)

            # this could be the lh. after the e-step
            lhs_iter[iter * 2] = lhs_iter[iter * 2 + 1]

            # print(f'\t Computed lh {int(time.time() - STARTTIME)}s')

            if iter > 0:
                # only accounts for dips after e-step
                if lhs_iter[iter * 2 + 1] < lhs_iter[(iter - 1) * 2 + 1]:
                    curr_dip = np.abs(lhs_iter[iter * 2 + 1] -
                                      lhs_iter[(iter - 1) * 2 + 1])

                    dips_iter[iter * 2] = np.asarray([curr_dip, iter])
                    dips_iter[iter * 2 + 1] = np.asarray([curr_dip, iter])

            if iter > 0 and np.abs(
                    (lhs_iter[(iter - 1) * 2 + 1] - lhs_iter[iter * 2 + 1])
                    / lhs_iter[(iter - 1) * 2 + 1]) < RTOL:
                if iter < n_iter - 1:
                    lhs_iter[iter * 2 + 1:] = lhs_iter[iter * 2 + 1]
                print(f'Run {run} : finished after {iter + 1} (max. {n_iter}) '
                      f'iterations')
                break

            """
            if lhs_iter[iter * 2 + 1] < lhs_iter[iter * 2]:
                curr_dip = np.abs(lhs_iter[iter * 2 + 1] -
                                  lhs_iter[iter * 2])

                dips_iter[iter * 2 + 1] = np.asarray([curr_dip, iter])

                if iter > 0:
                    print(f'{lhs_iter[iter * 2 + 1]} < '
                          f'{lhs_iter[iter * 2]}')
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
                lhs_iter, dips_iter)

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
            pro_w = dirichlet.rvs([2 * n_profiles] * n_profiles, n_clusters)
            # init cluster probabilities
            weights = np.random.randint(1, n_alns, n_clusters)
            cl_w = weights / weights.sum()

        params.append([prof, pro_w, cl_w])

    return params


def main(args):
    # ****************************** PARAMETERS ****************************** #

    em_config_path = args[0]
    em_config = read_config_file(em_config_path)
    n_alns, n_iter, n_runs, test, n_profiles, n_clusters, n_proc, fasta_in_path, \
    config_path_msa, profile_path, save_path = em_config.values()

    if not test:
        os.mkdir(f'{save_path}/lh')  # for saving likelihoods
        os.mkdir(f'{save_path}/init_weights')
        profiles = np.genfromtxt(profile_path, delimiter='\t').T

        config = read_config_file(config_path_msa)
        all_n_alns = config['data']['nb_alignments']
        alns, fastas, config['data'] = raw_alns_prepro([fasta_in_path],
                                                       config['data'],
                                                       take_quantiles=[True])

        print(f'Alignments loaded : {int(time.time() - STARTTIME)}s')

        sample_inds = np.arange(0, all_n_alns,
                                np.round(all_n_alns / n_alns))[:n_alns].astype(
            int)

        alns = [alns[0][ind] for ind in sample_inds]  # sample alignments for EM
        fastas = [fastas[0][ind] for ind in sample_inds]

        if save_path != '':
            np.savetxt(f'{save_path}/init_weights/real_fastanames4estim.txt',
                       fastas, delimiter=',', fmt='%s')

        print(config['data'])

        aa_counts = [count_aas([aln], 'sites').T for aln in alns]
        print(f'Count vectors generated : {int(time.time() - STARTTIME)}s')

        n_aas = aa_counts[0].shape[-1]

        print(f'\nEstimation on {len(alns)} MSAs\n')

    else:
        n_alns = 100
        n_sites = 40
        n_seqs = 40
        n_aas = 6

        n_profiles = 4
        n_clusters = 2

        # cluster_weights = np.asarray([0.1, 0.6, 0.3])
        cluster_weights = np.asarray([0.2, 0.8])
        profile_weights = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                                      [0.05, 0.05, 0.8, 0.1]])
        # [0.2, 0.15, 0.3, 0.35]])
        profiles = np.asarray([[0., 0., 0.25, 0.25, 0.5, 0.],
                               [0.05, 0.05, 0.05, 0.05, 0.4, 0.4],
                               [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                               [0.05, 0.05, 0.7, 0.05, 0.05, 0.1]])
        # profiles = np.asarray([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        #                       [0.005, 0.005, 0.005, 0.005, 0.49, 0.49],
        #                       [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
        #                       [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]])

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

        # **************************** OPTIMAL lh ************************ #

        pi, p_aln_cl = e_step(aa_counts, profiles, profile_weights,
                              cluster_weights)
        optimal_lh, _ = compute_vlb_fl(aa_counts, cluster_weights,
                                       profile_weights,
                                       profiles, pi, p_aln_cl)

    # determine initial parameters
    init_params = init_estimates(n_runs, n_clusters, n_profiles, n_aas, n_alns,
                                 test,
                                 true_params=[cluster_weights, profile_weights,
                                              profiles]
                                 if test else None)

    # estimate init profile weights from selected MSAs
    init_pro_weights_table = np.zeros((int(n_runs / 2) * n_clusters,
                                       n_profiles + 2))
    msa_freqs_table = np.zeros((int(n_runs / 2) * n_clusters, n_aas + 2))

    # determine msa and its aa frequencies
    msa_inds4pro_w_init = np.zeros((int(n_runs / 2), n_clusters))
    msa_freqs = count_aas(alns, level='msa')
    msa_freqs /= np.repeat(msa_freqs.sum(axis=1)[:, np.newaxis], 20, axis=1)
    msa_freqs = np.round(msa_freqs, 8)

    pca = PCA(n_components=2)
    pca_msa_freqs = pca.fit_transform(msa_freqs)

    pca_msa_freqs_c = pca_msa_freqs - pca_msa_freqs.mean(axis=0)
    dists = np.sum(pca_msa_freqs_c ** 2, axis=1) ** 0.5
    # center cluster
    msa_inds4pro_w_init[:, 0] = np.argsort(dists)[0:int(n_runs / 2)]

    if n_clusters > 1:
        r = np.quantile(dists, 0.8)
        # cluster angles
        angles = (360 / (n_clusters - 1)) * np.arange(1, n_clusters)
        # per run (use this method for half of the runs)
        angles = np.repeat(angles[np.newaxis, :], int(n_runs / 2), axis=0)
        # get rotation degrees
        rot_angles = (360 / (n_clusters - 1) / int(n_runs / 2))
        rot_angles *= np.arange(0, int(n_runs / 2))
        # rotate angles and normalize
        angles = np.add(angles, rot_angles[:, None])
        angles = (angles % 360 + 360) % 360

        thetas = np.deg2rad(angles)
    else:
        thetas = int(n_runs / 2) * [[]]

    for run in range(int(n_runs / 2)):
        for i, theta in enumerate(thetas[run]):
            target = pol2cart(r, theta)
            dists = np.sum((pca_msa_freqs_c - target) ** 2, axis=1) ** 0.5
            msa_inds4pro_w_init[run, i + 1] = int(np.argmin(dists))

        plt.scatter(pca_msa_freqs[:, 0], pca_msa_freqs[:, 1], color='lightblue',
                    s=1)
        plt.scatter(pca_msa_freqs[(msa_inds4pro_w_init[run]).astype(int), 0],
                    pca_msa_freqs[(msa_inds4pro_w_init[run]).astype(int), 1],
                    color='coral')
        plt.savefig(f'{save_path}/init_weights/init_msas_run{run + 1}.png')
        plt.close('all')

        run_table_inds = np.arange(run * n_clusters,
                                   run * n_clusters + n_clusters)
        msa_freqs_table[run_table_inds, :n_aas] = np.asarray(
            [msa_freqs[int(aln)] for aln in msa_inds4pro_w_init[run]])

        # EM on msa for init profile weights
        init_pro_w = np.zeros((n_clusters, n_profiles))
        for cl, aln in enumerate(msa_inds4pro_w_init[run]):
            rand_params = init_estimates(1, 1, n_profiles, n_aas, n_alns,
                                         test, true_params=None)[0]
            init_pro_w[cl] = em(rand_params, profiles, [aa_counts[int(aln)]],
                                n_iter)[0][1]
            # avoid too low weights
            # init_pro_w[cl][init_pro_w[cl] <= np.finfo(float).eps] = np.min(
            #     init_pro_w[cl][init_pro_w[cl] > np.finfo(float).eps])
            # init_pro_w[cl] /= np.sum(init_pro_w[cl])

            init_params[run][1] = init_pro_w

            if n_clusters > 1:
                cl_w = np.random.normal(loc=0.3 / (n_clusters - 1),
                                        scale=0.001, size=n_clusters - 1)
                init_params[run][2] = np.concatenate(([1 - cl_w.sum()], cl_w))

            init_pro_weights_table[run_table_inds, :n_profiles] = init_pro_w
            init_pro_weights_table[n_clusters * run + cl, -1] = cl + 1
            msa_freqs_table[n_clusters * run + cl, -1] = cl + 1
            init_pro_weights_table[run_table_inds, -2] = np.repeat(run + 1,
                                                                   n_clusters)
            msa_freqs_table[run_table_inds, -2] = np.repeat(run + 1, n_clusters)

    if save_path != "":
        header_str = ','.join(np.arange(1, n_profiles + 1).astype(str)) + \
                     ',run,cl'
        np.savetxt(f'{save_path}/init_weights/init_weights.csv',
                   init_pro_weights_table,
                   delimiter=',',
                   header=header_str, comments='')
        np.savetxt(f'{save_path}/init_weights/msa_freqs.csv',
                   msa_freqs_table,
                   delimiter=',', header=','.join(AAS) + ',run,cl', comments='')

    # maes_runs = {'cl.w.': [], 'pro.w.': [], 'pro.': []}
    estimates_runs = []
    lhs = np.zeros((n_runs, n_iter * 2))
    dips = np.zeros((n_runs, n_iter * 2, 2))

    if not test:
        try:
            print(f'Start EM : {int(time.time() - STARTTIME)}s')
            # result = em(init_params[0], aa_counts, n_iter, 1)
            # estimates_runs.append(result[0])
            # lhs[0] = result[1]
            # dips[0] = result[2]

            cl_w_runs = np.zeros((n_runs, n_clusters))
            pro_w_runs = np.zeros((n_runs, n_clusters, n_profiles))
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
                    lhs[run] = result[proc][1]
                    dips[run] = result[proc][2]

            best_lh_run = np.argmax(lhs[:, -1], axis=0)

            # best parameters
            if n_clusters > 1:
                # sort and compute MAEs (approx)
                sort_cl_w = np.argsort(cl_w_runs)
                cl_w_runs_sort = np.take_along_axis(cl_w_runs, sort_cl_w,
                                                    axis=-1)
                # sort by clusters
                pro_w_runs_sort = pro_w_runs[
                                  np.arange(len(cl_w_runs))[:, np.newaxis],
                                  sort_cl_w, :]
                # sort by profile weights
                # pro_w_runs_sort = np.take_along_axis(pro_w_runs_sort,
                # np.argsort(pro_w_runs_sort), axis=-1)

                inds = np.asarray(np.triu_indices(n_proc, k=1))  # pair indices
                dists_cl_w = np.mean(np.abs(
                    cl_w_runs_sort[inds[0], :] - cl_w_runs_sort[inds[1], :]))
                dists_pro_w = np.mean(np.abs(
                    pro_w_runs_sort[inds[0], :, :] - pro_w_runs_sort[inds[1], :,
                                                     :]))

                print(f'MAE of cluster weights : {dists_cl_w}\n'
                      f'MAE of profiles weights : {dists_pro_w}')

                # save best
                np.savetxt(f'{save_path}/cl_weights_best{best_lh_run + 1}.csv',
                           cl_w_runs[best_lh_run], delimiter=',')
                os.rename(f'{save_path}/cl_weights_{best_lh_run + 1}.csv',
                          f'{save_path}/cl_weights_best{best_lh_run + 1}.csv')
                for cl in range(n_clusters):
                    os.rename(f'{save_path}/cl{cl + 1}_pro_weights_'
                              f'{best_lh_run + 1}.csv',
                              f'{save_path}/cl{cl + 1}_pro_weights_best'
                              f'{best_lh_run + 1}.csv')

            elif n_clusters == 1:
                os.rename(f'{save_path}/pro_weights_{best_lh_run + 1}.csv',
                          f'{save_path}/pro_weights_best{best_lh_run + 1}.csv')

        except KeyboardInterrupt:
            print("Keyboard interrupt in main:")
        finally:
            print("cleaning up main")
    else:
        for run in range(n_runs):
            print(f'{run + 1}. Run')

            estimates_run, lhs_run, dips_run = em(init_params[run],
                                                  aa_counts, n_iter,
                                                  test=test)

            estimates_runs.append(estimates_run)
            lhs[run] = lhs_run
            dips[run] = dips_run

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

    np.savetxt(f'{save_path}/lh/likelihoods_{n_runs}runs_{n_iter}iter.csv', lhs,
               delimiter=',')
    lhs[lhs == -np.inf] = np.min(lhs[lhs > -np.inf])

    # **************************** PLOT RESULTS ***************************** #

    # dips[dips == np.inf] = 1  # TODO
    plt.plot(lhs[:, -1])
    axes = plt.gca()
    y_lims = axes.get_ylim()
    plt.close()
    plt.ylim(y_lims)
    plt.bar(np.arange(1, n_runs + 1), lhs[:, -1],
            tick_label=np.arange(1, n_runs + 1))
    plt.xlabel('EM')
    plt.ylabel('log-likelihood')
    plt.tight_layout()
    plt.savefig(f'{save_path}/lh/em_lh_bar.png')
    plt.close()

    dips_mask = dips[:, :, 0].copy()
    dips_mask[dips_mask > 0] = 1
    # determine edge colors for markers : red edge at dips
    edgecol_e = np.repeat(
        np.asarray(['blue'] * n_iter, dtype=str)[np.newaxis, :],
        n_runs, axis=0)
    edgecol_m = np.repeat(np.asarray(['orange'] * n_iter)[np.newaxis, :],
                          n_runs, axis=0)
    edgecol_e[dips_mask[:, np.arange(0, n_iter * 2, 2)].astype(bool)] = 'red'
    edgecol_m[dips_mask[:, np.arange(1, n_iter * 2, 2)].astype(bool)] = 'red'
    # e-step and m-step mask for lhs
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
                #                   lhs[run])
                axs[row, col].plot(np.arange(n_iter), lhs[run, m_step_mask])
                if test:
                    axs[row, col].hlines(y=optimal_lh, color='red',
                                         xmin=0.5,
                                         xmax=n_iter + 0.5)  # lh with given

                # axs[row, col].scatter(np.arange(0.5, n_iter),
                #                      lhs[run, e_step_mask],
                #                      c='blue', edgecolors=edgecol_e[run],
                #                      label='E-Step')
                axs[row, col].scatter(np.arange(n_iter),
                                      lhs[run, m_step_mask],
                                      c='orange', edgecolors=edgecol_m[run],
                                      label='M-Step')
                # annotate all dips > 1e-05
                move_annot = False
                # for x, y, dip in zip(np.arange(0.5, n_iter * 2 + 0.5, 0.5),
                #                     lhs[run], dips[run, :, 0]):
                for x, y, dip in zip(np.arange(n_iter), lhs[run, e_step_mask],
                                     dips[run, e_step_mask, 0]):
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
                axs[row, col].set_ylabel('lh')

                # if row == 0 and col == 0:
                #    ylims = axs[row, col].get_ylim()
                # else:
                #    axs[row, col].set_ylim(ylims)

                axs[row, col].set_title(f'Run {run + 1}')
                # enable legend when showing e-step lhs.
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

    fig.savefig(f'{save_path}/lh/sim_eval_em.png')

    plt.close(fig)

    print(f'Total runtime : {time.time() - STARTTIME}s')

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

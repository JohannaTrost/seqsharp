import os
import time
import warnings

from itertools import permutations
from scipy.stats import multinomial, dirichlet
from matplotlib import pylab as plt

import numpy as np

np.random.seed(72)


def e_step(data, profs, pro_w, cl_w):
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

    for aln in range(n_alns):
        n_sites = data[aln].shape[0]
        n_aas_site = data[aln].sum(axis=-1)

        prob_pro_site = np.asarray([multinomial.pmf(data[aln], n_aas_site,
                                                    profs[pro])
                                    for pro in range(n_profiles)])

        # site masks - inverse Eigenmatrix n_sites x n_sites
        site_masks = ~np.eye(n_sites, dtype=bool)

        for cl in range(n_clusters):

            # prob. for remaining sites
            log_probs_remaining_sites = np.zeros(n_sites)
            """
            for site, site_mask in enumerate(site_masks):
                prev_sum = 1
                for i, site_prob in enumerate(prob_pro_site[:, site_mask].T):
                    remaining_sites_prob = site_prob * pro_w[cl, :]
                    remaining_sites_prob *= prev_sum
                    prev_sum = np.sum(remaining_sites_prob)
                probs_remaining_sites[site] = prev_sum
            """
            for not_site, site_mask in enumerate(site_masks):
                probs_not_site = prob_pro_site[:, site_mask]
                log_probs_remaining_sites[not_site] = np.sum(
                    np.log(probs_not_site.T @ pro_w[cl, :]))
            """
            for not_site, site_mask in enumerate(site_masks):
                probs_not_site = prob_pro_site[:, site_mask]
                probs_remaining_sites[not_site] = np.prod(np.sum(probs_not_site *
                                                         np.repeat(
                                                             pro_w[cl, :,
                                                             np.newaxis],
                                                             n_sites - 1, axis=1),
                                                         axis=0))
            """

            log_pi[aln][:, cl, :] = np.repeat(
                log_probs_remaining_sites[:, np.newaxis],
                n_profiles, axis=1)
            log_pi[aln][:, cl, :] += np.log(cl_w[cl])

            for pro in range(n_profiles):
                log_pi[aln][:, cl, pro] += np.log(prob_pro_site[pro])
                log_pi[aln][:, cl, pro] += np.log(pro_w[cl, pro])

            if np.any(log_pi[aln] == -np.inf):
                warnings.warn(
                    f"In MSA {aln + 1} {np.sum(log_pi[aln] == -np.inf)}/"
                    f"{n_sites * n_clusters * n_profiles} log_pi's are "
                    f"-infinity")

                log_pi[aln][log_pi[aln] == -np.inf] = np.nextafter(-np.inf, 0)

            # log to prob
            max_logs = np.max(np.max(log_pi[aln], axis=2), axis=1)  # per site
            max_logs = np.repeat(max_logs[:, np.newaxis], n_clusters, axis=1)
            max_logs = np.repeat(max_logs[:, :, np.newaxis], n_profiles, axis=2)

            pi[aln] = np.exp(log_pi[aln] - max_logs)

    # normalization
    sum_over_profiles = [pi[i].sum(axis=2).sum(axis=1) for i in range(n_alns)]
    sum_over_profiles = [
        np.repeat(sum_over_profiles[i][:, np.newaxis], n_clusters,
                  axis=1)
        for i in range(n_alns)]
    sum_over_profiles = [
        np.repeat(sum_over_profiles[i][:, :, np.newaxis], n_profiles, axis=2)
        for i in range(n_alns)]

    pi = [pi[i] / sum_over_profiles[i] for i in range(n_alns)]

    # Profile-site-accuracy - without considering cluster
    correct_pro_site = []
    for aln in range(n_alns):
        n_sites = data[aln].shape[0]
        correct_pro_site += [np.sum(np.argmax(pi[aln][:, cl, :], axis=1) ==
                                    profile_site_asso[aln]) / n_sites
                             for cl in range(n_clusters)]
    prof_site_acc = np.mean(correct_pro_site)

    return pi, prof_site_acc


def compute_vlb_fl(data, alpha_clusters, alpha_profiles,
                   beta, gamma):
    n_alns = len(data)
    n_profiles = beta.shape[0]
    n_cluster = alpha_clusters.shape[0]

    # avoid division by 0
    alpha_profiles[alpha_profiles == 0] = np.finfo(float).eps
    alpha_clusters[alpha_clusters == 0] = np.finfo(float).eps

    # compute lower bound of full likelihood
    likelihood = 0
    for i in range(n_alns):
        # P(A_i | v_k) for all sites
        n_aas_site = data[i].sum(axis=-1)
        sites_profile_probs = [multinomial.pmf(data[i], n_aas_site,
                                               beta[k])
                               for k in range(n_profiles)]
        sites_profile_probs = np.asarray(sites_profile_probs)
        # avoid division by zero
        sites_profile_probs[sites_profile_probs == 0] = np.finfo(float).eps
        log_sites_profile_probs = np.log(sites_profile_probs)

        # site masks - inverse Eigenmatrix n_sites x n_sites
        sites_masks = ~np.eye(n_sites, dtype=bool)
        log_other_sites_clusters = np.zeros((n_sites, n_cluster))

        for c in range(n_cluster):
            # remaining sites for each site
            for not_site, sites_mask in enumerate(sites_masks):
                probs_not_site = sites_profile_probs[:, sites_mask]
                probs_not_site[probs_not_site == 0] = np.finfo(float).eps
                log_other_sites_clusters[not_site, c] = np.sum(np.log(
                    probs_not_site.T @ alpha_profiles[c, :]))

            for k in range(n_profiles):
                weights = (gamma[i][:, c, k])
                weights[weights == 0] = np.finfo(float).eps
                likelihood += np.sum(weights *
                                     (log_sites_profile_probs[k]
                                      + log_other_sites_clusters[:, c]
                                      + np.log(alpha_profiles[c, k])
                                      + np.log(alpha_clusters[c])))
                # likelihood -= np.sum(weights * np.log(weights))

    return likelihood


# ******************************** PARAMETERS ******************************** #

n_iter = 10
n_runs = 5

n_alns = 100
n_sites = 40
n_seqs = 30
#n_seqs = 1200
n_aas = 6

n_profiles = 4
n_clusters = 2

cluster_weights = np.asarray([0.2, 0.8])
profile_weights = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                              [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
profiles = np.asarray([[0., 0., 0.25, 0.25, 0.5, 0.],
                       [0.05, 0.05, 0.05, 0.05, 0.4, 0.4],
                       [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                       [0.05, 0.05, 0.7, 0.05, 0.05, 0.1]])
#profiles = np.asarray([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
#                       [0.005, 0.005, 0.005, 0.005, 0.49, 0.49],
#                       [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
#                       [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]])

# ******************************** SIMULATION ******************************** #

cluster_alns_asso = np.concatenate([int(cluster_weights[i] * n_alns) * [i]
                                    for i in range(n_clusters)], axis=0)
profile_site_asso = np.asarray([np.concatenate([int(profile_weights[cl,
                                                                    i] * n_sites) * [
                                                    i]
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

# ****************************** OPTIMAL likelihood ******************************* #

pi, acc = e_step(aa_counts, profiles, profile_weights, cluster_weights)
optimal_likelihood = compute_vlb_fl(aa_counts, cluster_weights, profile_weights,
                                    profiles, pi)

maes_runs = {'cl.w.': [], 'pro.w.': [], 'pro.': []}
estimates_runs = []
likelihoods = np.zeros((n_runs, n_iter * 2))

for run in range(n_runs):

    print(f'{run + 1}. Run')

    # *************************** INIT PARAMS **************************** #

    if run == 0:  # uniform
        estim_cluster_weights = np.ones(n_clusters) / n_clusters
        estim_profile_weights = np.ones((n_clusters, n_profiles)) / n_profiles
        estim_profiles = np.ones((n_profiles, n_aas)) / n_aas
    elif run == 1:  # correct params
        estim_cluster_weights = cluster_weights
        estim_profile_weights = profile_weights
        estim_profiles = profiles
    else:
        # init profiles
        estim_profiles = dirichlet.rvs([2 * n_aas] * n_aas, n_profiles)

        # init profile probabilities per cluster
        cluster_profile_counts = np.random.randint(1,
                                                   n_sites,
                                                   (n_clusters,
                                                    n_profiles))
        cluster_counts = cluster_profile_counts.sum(axis=1)
        cluster_counts_shaped = cluster_counts.repeat(
            [n_profiles] * n_clusters).reshape(n_clusters, n_profiles)
        estim_profile_weights = cluster_profile_counts / cluster_counts_shaped

        # init cluster probabilities
        weights = np.random.randint(1, n_alns, n_clusters)
        estim_cluster_weights = weights / weights.sum()

    estimates_iter = []

    for iter in range(n_iter):

        print(f'\tIteration : {iter + 1}')

        pi, acc = e_step(aa_counts, estim_profiles, estim_profile_weights,
                         estim_cluster_weights)

        likelihoods[run, iter * 2] = compute_vlb_fl(aa_counts,
                                                    estim_cluster_weights,
                                                    estim_profile_weights,
                                                    estim_profiles, pi)

        # ***************************** M-STEP ***************************** #

        # profiles
        aln_cl_profiles = np.zeros((n_alns, n_clusters, n_profiles, n_aas))
        for aln in range(n_alns):
            n_sites = aa_counts[aln].shape[0]
            for cl in range(n_clusters):
                weighted_counts = pi[aln][:, cl, :].T.dot(aa_counts[aln])
                aln_cl_profiles[aln, cl] = weighted_counts
                aln_cl_profiles[aln, cl] /= weighted_counts.sum(
                    axis=-1).reshape(-1, 1)

        estim_profiles = aln_cl_profiles.mean(axis=0).mean(axis=0)

        # profile weights
        cluster_probs_aln = np.asarray(
            [np.mean(np.sum(pi[aln], axis=2), axis=0) for aln in range(n_alns)])
        estim_cluster_alns_asso = \
            np.asarray([np.where(aln_cl_prob == np.max(aln_cl_prob))[0]
                        for aln_cl_prob in cluster_probs_aln]).T[0]

        sum_profile_porbs = [probs.sum(axis=2) for probs in pi]
        sum_profile_porbs = [
            np.repeat(probs[:, :, np.newaxis], n_profiles, axis=2)
            for probs in sum_profile_porbs]
        estim_profile_weights = np.mean(
            [(pi[i] / sum_profile_porbs[i]).mean(axis=0)
             for i in range(n_alns)],
            axis=0)
        """
        estim_profile_weights_alns = [np.mean(pi[aln], axis=0) for aln in
                                      range(n_alns)]
        estim_profile_weights_alns = np.asarray(
            [estim_profile_weights_alns[aln] /
             np.repeat(np.sum(
                 estim_profile_weights_alns[aln],
                 axis=1)[:, np.newaxis],
                       n_profiles, axis=1)
             for aln in range(n_alns)])

        estim_profile_weights = np.zeros((n_clusters, n_profiles))
        for cl in range(n_clusters):
            estim_pro_w_cl = estim_profile_weights_alns[
                np.where(estim_cluster_alns_asso == cl)]

            if len(estim_pro_w_cl) > 0:
                estim_profile_weights[cl] = np.mean(np.mean(estim_pro_w_cl,
                                                            axis=0),
                                                    axis=0)
            else:
                estim_profile_weights[cl] = np.zeros(n_profiles)
        """

        # cluster weights
        estim_cluster_weights = cluster_probs_aln
        estim_cluster_weights /= np.repeat(np.sum(cluster_probs_aln,
                                                  axis=1)[:, np.newaxis],
                                           n_clusters, axis=1)
        estim_cluster_weights = np.mean(estim_cluster_weights, axis=0)

        ''' 2. option
        estim_cluster_weights = np.bincount(estim_cluster_alns_asso) / n_alns
        '''
        # *************************** likelihood *************************** #

        likelihoods[run, iter * 2 + 1] = compute_vlb_fl(aa_counts,
                                                        estim_cluster_weights,
                                                        estim_profile_weights,
                                                        estim_profiles, pi)

        estimates_iter.append(
            [estim_profiles, estim_profile_weights, estim_cluster_weights])

    estimates_runs.append(estimates_iter)

    # *************************** ERROR *************************** #

    profiles_inds = list(range(n_profiles))
    profiles_permuts = np.asarray(list(set(permutations(profiles_inds))))

    mae_profiles = np.zeros((len(profiles_permuts)))
    mae_profile_weights = np.zeros((len(profiles_permuts)))

    for i, order in enumerate(profiles_permuts):
        # profile order according to profiles
        mae_profiles[i] = np.mean(np.abs(profiles - estim_profiles[order, :]))

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

    for pro, prow, clw in estimates_iter:
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

    maes_runs['cl.w.'].append(maes_cluster_weights)
    maes_runs['pro.w.'].append(maes_profile_weights)
    maes_runs['pro.'].append(maes_profiles)

# ****************************** PLOT RESULTS ******************************* #

# timestamp = time.time()
# save_path = f'../results/{timestamp}'
# os.mkdir(save_path)

likelihoods[likelihoods == -np.inf] = np.min(likelihoods[likelihoods > -np.inf])

best_likelihood_ind = np.argmax(likelihoods, axis=0)
n_rows, n_cols = n_runs, 2

fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                        figsize=(8., 8.))
for run in range(n_runs):
    axs[run, 0].plot(np.arange(0, n_iter, 0.5), likelihoods[run])
    axs[run, 0].hlines(y=optimal_likelihood, color='red',
                       xmin=0,
                       xmax=n_iter - 1)  # likelihood with given
    # parameters

    axs[run, 1].plot(maes_runs['cl.w.'][run], label='cluster weights')
    axs[run, 1].plot(maes_runs['pro.w.'][run], label='profiles weights')
    axs[run, 1].plot(maes_runs['pro.'][run], label='profiles')

    # x and y axis labels, y-limit
    axs[run, 0].set_xticks(np.arange(0, n_iter))
    axs[run, 1].set_xticks(np.arange(0, n_iter))
    axs[run, 0].set_ylabel('likelihood')
    axs[run, 1].set_ylabel('MAE')
    for col in range(n_cols):
        axs[run, col].set_xlabel('Iterations')

    axs[run, 1].legend()

    # titles
    if run == 0:
        axs[run, 0].set_title('Run with uniform initial params.')
    elif run == 1:
        axs[run, 0].set_title('Run with correct params. as initial '
                              'weights')
    else:
        axs[run, 0].set_title(f'Run {run + 1}')

fig.suptitle(f'Test EM : {n_runs} runs')

fig.tight_layout()

# fig.savefig(f'{save_path}/sim_eval_em.png')

# plt.close(fig)

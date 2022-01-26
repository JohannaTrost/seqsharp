import time

import numpy as np
from scipy.stats import multinomial, dirichlet

# ******************************** PARAMETERS ******************************** #

n_alns = 100
n_sites = 40
n_seqs = 1200
n_aas = 6

n_profiles = 4
n_clusters = 2

cluster_weights = np.asarray([0.2, 0.8])
profile_weights = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                              [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
profiles = np.asarray([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                       [0.005, 0.005, 0.005, 0.005, 0.49, 0.49],
                       [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                       [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]])

# ******************************** SIMULATION ******************************** #

cluster_alns_asso = np.concatenate([int(cluster_weights[i] * n_alns) * [i]
                                    for i in range(n_clusters)], axis=0)
profile_site_asso = [np.concatenate([int(profile_weights[cl, i] * n_sites) * [i]
                                     for i in range(n_profiles)], axis=0)
                     for cl in cluster_alns_asso]
profile_site_asso = np.asarray([
    np.concatenate((assos,
                    np.where(profile_weights[cl]
                             == profile_weights[cl].max())[0]))
    if len(assos) < n_sites else assos
    for cl, assos in zip(cluster_alns_asso, profile_site_asso)])

aa_counts = []
for aln in range(n_alns):
    sites_counts = np.zeros((n_sites, n_aas))
    for site in range(n_sites):
        sites_counts[site] = profiles[profile_site_asso[aln, site]] * n_seqs
    aa_counts.append(sites_counts)

# ********************************** E-STEP ********************************** #

# pi : probability for profile z and cluster c at site j of alignment i given
#      weights and profiles and aa counts
pi = [np.zeros((aa_counts[aln].shape[0], n_clusters, n_profiles))
      for aln in range(n_alns)]

correct_pro_site = []

for aln in range(n_alns):
    n_sites = aa_counts[aln].shape[0]
    n_aas_site = aa_counts[aln].sum(axis=-1)
    prob_pro_site = np.asarray([multinomial.pmf(aa_counts[aln], n_aas_site,
                                                profiles[pro])
                                for pro in range(n_profiles)])

    # site masks - inverse Eigenmatrix n_sites x n_sites
    site_masks = ~np.eye(n_sites, dtype=bool)

    for cl in range(n_clusters):

        # prob. for remaining sites
        probs_remaining_sites = np.zeros((n_sites))
        for site, site_mask in enumerate(site_masks):
            prev_sum = 1
            for site_prob in prob_pro_site[:, site_mask].T:
                remaining_sites_prob = site_prob * profile_weights[cl, :]
                remaining_sites_prob *= prev_sum
                prev_sum = np.sum(remaining_sites_prob)
            probs_remaining_sites[site] = prev_sum
        """
        for not_site, site_mask in enumerate(site_masks):
            probs_not_site = prob_pro_site[:, site_mask]
            probs_remaining_sites[not_site] = np.prod(
                probs_not_site.T @ profile_weights[cl, :])
        """
        """
        for not_site, site_mask in enumerate(site_masks):
            probs_not_site = prob_pro_site[:, site_mask]
            probs_remaining_sites[not_site] = np.prod(np.sum(probs_not_site *
                                                     np.repeat(
                                                         profile_weights[cl, :,
                                                         np.newaxis],
                                                         n_sites - 1, axis=1),
                                                     axis=0))
        """

        probs_remaining_sites[probs_remaining_sites == 0] = np.finfo(float).eps

        pi[aln][:, cl, :] = np.repeat(probs_remaining_sites[:, np.newaxis],
                                      n_profiles, axis=1)
        pi[aln][:, cl, :] *= cluster_weights[cl]

        for pro in range(n_profiles):
            pi[aln][:, cl, pro] *= prob_pro_site[pro] * profile_weights[cl, pro]

            correct_pro_site.append(
                np.all((pi[aln][:, cl, pro] == np.max(pi[aln][:, cl, pro]))
                       == (profile_site_asso[aln] == pro)))

# without considering cluster
prof_site_acc = np.sum(np.asarray(correct_pro_site) == True) / len(
    correct_pro_site)

# ********************************** M-STEP ********************************** #


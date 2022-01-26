import os
import time
from itertools import permutations

from scipy.stats import multinomial, dirichlet
from matplotlib import pylab as plt

import numpy as np

np.random.seed(72)


def init_params(n_aas, n_profiles, n_clusters, n_sites):
    # init profiles
    profiles = dirichlet.rvs([2 * n_aas] * n_aas, n_profiles)

    # init profile probabilities per cluster
    cluster_profile_counts = np.random.randint(1, n_sites,
                                               (n_clusters, n_profiles))
    cluster_counts = cluster_profile_counts.sum(axis=1)
    cluster_counts_shaped = cluster_counts.repeat(
        [n_profiles] * n_clusters).reshape(n_clusters, n_profiles)
    profiles_weights = cluster_profile_counts / cluster_counts_shaped

    # init cluster probabilities
    weights = np.random.randint(1, n_alns, n_clusters)
    cluster_weights = weights / weights.sum()

    return profiles, profiles_weights, cluster_weights


def generate_msa_aa_counts(n_alns, n_seqs, n_sites, profiles, profiles_weights,
                           cluster_weights):
    n_profiles = len(profiles)
    n_clusters = len(cluster_weights)
    n_aas = profiles.shape[-1]

    # n_sites_width = 2
    # n_sites_per_aln = np.random.gamma(n_sites_avg,
    #                                  n_sites_width,
    #                                  n_alns).astype(int)

    alns = []
    profile_choices = []

    # pick cluster for alignments
    # cluster_counts = np.round(cluster_weights * n_alns, 0).astype(int)
    # cluster_choices = np.repeat(list(range(n_clusters)), cluster_counts)
    # np.random.shuffle(cluster_choices)
    cluster_choices = np.random.choice(n_clusters, size=n_alns,
                                       p=cluster_weights)

    for aln_index in range(n_alns):

        # pick profile for sites
        aln_profile_weights = profiles_weights[cluster_choices[aln_index]]
        # profiles_counts = np.round(aln_profile_weights * n_sites, 0).astype(int)
        # profile_indices = np.repeat(list(range(n_profiles)), profiles_counts)
        # np.random.shuffle(profile_indices)

        # if len(profile_indices) < n_sites:
        #     missing_indices = np.random.choice(n_profiles,
        #                                        size=n_sites -
        #                                             len(profile_indices),
        #                                        p=aln_profile_weights)
        #     profile_indices = np.concatenate((profile_indices, missing_indices))
        # profile_indices = profile_indices[:n_sites]
        profile_indices = np.random.choice(n_profiles, size=n_sites,
                                           p=aln_profile_weights)

        aln = np.zeros((n_sites, n_aas), dtype=int)
        for j in range(n_sites):
            # pick amino acids
            #site_profile = profiles[profile_indices[j]]
            #aln[j] = np.round(site_profile * n_seqs, 0).astype(int)
            aa_indices = np.random.choice(n_aas, size=n_seqs,
                                          p=profiles[profile_indices[j]])
            aln[j] = np.bincount(np.sort(aa_indices),
                                 minlength=n_aas)  # counts

        profile_choices.append(profile_indices)
        alns.append(aln)

    return alns, profile_choices, cluster_choices


def accuracy(gamma, gamma_pred, n_sites, prob_threshold=0.5):
    n_alns = len(gamma)
    n_correct_sites = 0
    for i in range(n_alns):
        correct_sites_mask = np.argmax(gamma_pred[i], axis=1) == np.argmax(
            gamma[i], axis=1)
        n_toolow_prob = (gamma_pred[i][correct_sites_mask].max(
            axis=1) <= prob_threshold).sum()
        print(f'corret: {correct_sites_mask.sum()} too low: {n_toolow_prob}')
        n_correct_sites += correct_sites_mask.sum() - n_toolow_prob
    return n_correct_sites / n_sites


def eval_preds(profiles, profiles_weights, gamma, profiles_pred, weights_pred,
               gamma_pred, n_sites_per_aln):
    '''
    n_profiles = len(profiles)
    n_alns = len(gamma)

    # sort predicted profiles according to actual profiles
    profiles_pred_sorted = np.zeros(profiles.shape)
    weights_pred_sorted = np.zeros(n_profiles)
    gamma_pred_sorted = [np.zeros((n_sites, n_profiles))
                         for n_sites in n_sites_per_aln]
    for i, p in enumerate(profiles):
        # choose the predicted profile with the smallest mae to profile i
        pred_profile_index = np.argmin(
            ((np.asarray([p] * n_profiles) - profiles_pred) ** 2).mean(axis=1))
        profiles_pred_sorted[i] = profiles_pred[pred_profile_index]
        weights_pred_sorted[i] = weights_pred[pred_profile_index]
        for aln_index in range(n_alns):
            gamma_pred_sorted[aln_index][:, i] = gamma_pred[aln_index][:, pred_profile_index]
    '''
    mae_profiles = ((profiles - profiles_pred) ** 2).mean()
    mae_weights = ((profiles_weights - weights_pred) ** 2).mean()
    acc = accuracy(gamma, gamma_pred, n_sites_per_aln.sum())

    return acc, mae_profiles, mae_weights


class MultinomialExpectationMaximizer:
    def __init__(self, C, K, rtol=1e-3, max_iter=100, restarts=10):
        self._n_clusters = C
        self._n_profiles = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):  # TODO
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = (self._get_mixture_weight(alpha, k) *
                          self._multinomial_prob_aln_sites(X_test, beta[k]))
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):  # TODO
        if log_likelihood is None:
            log_likelihood = self.compute_log_likelihood(X_test, alpha, beta)
        N = X_test.shape[0]
        return np.log(N) * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_icl_bic(self, bic, gamma):  # TODO
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic - classification_entropy

    def _e_step(self, aa_counts, alpha_profiles, alpha_clusters, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        aa_counts: (A x N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (A x N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        K = beta.shape[0]
        C = alpha_clusters.shape[0]
        weighted_multi_prob = []

        for i in range(len(aa_counts)):
            n_sites = len(aa_counts[i])

            # P(A_i | v_k) for all sites
            n_aas_site = aa_counts[i].sum(axis=-1)
            sites_profile_probs = [multinomial.pmf(aa_counts[i], n_aas_site,
                                                   beta[k])
                                   for k in range(K)]
            sites_profile_probs = np.asarray(sites_profile_probs)

            # site masks - inverse Eigenmatrix n_sites x n_sites
            sites_masks = ~np.eye(n_sites, dtype=bool)

            aln_weighted_probs = np.zeros((n_sites, C, K))
            other_sites_clusters = np.zeros((n_sites, C))

            for c in range(C):
                # remaining sites for each site
                for not_site, sites_mask in enumerate(sites_masks):
                    probs_not_site = sites_profile_probs[:, sites_mask]
                    other_sites_clusters[not_site, c] = np.prod(
                        probs_not_site.T @ alpha_profiles[c, :])

                other_sites_clusters[other_sites_clusters[:, c] == 0, c] = \
                    np.finfo(float).eps

                for k in range(K):
                    aln_weighted_probs[:, c, k] = (sites_profile_probs[k, :]
                                                   * other_sites_clusters[:, c]
                                                   * alpha_clusters[c]
                                                   * alpha_profiles[c, k])
            # To avoid division by 0
            aln_weighted_probs[aln_weighted_probs == 0] = np.finfo(float).eps
            weighted_multi_prob.append(aln_weighted_probs)

        denum = [weighted_multi_prob[i].sum(axis=2).sum(axis=1)
                 for i in range(len(weighted_multi_prob))]
        # prepare array shape for division
        denum = [np.repeat(denum[i][:, np.newaxis], C, axis=1)
                 for i in range(len(weighted_multi_prob))]
        denum = [np.repeat(denum[i][:, :, np.newaxis], K, axis=2)
                 for i in range(len(weighted_multi_prob))]

        # normalize
        gamma = [weighted_multi_prob[i] / denum[i]
                 for i in range(len(weighted_multi_prob))]

        return gamma

    def _get_mixture_weight(self, alpha, k):
        return alpha[k]

    def _m_step(self, alns_aa_counts, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        aa_counts: (A x N x C), matrix of counts
        gamma: (A x N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha_clusters, alpha_profiles = self._m_step_alpha(gamma)

        # Compute beta
        beta = self._m_step_beta(alns_aa_counts, gamma)

        return alpha_clusters, alpha_profiles, beta

    def _m_step_alpha(self, gamma):

        # average over probabilities per site per cluster
        cluster_weights = np.mean([probs.sum(axis=2).mean(axis=0)
                                   for probs in gamma],
                                  axis=0)

        sum_profile_porbs = [probs.sum(axis=2) for probs in gamma]
        sum_profile_porbs = [
            np.repeat(probs[:, :, np.newaxis], n_profiles, axis=2)
            for probs in sum_profile_porbs]
        profile_weights = np.mean(
            [(gamma[i] / sum_profile_porbs[i]).mean(axis=0)
             for i in range(len(gamma))],
            axis=0)

        return cluster_weights, profile_weights

    def _m_step_beta(self, X, gamma):
        # weighted_counts = gamma.T.dot(aa_counts)
        n_cluster = gamma[0].shape[1]
        n_profiles = gamma[0].shape[2]
        n_aas = X[0].shape[-1]
        weighted_counts = np.zeros((n_profiles, n_aas))
        for aln in range(len(gamma)):
            for p in range(n_profiles):
                for c in range(n_cluster):
                    weighted_counts[p] += gamma[aln][:, c, p].T.dot(X[aln])
        # normalize
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb_fl(self, aa_counts, alpha_clusters, alpha_profiles,
                        beta, gamma):
        n_alns = len(aa_counts)
        n_profiles = beta.shape[0]
        n_cluster = alpha_clusters.shape[0]

        # avoid division by 0
        alpha_profiles[alpha_profiles == 0] = np.finfo(float).eps
        alpha_clusters[alpha_clusters == 0] = np.finfo(float).eps

        # compute lower bound of full likelihood
        loss = 0
        for i in range(n_alns):
            # P(A_i | v_k) for all sites
            n_aas_site = aa_counts[i].sum(axis=-1)
            sites_profile_probs = [multinomial.pmf(aa_counts[i], n_aas_site,
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
                    log_other_sites_clusters[not_site, c] = np.sum(np.log(
                        probs_not_site.T @ alpha_profiles[c, :]))

                for k in range(n_profiles):
                    weights = (gamma[i][:, c, k])
                    loss += np.sum(weights *
                                   (log_sites_profile_probs[k]
                                    + log_other_sites_clusters[:, c]
                                    + np.log(alpha_profiles[c, k])
                                    + np.log(alpha_clusters[c])))
                    loss -= np.sum(weights * np.log(weights))

        return loss

    def _init_params(self, X):
        n_aas = X[0].shape[-1]
        n_sites = X[0].shape[0]
        n_alns = len(X)

        # init profiles
        profiles = dirichlet.rvs([2 * n_aas] * n_aas, self._n_profiles)

        # init profile probabilities per cluster
        cluster_profile_counts = np.random.randint(1,
                                                   n_sites,
                                                   (self._n_clusters,
                                                    self._n_profiles))
        cluster_counts = cluster_profile_counts.sum(axis=1)
        cluster_counts_shaped = cluster_counts.repeat(
            [self._n_profiles] * n_clusters).reshape(self._n_clusters,
                                                     self._n_profiles)
        profiles_weights = cluster_profile_counts / cluster_counts_shaped

        # init cluster probabilities
        weights = np.random.randint(1, n_alns, self._n_clusters)
        cluster_weights = weights / weights.sum()

        return profiles, profiles_weights, cluster_weights

    def _train_once(self, alns_aa_counts, target_profiles,
                    target_alpha_profiles, target_alpha_clusters,
                    init_qpv=None):
        '''
        Runs one full cycle of the EM algorithm
        :param alns_aa_counts: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        if init_qpv is None:
            beta, alpha_profiles, alpha_clusters = self._init_params(
                alns_aa_counts)
        else:
            beta, alpha_profiles, alpha_clusters = init_qpv

        gamma = None
        losses = []
        parameters = []

        for it in range(self._max_iter):
            print(it)

            # beta = target_profiles
            alpha_profiles = target_alpha_profiles
            alpha_clusters = target_alpha_clusters

            parameters.append([alpha_clusters, alpha_profiles, beta])

            gamma = self._e_step(alns_aa_counts, alpha_profiles, alpha_clusters,
                                 beta)
            alpha_clusters, alpha_profiles, beta = self._m_step(alns_aa_counts,
                                                                gamma)

            # evaluation

            loss = self._compute_vlb_fl(alns_aa_counts, alpha_clusters,
                                        alpha_profiles, beta, gamma)

            losses.append(loss)

            print(f'Loss: {loss}')
            # if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
            #     print(f'Finished after {it + 1} training cycles')
            #     break

        return parameters, gamma, np.asarray(losses)

    def fit(self, alns_aa_counts, profiles, target_alpha_profiles,
            target_alpha_clusters):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        :param alns_aa_counts: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_mae_cluster = float('inf')
        best_mae_profile = float('inf')
        best_alpha_cluster = None
        best_alpha_profiles = None
        best_beta = profiles
        best_gamma = None
        losses = None
        best_maes = None
        alphas_cluster = []
        alphas_profiles = []
        gammas = []

        for it in range(self._restarts):
            print('iteration %i' % it)
            estimates, gamma, loss = self._train_once(alns_aa_counts, profiles,
                                                      target_alpha_profiles,
                                                      target_alpha_clusters)

            if maes['cluster'][-1] < best_mae_cluster:
                print(
                    f'better cluster mae (iteration {it}) {best_mae_cluster} vs. {maes["cluster"][-1]}')
                best_mae_cluster = maes["cluster"][-1]
            if maes['profile'][-1] < best_mae_profile:
                print(
                    f'better profile mae (iteration {it}) {best_mae_profile} vs. {maes["profile"][-1]}')
                best_mae_profile = maes["profile"][-1]
            if loss[-1] > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss[-1]))
                best_loss = loss[-1]
                best_mae_cluster = maes["cluster"][-1]
                best_mae_profile = maes["profile"][-1]
                best_alpha_cluster = alpha_cluster
                best_alpha_profiles = alpha_profiles
                best_beta = beta
                best_gamma = gamma
                losses = loss
                best_maes = maes
            alphas_cluster.append(alpha_cluster)
            alphas_profiles.append(alpha_profiles)
            gammas.append(gamma)

        return (
            losses, best_maes, best_alpha_cluster, best_alpha_profiles,
            best_beta,
            best_gamma, alphas_cluster, alphas_profiles, gammas)


# init simulation parameters
n_sim = 1
n_trials = 5
max_iter = 15
n_alns = 100
n_seqs = 100
n_clusters = 2
n_profiles = 4
n_sites = 30
n_aas = 6

print(f'{n_sim} simulations\n{n_trials} trials\n{n_alns} MSAs\n{n_seqs} '
      f'sequences per MSA\n{n_sites} sites per MSA\n{n_aas} amino acids\n'
      f'{n_clusters} clusters \n{n_profiles} profiles\n')

timestamp = time.time()
save_path = f'../results/{timestamp}'
os.mkdir(save_path)

# in evaluation values
accs = np.zeros((n_sim, n_trials, 2))
maes_sims = {'cl.w.': [], 'pro.w.': [], 'pro.': []}
losses_sims = []

for ind_sim in range(n_sim):

    np.random.seed(ind_sim)

    # geneate profiles and weights
    # profiles, profiles_weights, cluster_weights = init_params(n_aas,
    #                                                          n_profiles,
    #                                                          n_clusters,
    #                                                          n_sites)
    cluster_weights = np.asarray([0.2, 0.8])
    profiles_weights = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                                  [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
    profiles = np.asarray([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                           [0.005, 0.005, 0.005, 0.005, 0.49, 0.49],
                           [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                           [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]])

    # generate alignments
    simulation = generate_msa_aa_counts(n_alns, n_seqs, n_sites, profiles,
                                        profiles_weights, cluster_weights)

    train_aa_counts, profile_choices, cluster_choices = simulation

    print('_________________________________________________')
    print(f'Simulation {ind_sim + 1}')

    maes_trials = {'cl.w.': [], 'pro.w.': [], 'pro.': []}
    losses_trials = []

    # compute full log-likelihood with given parameters
    gamma_given_params = model._e_step(train_aa_counts, profiles_weights,
                                       cluster_weights, profiles)
    given_params_loss = model._compute_vlb_fl(train_aa_counts,
                                              cluster_weights,
                                              profiles_weights, profiles,
                                              gamma_given_params)

    np.random.seed(72)

    for ind_trial in range(n_trials):

        print('_________________________________________________')
        print(f'\tTrial {ind_trial + 1}')

        model = MultinomialExpectationMaximizer(n_clusters, n_profiles,
                                                restarts=10, rtol=1e-8,
                                                max_iter=max_iter)

        if ind_trial == 0:
            alpha_clusters = np.ones((n_clusters)) / n_clusters
            alpha_profiles = np.ones((n_clusters, n_profiles)) / n_profiles
            beta = np.ones((n_profiles, n_aas)) / n_aas
            init_qpv = [beta, alpha_profiles, alpha_clusters]
            estimates, gamma, train_losses = model._train_once(train_aa_counts,
                                                               profiles,
                                                               profiles_weights,
                                                               cluster_weights,
                                                               init_qpv)
        elif ind_trial == 1:
            init_qpv = [profiles, profiles_weights, cluster_weights]
            estimates, gamma, train_losses = model._train_once(train_aa_counts,
                                                               profiles,
                                                               profiles_weights,
                                                               cluster_weights,
                                                               init_qpv)
        else:
            estimates, gamma, train_losses = model._train_once(train_aa_counts,
                                                               profiles,
                                                               profiles_weights,
                                                               cluster_weights)

        alpha_clusters, alpha_profiles, beta = estimates[-1]

        # best_train_loss, best_train_maes, alpha_clusters, alpha_profiles, best_beta, gamma, alphas_cluster, alphas_profiles, gammas = model.fit(
        #    train_aa_counts, profiles, profiles_weights, cluster_weights)
        # test e-step and m-step
        # gamma = model._e_step(train_aa_counts, profiles_weights, cluster_weights,
        #                       profiles)
        # alpha_clusters, alpha_profiles, beta = model._m_step(train_aa_counts, gamma)
        """
        em_profile_choice = [
            gamma[i][:, cluster_choices[i], :].argmax(axis=-1)
            for i in range(n_alns)]

        accs_site_prof = [
            (profile_choices[i] == em_profile_choice[i]).sum() /
            n_sites for i in range(n_alns)]
        accs_aln_cluster = [(gamma[i].sum(axis=2).argmax(axis=1) ==
                             cluster_choices[i]).sum() / n_sites
                            for i in range(n_alns)]
        """
        """
        # get optimal profile and cluster order
        # weighted profiles for given profiles and weights
        weighted_profile_weights = (profiles_weights
                                    * np.repeat(cluster_weights[:, np.newaxis],
                                                n_profiles, axis=1))
        weighted_profiles = profiles * np.repeat(weighted_profile_weights
                                                 [:, :, np.newaxis],
                                                 n_aas, axis=2)

        # weighted profiles for estimated profiles and weights
        weighted_alpha_profiles = (alpha_profiles
                                   * np.repeat(alpha_clusters[:, np.newaxis],
                                               n_profiles, axis=1))
        weighted_beta = beta * np.repeat(weighted_alpha_profiles
                                         [:, :, np.newaxis],
                                         n_aas, axis=2)

        # find best cluster and profile order to match given parameters
        clusters_inds = list(range(n_clusters))
        clusters_permuts = np.asarray(list(set(permutations(clusters_inds))))
        profiles_inds = list(range(n_profiles))
        profiles_permuts = np.asarray(list(set(permutations(profiles_inds))))

        best_mae_weighted_profiles = np.inf
        best_cluster_order = None
        best_profile_order = None
        for cluster_order in clusters_permuts:
            reorderd_params = weighted_beta[cluster_order]
            for profile_order in profiles_permuts:
                rep_order = np.repeat(profile_order[np.newaxis],
                                      n_clusters, axis=0)
                rep_order = np.repeat(rep_order[:, :, np.newaxis], n_aas,
                                      axis=2)
                reorderd_params = np.take_along_axis(reorderd_params, rep_order,
                                                     axis=1)
                mae = np.mean(np.abs(weighted_profiles - reorderd_params))
                if mae < best_mae_weighted_profiles:
                    best_mae_weighted_profiles = mae
                    best_cluster_order = cluster_order
                    best_profile_order = profile_order
        """
        profiles_inds = list(range(n_profiles))
        profiles_permuts = np.asarray(list(set(permutations(profiles_inds))))

        mae_profiles = np.zeros((len(profiles_permuts)))
        mae_profile_weights = np.zeros((len(profiles_permuts)))

        for i, order in enumerate(profiles_permuts):
            # profile order according to profiles
            mae_profiles[i] = np.mean(np.abs(profiles - beta[order, :]))

        ind_min_mae = np.argmin(mae_profiles)
        best_profile_order = profiles_permuts[ind_min_mae, :]

        # mae for profile weights given order obtained by profile maes
        sorted_alpha_profiles = np.take_along_axis(
            alpha_profiles,
            np.repeat(best_profile_order[np.newaxis], n_clusters, axis=0),
            axis=1)

        # get optimal cluster order
        clusters_inds = list(range(n_clusters))
        clusters_permuts = np.asarray(list(set(permutations(clusters_inds))))

        mae_profile_weights = np.zeros((len(clusters_permuts)))

        for i, order in enumerate(clusters_permuts):
            # cluster order according to profile weights
            mae_profile_weights[i] = np.mean(np.abs(profiles_weights -
                                                    sorted_alpha_profiles[
                                                        order]))
        ind_min_mae = np.argmin(mae_profile_weights)
        best_cluster_order = clusters_permuts[ind_min_mae, :]

        # compute errors for all iterations given best profile and cluster order
        maes_profiles, maes_profile_weights, maes_cluster_weights = [], [], []
        for clw, prow, pro in estimates:
            # compute profiles error
            mae_profiles = np.mean(np.abs(profiles
                                          - pro[best_profile_order, :]))
            # compute profiles weights error
            reorderd_alpha_profiles = prow[best_cluster_order]
            reorderd_alpha_profiles = np.take_along_axis(
                reorderd_alpha_profiles,
                np.repeat(best_profile_order[np.newaxis], n_clusters, axis=0),
                axis=1)
            mae_profile_weights = np.mean(np.abs(profiles_weights
                                                 - reorderd_alpha_profiles))
            # compute cluster weights error
            mae_cluster_weights = np.mean(np.abs(cluster_weights
                                                 - clw[best_cluster_order]))

            maes_profiles.append(mae_profiles)
            maes_profile_weights.append(mae_profile_weights)
            maes_cluster_weights.append(mae_cluster_weights)

        maes_trials['cl.w.'].append(maes_cluster_weights)
        maes_trials['pro.w.'].append(maes_profile_weights)
        maes_trials['pro.'].append(maes_profiles)

        losses_trials.append(train_losses)

        accs[ind_sim, ind_trial] = np.asarray([np.mean(accs_site_prof),
                                               np.mean(accs_aln_cluster)])

    maes_sims['cl.w.'].append(maes_trials['cl.w.'])
    maes_sims['pro.w.'].append(maes_trials['pro.w.'])
    maes_sims['pro.'].append(maes_trials['pro.'])
    losses_sims.append(losses_trials)

    ######### PLOT RESULTS #########

    best_loss_ind = np.argmax([trial_loss[-1] for trial_loss in losses_trials])
    n_rows, n_cols = n_trials, 2

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                            figsize=(12., 12.))
    for trial in range(n_trials):
        axs[trial, 0].plot(losses_trials[trial])
        axs[trial, 0].hlines(y=given_params_loss, color='red',
                             xmin=0,
                             xmax=max_iter)  # loss with given parameters

        axs[trial, 1].plot(maes_trials['cl.w.'][trial], label='cluster weights')
        axs[trial, 1].plot(maes_trials['pro.w.'][trial], label='profiles '
                                                               'weights')
        axs[trial, 1].plot(maes_trials['pro.'][trial], label='profiles')

        # x and y axis labels, y-limit
        axs[trial, 0].set_ylabel('Loss')
        axs[trial, 1].set_ylabel('MAE')
        for col in range(n_cols):
            axs[trial, col].set_xlabel('Iterations')

        axs[trial, 1].legend()

        # titles
        if trial == 0:
            axs[trial, 0].set_title('Run with uniform initial params.')
        elif trial == 1:
            axs[trial, 0].set_title('Run with correct params. as initial '
                                    'weights')
        else:
            axs[trial, 0].set_title(f'Run {trial + 1}')

    fig.suptitle(f'Test EM : {n_trials} trials for simulation {ind_sim}')

    fig.tight_layout()

    fig.savefig(f'{save_path}/{ind_sim}sim_eval_em.png')

    plt.close(fig)

print(f'\n--------- FOR {n_sim} SIMULATIONS with {n_trials} TRIALS ---------')
print(f'MSA-cluster-association accuracy : ')
print(f'\t avg.{np.round(accs[:, :, 1].mean(axis=0), 4)}')
print(f'\t min.{np.round(accs[:, :, 1].min(axis=0), 4)}')
print(f'\t max.{np.round(accs[:, :, 1].mean(axis=0), 4)}')
print(f'Site-profile-association accuracy : ')
print(f'\t avg.{np.round(accs[:, :, 0].mean(axis=0), 4)}')
print(f'\t min.{np.round(accs[:, :, 0].min(axis=0), 4)}')
print(f'\t max.{np.round(accs[:, :, 0].max(axis=0), 4)}')
print(f'Profile weights MAE : ')
print(f'\t avg.{np.round(np.mean(maes_sims["pro.w."], axis=0), 4)}')
print(f'\t min.{np.round(np.min(maes_sims["pro.w."], axis=0), 4)}')
print(f'\t max.{np.round(np.max(maes_sims["pro.w."], axis=0), 4)}')
print(f'Cluster weights MAE : ')
print(f'\t avg.{np.round(np.mean(maes_sims["cl.w."], axis=0), 4)}')
print(f'\t min.{np.round(np.min(maes_sims["cl.w."], axis=0), 4)}')
print(f'\t max.{np.round(np.max(maes_sims["cl.w."], axis=0), 4)}')
print(f'Profiles MAE : ')
print(f'\t avg.{np.round(np.mean(maes_sims["pro."], axis=0), 4)}')
print(f'\t min.{np.round(np.min(maes_sims["pro."], axis=0), 4)}')
print(f'\t max.{np.round(np.max(maes_sims["pro."], axis=0), 4)}')

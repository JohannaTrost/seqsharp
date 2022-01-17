import os
import time

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
            site_profile = profiles[profile_indices[j]]
            aln[j] = np.round(site_profile * n_seqs, 0).astype(int)
            # aa_indices = np.random.choice(n_aas, size=n_seqs,
            #                               p=profiles[profile_indices[j]])
            # aln[j] = np.bincount(np.sort(aa_indices),
            #                      minlength=n_aas)  # counts

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

    def _remaining_sites_prob(self, counts, n_aas_sites, beta, alpha):
        n_sites = counts.shape[0]
        n_profiles = beta.shape[0]

        # mutinomial probabilities for all sites and all profiles
        probs = [multinomial.pmf(counts, n_aas_sites, profile) * alpha[j]
                 for j, profile in enumerate(beta)]
        probs = np.asarray(probs)

        # dynamic computation of probability for all remaining sites
        memo_mat = np.zeros((n_sites, n_profiles))  # memoization matrix
        for site in range(len(counts)):
            prev_sum = memo_mat[site - 1, :].sum() if site > 0 else 1
            for profile in range(len(beta)):
                memo_mat[site, profile] = prev_sum * probs[site, profile]

        return memo_mat[-1, :].sum()

    def _multinomial_prob_aln_sites(self, counts, beta, alpha, profile,
                                    log=False):

        """Evaluates the multinomial probabilities for given sites (aa counts)
        counts: (N x C), matrix of aa counts per site of one MSA
        beta: (K x C), vector of multinomial parameters, profiles
        alpha: (K x C), profile weights for a given cluster
        profile: (C), index of a profile in beta
        Returns:
        p: (N), scalar values for the probabilities of observing each site
                given a profile
        """
        n_sites = len(counts)

        probs_sites = np.zeros((n_sites))

        for j, site in enumerate(counts):
            sites_mask = np.asarray(range(n_sites)) != j
            n = site.sum(axis=-1)

            # compute probability per site considering the rest of the alignment
            other_sites_prob = self._remaining_sites_prob(counts[sites_mask],
                                                          beta, alpha)
            if log:
                site_prob = multinomial.logpmf(site, n, beta[profile])
                # avoid division by zero
                if other_sites_prob == 0:
                    other_sites_prob = np.finfo(float).eps
                probs_sites[j] = site_prob + np.log(other_sites_prob)
            else:
                site_prob = multinomial.pmf(site, n, beta[profile])
                probs_sites[j] = site_prob * other_sites_prob

        return probs_sites

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
                        probs_not_site.T@alpha_profiles[c, :])

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
        # n_sites = np.sum([probs.sum() for probs in gamma])

        # average over probabilities per site per cluster
        cluster_weights = np.mean([probs.sum(axis=2).mean(axis=0)
                                   for probs in gamma],
                                  axis=0)
        # trying to sum in denominator start from a single site !!
        # for i in range(len(gamma[0])):
        #     cluster_prob_sum = gamma[i][:, :, :].sum(axis=1)
        #     clw0 = [[gamma[0][i, j, :] / cl_sum[i] for j in range(n_clusters)]
        #             for i in range(len(gamma[0]))]

        sum_profile_porbs = [probs.sum(axis=2) for probs in gamma]
        sum_profile_porbs = [np.repeat(probs[:, :, np.newaxis], n_profiles, axis=2)
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
            cluster_aln_prob = gamma[i].sum(axis=2).mean(axis=0)
            loss += cluster_aln_prob.T.dot(np.log(cluster_weights))

            # P(A_i | v_k) for all sites
            n_aas_site = aa_counts[i].sum(axis=-1)
            sites_profile_probs = [multinomial.pmf(aa_counts[i], n_aas_site,
                                                   beta[k])
                                   for k in range(n_profiles)]
            sites_profile_probs = np.asarray(sites_profile_probs)
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
                    loss += np.sum(weights * log_sites_profile_probs)
                    loss += np.sum(weights * log_other_sites_clusters[:, c])
                    loss += np.sum(weights * np.log(alpha_profiles[c, k]))
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

        loss = -float('inf')
        mae_cluster_weights = float('inf')
        mae_profile_weights = float('inf')
        mae_profiles = float('inf')
        gamma = None
        losses = []
        maes = {'cl.w.': [], 'pro.w.': [], 'pro.': []}

        for it in range(self._max_iter):

            print(it)

            prev_loss = loss
            prev_mae_cluster_weights = mae_cluster_weights
            prev_mae_profile_weights = mae_profile_weights
            prev_mae_profiles = mae_profiles

            gamma = self._e_step(alns_aa_counts, alpha_profiles, alpha_clusters,
                                 beta)
            alpha_clusters, alpha_profiles, beta = self._m_step(alns_aa_counts,
                                                                gamma)

            # evaluation

            loss = self._compute_vlb_fl(alns_aa_counts, alpha_clusters,
                                        alpha_profiles, beta, gamma)

            mae_cluster_weights = np.abs(target_alpha_clusters
                                         - alpha_clusters).mean()
            mae_profile_weights = np.abs(target_alpha_profiles
                                         - alpha_profiles).mean()
            mae_profiles = np.abs(target_profiles - beta).mean()

            if loss < prev_loss:
                print(f'loss ({loss}) < prev_loss ({prev_loss})')
            if mae_cluster_weights > prev_mae_cluster_weights:
                print(f'cl.w.mae ({mae_cluster_weights}) > prev.cl.w.mae '
                      f'({prev_mae_cluster_weights})')
            if mae_profile_weights > prev_mae_profile_weights:
                print(
                    f'pro.w.mae. ({mae_profile_weights}) > prev.pro.w.mae ({prev_mae_profile_weights})')
            if mae_profiles > prev_mae_profiles:
                print(
                    f'pro.mae ({mae_profiles}) > prev.pro.mae ({prev_mae_profiles})')

            losses.append(loss)
            maes['cl.w.'].append(mae_cluster_weights)
            maes['pro.w.'].append(mae_profile_weights)
            maes['pro.'].append(mae_profiles)

            print('Loss: %f' % loss)
            # if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
            #     print(f'Finished after {it + 1} training cycles')
            #     break

        maes['cl.w.'] = np.asarray(maes['cl.w.'])
        maes['pro.w.'] = np.asarray(maes['pro.w.'])
        maes['pro.'] = np.asarray(maes['pro.'])

        return alpha_clusters, alpha_profiles, beta, gamma, np.asarray(
            losses), maes

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
            alpha_cluster, alpha_profiles, beta, gamma, loss, maes = \
                self._train_once(alns_aa_counts, profiles,
                                 target_alpha_profiles, target_alpha_clusters)

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
n_sim = 10
n_trials = 5
n_alns = 50
n_seqs = 100
n_clusters = 3
n_profiles = 6
n_sites = 30
n_aas = 4

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
    profiles, profiles_weights, cluster_weights = init_params(n_aas,
                                                              n_profiles,
                                                              n_clusters,
                                                              n_sites)
    # generate alignments
    simulation = generate_msa_aa_counts(n_alns, n_seqs, n_sites, profiles,
                                        profiles_weights, cluster_weights)

    print('_________________________________________________')
    print(f'Simulation {ind_sim + 1}')

    maes_trials = {'cl.w.': [], 'pro.w.': [], 'pro.': []}
    losses_trials = []

    np.random.seed(72)

    for ind_trial in range(n_trials):

        print('_________________________________________________')
        print(f'\tTrial {ind_trial + 1}')

        train_aa_counts, profile_choices, cluster_choices = simulation

        model = MultinomialExpectationMaximizer(n_clusters, n_profiles,
                                                restarts=10, rtol=1e-8,
                                                max_iter=15)

        if ind_trial != 0:
            alpha_clusters, alpha_profiles, beta, gamma, train_losses, maes = \
                model._train_once(train_aa_counts, profiles, profiles_weights,
                                  cluster_weights)
        else:
            alpha_clusters = np.ones((n_clusters)) / n_clusters
            alpha_profiles = np.ones((n_clusters, n_profiles)) / n_profiles
            beta = np.ones((n_profiles, n_aas)) / n_aas
            init_qpv = [beta, alpha_profiles, alpha_clusters]

            alpha_clusters, alpha_profiles, beta, gamma, train_losses, maes = \
                model._train_once(train_aa_counts, profiles, profiles_weights,
                                  cluster_weights, init_qpv)

        # best_train_loss, best_train_maes, alpha_clusters, alpha_profiles, best_beta, gamma, alphas_cluster, alphas_profiles, gammas = model.fit(
        #    train_aa_counts, profiles, profiles_weights, cluster_weights)
        # test e-step and m-step
        #gamma = model._e_step(train_aa_counts, profiles_weights, cluster_weights,
        #                       profiles)
        #alpha_clusters, alpha_profiles, beta = model._m_step(train_aa_counts, gamma)

        em_profile_choice = [
            gamma[i][:, cluster_choices[i], :].argmax(axis=-1)
            for i in range(n_alns)]

        accs_site_prof = [
            (profile_choices[i] == em_profile_choice[i]).sum() /
            n_sites for i in range(n_alns)]
        accs_aln_cluster = [(gamma[i].sum(axis=2).argmax(axis=1) ==
                             cluster_choices[i]).sum() / n_sites
                            for i in range(n_alns)]

        maes_trials['cl.w.'].append(maes['cl.w.'])
        maes_trials['pro.w.'].append(maes['pro.w.'])
        maes_trials['pro.'].append(maes['pro.'])
        losses_trials.append(train_losses)
        accs[ind_sim, ind_trial] = np.asarray([np.mean(accs_site_prof),
                                               np.mean(accs_aln_cluster)])

    maes_sims['cl.w.'].append(maes_trials['cl.w.'])
    maes_sims['pro.w.'].append(maes_trials['pro.w.'])
    maes_sims['pro.'].append(maes_trials['pro.'])
    losses_sims.append(losses_trials)

    ######### PLOT RESULTS #########

    best_loss_ind = np.argmax([trial_loss[-1]
                               for trial_loss in losses_trials])

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True,
                            figsize=(12., 6.))

    axs[0, 0].plot(losses_trials[0], label=f'uniform init. param.',
                   linewidth=3.0)
    axs[0, 1].plot(maes_trials["cl.w."][0],
                   label=f'uniform init. param.',
                   linewidth=3.0)
    axs[1, 0].plot(maes_trials["pro.w."][0],
                   label=f'uniform init. param.',
                   linewidth=3.0)
    axs[1, 1].plot(maes_trials["pro."][0],
                   label=f'uniform init. param.',
                   linewidth=3.0)

    for trial in range(1, n_trials):
        if trial == best_loss_ind:
            axs[0, 0].plot(losses_trials[trial], label=f'{trial} best loss',
                           linewidth=3.0)
            axs[0, 1].plot(maes_trials["cl.w."][trial],
                           label=f'{trial} best loss',
                           linewidth=3.0)
            axs[1, 0].plot(maes_trials["pro.w."][trial],
                           label=f'{trial} best loss',
                           linewidth=3.0)
            axs[1, 1].plot(maes_trials["pro."][trial],
                           label=f'{trial} best loss',
                           linewidth=3.0)
        else:
            axs[0, 0].plot(losses_trials[trial], label=trial)
            axs[0, 1].plot(maes_trials["cl.w."][trial], label=trial)
            axs[1, 0].plot(maes_trials["pro.w."][trial], label=trial)
            axs[1, 1].plot(maes_trials["pro."][trial], label=trial)

    fig.suptitle(f'Test EM : {n_trials} trials for simulation {ind_sim}')
    axs[0, 0].set_title('Loss (lower bound of log-likelihood)')
    axs[0, 1].set_title(f'MAE cluster weights {cluster_weights}')
    axs[1, 0].set_title('MAE profile weights')
    axs[1, 1].set_title('MAE profiles')

    max_ylim = np.max(list(maes_trials.values()))
    max_ylim = np.true_divide(np.ceil(max_ylim * 10), 10)
    ylims = (0, max_ylim)

    axs[0, 1].set_ylim(ylims)
    axs[1, 0].set_ylim(ylims)
    axs[1, 1].set_ylim(ylims)

    fig.tight_layout()

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

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

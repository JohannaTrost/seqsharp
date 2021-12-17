import random

from scipy.stats import multinomial, dirichlet
import numpy as np


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
        # choose the predicted profile with the smallest mse to profile i
        pred_profile_index = np.argmin(
            ((np.asarray([p] * n_profiles) - profiles_pred) ** 2).mean(axis=1))
        profiles_pred_sorted[i] = profiles_pred[pred_profile_index]
        weights_pred_sorted[i] = weights_pred[pred_profile_index]
        for aln_index in range(n_alns):
            gamma_pred_sorted[aln_index][:, i] = gamma_pred[aln_index][:, pred_profile_index]
    '''
    mse_profiles = ((profiles - profiles_pred) ** 2).mean()
    mse_weights = ((profiles_weights - weights_pred) ** 2).mean()
    acc = accuracy(gamma, gamma_pred, n_sites_per_aln.sum())

    return acc, mse_profiles, mse_weights


def dice(prob_lst, rolls=100, trials=100):
    data = []
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
        noise = 0.05
        newdata = (np.atleast_2d(
            [[vals.count(val) for val in set(vals)]] * trials) *
                   (1 - (np.random.random((trials, 6)) * noise))).astype(int)

        if len(data) > 1:
            data = np.concatenate((data, newdata))
        else:
            data = newdata + 0
    return data


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
            mn_probs_k = self._get_mixture_weight(alpha,
                                                  k) * self._multinomial_prob_aln_sites(
                X_test, beta[k])
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

    def _remaining_sites_prob(self, counts, beta, alpha, log=False):
        n_sites = counts.shape[0]
        n_profiles = beta.shape[0]

        memo_mat = np.zeros((n_sites, n_profiles))  # memoization matrix

        for i, site in enumerate(counts):
            n = site.sum(axis=-1)
            prev_sum = memo_mat[i-1, :].sum() if i > 0 else 1
            for j, profile in enumerate(beta):
                if log:
                    prob = multinomial.logpmf(site, n, profile) * alpha[j]
                else:
                    prob = multinomial.pmf(site, n, profile) * alpha[j]
                memo_mat[i, j] = prev_sum * prob

        return memo_mat[-1, :].sum()



    def _multinomial_prob_aln_sites(self, counts, beta, alpha, profile, log=False):

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

            # compute 1 probability per site considering the rest of the
            # alignment
            if log:  # TODO: check again log version
                site_prob = multinomial.logpmf(site, n, beta[profile])
                other_sites_prob = self._remaining_sites_prob(
                    counts[sites_mask], beta, alpha, log=True)
                probs_sites[j] = site_prob + other_sites_prob
            else:
                site_prob = multinomial.pmf(site, n, beta[profile])
                other_sites_prob = self._remaining_sites_prob(
                    counts[sites_mask], beta, alpha)
                probs_sites[j] = site_prob * other_sites_prob

        return probs_sites

    def _e_step(self, X, alpha_profiles, alpha_clusters, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        alns_aa_counts: (A x N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (A x N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        K = beta.shape[0]
        C = alpha_clusters.shape[0]
        weighted_multi_prob = []
        for i in range(len(X)):

            print(f'{i + 1}/{n_alns} MSAs')

            aln_weighted_probs = np.zeros((len(X[i]), C, K))
            for c in range(C):
                for k in range(K):
                    aln_weighted_probs[:, c, k] = \
                        (alpha_profiles[c, k] * alpha_clusters[c] *
                         self._multinomial_prob_aln_sites(X[i], beta,
                                                          alpha_profiles[c], k))
            weighted_multi_prob.append(aln_weighted_probs)

        # To avoid division by 0
        for i in range(len(weighted_multi_prob)):
            weighted_multi_prob[i][weighted_multi_prob[i] == 0] = np.finfo(
                float).eps

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

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        alns_aa_counts: (A x N x C), matrix of counts
        gamma: (A x N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = self._m_step_alpha(gamma)

        # Compute beta
        # beta = self._m_step_beta(X, gamma)
        beta = None

        return alpha, beta

    def _m_step_alpha(self, gamma):
        total = np.sum([probs.sum() for probs in gamma])
        profile_prob = np.sum([probs.sum(axis=-2).sum(axis=0) for probs in gamma],
                              axis=0)
        alpha_profiles = profile_prob / total

        return alpha

    def _m_step_beta(self, X, gamma):
        # weighted_counts = gamma.T.dot(alns_aa_counts)
        K = gamma[0].shape[1]
        C = X[0].shape[-1]
        weighted_counts = np.zeros((K, C))
        for i in range(len(gamma)):
            for j, site_prob in enumerate(gamma[i]):
                for cluster in range(K):
                    weighted_counts[cluster] += site_prob[cluster] * X[i][j]
        # normalize
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb(self, alns_aa_counts, alpha, beta, gamma):
        """
        Computes the variational lower bound
        alns_aa_counts: (A x N x C), data points
        alpha: (K) or (AxK) with individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (A x N x K), posterior probabilities for objects clusters assignments
        Returns value of variational lower bound
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = [w[:, k] for w in gamma]
            log_probs = self._multinomial_prob_aln_sites(alns_aa_counts, beta, alpha,
                                                         beta[k],
                                                         log=True)
            loss += np.sum(
                [np.sum(weights[i] * (np.log(alpha[k]) + log_probs[i]))
                 for i in range(len(weights))])
            loss -= np.sum([np.sum(w * np.log(w)) for w in weights])
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
                                                   (self.n_clusters,
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

    def _train_once(self, X, profiles):
        '''
        Runs one full cycle of the EM algorithm
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        loss = float('inf')
        beta, alpha_profiles, alpha_clusters = self._init_params(X)
        gamma = None
        losses = np.zeros(self._max_iter)

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha_profiles, alpha_clusters, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            losses[it] = loss

            print('Loss: %f' % loss)
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                print(f'Finished after {it + 1} training cycles')
                break
        return alpha, beta, gamma, losses

    def fit(self, alns_aa_counts, profiles=None):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        :param alns_aa_counts: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_alpha_cluster = None
        best_alpha_profiles = None
        best_beta = profiles
        best_gamma = None
        losses = None

        for it in range(self._restarts):
            print('iteration %i' % it)
            alpha_cluster, alpha_profiles, beta, gamma, loss = \
                self._train_once(alns_aa_counts, profiles)
            if loss[-1] > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss[-1]))
                best_loss = loss[-1]
                best_alpha_cluster = alpha_cluster
                best_alpha_profiles = alpha_profiles
                best_beta = beta
                best_gamma = gamma
                losses = loss

        return (losses, best_alpha_cluster, best_alpha_profiles, best_beta,
                best_gamma)


# init simulation parameters
n_trials = 10
n_alns = 3
n_seqs = 100
n_clusters = 3
n_profiles = 4
n_sites = 30
n_aas = 6

print(f'{n_trials} trials\n{n_alns} MSAs\n{n_seqs} sequences per MSA\n'
      f'{n_sites} sites per MSA\n{n_aas} amino acids\n'
      f'{n_clusters} clusters \n{n_profiles} profiles\n')

# in evaluation values
accs = np.zeros((n_trials))
# mses_profiles = np.zeros((n_trials))
# mses_weights = np.zeros((n_trials))

for i in range(n_trials):

    print('_________________________________________________')
    print(f'TRIAL {i + 1}')

    # geneate profiles and weights
    profiles, profiles_weights, cluster_weights = init_params(n_aas,
                                                              n_profiles,
                                                              n_clusters,
                                                              n_sites)
    # generate alignments
    simulation = generate_msa_aa_counts(n_alns, n_seqs, n_sites, profiles,
                                        profiles_weights, cluster_weights)

    train_aa_counts, profile_choices, cluster_choices = simulation

    model = MultinomialExpectationMaximizer(n_clusters, n_profiles,
                                            restarts=10, rtol=1e-5)

    # test e-step
    gamma = model._e_step(train_aa_counts, profiles_weights, cluster_weights,
                          profiles)
    em_profile_choice = [gamma[i][:, cluster_choices[i], :].argmax(axis=-1) for
                         i in range(n_alns)]
    accs_per_aln = [(profile_choices[i] == em_profile_choice[i]).sum() / n_sites
                    for i in range(n_alns)]
    print(f'\nMSAs mean acc. : {np.mean(accs_per_aln)}')
    print(f'min acc. : {np.min(accs_per_aln)}')
    print(f'max acc. : {np.max(accs_per_aln)}')

    accs[i] = np.mean(accs_per_aln)

print(f'\n--------- FOR {n_trials} TRIALS ---------')
print(f'mean acc. : {np.mean(accs)}')
print(f'min trial acc. : {np.min(accs)}')
print(f'max trial acc. : {np.max(accs)}')

"""
    best_train_loss, best_alpha, best_beta, best_gamma = model.fit(
        train_aa_counts, profiles)

    acc, mse_profiles, mse_weights = eval_preds(profiles,
                                                profiles_weights,
                                                gamma,
                                                best_beta,
                                                best_alpha,
                                                best_gamma,
                                                n_sites)
    accs[i] = acc
    mses_profiles[i] = mse_profiles
    mses_weights[i] = mse_weights
"""
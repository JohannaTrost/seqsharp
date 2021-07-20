import random
import os
import time
import multiprocessing

from scipy.stats import multinomial, dirichlet
import numpy as np
import matplotlib.pyplot as plt


class MultinomialExpectationMaximizer:
    def __init__(self, K=263, rtol=1e-3, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def compute_log_likelihood(self, X_test, alpha, beta):
        mn_probs = np.zeros(X_test.shape[0])
        for k in range(beta.shape[0]):
            mn_probs_k = (self._get_mixture_weight(alpha, k) *
                          self._multinomial_prob(X_test, beta[k]))
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def compute_bic(self, X_test, alpha, beta, log_likelihood=None):
        if log_likelihood is None:
            log_likelihood = self.compute_log_likelihood(X_test, alpha, beta)
        N = X_test.shape[0]
        return np.log(N) * (alpha.size + beta.size) - 2 * log_likelihood

    def compute_icl_bic(self, bic, gamma):
        classification_entropy = -(np.log(gamma.max(axis=1))).sum()
        return bic - classification_entropy

    def _multinomial_prob(self, counts, beta, log=False):
        """
        Evaluates the multinomial probability for a given vector of counts
        counts: (N x C), matrix of counts
        beta: (C), vector of multinomial parameters for a specific cluster k
        Returns:
        p: (N), scalar values for the probabilities of observing each count vector given the beta parameters
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, beta)
        if log:
            return m.logpmf(counts)
        return m.pmf(counts)

    def _e_step(self, X, alpha, beta):
        """
        Performs E-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        alpha: (K) or (NxK) in the case of individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        Returns:
        gamma: (N x K), posterior probabilities for objects clusters assignments
        """
        # Compute gamma
        N = X.shape[0]
        K = beta.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = self._get_mixture_weight(alpha, k) * self._multinomial_prob(X, beta[k])

        # To avoid division by 0
        weighted_multi_prob[weighted_multi_prob == 0] = np.finfo(float).eps

        denum = weighted_multi_prob.sum(axis=1)
        gamma = weighted_multi_prob / denum.reshape(-1, 1)

        return gamma

    def _get_mixture_weight(self, alpha, k):
        return alpha[k]

    def _m_step(self, X, gamma):
        """
        Performs M-step on MNMM model
        Each input is numpy array:
        X: (N x C), matrix of counts
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns:
        alpha: (K), mixture component weights
        beta: (K x C), mixture categories weights
        """
        # Compute alpha
        alpha = self._m_step_alpha(gamma)

        # Compute beta
        beta = self._m_step_beta(X, gamma)

        return alpha, beta

    def _m_step_alpha(self, gamma):
        alpha = gamma.sum(axis=0) / gamma.sum()
        return alpha

    def _m_step_beta(self, X, gamma):
        weighted_counts = gamma.T.dot(X)
        beta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)
        return beta

    def _compute_vlb(self, X, alpha, beta, gamma):
        """
        Computes the variational lower bound
        X: (N x C), data points
        alpha: (K) or (NxK) with individual weights, mixture component weights
        beta: (K x C), multinomial categories weights
        gamma: (N x K), posterior probabilities for objects clusters assignments
        Returns value of variational lower bound
        """
        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(
                weights * (np.log(self._get_mixture_weight(alpha, k)) +
                           self._multinomial_prob(X, beta[k], log=True)))
            loss -= np.sum(weights * np.log(weights))
        return loss

    def _init_params(self, X):
        C = X.shape[1]
        weights = np.random.randint(1, 6, self._K)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * C] * C, self._K)
        return alpha, beta

    def _train_once(self, X):
        '''
        Runs one full cycle of the EM algorithm
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        loss = float('inf')
        alpha, beta = self._init_params(X)
        gamma = None
        losses = np.zeros(self._max_iter)

        for it in range(self._max_iter):
            prev_loss = loss
            gamma = self._e_step(X, alpha, beta)
            alpha, beta = self._m_step(X, gamma)
            loss = self._compute_vlb(X, alpha, beta, gamma)
            losses[it] = loss

            print('Loss: %f' % loss)
            if it > 0 and np.abs((prev_loss - loss) / prev_loss) < self._rtol:
                print(f'Finished after {it + 1} training cycles')
                break
        return alpha, beta, gamma, losses

    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        :param X: (N, C), matrix of counts
        :return: The best parameters found along with the associated loss
        '''
        best_loss = -float('inf')
        best_alpha = None
        best_beta = None
        best_gamma = None
        losses = None

        p_dir = '/beegfs/data/jtrost/mlaa/profile_estimation'

        result_path = f'{p_dir}/profiles/{time.time()}-profiles.csv'

        for it in range(self._restarts):
            print('iteration %i' % it)
            alpha, beta, gamma, loss = self._train_once(X)
            if loss[-1] > best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss[-1]))
                best_loss = loss[-1]
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma
                losses = loss

                save_profiles(best_beta, best_alpha, losses, result_path)

        return losses, best_alpha, best_beta, best_gamma


def save_profiles(best_beta, best_alpha, losses, result_path):
    result = np.concatenate((best_beta, np.asarray([best_alpha]).T), axis=1)
    losses_padded = np.concatenate(
        (losses, np.zeros(result.shape[0] - losses.shape[0])))
    result = np.concatenate((result, np.asarray([losses_padded]).T), axis=1)

    np.savetxt(result_path,
               result,
               delimiter=',',
               header='A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,weights,loss',
               comments='')


def main():

    nb_cores = 32  # psutil.cpu_count(logical=False)

    p_dir = '/beegfs/data/jtrost/mlaa/profile_estimation'

    counts_files = os.listdir(f'{p_dir}/counts')

    start = time.time()

    process_pool = multiprocessing.Pool(nb_cores)
    result = process_pool.starmap(np.genfromtxt,
                                  [(f'{p_dir}/counts/{file}', float, '#', ',')
                                   for
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

    model = MultinomialExpectationMaximizer(restarts=1, rtol=1e-10)
    model.fit(aa_counts_sites)
    # alpha_test = best_alpha

    # log_likelihood = model.compute_log_likelihood(test, alpha_test, best_beta)
    # bic = model.compute_bic(test, best_alpha, best_beta, log_likelihood)
    # icl_bic = model.compute_icl_bic(bic, best_gamma)

    # print('ideal number of distributions=%i' % (np.argmin(bics) + 1))
    # print('betas for this selection: \n', best_betas[np.argmin(bics)])

if __name__ == '__main__':
    main()

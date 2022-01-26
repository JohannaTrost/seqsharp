import random
import os
import time
import multiprocessing

from scipy.stats import multinomial, dirichlet
import numpy as np
import matplotlib.pyplot as plt


def count_aas(data, level, save=True):
    # pid = os.getpid()
    # print(f'starting process {pid}')

    aas = 'ARNDCQEGHILKMFPSTWYV'
    aa_counts_genes = []
    nb_sites = 0

    for aln in data:
        nb_seqs = len(aln)
        seq_len = len(aln[0])

        # transform alignment into array to make sites accessible
        aln_arr = np.empty((nb_seqs, seq_len), dtype='<U1')
        for j in range(nb_seqs):
            aln_arr[j, :] = np.asarray([aa for aa in aln[j]])

        if level == 'sites':
            aa_counts = np.zeros((len(aas), seq_len))
            # count aa at each site
            for site_ind in range(seq_len):
                site = ''.join([aa for aa in aln_arr[:, site_ind]])
                for i, aa in enumerate(aas):
                    aa_counts[i, site_ind] = site.count(aa)
            nb_sites += seq_len
        elif level == 'genes':
            aa_counts = np.zeros((len(aas), nb_seqs))
            # count aa for each gene
            for gene_ind in range(nb_seqs):
                gene = ''.join([aa for aa in aln_arr[gene_ind, :]])
                for i, aa in enumerate(aas):
                    aa_counts[i, gene_ind] = gene.count(aa)
        if len(aa_counts_genes) == 0:
            aa_counts_genes = aa_counts
        else:
            aa_counts_genes = np.concatenate((aa_counts_genes, aa_counts),
                                             axis=1)

    if save:
        np.savetxt(f'../counts/{os.getpid()}-counts-{nb_sites}sites.csv',
                   np.asarray(aa_counts),
                   delimiter=',',
                   fmt='%1.1f')
        print(f'Successfully saved {nb_sites} sites.\n')
    else:
        return aa_counts_genes


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
        alns_aa_counts: (N x C), matrix of counts
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
        alns_aa_counts: (N x C), matrix of counts
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
        alns_aa_counts: (N x C), data points
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
        weights = np.random.randint(1, 20, self._K)
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
    aa_freqs = np.genfromtxt('../../data/aafreqs_real_sites.csv', delimiter=',')
    aa_freqs = aa_freqs[0:100000, :20]
    np.random.shuffle(aa_freqs)

    train_size = int(len(aa_freqs) * 0.9)
    train = aa_freqs[:train_size]
    test = aa_freqs[train_size:]

    ks = list(range(1, 11))
    bics = []
    best_betas = []
    for k in ks:
        model = MultinomialExpectationMaximizer(k, restarts=10, rtol=1e-4)
        best_train_loss, best_alpha, best_beta, best_gamma = model.fit(train)
        alpha_test = best_alpha

        log_likelihood = model.compute_log_likelihood(test, alpha_test,
                                                      best_beta)
        # basian information content
        bic = model.compute_bic(test, best_alpha, best_beta, log_likelihood)
        icl_bic = model.compute_icl_bic(bic, best_gamma)

        bics.append(bic)
        best_betas.append(best_beta)

    print('ideal number of distributions=%i' % (np.argmin(bics) + 1))
    print('betas for this selection: \n', best_betas[np.argmin(bics)])

    plt.figure()
    plt.plot(ks, bics)
    plt.show()

if __name__ == '__main__':
    main()

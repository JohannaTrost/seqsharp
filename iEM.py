import os.path
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pylab as plt

from preprocessing import alns_from_fastas
from stats import count_aas
from hEM import init_estimates, em, full_log_lk, theoretical_cl_freqs, \
    lk_per_site

# load alignments
from utils import load_weights

data_path = 'data/real_fasta_sample_30'
raw_data = alns_from_fastas(data_path)[0]
counts = [count_aas([aln], 'sites').T for aln in raw_data]

# load profiles
profiles = np.genfromtxt(
    f'results/profiles_weights/profiles/64-edcluster-profiles.tsv',
    delimiter='\t').T

# set params
n_runs, n_iter = 5, 70
n_clusters, n_profiles, n_aas, n_alns, test = 30, 64, 20, 30, False
save = '../results/profiles_weights/iEM_30cl_30aln_5runs'
if not os.path.exists(save):
    os.mkdir(save)

# set init params
np.random.seed(72)
# init_params = init_estimates(n_runs, n_clusters, n_profiles, n_aas, n_alns,
#                             test, true_params=None)

init_pro_w = np.zeros((n_runs, n_clusters, n_profiles))
init_cl_w = np.zeros((n_runs, n_clusters))
for run in range(n_runs):
    for cl_aln in range(n_clusters):
        init_pro_w[run, cl_aln] = \
            init_estimates(1, 1, n_profiles, n_aas, n_alns, test,
                           true_params=None)[0][1]
        weights = np.random.randint(1, n_alns, n_clusters)
        init_cl_w[run] = weights / weights.sum()

# ---------- independent EMs for each msa
pro_w_runs = np.zeros((n_runs, n_clusters, n_profiles))
lks_iem_other_init, lks_hem_other_init = np.zeros(n_runs), np.zeros(n_runs)

for run in range(n_runs):
    for cl_aln in range(n_clusters):
        inits = [profiles, np.array([init_pro_w[run, cl_aln]]), np.ones(1)]
        # inits = [init_params[run][0], np.array([init_params[run][1][cl_aln]]),
        #         np.ones(1)]
        res = em(inits, profiles, [counts[cl_aln]], n_iter)

        pro_w_runs[run, cl_aln] = res[0][1]

        # save
        np.savetxt(f'{save}/cl{cl_aln + 1}_pro_weights_{run + 1}.csv',
                   pro_w_runs[run, cl_aln], delimiter=',')

    np.savetxt(f'{save}/cl_weights_{run + 1}.csv',
               np.ones(n_clusters) / n_clusters, delimiter=',')

    # ---------- lk & vlb
    lks_iem_other_init[run] = full_log_lk(counts, profiles, res[0][1])
    lks_hem_other_init[run] = full_log_lk(counts, profiles, res[0][1],
                                          np.ones(n_clusters) / n_clusters)

# ---------- hEM with multiple clusters

lks_hem_multicl = np.zeros(n_runs)
n_proc = n_runs
runs = np.arange(1, n_runs + 1)
save_path_hem = '../results/profiles_weights/hEM_30cl_30aln_5runs'
if not os.path.exists(save_path_hem):
    os.mkdir(save_path_hem)

process_pool = multiprocessing.Pool(n_proc)
result = process_pool.starmap(em,
                              zip([[profiles, init_pro_w[run], init_cl_w[run]]
                                   for run in range(n_runs)],
                                  [profiles] * n_proc,
                                  [counts] * n_proc,
                                  [n_iter] * n_proc, runs,
                                  [test] * n_proc,
                                  [save_path_hem] * n_proc))
for proc, run in zip(range(n_proc), runs):
    lks_hem_multicl[run] = result[proc][3][-1]

# plot lks
plt.plot(lks_hem_other_init[np.argsort(lks_hem_multicl)], color='c',
         label='hEM iEM parameters', linestyle='',
         marker='.')
plt.plot(lks_hem_multicl[np.argsort(lks_hem_multicl)], color='coral',
         label='hEM', linestyle='',
         marker='.')
plt.ylabel('loglk')
plt.xlabel('EM runs with different initial parameters')
plt.legend()
plt.tight_layout()
plt.savefig('../results/hem_hem_iem_params_plt_other_init_5runs.png', dpi=100)
plt.close('all')

# run hem with iem params as inits and uniform cluster weights
iem_cl_w_runs, iem_pro_w_runs = load_weights(
    '../results/profiles_weights/iEM_30cl_30aln_30runs', 30, 30, 64)

lks_hem_iem_init = np.zeros(n_runs)
n_proc = n_runs
runs = np.arange(1, n_runs + 1)
save_path_hem = '../results/profiles_weights/hEM_iEMinits_unif_init_cl_w_30cl_30aln_30runs'
if not os.path.exists(save_path_hem):
    os.mkdir(save_path_hem)

process_pool = multiprocessing.Pool(n_proc)
result = process_pool.starmap(em,
                              zip([[profiles, iem_pro_w_runs[run],
                                    iem_cl_w_runs[run]]
                                   for run in range(n_runs)],
                                  [profiles] * n_proc,
                                  [counts] * n_proc,
                                  [n_iter] * n_proc, runs,
                                  [test] * n_proc,
                                  [save_path_hem] * n_proc))
for proc, run in zip(range(n_proc), runs):
    lks_hem_iem_init[run] = result[proc][3][-1]

# hEM with iEM inits -> nans after 10th iteration
iem_cl_w_runs, iem_pro_w_runs = load_weights(
    '../results/profiles_weights/iEM_30cl_30aln_30runs', 30, 30, 64)
n_iter = 10

np.random.seed(72)
rand_params = init_estimates(n_runs, n_clusters, n_profiles, n_aas, n_alns,
                             False)
res = em([profiles, iem_pro_w_runs[0], rand_params[0][2]],
         profiles, counts, n_iter)

pi, p_aln_cl = e_step(counts, res[0][0], res[0][1], res[0][2])
estim_cluster_weights, estim_profile_weights, estim_profiles = m_step(
    pi, p_aln_cl, counts)

# load weights hem

hem_weights_path = "results/profiles_weights/hEM_10cl10aln_15runs_originit"
run = 'best3'

hem_pro_w = np.asarray(
    [np.genfromtxt(f'{hem_weights_path}/cl{cl}_pro_weights_{run}.csv',
                   delimiter=',') for cl in range(1, 11)])
hem_cl_w = np.genfromtxt(f'{hem_weights_path}/cl_weights_{run}.csv',
                         delimiter=',')

th_cl_w = theoretical_cl_freqs(profiles, hem_pro_w)
p_aln_given_cl = np.asarray([[lk_per_site(aln, profiles, cl_pro_w)
                              for cl_pro_w in hem_pro_w] for aln in counts])

""" init parameter set to circle on PCA etc.

# estimate init profile weights from selected MSAs
n_runs_select = 0  # TODO
if n_runs_select > 0:
    init_pro_weights_table = np.zeros((n_runs_select * n_clusters,
                                       n_profiles + 2))
    msa_freqs_table = np.zeros((n_runs_select * n_clusters, n_aas + 2))

    # determine msa and its aa frequencies
    msa_inds4pro_w_init = np.zeros((n_runs_select, n_clusters))
    msa_freqs = count_aas(alns, level='msa')
    msa_freqs /= np.repeat(msa_freqs.sum(axis=1)[:, np.newaxis], 20, axis=1)
    msa_freqs = np.round(msa_freqs, 8)

    pca = PCA(n_components=2)
    pca_msa_freqs = pca.fit_transform(msa_freqs)

    # center and 70% circle on PC 1 and 2
    pca_msa_freqs_c = pca_msa_freqs - pca_msa_freqs.mean(axis=0)
    msa_inds4pro_w_init[:2, :] = select_init_msa(pca_msa_freqs_c, 2, n_clusters)
    # set cluster weights accordingly
    if n_clusters > 1:
        for run in range(2):
            cl_w = np.random.normal(loc=0.3 / (n_clusters - 1),
                                    scale=0.001, size=n_clusters - 1)
            init_params[run][2] = np.concatenate(([1 - cl_w.sum()], cl_w))

    # kmeans on PC1 and 2
    seeds = [3874, 98037]
    for run in range(int(2)):
        kmeans_pcs = KMeans(n_clusters=n_clusters, random_state=seeds[run]).fit(
            pca_msa_freqs_c)
        cl_coord = kmeans_pcs.cluster_centers_
        for i, target in enumerate(cl_coord):
            dists = np.sum((pca_msa_freqs_c - target) ** 2, axis=1) ** 0.5
            msa_inds4pro_w_init[run+2, i] = int(np.argmin(dists))
        # set em init cluster weights
        init_params[run+2][2] = np.asarray([sum(kmeans_pcs.labels_ == i) / n_alns
                                            for i in range(n_clusters)])

    # kmeans on MSA AA frequencies
    msa_freqs_c = msa_freqs - msa_freqs.mean(axis=0)
    for run in range(int(2)):
        kmeans_freqs = KMeans(n_clusters=n_clusters, random_state=seeds[run]).fit(
            msa_freqs_c)
        cl_freqs = kmeans_freqs.cluster_centers_
        for i, target in enumerate(cl_freqs):
            # cluster freqs could be used directly as starting MSAs
            dists = np.sum((msa_freqs_c - target) ** 2, axis=1) ** 0.5
            msa_inds4pro_w_init[run+4, i] = int(np.argmin(dists))
        # set em init cluster weights
        init_params[run + 4][2] = np.asarray(
            [sum(kmeans_freqs.labels_ == i) / n_alns
             for i in range(n_clusters)])

    for run in range(n_runs_select):
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

            init_pro_weights_table[run_table_inds, :n_profiles] = init_pro_w
            init_pro_weights_table[n_clusters * run + cl, -1] = cl + 1
            msa_freqs_table[n_clusters * run + cl, -1] = cl + 1
            init_pro_weights_table[run_table_inds, -2] = np.repeat(run + 1,
                                                                   n_clusters)
            msa_freqs_table[run_table_inds, -2] = np.repeat(run + 1, n_clusters)
    """

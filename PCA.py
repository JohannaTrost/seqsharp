import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pylab as plt

from utils import read_config_file
from preprocessing import raw_alns_prepro
from stats import get_aa_freqs


def loadings(coeff, labels=None):
    n = coeff.shape[0]
    plt.figure()
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], alpha=0.5)
        plt.text(coeff[i, 0] + 0.1, coeff[i, 1] + 0.1, labels[i],
                 ha='center', va='center')
    plt.xlim(np.min(coeff[:, 0]), np.max(coeff[:, 0]))
    plt.ylim(np.min(coeff[:, 1]), np.max(coeff[:, 1]))
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


real_fasta_path = '/home/jtrost/data/fasta_no_gaps'
sim_fasta_path = '/home/jtrost/data/ocaml_fasta_263hog'
noprof_sim_fasta_path = '/home/jtrost/data/ocaml_fasta_263hog_aa_w1p'

config = read_config_file(
    '/mnt/Clusterdata/mlaa/configs/config-kernel1-globalpool'
    '-lin.json')

alns, fastas, config['data'] = raw_alns_prepro([real_fasta_path,
                                                sim_fasta_path],
                                               config['data'],
                                               shuffle=False)

config = read_config_file(
    '/mnt/Clusterdata/mlaa/configs/config-kernel1-globalpool'
    '-lin.json')
noprof_alns, fastas, config['data'] = raw_alns_prepro([real_fasta_path,
                                                       noprof_sim_fasta_path],
                                                      config['data'],
                                                      shuffle=False)

real_aas = get_aa_freqs(alns[0], gaps=False, dict=False)
sim_aas = get_aa_freqs(alns[1], gaps=False, dict=False)
noprof_sim_aas = get_aa_freqs(noprof_alns[1], gaps=False, dict=False)

x = np.concatenate((real_aas, sim_aas), axis=0)
x_noprof = np.concatenate((real_aas, noprof_sim_aas), axis=0)

# principal components real vs. sim with 263 profiles
pca_aafreqs = PCA(n_components=2)
principalComponents_aafreqs = pca_aafreqs.fit_transform(x)

var_pc1 = np.round(pca_aafreqs.explained_variance_ratio_[0] * 100, 2)
var_pc2 = np.round(pca_aafreqs.explained_variance_ratio_[1] * 100, 2)

# principal components real vs. sim 1 profile per simulation
np_pca_aafreqs = PCA(n_components=2)
np_principalComponents_aafreqs = np_pca_aafreqs.fit_transform(x_noprof)

np_var_pc1 = np.round(np_pca_aafreqs.explained_variance_ratio_[0] * 100, 2)
np_var_pc2 = np.round(np_pca_aafreqs.explained_variance_ratio_[1] * 100, 2)

# Call the function. Use only the 2 PCs.
aas = list('ARNDCQEGHILKMFPSTWYVX') + ['other']
coeff = np.transpose(np_pca_aafreqs.components_[0:2, :])
loadings(coeff, aas)


# visualization
fig, axs = plt.subplots(2, 2, sharex=True, figsize=(24., 16.))

# PCA real vs. simulated 263 profiles
axs[0, 0].set_xlabel(f'Principal Component - 1 ({var_pc1}%)')
axs[0, 0].set_ylabel(f'Principal Component - 2 ({var_pc2}%)')
axs[0, 0].set_title("PCA of avg aa frequencies: real vs. simulated (263 "
                    "profiles)")

axs[0, 0].scatter(principalComponents_aafreqs[0:len(real_aas), 0],
                  principalComponents_aafreqs[0:len(real_aas), 1],
                  c='b', alpha=0.4, s=5)
axs[0, 0].scatter(principalComponents_aafreqs[len(real_aas):, 0],
                  principalComponents_aafreqs[len(real_aas):, 1],
                  c='g', alpha=0.4, s=5)

axs[0, 0].legend(['real', 'sim'], prop={'size': 15})

# real vs. simulated all hogenom frequency vectors
axs[0, 1].set_xlabel(f'Principal Component - 1 ({np_var_pc1}%)')
axs[0, 1].set_ylabel(f'Principal Component - 2 ({np_var_pc2}%)')
axs[0, 1].set_title("PCA of avg aa frequencies: real vs. simulated (1 hogenom "
                    "profiles/aln)")

axs[0, 1].scatter(np_principalComponents_aafreqs[0:len(real_aas), 0],
                  np_principalComponents_aafreqs[0:len(real_aas), 1],
                  c='b', alpha=0.4, s=5)
axs[0, 1].scatter(np_principalComponents_aafreqs[len(real_aas):, 0],
                  np_principalComponents_aafreqs[len(real_aas):, 1],
                  c='g', alpha=0.4, s=5)

axs[0, 1].legend(['real', 'sim'], prop={'size': 15})
# visualization
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(24., 12.))

# PCA real vs. simulated 263 profiles
axs[0, 0].set_xlabel(f'Principal Component - 1 ({var_pc1}%)')
axs[0, 0].set_ylabel(f'Principal Component - 2 ({var_pc2}%)')
axs[0, 0].set_title("PCA of avg aa frequencies: real vs. simulated (263 "
                    "profiles)")

axs[0, 0].scatter(principalComponents_aafreqs[0:len(real_aas), 0],
                  principalComponents_aafreqs[0:len(real_aas), 1],
                  c='b', alpha=0.4, s=5)
axs[0, 0].scatter(principalComponents_aafreqs[len(real_aas):, 0],
                  principalComponents_aafreqs[len(real_aas):, 1],
                  c='g', alpha=0.4, s=5)

axs[0, 0].legend(['real', 'sim'], prop={'size': 15})

# real vs. simulated all hogenom frequency vectors
axs[0, 1].set_xlabel(f'Principal Component - 1 ({np_var_pc1}%)')
axs[0, 1].set_ylabel(f'Principal Component - 2 ({np_var_pc2}%)')
axs[0, 1].set_title("PCA of avg aa frequencies: real vs. simulated (1 hogenom "
                    "profiles/aln)")

axs[0, 1].scatter(np_principalComponents_aafreqs[0:len(real_aas), 0],
                  np_principalComponents_aafreqs[0:len(real_aas), 1],
                  c='b', alpha=0.4, s=5)
axs[0, 1].scatter(np_principalComponents_aafreqs[len(real_aas):, 0],
                  np_principalComponents_aafreqs[len(real_aas):, 1],
                  c='g', alpha=0.4, s=5)

axs[0, 1].legend(['real', 'sim'], prop={'size': 15})

# only simulations with 263 profiles
axs[1, 1].set_xlabel(f'Principal Component - 1 ({np_var_pc1}%)')
axs[1, 1].set_ylabel(f'Principal Component - 2 ({np_var_pc2}%)')
axs[1, 1].set_title("PCA of avg aa frequencies: simulations (1 hogenom "
                    "profiles/aln)")

axs[1, 1].scatter(np_principalComponents_aafreqs[0:len(real_aas), 0],
                  np_principalComponents_aafreqs[0:len(real_aas), 1],
                  c='b', alpha=0.4, s=5)
axs[1, 1].scatter(np_principalComponents_aafreqs[len(real_aas):, 0],
                  np_principalComponents_aafreqs[len(real_aas):, 1],
                  c='g', alpha=0.4, s=5)

axs[0, 1].legend(['real', 'sim'], prop={'size': 15})

plt.tight_layout()
plt.savefig('../figs/PCA_aafreqs.png')
plt.tight_layout()
plt.savefig('../figs/PCA_aafreqs.png')

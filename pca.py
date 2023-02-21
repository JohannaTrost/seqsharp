# load data
from preprocessing import aa_freq_samples
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pylab as plt
from sklearn.decomposition import PCA
import numpy as np

sim_dir = '../../data/simulations'
fasta_paths = ['../../data/hogenom_fasta',
               f'{sim_dir}/alisim_poisson_gapless',
               f'{sim_dir}/alisim_lg_gapless',
               f'{sim_dir}/alisim_lg_c60_gapless',
               f'{sim_dir}/alisim_lg_s0256_gapless',
               f'{sim_dir}/alisim_lg_s0256_g4_gapless',
               f'{sim_dir}/alisim_lg_s0256_gc_gapless']
# alns, fastas, _  = raw_alns_prepro(fasta_paths)
freqs = aa_freq_samples('../../data',
                        [d.split(f'../../data/')[1] for d in fasta_paths],
                        sample_prop=0.5, n_alns=None, levels=['sites'])

# get theoretical msa dots - weights@profiles w_files = [f for f in
# os.listdir("../results/profiles_weights/sim_edcl64_1cl_1aln") if 'best' in
# f and 'pro' in f] weights = [np.genfromtxt(
# f'../results/profiles_weights/sim_edcl64_1cl_1aln/{f}', delimiter=',') for
# f in w_files] weights = np.asarray(weights) profiles = np.genfromtxt(
# '../results/profiles_weights/profiles/64-edcluster-profiles.tsv',
# delimiter='\t') th_msa = np.matmul(profiles, weights.T).T

pca = make_pipeline(StandardScaler(), PCA(n_components=2))
pca_real_site_freqs = pca.fit_transform(freqs['hogenom_fasta'][:20].T)
pca_sim_freqs = [pca.transform(freqs[d.split('../../data/simulations/')[1]][:20].T)
                 for d in fasta_paths[1:]]
# pca_msa_th_freqs = pca.transform(th_msa)

# tsne = TSNE()
# tsne_msa_freqs = tsne.fit_transform(msa_freqs)
# tsne_msa_sim_freqs = tsne.fit_transform(msa_sim_freqs)
# tsne_msa_th_freqs = tsne.fit_transform(th_msa)

s = 20


def make_heatmap(xx, yy, s=25):
    plot_this = np.zeros((s, s))

    o_min = np.min([xx.min(), yy.min()])
    xx -= o_min
    yy -= o_min

    o_max = np.max([xx.max(), yy.max()])
    xx /= o_max
    xx *= s - 1

    yy /= o_max
    yy *= s - 1

    for x_pp, y_pp in zip(xx, yy):
        plot_this[int(y_pp), int(x_pp)] += 1

    plot_this[plot_this == 0] = np.NaN

    return plot_this


def plot_pca(data, ps=3, alpha=0.6, clim=60, save=None, titles=None):
    xlim = (np.min([d[:, 0].min() for d in data]),
            np.max([d[:, 0].max() for d in data]))
    ylim = (np.min([d[:, 1].min() for d in data]),
            np.max([d[:, 1].max() for d in data]))

    fig, axs = plt.subplots(ncols=len(data), nrows=1, figsize=(12., 4.5))
    for i in range(len(data)):
        # axs[i, 1].scatter(data[i][:, 0], data[i][:, 1], s=ps, color='coral',
        #                  alpha=alpha)
        # axs[i, 1].set_xlim(xlim)
        # axs[i, 1].set_ylim(ylim)

        x, y = data[i][:, 0].copy(), data[i][:, 1].copy()
        plot_this = make_heatmap(x, y, s)
        hm = axs[i].imshow(plot_this, clim=[0, clim],
                           cmap='turbo', interpolation='bilinear')
        axs[i].invert_yaxis()
        if titles is not None:
            axs[i].set_title(titles[i])
        axs[i].axis('off')

        # confidence_ellipse(data[0][:, 0], data[0][:, 1], axs[i, 1], n_std=2,
        #               edgecolor='coral')
    # fig.delaxes(axs[2][1])
    fig.colorbar(hm, ax=axs[i])
    plt.tight_layout()
    plt.savefig(save)
    plt.close('all')


plot_names = ['Empirical sites',
              'Poisson', 'LG', 'LG+C60', 'LG+S256', 'LG+S256+G4', 'LG+S256+GC']
plot_pca([pca_real_site_freqs] + pca_sim_freqs,
         save='../figs/emp_sim_pca_hm_site_freqs.pdf',
         titles=plot_names,
         clim=20000)

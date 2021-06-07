import os
import random
import sys
import time
import torch
import numpy as np
from preprocessing import data_prepro

def mmse_pairs(align1, align2, max_gb=16, max_data=5e6):
    if len(align1) < len(align2):
        tmp = align1
        align1 = align2
        align2 = tmp

    i_al12, i_al21 = np.meshgrid(range(len(align1)), range(len(align2)))  # combinations of indices
    i_al12, i_al21 = i_al12.flatten(), i_al21.flatten()

    if len(i_al12) > max_data:
        sel_data = np.random.permutation(len(i_al12))[:int(max_data)]
        i_al12 = i_al12[sel_data]
        i_al21 = i_al21[sel_data]

    split_into = (len(i_al12) *  # number of pairs
                  8 *  # byte per float64
                  align1.shape[1] *  # number of dimensions
                  align1.shape[2] *  # sequence length
                  2 *  # 2 alignments
                  10 ** -9 //  # convert to gigabyte
                  max_gb + 1)  # ceil to integer depending on available memory

    split_i_al12 = np.array_split(i_al12, split_into)
    split_i_al21 = np.array_split(i_al21, split_into)

    num_pairs = len(i_al12)
    mse = 0

    for al12, al21 in zip(split_i_al12, split_i_al21):
        diff = align1[al12, :, :] - align2[al21, :, :]

        path = np.einsum_path('...jk,...jk->...k', diff, diff,
                              optimize='optimal')[0]

        mse += np.mean(np.einsum('...jk,...jk->...k', diff, diff,
                                optimize=path))
        """
        mse += (np.mean(  # mean ...
            np.sum(  # ... squared error (MSE)
                (align1[al12, :, :] - align2[al21, :, :]) ** 2,
                axis=1),
            axis=1) / num_pairs).sum()  # average over all pairs
        """
    return mse  # average over all pair combinations


def distances(alns, fastas, aln_ind, pairs=False, max_gb=16, save_path=None):
    alns_chunks = [alns[i:i + 3] for i in range(0, len(alns), 3)]
    fastas_chunks = [fastas[i:i + 3] for i in range(0, len(fastas), 3)]
    print(fastas_chunks)
    for i, aln1 in enumerate(alns_chunks[aln_ind]):
        dists = []
        m_time = 0
        print(f'aln1 : {len(aln1)} pairs vs. all')
        for aln2 in alns:
            start = time.time()
            if pairs:
                dists.append(mmse_pairs(aln1, aln2, max_gb=max_gb))
            else:
                dists.append(mmse(aln1, aln2) / len(alns))
            m_time += round(time.time() - start, 3)
        print(f'mean time : {m_time / len(alns)}')

        if save_path is not None:
            with open(f'/beegfs/data/jtrost/mlaa/{aln_ind}-aln-dists-p.txt', 'a', ) as file:
                file.write(f'{str(dists)[1:-1]},{fastas_chunks[aln_ind][i]}\n')


def mmse(align1, align2):
    return np.mean(np.sum((align1 - align2)**2, axis=0))


def main(args):
    print(f'This is working\n')
    aln_ind = int(args[0])
    max_gb = int(args[1])
    print(f'aln_ind: {aln_ind}, max_gb: {max_gb}')

    # data specific parameters
    real_fasta_path = 'data/fasta_no_gaps'
    sim_fasta_path = 'data/sim_fasta_seqs'
    model_path = 'results/cnn-29-Apr-2021-20:38:57.134481-real-sim'

    nb_protein_families = 63  # number of multiple aligns
    min_seqs_per_align, max_seqs_per_align = 4, 300
    seq_len = 300

    nb_folds = 10

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ------------------------------------------------- data preparation ------------------------------------------------- #
    real_pairs_per_align, sim_pairs_per_align, fastas_real_p, fastas_sim_p = data_prepro(real_fasta_path, sim_fasta_path,
                                                                                     nb_protein_families,
                                                                                     min_seqs_per_align,
                                                                                     max_seqs_per_align, seq_len, pairs=True)

    distances(real_pairs_per_align, fastas_real_p, aln_ind, pairs=True, max_gb=max_gb)

if __name__ == '__main__':
    main(main(sys.argv[1:]))


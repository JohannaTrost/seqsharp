"""Functions and classes for data preprocessing

Provides functions to turn multiple alignments from fasta files
into 'neural-network-readable' representations that can be
transformed into tensor datasets using
the child classes of *torch.utils.data.Dataset*
"""

import time
import os
import random
import warnings

import psutil
import torch
import multiprocessing as mp
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset

from stats import generate_aln_stats_df, get_nb_sites, nb_seqs_per_alns, \
    count_aas
from utils import split_lst, flatten_lst

warnings.simplefilter("ignore", DeprecationWarning)

""" amino acids 
alanine       : A |  glutamine     : Q | leucine       : L | serine    : S |
arginine      : R |  glutamic acid : E | lysine        : K | threonine : T |
asparagine    : N |  glycine       : G | methionine    : M | tryptophan: W |
aspartic acid : D |  histidine     : H | phenylalanine : F | tyrosine  : Y |
cysteine      : C |  isoleucine    : I | proline       : P | valine    : V |
unknown : X 
gap/indel : - 
"""

ENCODER = str.maketrans('BZJUO' + 'ARNDCQEGHILKMFPSTWYV' + 'X-',
                        '\x00' * 5 + '\x01\x02\x03\x04\x05\x06\x07\x08\t\n'
                                     '\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14' +
                        '\x15\x16')

THREADS = psutil.cpu_count(logical=False)


def seq2index(seq):
    """Transforms amino acid sequence to integer sequence

    Translates sequence to byte sequence and finally to an
    integer sequence that could be converted to one-hot encodings
    of amino acids (including gaps and unknown amino acids)

    :param seq: protein sequence as string
    :return: array of integers from 0 to 22
    """

    seq = seq.translate(ENCODER)
    return np.fromstring(seq, dtype='uint8').astype('int64')


def index2code(index_seq):
    """Transforms array of indices into one-hot encoded arrays

    example:
        array([5, 13]) ->
        array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]])

    :param index_seq: array of integers from 0 to 22
    :return: 2D array of 0s and 1s
    """

    seq_enc = np.zeros((index_seq.size, 23))
    seq_enc[np.arange(index_seq.size), index_seq] = 1
    return seq_enc


def aln_from_fasta(filename):
    """Gets aligned sequences from given file

    :param filename: <path/to/> alignments (string)
    :param nb_seqs: number of sequences in alignment (integer)
    :return: list of strings
    """

    alned_seqs_raw = [str(seq_record.seq) for seq_record in
                      SeqIO.parse(filename, "fasta")]
    return alned_seqs_raw


def alns_from_fastas(fasta_dir, take_quantiles=False, nb_alns=None):
    """Extracts alignments from fasta files in given directory

    :param fasta_dir: <path/to/> fasta files
    :param nb_alns: number of alignments
    :return: list of aligned sequences (string list),
             list of alignment identifiers (strings)
    """

    fasta_files = np.asarray(os.listdir(fasta_dir))
    # shuffle
    fasta_files = fasta_files[np.random.permutation(
        np.arange(0, len(fasta_files)))]

    nb_seqs = np.zeros(len(fasta_files))
    seq_lens = np.zeros(len(fasta_files))

    for i, file in enumerate(fasta_files):
        aln = aln_from_fasta(fasta_dir + '/' + file)
        nb_seqs[i] = len(aln)
        if len(aln) > 0:
            seq_lens[i] = len(aln[0])

    inds_non0 = np.where(nb_seqs != 0)[0][np.where(nb_seqs != 0)[0] ==
                                          np.where(seq_lens != 0)[0]]
    neq_nb_len = np.where(nb_seqs != 0)[0] != np.where(seq_lens != 0)[0]
    inds_non0 = np.concatenate((inds_non0,
                                np.where(nb_seqs != 0)[0][neq_nb_len],
                                np.where(seq_lens != 0)[0][neq_nb_len]))
    # filter MSAs without sequences
    fasta_files = fasta_files[inds_non0]
    nb_seqs = nb_seqs[inds_non0]
    seq_lens = seq_lens[inds_non0]

    if take_quantiles:
        # such that min = 4 for 6000 MSAs and max < 400
        q_ns = (np.quantile(nb_seqs, 0.39), np.quantile(nb_seqs, 0.998))
        q_sl = (np.quantile(seq_lens, 0.05), np.quantile(seq_lens, 0.95))

        print(f'\nq_ns : {q_ns} (0.25, 0.998)\nq_sl : {q_sl} (0.05, 0.95)\n')

        ind_q_ns = np.where((q_ns[0] <= nb_seqs) & (nb_seqs <= q_ns[1]))[0]
        ind_q_sl = np.where((q_sl[0] <= seq_lens[ind_q_ns]) &
                            (seq_lens[ind_q_ns] <= q_sl[1]))[0]

        # filter : keep MSAs with seq. len. and n.seq. within above quantiles
        nb_seqs = nb_seqs[ind_q_ns][ind_q_sl]
        seq_lens = seq_lens[ind_q_ns][ind_q_sl]
        fasta_files = fasta_files[ind_q_ns][ind_q_sl]

    if nb_alns is not None:
        if (nb_alns - len(fasta_files)) > 0:  # we get less MSAs than wanted
            print('Only {} / {} fasta files taken into account.'.format(
                len(fasta_files), nb_alns))
        else:  # restrict number of MSAs as demanded
            fasta_files = fasta_files[:nb_alns]
            nb_seqs = nb_seqs[:nb_alns]
            seq_lens = seq_lens[:nb_alns]

    alns, fastas = [], []
    for file in fasta_files:
        seqs = aln_from_fasta(fasta_dir + '/' + file)
        alns.append(seqs)
        fastas.append(file.split('.')[0])

    if len(alns) > 0:
        out_stats = {'n_seqs_min': nb_seqs.min(),
                     'n_seqs_max': nb_seqs.max(),
                     'n_seqs_avg': nb_seqs.mean(),
                     'seq_lens_min': seq_lens.min(),
                     'seq_lens_max': seq_lens.max(),
                     'seq_lens_avg': seq_lens.mean()}
    else:
        out_stats = {}

    if take_quantiles:
        print('After filtering quantiles:')
    print(out_stats)

    return alns, fastas, out_stats


def remove_gaps(alns):
    """Removes columns with gaps from given raw alignments (not yet encoded)

    :param alns: list of list of amino acid sequences (2D list strings)
    """

    alns_no_gaps = []

    for aln in alns:
        aln = np.asarray([list(seq) for seq in aln])
        remove_columns = np.any(aln == '-', axis=0)
        aln_no_gaps = aln[:, np.invert(remove_columns)]

        if np.any(remove_columns):
            aln_no_gaps = [''.join([aa for aa in seq]) for seq in aln_no_gaps]
            alns_no_gaps.append(aln_no_gaps)
        else:
            alns_no_gaps.append(aln)

    return alns_no_gaps


def encode_aln(alned_seqs_raw, seq_len, padding=''):
    """Turns aligned sequences into (padded) one-hot encodings

    Trims/pads the alignment to a certain number of sites (*seq_len*)
    If sequences are > *seq_len* only the middle part of the sequence is taken
    If sequences are < *seq_len* they will be padded at both end either with
    zeros or random amino acids (according to *padding*)

    :param alned_seqs_raw: list of amino acid sequences (strings)
    :param seq_len: number of sites (integer)
    :param padding: 'data' or else padding will use zeros
    :return: one-hot encoded alignment (3D array)
    """

    # encode sequences and limit to certain seq_len (seq taken from the middle)
    if len(alned_seqs_raw[0]) > seq_len:
        diff = len(alned_seqs_raw[0]) - seq_len  # overhang
        start = int(np.floor((diff / 2)))
        end = int(-np.ceil((diff / 2)))
        seqs = np.asarray([index2code(seq2index(seq[start:end])).T
                           for seq in alned_seqs_raw])
    else:
        seqs = np.asarray([index2code(seq2index(seq)).T
                           for seq in alned_seqs_raw])

    if len(alned_seqs_raw[0]) < seq_len:  # padding
        pad_size = (seq_len - len(alned_seqs_raw[0]))
        pad_before = pad_size // 2
        pad_after = pad_size // 2 if pad_size % 2 == 0 else pad_size // 2 + 1

        seqs_shape = list(seqs.shape)
        seqs_shape[2] += pad_after + pad_before
        seqs_new = np.zeros(seqs_shape)

        if padding == 'data':  # pad with random amino acids
            seqs_new = np.asarray([index2code(np.random.randint(0,
                                                                seqs_shape[1],
                                                                seqs_shape[
                                                                    2])).T
                                   for _ in range(seqs_shape[0])])
        elif padding == 'gaps':
            seqs_new = np.asarray([index2code(seq2index(seq)).T
                                   for seq in ['-' * seq_len] * seqs_shape[1]])

        seqs_new[:, :, pad_before:-pad_after] = seqs + 0
        return seqs_new
    else:
        return seqs


def seq_pair_worker(alns):
    """Generates pair representation for given alignments"""

    return [make_seq_pairs(aln) for aln in alns]


def get_aln_representation(alned_seqs):
    """Returns proportions of amino acids at each site of the alignment"""

    return np.sum(alned_seqs, axis=0) / len(alned_seqs)


def make_seq_pairs(alned_seqs):
    """Returns pairwise alignment representation

    Representation consists of the sum of 2 pairs and the amino acid
    proportions at each site for the rest of the sequences

    :param alned_seqs: one-hot encoded aligned sequences (3D array)
    :return: pair representation (3D array (pair, aa, sites))
    """

    inds = np.asarray(np.triu_indices(len(alned_seqs), k=1))  # pair indices

    sums = alned_seqs[inds[0], :, :] + alned_seqs[inds[1], :, :]
    sum_all_seqs = np.sum(alned_seqs, axis=0)
    aa_prop_no_pair = (sum_all_seqs - sums) / (len(alned_seqs) - 2)
    return np.concatenate((sums / 2, aa_prop_no_pair), axis=1)


def make_pairs_from_alns_mp(alns):
    """Generates pair representations using multi threading"""

    threads = len(alns) if THREADS > len(alns) else THREADS + 0

    alns_chunks = list(split_lst(alns, threads))

    work = [alns_chunks[i] for i in range(threads)]

    with mp.Pool() as pool:
        res = pool.map(seq_pair_worker, work)

    pairs_per_aln = flatten_lst(res)

    return pairs_per_aln


class TensorDataset(Dataset):
    """Empirical and simulated alignment representations and their labels

        Attributes
        ----------
        data : FloatTensor
            alignment representations
            (either for sequence pairs or full alignments)
        labels : FloatTensor
            0 for empirical, 1 for simulated
    """

    def __init__(self, real_alns, sim_alns, pairs=False):
        nb_real = len(real_alns)
        nb_sim = len(sim_alns)

        data = real_alns + sim_alns

        if pairs:
            # remove first dimension from (aln, pair, chnls, sites) array
            data = np.concatenate(data)
            nb_real = np.sum([len(aln) for aln in real_alns])
            nb_sim = np.sum([len(aln) for aln in sim_alns])

        data = np.asarray(data, dtype='float32')

        self.labels = torch.FloatTensor([0] * nb_real + [1] * nb_sim)
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)


class DatasetAln(Dataset):
    """Generates dataset from pair representation for one alignment

    To be able to evaluate the performance per alignment when using the
    pair representation, because here the information about the membership
    of a pair to an alignment is usually lost
    """

    def __init__(self, aln, real):
        self.data = torch.from_numpy(np.asarray(aln)).float()
        labels = [0] * len(aln) if real else [1] * len(aln)
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)


def raw_alns_prepro(fasta_paths,
                    params,
                    take_quantiles=None,
                    shuffle=False):
    """Loads and preprocesses raw (not encoded) alignments

    :param take_quantiles: indicate possible reduction of number of sequences
                           per alignment
    :param fasta_paths: <path(s)/to> empirical/simulated fasta files
                        (list/tuple string)
    :param params: parameters for preprocessing (dictionary)
    :return: ids, preprocessed raw alignments, updated param. dict.
    """

    if take_quantiles is None:
        take_quantiles = [False] * len(fasta_paths)

    nb_alns, min_nb_seqs, max_nb_seqs, seq_len, padding = params.values()

    print("Loading alignments ...")

    # load sets of multiple alned sequences
    alns, fastas, lims = [], [], []
    for i, path in enumerate(fasta_paths):
        path = str(path)
        sim_cl_dirs = [dir for dir in os.listdir(path)
                       if os.path.isdir(os.path.join(path, dir))]
        if len(sim_cl_dirs) > 0:  # there are multiple clusters
            sim_alns, sim_fastas, sim_lims = [], [], {}
            for dir in sim_cl_dirs:
                sim_data = alns_from_fastas(f'{path}/{dir}', take_quantiles[i],
                                            nb_alns)
                sim_alns += sim_data[0]
                sim_fastas += sim_data[1]
                if len(sim_lims) == 0:
                    for k, v in sim_data[2].items():
                        sim_lims[k] = [v]
                else:
                    sim_lims_tmp = {}
                    for k in sim_lims.keys():
                        sim_lims_tmp[k] = sim_lims[k] + [sim_data[2][k]]
                    sim_lims = sim_lims_tmp.copy()
            if len(sim_alns) > nb_alns:
                inds = np.random.choice(np.arange(len(sim_alns)), size=nb_alns,
                                        replace=False)
                raw_data = [[sim_alns[ind] for ind in inds],
                            [sim_fastas[ind] for ind in inds],
                            sim_lims]
            else:
                raw_data = [sim_alns, sim_fastas, sim_lims]
        else:
            raw_data = alns_from_fastas(path, take_quantiles[i], nb_alns)
        nb_alns = len(raw_data[0])
        alns.append(raw_data[0])
        fastas.append(raw_data[1])
        lims.append(raw_data[2])

    if len(alns) == 2 and len(fastas) == 2:  # if there is simulated data
        print(f"avg. seqs. len. : {lims[1]['seq_lens_avg']} (sim.) vs. "
              f"{lims[0]['seq_lens_avg']} (emp.)")
        print(f"avg. n.seqs. : {lims[1]['n_seqs_avg']} (sim.) vs. "
              f"{lims[0]['n_seqs_avg']} (emp.)")

        assert len(alns[0]) == len(alns[1]), f' {len(alns[0])} == {len(alns[1])}'
        """
        # sort simulated data by sequence length
        ind_s = np.argsort(get_nb_sites(alns[1]))

        alns[1] = [alns[1][i] for i in ind_s]
        fastas[1] = [fastas[1][i] for i in ind_s]

        # keeping same amount of simulated alns from the "middle"(regarding
        # lengths)
        start = len(alns[0]) // 2
        alns[1] = alns[1][start:start + len(alns[0])]
        fastas[1] = fastas[1][start:start + len(alns[0])]

        # shuffling the sorted simulated alignments and their ids
        indices = np.arange(len(alns[1]))
        np.random.shuffle(indices)
        alns[1] = [alns[1][i] for i in indices]
        fastas[1] = [fastas[1][i] for i in indices]
        """

    # ensure same number of MSAs for all data sets
    for i in range(len(alns)):
        inds = np.random.choice(range(len(alns[i])), nb_alns, replace = False)
        alns[i] = [alns[i][ind] for ind in inds]

    params['nb_sites'] = int(min(seq_len, lims[0]['seq_lens_max']))

    nb_seqs = np.asarray(nb_seqs_per_alns(alns))
    params['max_seqs_per_align'] = np.max(nb_seqs, axis=1).astype(int)
    params['min_seqs_per_align'] = np.min(nb_seqs, axis=1).astype(int)
    params['nb_alignments'] = nb_alns

    if shuffle:  # shuffle sites/columns of alignments
        for i in range(len(alns)):
            for j in range(len(alns[i])):
                aln = np.asarray([list(seq) for seq in alns[i][j]])
                aln[:, :] = aln[:, np.random.permutation(range(aln.shape[1]))]

                alns[i][j] = [''.join([aa for aa in seq]) for seq in aln]

    return alns, fastas, params


def get_representations(alns,
                        fastas,
                        params,
                        pairs=False,
                        csv_path=None):
    """Encodes alignments and generates their representations

    :param fastas: a set of lists of alignment identifiers (2D string list )
    :param alns: preprocessed raw alignment sets (3D string list)
    :param params: parameters for preprocessing (dictionary)
    :param pairs: choose representation by pairs if true (boolean)
    :param csv_path: <path/to> store csv file with info about alignments
    :return: alignment representations
    """

    nb_alns, min_nb_seqs, max_nb_seqs, seq_len, padding = params.values()

    if pairs:

        print("Pairing sequences ...")

        start = time.time()

        alns_reprs_pairs = []
        for alns_set in alns:
            alns_reprs_pairs.append([make_seq_pairs(encode_aln(
                aln, seq_len, padding=padding)) for aln in alns_set])

        print(f'Finished pairing after {round(start - time.time(), 2)}s\n')

        if csv_path is not None:
            generate_aln_stats_df(fastas, alns, seq_len,
                                  None, is_sim=[0, 1] if len(alns) == 2 else [])

        return alns_reprs_pairs

    else:

        print("Generating alignment representations ...")

        # make pairs !additional dim for each multiple alingment needs flatteing
        # before passed to CNN!
        alns_reprs = []
        for alns_set in alns:
            alns_reprs.append([get_aln_representation(encode_aln(
                aln, seq_len, padding=padding)) for aln in alns_set])

        if csv_path is not None:
            generate_aln_stats_df(fastas, alns, seq_len,
                                  alns_reprs,
                                  is_sim=[0, 1] if len(alns) == 2 else [],
                                  csv_path=csv_path)

        return alns_reprs


def aa_freq_samples(in_dir, out_dir, data_dirs, sample_prop, n_alns, levels):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i in range(len(data_dirs)):
        data_dir = data_dirs[i]
        print(data_dir)

        if os.path.exists(f'{in_dir}/{data_dir}'):
            cl_dirs = ['/' + name for name in os.listdir(f'{in_dir}/{data_dir}')
                       if
                       os.path.isdir(
                           os.path.join(f'{in_dir}/{data_dir}', name))]
            alns_samples, cl_assign = [], []
            if len(cl_dirs) == 0:
                alns = alns_from_fastas(f'{in_dir}/{data_dir}',
                                        take_quantiles=True
                                        if data_dir == 'fasta_no_gaps'
                                        else False, nb_alns=n_alns)[0]
                # sampling
                sample_size = np.round(sample_prop * len(alns)).astype(int)
                sample_inds = np.random.randint(0, len(alns), sample_size)
                alns_samples = [alns[ind] for ind in sample_inds]
                cl_assign = [1] * sample_size
            else:
                for cl, cl_dir in enumerate(cl_dirs):
                    alns = alns_from_fastas(f'{in_dir}/{data_dir}{cl_dir}',
                                            take_quantiles=False,
                                            nb_alns=n_alns)[0]
                    # sampling
                    sample_size = np.round(sample_prop * len(alns)).astype(int)
                    sample_inds = np.random.randint(0, len(alns), sample_size)
                    alns_samples += [alns[ind] for ind in sample_inds]
                    cl_assign += [cl + 1] * sample_size

            seq_lens = np.asarray([len(aln[0]) for aln in alns_samples])
            n_seqs = np.asarray([len(aln) for aln in alns_samples])

            # frequency vectors on MSA level
            if 'msa' in levels:
                counts_alns = count_aas(alns_samples, level='msa')
                counts_alns /= np.repeat(counts_alns.sum(axis=1)[:, np.newaxis],
                                         20,
                                         axis=1)
                freqs_alns = np.round(counts_alns, 8)
                cl_assign = np.asarray(cl_assign).reshape(-1, 1)

                table = np.concatenate((freqs_alns, cl_assign), axis=1)
                np.savetxt(f"{out_dir}/{data_dir.split('/')[-1]}_alns.csv",
                           table,
                           delimiter=",",
                           header='A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,cl',
                           comments='')

            # frequency vectors on gene level
            if 'genes' in levels:
                counts_genes = count_aas(alns_samples, level='genes')
                div_seq_lens = np.repeat(np.asarray(seq_lens), n_seqs)
                freqs_genes = np.round(counts_genes / div_seq_lens, 8)

                aln_assign = np.repeat(np.arange(1, len(alns_samples) + 1),
                                       n_seqs).reshape(1, -1)
                cl_assign = np.repeat(cl_assign, n_seqs).reshape(1, -1)

                table = np.concatenate((freqs_genes, aln_assign, cl_assign),
                                       axis=0)
                np.savetxt(f"{out_dir}/{data_dir.split('/')[-1]}_genes.csv",
                           table.T,
                           delimiter=",",
                           header='A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,aln,cl',
                           comments='')

            # frequency vectors on site level
            if 'sites' in levels:
                counts_sites = count_aas(alns_samples, level='sites')
                div_n_seqs = np.repeat(np.asarray(n_seqs), seq_lens)
                freqs_sites = np.round(counts_sites / div_n_seqs, 8)

                aln_assign = np.repeat(np.arange(1, len(alns_samples) + 1),
                                       seq_lens).reshape(1, -1)
                cl_assign = np.repeat(cl_assign, seq_lens).reshape(1, -1)

                table = np.concatenate((freqs_sites, aln_assign, cl_assign),
                                       axis=0)
                np.savetxt(f"{out_dir}/{data_dir.split('/')[-1]}_sites.csv",
                           table.T,
                           delimiter=",",
                           header='A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,'
                                  'aln,cl',
                           comments='')
        else:
            warnings.warn(f'{in_dir}/{data_dir} does not exist')

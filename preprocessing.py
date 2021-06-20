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

from stats import generate_aln_stats_df, get_nb_sites, nb_seqs_per_alns
from utils import split_lst, flatten_lst

warnings.simplefilter("ignore", DeprecationWarning)

""" amino acids 
alanine       : A |  glutamine     : Q | leucine       : L | serine    : S |
arginine      : R |  glutamic acid : E | lysine        : K | threonine : T |
asparagine    : N |  glycine       : G | methionine    : M | tryptophan: W |
aspartic acid : D |  histidine     : H | phenylalanine : F | tyrosine  : Y |
cysteine      : C |  isoleucine    : I | proline       : P | valine    : V |
unknown : X 
gap : - 
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


def aln_from_fasta(filename, nb_seqs):
    """Gets aligned sequences from given file

    :param filename: <path/to/> alignments (string)
    :param nb_seqs: number of sequences in alignment (integer)
    :return: list of strings
    """

    alned_seqs_raw = [str(seq_record.seq) for seq_record in
                      SeqIO.parse(filename, "fasta")][:nb_seqs]
    return alned_seqs_raw


def alns_from_fastas(fasta_dir, min_nb_seqs, max_nb_seqs, nb_alns):
    """Gets alignments from fasta files in given directory

    :param fasta_dir: <path/to/> fasta files
    :param min_nb_seqs: min. required number of aligned sequences
    :param max_nb_seqs: max. number of aligned sequences taken
    :param nb_alns: number of alignments
    :return: list of aligned sequences (string list),
             list of alignment identifiers (strings)
    """

    fasta_files = os.listdir(fasta_dir)
    random.shuffle(fasta_files)
    alns = []
    fastas = []
    i = 0

    while len(alns) < nb_alns and i < len(fasta_files):
        seqs = aln_from_fasta(fasta_dir + '/' + fasta_files[i], max_nb_seqs)
        if len(seqs) >= min_nb_seqs:
            alns.append(seqs)
            fastas.append(fasta_files[i].split('.')[0])
        i += 1

    if (nb_alns - len(alns)) != 0:
        print('Only {} / {} fasta files taken into account.'.format(
            len(alns), nb_alns))

    return alns, fastas


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

    def __init__(self, real_alns, sim_alns, shuffle=False,
                 pairs=False):

        nb_real = len(real_alns)
        nb_sim = len(sim_alns)

        data = real_alns + sim_alns

        if pairs:
            # remove first dimension from (aln, pair, chnls, sites) array
            data = np.concatenate(data)
            nb_real = np.sum([len(aln) for aln in real_alns])
            nb_sim = np.sum([len(aln) for aln in sim_alns])

        data = np.asarray(data, dtype='float32')

        if shuffle:  # shuffle columns/sites of empirical alignments
            for i in range(nb_real):
                data[i, :, :] = data[i, :, np.random.permutation(
                    range(data.shape[2]))].swapaxes(0, 1)

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


def data_prepro(fasta_paths, params, pairs=False, take_quantiles=True,
                csv_path=None):
    """Loads alignments and generates representations for the network

    :param take_quantiles: indicate possible reduction of number of sequences
                           per alignment
    :param fasta_paths: <path(s)/to> empirical/simulated fasta files
                        (list/tuple string)
    :param params: parameters for preprocessing (dictionary)
    :param pairs: choose representation by pairs if true (boolean)
    :param csv_path: <path/to> store csv file with info about alignments
    :return: alignment representations and ids, number of sites
    """

    nb_alns, min_nb_seqs, max_nb_seqs, seq_len, padding = params.values()

    print("Loading alignments ...")

    # load sets of multiple alned sequences
    alns, fastas = [], []
    for path in fasta_paths:
        raw_data = alns_from_fastas(path, min_nb_seqs, max_nb_seqs, nb_alns)
        alns.append(raw_data[0])
        fastas.append(raw_data[1])

    if take_quantiles:
        # removing sequences where length is < 0.25 quantile or > 0.75 quantile
        seq_lens = np.asarray(get_nb_sites(alns[0]))
        q1 = np.quantile(seq_lens, 0.25)
        q2 = np.quantile(seq_lens, 0.75)

        alns[0] = [alns[0][i] for i in range(len(seq_lens))
                   if q1 <= seq_lens[i] <= q2]
        fastas[0] = [fastas[0][i] for i in range(len(seq_lens))
                     if q1 <= seq_lens[i] <= q2]

        if len(alns) == 2 and len(fastas) == 2:  # if there is simulated data

            # sort simulated data by sequence length
            ind_s = np.argsort(get_nb_sites(alns[1]))

            alns[1] = [alns[1][i] for i in ind_s]
            fastas[1] = [fastas[1][i] for i in ind_s]

            # keeping same amount of simulated alns from the "middle"(regarding
            # lengths)
            start = len(alns[0]) // 2

            alns[1] = alns[1][start:start + len(alns[0])]
            fastas[1] = fastas[1][start:start + len(alns[0])]

            assert len(alns[0]) == len(alns[1]), (f'{len(alns[1])} simulated '
                                                  f'alignments vs. '
                                                  f'{len(alns[0])} '
                                                  f'empirical alignments ')

            # shuffling the sorted simulated alignments and their ids
            indices = np.arange(len(alns[1]))
            np.random.shuffle(indices)
            alns[1] = [alns[1][i] for i in indices]
            fastas[1] = [fastas[1][i] for i in indices]

        params['nb_sites'] = int(min(seq_len, np.max(get_nb_sites(alns))))

    nb_seqs = nb_seqs_per_alns(alns)
    params['max_seqs_per_align'] = int(np.max(nb_seqs))
    params['min_seqs_per_align'] = int(np.min(nb_seqs))
    params['nb_alignments'] = len(alns[0])

    # generate alignment representations
    if pairs:

        print("Pairing sequences ...")

        start = time.time()

        alns_reprs_pairs = []
        for alns_set in alns:
            alns_reprs_pairs.append([make_seq_pairs(encode_aln(
                aln, params['nb_sites'], padding=padding)) for aln in alns_set])

        print(f'Finished pairing after {round(start - time.time(), 2)}s\n')

        if csv_path is not None:
            generate_aln_stats_df(fastas, alns, params['nb_sites'],
                                  None, is_sim=[0, 1] if len(alns) == 2 else [])

        return *alns_reprs_pairs + fastas, params

    else:

        print("Generating alignment representations ...")

        # make pairs !additional dim for each multiple alingment needs flatteing
        # before passed to CNN!
        alns_reprs = []
        for alns_set in alns:
            alns_reprs.append([get_aln_representation(encode_aln(
                aln, params['nb_sites'], padding=padding)) for aln in alns_set])

        if csv_path is not None:
            generate_aln_stats_df(fastas, alns, params['nb_sites'],
                                  alns_reprs,
                                  is_sim=[0, 1] if len(alns) == 2 else [],
                                  csv_path=csv_path)

        return *alns_reprs + fastas, params

import time

import pandas as pd
import psutil
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset
import os, random
import warnings
import multiprocessing as mp
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
                        '\x00' * 5 + '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14' +
                        '\x15\x16')

THREADS = psutil.cpu_count(logical=False)


def seq2index(seq):
    """
    Translates sequence to byte sequence and finally to an integer sequence that
    could be converted to one-hot encodings of the amino acids
    (including gaps and unknown amino acids)
    :param seq: protein sequence as string
    :return: array of numbers from 0 to 22
    """
    seq = seq.translate(ENCODER)
    return np.fromstring(seq, dtype='uint8').astype('int64')


def index2code(index_seq):
    seq_enc = np.zeros((index_seq.size, 23))
    seq_enc[np.arange(index_seq.size), index_seq] = 1
    return seq_enc


def aln_from_fasta(filename, nb_seqs):
    # load sequences
    alned_seqs_raw = [str(seq_record.seq) for seq_record in SeqIO.parse(filename, "fasta")][:nb_seqs]
    return alned_seqs_raw


def alns_from_fastas(dir, min_nb_seqs, max_nb_seqs, nb_alns):

    fasta_files = os.listdir(dir)
    random.shuffle(fasta_files)
    alns = []
    fastas = []
    i = 0

    while len(alns) < nb_alns and i < len(fasta_files):
        seqs = aln_from_fasta(dir + '/' + fasta_files[i], max_nb_seqs)
        if len(seqs) >= min_nb_seqs:
            alns.append(seqs)
            fastas.append(fasta_files[i].split('.')[0])
        i += 1

    if (nb_alns - len(alns)) != 0:
        print('Only {} / {} fasta files could be taken into account. The rest '
              ' contains less than the minimum of {} sequences.'.format(
            len(alns), nb_alns, min_nb_seqs))

    return alns, fastas


def encode_aln(alned_seqs_raw, seq_len, padding=''):
    # encode sequences and limit to certain seq_len (seq taken from the middle)
    middle = len(alned_seqs_raw[0]) // 2
    start = max(0, (middle - seq_len // 2))
    end = min(len(alned_seqs_raw[0]), (middle + seq_len // 2) + 1)
    seqs = np.asarray([index2code(seq2index(seq[start:end])).T
                       for seq in alned_seqs_raw])

    if len(alned_seqs_raw[0]) < seq_len: # padding
        pad_size = (seq_len - len(alned_seqs_raw[0]))
        pad_before = pad_size // 2
        pad_after = pad_size // 2 if pad_size % 2 == 0 else pad_size // 2 + 1

        seqs_shape = list(seqs.shape)
        seqs_shape[2] += pad_after + pad_before
        seqs_new = np.zeros(seqs_shape)

        if padding == 'data':
            seqs_new = np.asarray([index2code(np.random.randint(0, seqs_shape[1], seqs_shape[2])).T
                                   for _ in range(seqs_shape[0])])

        seqs_new[:, :, pad_before:-pad_after] = seqs + 0
        return seqs_new

    return seqs


def seq_pair_worker(alns):
    return [make_seq_pairs(aln) for aln in alns]


def make_aln_representation(alned_seqs):

    return np.sum(alned_seqs, axis=0) / len(alned_seqs)


def make_seq_pairs(alned_seqs):

    inds = np.asarray(np.triu_indices(len(alned_seqs), k=1))

    sums = alned_seqs[inds[0], :, :] + alned_seqs[inds[1], :, :]
    sum_all_seqs = np.sum(alned_seqs, axis=0)
    aa_prop_no_pair = (sum_all_seqs - sums) / len(alned_seqs)
    return np.concatenate((sums / 2, aa_prop_no_pair), axis=1)


def make_pairs_from_alns_mp(alns):

    if THREADS > len(alns):
        threads = len(alns)
    else:
        threads = THREADS + 0

    alns_chunks = list(split_lst(alns, threads))

    work = [alns_chunks[i] for i in range(threads)]

    with mp.Pool() as pool:
        res = pool.map(seq_pair_worker, work)

    pairs_per_aln = flatten_lst(res)

    return pairs_per_aln


class TensorDataset(Dataset):
    def __init__(self, real_alns, sim_alns, shuffle=False,
                 pairs=False):

        data, labels = self._build_dataset(real_alns, sim_alns, shuffle,
                                           pairs)

        self.data = torch.from_numpy(np.asarray(data)).float()
        self.labels = torch.from_numpy(np.asarray(labels)).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)

    def _build_dataset(self, real, sim, shuffle, pairs):

        nb_real = len(real)
        nb_sim = len(sim)

        data = np.asarray(real + sim)

        if pairs:
            data = np.concatenate(data)
            nb_real = np.sum([len(aln) for aln in real])
            nb_sim = np.sum([len(aln) for aln in sim])

        if shuffle:
            for i in range(nb_real):
                data[i, :, :] = data[i, :, np.random.permutation(
                    range(data.shape[2]))].swapaxes(0, 1)

        labels = [0]*nb_real + [1]*nb_sim

        return data, np.array(labels)


class DatasetAln(Dataset):
    def __init__(self, aln, real):

        self.data = torch.from_numpy(np.asarray(aln)).float()
        labels = [0] * len(aln) if real else [1] * len(aln)
        self.labels = torch.from_numpy(np.asarray(labels)).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)


def mmse(aln1, aln2):
    return np.mean(np.sum((aln1 - aln2) ** 2, axis=0))


def padding(alns, max_seq_len=300):
    paddings = []
    for aln in alns:
        seq_len = len(aln[0])
        paddings.append(max(max_seq_len - seq_len, 0))
    return paddings


def nb_seqs_per_alns(alns):
    return [len(aln) for aln in alns]


def seq_len_per_alns(alns):
    return [len(aln[0]) for aln in alns]


def get_aa_freqs(alns):
    aas = 'ARNDCQEGHILKMFPSTWYVX-'
    aa_freqs_alns = []

    for aln in alns:
        freqs = np.zeros(22)
        for seq in aln:
            for i, aa in enumerate(aas):
                freqs[i] += seq.count(aa)
        freqs /= (len(aln)*len(aln[0]))  # get proportions

        if sum(freqs) != 1:  # get sum as close to 1 as possible
            freqs[np.random.randint(20)] += 1-sum(freqs)

        aa_freqs_alns.append({aas[i]: freqs[i] for i in range(len(aas))})

    return aa_freqs_alns


def distance_stats(dists):
    masked_dists = np.ma.masked_equal(dists, 0.0, copy=False)
    mean_mse = masked_dists.mean(axis=1).data
    max_mse = masked_dists.max(axis=1).data
    min_mse = masked_dists.min(axis=1).data

    return {'mean': mean_mse, 'max': max_mse, 'min': min_mse}


def generate_aln_stats_dict(real_fastas, sim_fastas, real_alns, sim_alns,
                            max_seq_len, real_alns_repr, sim_alns_repr,
                            csv_path=None):
    if real_alns_repr is not None:
        dists_real = np.asarray([[mmse(aln1, aln2) for aln2 in real_alns_repr]
                            for aln1 in real_alns_repr])
        dists_sim = np.asarray([[mmse(aln1, aln2) for aln2 in sim_alns_repr]
                            for aln1 in sim_alns_repr])
        mse_stats_real = distance_stats(dists_real)
        mse_stats_sim = distance_stats(dists_sim)
    else:
        mean_mse = [None] * (len(sim_fastas) + len(real_fastas))
        max_mse = [None] * (len(sim_fastas) + len(real_fastas))
        min_mse = [None] * (len(sim_fastas) + len(real_fastas))
        mse_stats = {'mean': mean_mse, 'max': max_mse, 'min': min_mse}
    
    dat_dict = {'id': real_fastas + sim_fastas,
                'aa_freqs': get_aa_freqs(real_alns + sim_alns),
                'padding': padding(real_alns + sim_alns, max_seq_len),
                'number_seqs': nb_seqs_per_alns(real_alns + sim_alns),
                'seq_length': seq_len_per_alns(real_alns + sim_alns),
                'mean_mse_all': mse_stats_real['mean'] + mse_stats_sim['mean'],
                'max_mse_all': mse_stats_real['max'] + mse_stats_sim['max'],
                'min_mse_all': mse_stats_real['min'] + mse_stats_sim['min'],
                'simulated' : [0] * len(real_alns) + [1] * len(sim_alns)
                }

    df = pd.DataFrame(dat_dict)

    if csv_path is not None:
        csv_string = df.to_csv(index=False)
        with open(csv_path, 'w') as file:
            file.write(csv_string)

    return df


def data_prepro(real_fasta_path, sim_fasta_path, nb_alns, min_nb_seqs,
                max_nb_seqs,  seq_len, pairs=False, csv_path=None):

    print("Loading alignments ...")
    # load sets of multiple alned sequences
    raw_real_alns, fastas_real = alns_from_fastas(real_fasta_path, min_nb_seqs,
                                                  max_nb_seqs, nb_alns)
    raw_sim_alns, fastas_sim = alns_from_fastas(sim_fasta_path, min_nb_seqs,
                                                max_nb_seqs, nb_alns)

    print("Encoding alignments ...")
    # one-hot encode sequences shape: (nb_alns, nb_seqs, amino acids, seq_length)

    enc_real_alns = [encode_aln(aln, seq_len, padding='data') for aln in
                   raw_real_alns]
    enc_sim_alns = [encode_aln(aln, seq_len, padding='data') for aln in
                  raw_sim_alns]

    if pairs:
        print("Pairing sequences ...")
        start = time.time()
        # make pairs !additional dim for each multiple alingment needs to be flattened before passed to CNN!
        real_alns_repr_p = [make_seq_pairs(aln) for aln in enc_real_alns]
        sim_alns_repr_p = [make_seq_pairs(aln) for aln in enc_sim_alns]
        print(f'Finished pairing after {round(time.time() - start, 2)}s\n')

        if csv_path is not None:
            generate_aln_stats_dict(fastas_sim, fastas_real, raw_real_alns,
                                    raw_sim_alns, seq_len, None, None)

        return real_alns_repr_p, sim_alns_repr_p, fastas_real, fastas_sim
    else:
        print("alnment representations ...")
        start = time.time()
        # make pairs !additional dim for each multiple alingment needs flatteing
        # before passed to CNN!
        real_alns_repr = [make_aln_representation(aln) for aln in enc_real_alns]
        sim_alns_repr = [make_aln_representation(aln) for aln in enc_sim_alns]
        print(f'Finished pairing after {round(time.time() - start, 2)}s\n')

        if csv_path is not None:
            generate_aln_stats_dict(fastas_real, fastas_sim, raw_real_alns,
                                    raw_sim_alns, seq_len, real_alns_repr,
                                    sim_alns_repr, csv_path)

        return real_alns_repr, sim_alns_repr, fastas_real, fastas_sim





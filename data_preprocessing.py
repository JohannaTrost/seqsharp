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
    Translates sequence to byte sequence and finally to an integer sequence that could be converted to
    one-hot encodings of the amino acids (including gaps and unknown amino acids)
    :param seq: protein sequence as string
    :return: array of numbers from 0 to 22
    """
    seq = seq.translate(ENCODER)
    return np.fromstring(seq, dtype='uint8').astype('int64')


def index2code(index_seq):
    seq_enc = np.zeros((index_seq.size, 23))
    seq_enc[np.arange(index_seq.size), index_seq] = 1
    return seq_enc


def align_from_fasta(filename, nb_seqs):
    # load sequences
    aligned_seqs_raw = [str(seq_record.seq) for seq_record in SeqIO.parse(filename, "fasta")][:nb_seqs]
    return aligned_seqs_raw


def aligns_from_fastas(dir, min_seqs_per_align, max_seqs_per_align, nb_aligns):

    fasta_files = os.listdir(dir)
    random.shuffle(fasta_files)
    aligns = []
    i = 0

    while len(aligns) < nb_aligns and i < len(fasta_files):
        seqs = align_from_fasta(dir + '/' + fasta_files[i], max_seqs_per_align)
        if len(seqs) >= min_seqs_per_align:
            aligns.append(seqs)
        i += 1

    if (nb_aligns - len(aligns)) != 0:
        print('Only {} / {} fasta files could be taken into account. The rest '
              ' contains less than the minimum of {} sequences.'.format(
            len(aligns), nb_aligns, min_seqs_per_align))

    return aligns


def encode_align(aligned_seqs_raw, seq_len, padding=''):
    # encode sequences and limit to certain seq_len (seq taken from the middle)
    middle = len(aligned_seqs_raw[0]) // 2
    start = max(0, (middle - seq_len // 2))
    end = min(len(aligned_seqs_raw[0]), (middle + seq_len // 2))
    seqs = np.asarray([index2code(seq2index(seq[start:end])).T for seq in aligned_seqs_raw])

    if len(aligned_seqs_raw[0]) < seq_len: # padding
        pad_size = (seq_len - len(aligned_seqs_raw[0]))
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


def seq_pair_worker(aligns):
    return [make_seq_pairs(align) for align in aligns]


def make_seq_pairs(aligned_seqs):

    inds = np.asarray(np.triu_indices(len(aligned_seqs), k=1))

    sums = aligned_seqs[inds[0], :, :] + aligned_seqs[inds[1], :, :]
    sum_all_seqs = np.sum(aligned_seqs, axis=0)
    aa_prop_no_pair = (sum_all_seqs - sums) / len(aligned_seqs)
    return np.concatenate((sums / 2, aa_prop_no_pair), axis=1)


def make_pairs_from_aligns_mp(aligns):

    if THREADS > len(aligns):
        threads = len(aligns)
    else:
        threads = THREADS + 0

    aligns_chunks = list(split_lst(aligns, threads))

    work = [aligns_chunks[i] for i in range(threads)]

    with mp.Pool() as pool:
        res = pool.map(seq_pair_worker, work)

    pairs_per_align = flatten_lst(res)

    return pairs_per_align


class TensorDataset(Dataset):
    def __init__(self, real_aligns, sim_aligns=None, seq_len=500, shuffle=False):

        if sim_aligns is not None:

            data, labels = self._build_dataset(real_aligns, sim_aligns, shuffle)
        else:
            data, labels = self._build_dataset_class_per_align(real_aligns,
                                                               seq_len)

        data = torch.from_numpy(np.asarray(data)).float()
        labels = torch.from_numpy(np.asarray(labels)).float()

        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)

    def _build_dataset(self, real_pairs, sim_pairs, shuffle):

        nb_real_pairs = np.sum([len(align) for align in real_pairs])
        nb_sim_pairs = np.sum([len(align) for align in sim_pairs])

        data = real_pairs + sim_pairs
        data = np.concatenate(data)

        if shuffle:
            for i in range(nb_real_pairs):

                data[i, :, :] = data[i, :, np.random.permutation(range(data.shape[2]))].swapaxes(0, 1)

        labels = [0]*nb_real_pairs + [1]*nb_sim_pairs

        return data, np.array(labels)

    def _build_dataset_class_per_align(self, aligns, seq_len):
        # init dataset list
        data = []
        labels = []

        for label, seqs in enumerate(aligns):

            # combine sequences to pairs
            seq_pairs = make_seq_pairs(encode_align(seqs, seq_len))

            # populate dataset
            data += seq_pairs.tolist()
            labels += (label * np.ones(seq_pairs.shape[0])).tolist()

        return data, labels




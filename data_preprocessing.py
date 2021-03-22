from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import Dataset
import os, random
import warnings
import time
import itertools

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
AA = np.array(
    ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])
ENCODER = str.maketrans('BZJUO' + 'ARNDCQEGHILKMFPSTWYV' + 'X-',
                        '\x00' * 5 + '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14' +
                        '\x15\x16')

MIN_SEQS_PER_ALIGN = 50


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


def alignment_from_fasta(filename, nb_seqs):
    # load sequences
    aligned_seqs_raw = [str(seq_record.seq) for seq_record in SeqIO.parse(filename, "fasta")][:nb_seqs]
    return aligned_seqs_raw


def alignments_from_fastas(dir, nb_seqs_per_align, nb_alignments):
    fasta_files = os.listdir(dir)
    random.shuffle(fasta_files)
    alignments = []
    i = 0
    while len(alignments) < nb_alignments:
        seqs = alignment_from_fasta(dir + '/' + fasta_files[i], nb_seqs_per_align)
        if len(seqs) == nb_seqs_per_align:
            alignments.append(seqs)
        i += 1
    return alignments


def encode_alignment(aligned_seqs_raw, seq_len):
    # encode sequences and limit to certain seq_len (seq taken from the middle)
    middle = len(aligned_seqs_raw[0]) // 2
    start = max(0, (middle - seq_len // 2))
    end = min(len(aligned_seqs_raw[0]), (middle + seq_len // 2))
    seqs = np.array([index2code(seq2index(seq[start:end])).T for seq in aligned_seqs_raw])
    return seqs


def make_seq_pairs_tensor(aligned_seqs):
    nb_seqs = len(aligned_seqs)
    seq_pairs = []
    sum_all_seqs = np.sum(aligned_seqs, axis=0)
    for i in range(nb_seqs):
        for j in range(i + 1, nb_seqs):  # loops for pairs
            sums = aligned_seqs[i, :, :] + aligned_seqs[j, :, :]
            aa_prop_no_pair = (sum_all_seqs - sums) / nb_seqs
            # diffs = (aligned_seqs[i, :, :] - aligned_seqs[j, :, :])
            seq_pairs.append(np.concatenate((sums, aa_prop_no_pair), axis=0))

    seq_pairs_tensor = torch.from_numpy(np.asarray(seq_pairs))
    """
    Vectorized Solution (about 10% slower)
    ms_start = time.time()
    nb_seqs = len(seqs)
    ij = np.asarray(list(itertools.combinations(range(seqs.shape[0]), 2)))
    aa_prop_no_pair = seqs.sum(axis=0)[np.newaxis, :, :].repeat(ij.shape[0], 0)
    p1 = seqs[ij[:, 0], :, :]
    p2 = seqs[ij[:, 1], :, :]
    seq_pairs_sum = p1 + p2
    aa_prop_no_pair -= seq_pairs_sum
    aa_prop_no_pair = aa_prop_no_pair / nb_seqs
    seq_pairs_diff = p1 - p2
    print(time.time()-ms_start)
    """
    return seq_pairs_tensor


class TensorDataset(Dataset):
    def __init__(self, data, target):
        assert data.size(0) == target.size(0)
        self.data = data
        self.target = target.long()

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.size(0)


def build_dataset(alignments, min_seq_len):
    # init dataset tensor/list
    data = torch.from_numpy(np.array([]))
    labels = torch.from_numpy(np.array([]))

    for label, seqs in enumerate(alignments):
        seq_pairs = make_seq_pairs_tensor(encode_alignment(seqs, min_seq_len))
        data = torch.cat((data, seq_pairs), 0)
        labels = torch.cat((labels, label * torch.ones(seq_pairs.shape[0], dtype=torch.float64)),
                           0)  # same label for all seq pairs from the current alignment

    return TensorDataset(data, labels)



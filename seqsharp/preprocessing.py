"""Functions and classes for data preprocessing

Provides functions to turn multiple alignments from fasta files
into 'neural-network-readable' representations that can be
transformed into tensor datasets using
the child classes of *torch.utils.data.Dataset*
"""
import errno
import os

import psutil
import torch
import multiprocessing as mp

from Bio import Phylo, SeqIO, Seq
from torch.utils.data import Dataset

from stats import *
from utils import split_lst, flatten_lst

warnings.simplefilter("ignore", DeprecationWarning)

PROTEIN_ENCODER = str.maketrans('ARNDCQEGHILKMFPSTWYV' + '-',
                                '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
                                '\x0c\r\x0e\x0f\x10\x11\x12\x13' + '\x14')
DNA_ENCODER = str.maketrans('ACGT-', '\x00\x01\x02\x03\x04')

PROTEIN_EMP_ALPHABET = 'ARNDCQEGHILKMFPSTWYV-BZJUOX'
DNA_EMP_ALPHABET = 'AGCTNDHVBRYKMSW-X*'

PROTEIN_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'
DNA_ALPHABET = 'ACGT'

PROTEIN_AMBIG = {'B': ['N', 'D'],
                 'Z': ['Q', 'E'],
                 'J': PROTEIN_ALPHABET,
                 'U': PROTEIN_ALPHABET,
                 'O': PROTEIN_ALPHABET,
                 'X': PROTEIN_ALPHABET}
DNA_AMBIG = {'*': ['A', 'G', 'C', 'T'],
             'X': ['A', 'G', 'C', 'T'],
             'N': ['A', 'G', 'C', 'T'], 'D': ['G', 'A', 'T'],
             'H': ['A', 'C', 'T'], 'V': ['G', 'C', 'A'],
             'B': ['G', 'T', 'C'], 'R': ['G', 'A'],
             'Y': ['C', 'T'], 'K': ['G', 'T'], 'M': ['A', 'C'],
             'S': ['G', 'C'], 'W': ['A', 'T']}

THREADS = psutil.cpu_count(logical=False)


def rearrange_tree(root):
    """Arranges tree such that it accepted by Seq-Gen simulator

    :param root: root of Phylo tree
    """

    for i, clade in enumerate(root.clades):
        if i == 0 and not clade.is_terminal():
            break
        elif i > 0 and not clade.is_terminal():
            tmp = root.clades[0]
            root.clades[0] = clade
            root.clades[i] = tmp
            break


def tree_remove_confidence(clade):
    """Sets confidence of all nodes to None"""

    clade.confidence = None
    for child in clade.clades:
        tree_remove_confidence(child)


def adapt_newick_format(in_path, out_path):
    """Reformat newick tree for Seq-Gen simulations"""
    tree = Phylo.read(in_path, 'newick')

    tree_remove_confidence(tree.root)
    rearrange_tree(tree.root)

    Phylo.write(tree, out_path, 'newick')

    print('Saved formatted tree to ' + out_path)


def remove_gaps(fasta_in, fasta_out):
    """Remove columns with gaps from given fasta file

    :param fasta_in: </path/to> fasta file
    :param fasta_out:  </path/to> new fasta file
    """

    aln_records = [rec for rec in SeqIO.parse(fasta_in, "fasta")]
    aln = np.asarray([np.asarray(list(rec.seq)) for rec in aln_records])

    remove_columns = np.any(aln == '-', axis=0)
    aln_no_gaps = aln[:, np.invert(remove_columns)]

    if aln_no_gaps.shape[1] > 0:

        aln_no_gaps = [''.join([aa for aa in seq]) for seq in aln_no_gaps]

        # update alignment with sequences without gaps
        for i, rec in enumerate(aln_records):
            rec.seq = Seq.Seq(aln_no_gaps[i])

        SeqIO.write(aln_records, fasta_out, "fasta")

    else:
        print(f'No columns left after removing gaps. {fasta_out} not saved.')


def is_mol_type(aln, molecule_type='protein'):
    """Check if given MSA is DNA

    :param aln: list (n_seq) of strings (sequences)
    :return: True if it is DNA according to DNA_EMP_ALPHABET False otherwise
    """

    if molecule_type == 'protein':
        alphabet = PROTEIN_EMP_ALPHABET
    elif molecule_type == 'DNA':
        alphabet = DNA_EMP_ALPHABET

    return set(list(''.join(aln))).issubset(set(list(alphabet)))


def seq2index(seq, molecule_type='protein'):
    """Transforms amino acid sequence to integer sequence

    Translates sequence to byte sequence and finally to an
    integer sequence that could be converted to one-hot encodings
    of amino acids (including gaps and unknown amino acids)

    :param seq: protein sequence as string
    :param molecule_type: either 'DNA' or 'protein' sequences
    :return: array of integers from 0 to 22
    """

    if molecule_type == 'protein':
        seq = seq.translate(PROTEIN_ENCODER)
    elif molecule_type == 'DNA':
        seq = seq.translate(DNA_ENCODER)

    return np.fromstring(seq, dtype='uint8').astype('int64')


def index2code(index_seq, molecule_type='protein'):
    """Transforms array of indices into one-hot encoded arrays

    example:
        array([5, 13]) ->
        array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]])

    :param index_seq: array of integers from 0 to 22
    :param molecule_type: either 'DNA' or 'protein' sequences
    :return: 2D array of 0s and 1s
    """

    if molecule_type == 'protein':
        seq_enc = np.zeros((index_seq.size, 21))
    elif molecule_type == 'DNA':
        seq_enc = np.zeros((index_seq.size, 5))

    seq_enc[np.arange(index_seq.size), index_seq] = 1

    return seq_enc


def replace_ambig_chars(seq, molecule_type='protein'):
    """Replace all ambiguous nucleotides/amino acids by randomly drawn
    nucleotides (A,C,G or T)/ 20 amino acids

    :param seq: sequence
    :return: cleaned sequence
    """

    if molecule_type == 'protein':
        repl_dict = PROTEIN_AMBIG
    elif molecule_type == 'DNA':
        repl_dict = DNA_AMBIG

    seq_arr = np.asarray(list(seq))
    for mol in repl_dict.keys():
        mol_ind = np.where(seq_arr == mol)[0]
        repl_nucs = np.random.choice(repl_dict[mol], len(mol_ind))
        seq_arr[mol_ind] = repl_nucs

    return ''.join(seq_arr)


def remove_ambig_pos_sites(aln, molecule_type):
    """Replace all ambiguous nucleotides/amino acids by randomly drawn
    nucleotides (A,C,G or T)/ 20 amino acids

    :param aln: (n_sites list of strings) MSA
    :param molecule_type: either 'protein' or 'DNA' sequence
    :return: cleaned sequence
    """

    if molecule_type == 'protein':
        repl_dict = PROTEIN_AMBIG
    elif molecule_type == 'DNA':
        repl_dict = DNA_AMBIG

    aln_arr = np.asarray([list(seq) for seq in aln])
    for mol in repl_dict.keys():
        aln_arr = aln_arr[:, np.all(aln_arr != mol, axis=0)]

    return [''.join([aa for aa in seq]) for seq in aln_arr]


def load_msa(filename):
    """Gets aligned sequences from given file

    :param filename: <path/to/> alignments (string)
    :return: list of strings
    """
    if filename.endswith('.fa') or filename.endswith('.fasta'):
        format = 'fasta'
    elif filename.endswith('.phy'):
        format = 'phylip'
    else:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'File format not recognized: '
                         f'{filename.split(".")[-1]}')

    alned_seqs_raw = [str(seq_record.seq) for seq_record in
                      SeqIO.parse(open(filename, encoding='utf-8'), format)]
    return alned_seqs_raw


def load_alns(fasta_dir, quantiles=False, n_alns=None, seq_len=None,
        molecule_type='protein', rem_ambig_chars='remove'):
    """Extracts alignments from fasta files in given directory

    :param rem_ambig_chars: indicate how to remove ambiguous letters: either
    replace randomly 'repl_unif' or remove respective sites 'remove'
    :param fasta_dir: <path/to/> fasta files
    :param quantiles: if True keep MSAs where seq. len. and n. seq.
    within quantiles
    :param n_alns: number of alignments
    :param molecule_type: either protein or DNA sequences
    :return: list of aligned sequences (string list),
             list of alignment identifiers (strings)
    """

    # load fasta filenames
    if fasta_dir.endswith('.txt'):
        # files in txt file must be in same directory as data!
        fasta_files = np.genfromtxt(fasta_dir, dtype=str)
        if not fasta_files[0].endswith('.fasta'):
            fasta_files = np.core.defchararray.add(fasta_files,
                                                   np.repeat('.fasta',
                                                             len(fasta_files)))
        fasta_dir = '/'.join(fasta_dir.split('/')[:-1])
    else:
        fasta_files = np.asarray(os.listdir(fasta_dir))

    if len(fasta_files) == 0:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'No fasta files in directory {fasta_dir}')

    if molecule_type == 'protein':
        ambig_chars = PROTEIN_AMBIG.keys()
    else:
        ambig_chars = DNA_AMBIG.keys()

    # load and preprocess MSAs
    alns, fastas = [], []
    frac_ambig_mol_sites = []
    count_empty, count_wrong_mol_type, count_too_long = 0, 0, 0
    for file in tqdm(fasta_files):
        aln = load_msa(fasta_dir + '/' + file)
        if len(aln) > 0:  # check if no sequences
            # check if num sites
            if len(aln[0]) > 0:
                # exceeds max seq len
                if seq_len is None or len(aln[0]) <= seq_len:
                    # clean up
                    if is_mol_type(aln, molecule_type):
                        # deal with ambiguous letters
                        frac_ambig = get_frac_sites_with(ambig_chars, aln)
                        if frac_ambig > 0:
                            if rem_ambig_chars == 'remove':
                                aln = remove_ambig_pos_sites(aln, molecule_type)
                            elif rem_ambig_chars == 'repl_unif':
                                aln = replace_ambig_chars(','.join(aln),
                                                          molecule_type)
                                aln = aln.split(',')

                        if len(aln[0]) > 10:  # exclude alns with <=10 sites
                            frac_ambig_mol_sites.append(frac_ambig)
                            alns.append(aln)
                            fastas.append(file)
                        else:
                            count_empty += 1
                            print(f'{file} empty after cleaning: '
                                  f'{frac_ambig} amig. letters')
                    else:
                        count_wrong_mol_type += 1
                else:
                    count_too_long += 1
            else:
                count_empty += 1
        else:
            count_empty += 1

        if n_alns is not None and n_alns == len(alns):
            break

    fastas = np.asarray(fastas)
    frac_ambig_mol_sites = np.asarray(frac_ambig_mol_sites)

    print('\n')
    if count_empty > 0:
        print(f'{count_empty} empty file(s)')
    if count_wrong_mol_type > 0:
        print(f'{count_wrong_mol_type} file(s) did not contain '
              f'{molecule_type} sequences')
    if count_too_long > 0:
        print(f'{count_too_long} file(s) have too long sequences '
              f'>{seq_len}')
    if np.sum(frac_ambig_mol_sites != 0) > 0:
        print(f'In {np.sum(frac_ambig_mol_sites != 0)} out of {len(alns)} MSAs '
              f'{np.round((np.sum(frac_ambig_mol_sites) / len(frac_ambig_mol_sites)) * 100, 2)}% sites '
              f'include ambiguous letters.')

    n_seqs = np.asarray(get_n_seqs_per_msa(alns))  # msa rows
    n_sites = np.asarray(get_n_sites_per_msa(alns))  # msa columns

    if quantiles:  # optional
        # such that min = 4 for 6000 MSAs and max < 400
        q_ns = (np.quantile(n_seqs, 0.39), np.quantile(n_seqs, 0.998))
        q_sl = (np.quantile(n_sites, 0.05), np.quantile(n_sites, 0.95))

        print(f'\nq_ns : {q_ns} (0.25, 0.998)\nq_sl : {q_sl} (0.05, 0.95)\n')

        ind_q_ns = np.where((q_ns[0] <= n_seqs) & (n_seqs <= q_ns[1]))[0]
        ind_q_sl = np.where((q_sl[0] <= n_sites[ind_q_ns]) &
                            (n_sites[ind_q_ns] <= q_sl[1]))[0]

        # filter : keep MSAs with seq. len. and n.seq. within above quantiles
        n_seqs = n_seqs[ind_q_ns][ind_q_sl]
        n_sites = n_sites[ind_q_ns][ind_q_sl]
        fastas = fastas[ind_q_ns][ind_q_sl]
        alns = [alns[i] for i in ind_q_ns]
        alns = [alns[i] for i in ind_q_sl]

    if n_alns is not None:  # optional
        if (n_alns - len(fastas)) > 0:  # we get less MSAs than wanted
            print(f'Only {len(fastas)} / {n_alns} fasta files taken into '
                  f'account.')
        else:  # restrict number of MSAs to n_alns
            fastas = fastas[:n_alns]
            n_seqs = n_seqs[:n_alns]
            n_sites = n_sites[:n_alns]
            alns = alns[:n_alns]

    print(f'\n{len(alns)} MSAs loaded from {fasta_dir}\n')

    if len(alns) > 0:
        out_stats = {'n_seqs_min': n_seqs.min(),
                     'n_seqs_max': n_seqs.max(),
                     'n_seqs_avg': n_seqs.mean(),
                     'seq_lens_min': n_sites.min(),
                     'seq_lens_max': n_sites.max(),
                     'seq_lens_avg': n_sites.mean()}
    else:
        out_stats = {}

    if quantiles:
        print('After filtering quantiles:')
    print(out_stats)

    return alns, fastas, out_stats


def load_msa_reprs(path, pairs, n_alns=None):
    """Load msa representations from a directory, TODO handel pair repr

    :param pairs: True if pair representation is given False otherwise
    :param path: <path/to/dir> directory containing csv files with msa reprs.
    :return: list of msa reprs., list (n_alns) with filenames without extension
    """

    if n_alns is not None and n_alns != '':
        files = os.listdir(path)[:n_alns]
    else:
        files = os.listdir(path)
    filenames = []
    msa_reprs = []
    for file in files:
        print(file)
        filenames.append(file.split('.')[0])
        if not pairs:
            msa_reprs.append(np.genfromtxt(f'{path}/{file}', delimiter=','))
        else:
            raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                             'load representations not yet possible for pair '
                             'representations')
    return msa_reprs, filenames


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


def encode_aln(alned_seqs_raw, seq_len, padding='', molecule_type='protein'):
    """Transforms aligned sequences to (padded) one-hot encodings

    Trims/pads the alignment to a certain number of sites (*seq_len*)
    If sequences are > *seq_len* only the middle part of the sequence is taken
    If sequences are < *seq_len* they will be padded at both end either with
    zeros or random amino acids (according to *padding*)

    :param molecule_type: either protein or DNA sequences
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
        seqs = np.asarray([index2code(seq2index(seq[start:end],
                                                molecule_type),
                                      molecule_type).T
                           for seq in alned_seqs_raw])
    else:
        seqs = np.asarray([index2code(seq2index(seq, molecule_type),
                                      molecule_type).T
                           for seq in alned_seqs_raw])

    if len(alned_seqs_raw[0]) < seq_len:  # padding
        pad_size = int(seq_len - len(alned_seqs_raw[0]))
        pad_before = int(pad_size // 2)
        pad_after = (int(pad_size // 2) if pad_size % 2 == 0
                     else int(pad_size // 2 + 1))

        seqs_shape = list(seqs.shape)
        seqs_shape[2] = int(seqs_shape[2] + pad_after + pad_before)
        seqs_new = np.zeros(seqs_shape)

        if padding == 'data':  # pad with random amino acids
            seqs_new = np.asarray([index2code(np.random.randint(0,
                                                                seqs_shape[1],
                                                                seqs_shape[
                                                                    2]),
                                              molecule_type).T
                                   for _ in range(seqs_shape[0])])
        elif padding == 'gaps':
            pad_data = np.repeat('-' * int(seq_len), seqs_shape[1])
            seqs_new = np.asarray([index2code(seq2index(seq, molecule_type),
                                              molecule_type).T
                                   for seq in pad_data])

        seqs_new[:, :, pad_before:-pad_after] = seqs + 0
        return seqs_new
    else:
        return seqs


def seq_pair_worker(alns):
    """Legacy function Generates pair representation for given alignments"""

    return [make_seq_pairs(aln) for aln in alns]


def get_aln_repr(alned_seqs):
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
    """Legacy function Generates pair representations using multi threading"""

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

    def __init__(self, data, labels, pairs=False):
        if pairs:  # remove MSA dimension
            n_real = np.sum([len(data[i]) for i, label in enumerate(labels)
                             if label == 0])
            n_sim = np.sum([len(data[i]) for i, label in enumerate(labels)
                            if label == 1])
            labels = [0] * n_real + [1] * n_sim
            # remove first dimension from (aln, pair, chnls, sites) array
            data = np.concatenate(data)

        self.labels = torch.FloatTensor(labels)
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


def raw_alns_prepro(fasta_paths, n_alns=None, seq_len=None,
        quantiles=None, shuffle=False,
        molecule_type='protein'):
    """Loads and preprocesses raw (not encoded) alignments

    :param shuffle: shuffle sites if True
    :param molecule_type: indicate if either 'protein' or 'DNA' sequences given
    :param quantiles: if True keep MSAs where seq. len. and n. seq.
                      within quantiles
    :param fasta_paths: <path(s)/to> empirical/simulated fasta files
                        (list/tuple of strings)
    :param params: parameters for preprocessing (dictionary)
    :return: ids, preprocessed string alignments, updated param. dict.
    """

    quantiles = [False] * len(fasta_paths) if quantiles is None else quantiles
    n_alns = None if n_alns == '' else n_alns
    seq_len = None if seq_len == '' else seq_len

    params = {}

    print("Loading alignments ...")

    # load sets of multiple aligned sequences
    alns, fastas, stats = [], [], []
    for i, path in enumerate(fasta_paths):
        path = str(path)
        size = n_alns[i] if isinstance(n_alns, list) else n_alns

        # in case of simulations with multiple MSA clusters each cluster has dir
        sim_cl_dirs = [dir for dir in os.listdir(path)
                       if os.path.isdir(os.path.join(path, dir))]
        if len(sim_cl_dirs) > 0:  # there are multiple clusters
            sim_alns, sim_fastas, sim_stats = [], [], {}
            for dir in sim_cl_dirs:
                sim_data = load_alns(f'{path}/{dir}', quantiles[i],
                                     size, seq_len,
                                     molecule_type=molecule_type)
                # concat remove cluster dimension
                sim_alns += sim_data[0]
                sim_fastas += sim_data[1]

                # get stats from current cluster
                if len(sim_stats) == 0:  # first cluster
                    for k, v in sim_data[2].items():
                        sim_stats[k] = [v]
                else:
                    for k in sim_stats.keys():
                        sim_stats[k] += [sim_data[2][k]]

            if size is not None and len(sim_alns) > size:
                inds = np.random.permutation(len(sim_alns))[:size]
                raw_data = [[sim_alns[ind] for ind in inds],
                            [sim_fastas[ind] for ind in inds],
                            sim_stats]
            else:
                raw_data = [sim_alns, sim_fastas, sim_stats]
        else:  # empirical data set or simulations without MSA clusters
            raw_data = load_alns(path, quantiles[i], size, seq_len,
                                 molecule_type=molecule_type)

        alns.append(raw_data[0])
        fastas.append(raw_data[1])
        stats.append(raw_data[2])

    if len(alns) == 2 and len(fastas) == 2:
        print(f"avg. seqs. len. : {stats[1]['seq_lens_avg']} (sim.) vs. "
              f"{stats[0]['seq_lens_avg']} (emp.)")
        print(f"avg. n.seqs. : {stats[1]['n_seqs_avg']} (sim.) vs. "
              f"{stats[0]['n_seqs_avg']} (emp.)")

    # ensure same number of MSAs for all data sets
    if n_alns is not None and not isinstance(n_alns, list):
        for i in range(len(alns)):
            inds = np.random.choice(range(len(alns[i])), n_alns, replace=False)
            alns[i] = [alns[i][ind] for ind in inds]
            fastas[i] = [fastas[i][ind] for ind in inds]

    params['nb_sites'] = int(np.max([ds_stats['seq_lens_max']
                                     for ds_stats in stats]))
    # before it was int(min(seq_len, stats[0]['seq_lens_max'])), why?
    n_msa_ds = len(alns)
    nb_seqs = [get_n_seqs_per_msa(alns[i]) for i in range(n_msa_ds)]
    params['max_seqs_per_align'] = [int(max(nb_seqs[i]))
                                    for i in range(n_msa_ds)]
    params['min_seqs_per_align'] = [int(min(nb_seqs[i]))
                                    for i in range(n_msa_ds)]
    params['n_alignments'] = [len(alns[i]) for i in range(n_msa_ds)]

    if shuffle:  # shuffle sites/columns of alignments
        alns = shuffle_sites(alns)
    return alns, fastas, params


def shuffle_sites(msa_ds):
    """Shuffle sites within MSAs of MSA data sets"""

    for i in range(len(msa_ds)):  # data set
        for j in range(len(msa_ds[i])):  # MSA
            aln = np.asarray([list(seq) for seq in msa_ds[i][j]])
            aln[:, :] = aln[:, np.random.permutation(range(aln.shape[1]))]
            msa_ds[i][j] = [''.join([aa for aa in seq]) for seq in aln]
    return msa_ds


def make_msa_reprs(alns, seq_len, pad='zeros', molecule_type='protein'):
    """Encodes alignments and generates their representations

    :param alns: preprocessed raw alignment sets (3D string list)
    :param seq_len: max. sequence length, used for padding
    :param pad: padding type, default: zeros
    :param molecule_type: either protein or DNA sequences
    :return: alignment representations
    """

    alns_reprs = []

    print("Generating alignment representations ...")

    for alns_set in tqdm(alns):
        if seq_len == 1:
            alns_reprs.append(get_msa_compositions(alns_set))
        else:
            alns_reprs.append([get_aln_repr(
                encode_aln(alns, seq_len, pad, molecule_type))
                for alns in alns_set])

    return alns_reprs


def msa_comp2df(data_collection_path, save='', molecule_type='protein'):
    comps = get_msa_compositions(load_alns(data_collection_path,
                                           molecule_type=molecule_type)[0],
                                 molecule_type=molecule_type)
    df_comps = pd.DataFrame(comps,
                            columns=list(PROTEIN_ALPHABET)
                            if molecule_type == 'protein'
                            else list(DNA_ALPHABET))
    if save is not None and save != '':
        df_comps.to_csv(save, index=False)
    return df_comps

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
from Bio import Phylo, SeqIO, Seq
from torch.utils.data import Dataset

from .stats import *
from .utils import pickle_read

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
    """Sets confidence of all nodes to None

    :param clade: monophyletic group (Clade in Phylo.tree)
    :return: None
    """

    clade.confidence = None
    for child in clade.clades:
        tree_remove_confidence(child)


def adapt_newick_format(in_path, out_path):
    """Reformat newick tree for Seq-Gen simulations

    :param in_path: <path/to/input/tree>
    :param out_path: <path/to/output/tree>
    :return: None
    """
    tree = Phylo.read(in_path, 'newick')

    tree_remove_confidence(tree.root)
    rearrange_tree(tree.root)

    Phylo.write(tree, out_path, 'newick')

    print('Saved formatted tree to ' + out_path)


def remove_gaps(fasta_in, fasta_out):
    """Remove columns with gaps from given fasta file

    :param fasta_in: </path/to> fasta file
    :param fasta_out:  </path/to> new fasta file
    :return: None
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
    """Check if given MSA is DNA or AA

    :param molecule_type: either 'DNA' or 'protein' sequences
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

    :param molecule_type: either 'DNA' or 'protein' sequences
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
    """Replace all ambiguous nucleotides/amino acids by randomly sampled
    nucleotides (A,C,G or T)/ 20 amino acids

    :param aln: (n_sites list of strings) MSA
    :param molecule_type: either 'protein' or 'DNA' sequence
    :return: filtered sequence
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
    """Load MSA from given file

    :param filename: <path/to/> alignment (string)
    :return: MSA as list of strings
    """
    if filename.endswith('.fa') or filename.endswith('.fasta'):
        format_name = 'fasta'
    elif filename.endswith('.phy'):
        format_name = 'phylip'
    else:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'File extension not recognized: '
                         f'{filename.split(".")[-1]}')

    msa_raw = [str(seq_record.seq)
               for seq_record in SeqIO.parse(open(filename, encoding='utf-8'),
                                             format_name)]
    return msa_raw


def preprocess_msa(aln, msa_len, molecule_type, rem_ambig_chars):
    """

    :param aln: alignment (list of strings)
    :param msa_len: number of sites in *aln*
    :param molecule_type: either 'DNA' or 'protein' sequences
    :param rem_ambig_chars: if 'repl_unif' randomly replace ambiguous chars or
                            remove respective sites if 'remove'
    :return: tuple of preprocessed aln or None,
             frac_ambig: fraction of sites with ambiguous chars,
             msa_is: bool dict showing if MSA is empty, too long or has wrong
             molecule type
    """

    msa_is = {'empty': False, 'empty_rem': False, 'wrong_mol_type': False,
              'too_long': False}
    if len(aln) > 0:  # check if no sequences
        if len(aln[0]) > 0:  # check if no sites
            # check if MSA exceeds max num. of sites
            if msa_len is None or len(aln[0]) <= msa_len:
                if is_mol_type(aln, molecule_type):
                    # deal with ambiguous letters
                    if molecule_type == 'protein':
                        ambig_chars = PROTEIN_AMBIG.keys()
                    elif molecule_type == 'DNA':
                        ambig_chars = DNA_AMBIG.keys()
                    frac_ambig = get_frac_sites_with(ambig_chars, aln)
                    if frac_ambig > 0:
                        if rem_ambig_chars == 'remove':
                            aln = remove_ambig_pos_sites(aln, molecule_type)
                        elif rem_ambig_chars == 'repl_unif':
                            aln = replace_ambig_chars(','.join(aln),
                                                      molecule_type)
                            aln = aln.split(',')
                    else:
                        msa_is['empty_rem'] = True
                else:
                    msa_is['wrong_mol_type'] = True
            else:
                msa_is['too_long'] = True
        else:
            msa_is['empty'] = True
    else:
        msa_is['empty'] = True

    # exclude alns with <=2 sites
    return aln if len(aln[0]) > 2 else None, frac_ambig, msa_is


def load_alns(data_dir, n_alns=None, msa_len=None, molecule_type='protein',
        rem_ambig_chars='remove'):
    """Extracts alignments from files in given directory

    :param data_dir: <path/to/> MSA files
    :param n_alns: number of alignments
    :param msa_len: number of sites
    :param molecule_type: either protein or DNA sequences
    :param rem_ambig_chars: indicate how to remove ambiguous letters: either
                            replace randomly 'repl_unif' or remove respective
                            sites 'remove'
    :return: list of MSAs (string list), list of MSA identifiers (string list),
             stats of No. of sites/sequences in data collection (pd.Dataframe)
    """

    file_order_path = os.path.join(data_dir, 'file_order.txt')
    if os.path.exists(file_order_path):
        # use MSA order for pretrained models
        msa_files = np.genfromtxt(file_order_path, dtype=None, encoding=None)
    else:
        msa_files = np.asarray(sorted(os.listdir(data_dir)))

    n_files = len(msa_files)

    if n_files == 0:
        raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                         f'No files (with .fa, .fasta, .phy extension) '
                         f'in directory {data_dir}')

    # load and preprocess MSAs
    alns, files = [], []
    frac_ambig_mol_sites = []
    cnts = {'empty': 0, 'empty_rem': 0, 'wrong_mol_type': 0, 'too_long': 0}
    for file in tqdm(msa_files):
        aln = load_msa(os.path.join(data_dir, file))
        aln, frac_ambig, msa_is = preprocess_msa(aln, msa_len, molecule_type,
                                                 rem_ambig_chars)
        if aln is not None:
            alns.append(aln)
            files.append(file)
        frac_ambig_mol_sites.append(frac_ambig)
        # count empty, too long MSAs or MSAs of wrong molecule type
        for key in cnts.keys():
            cnts[key] += msa_is[key]

        if n_alns is not None and n_alns == len(alns):
            break

    files = np.asarray(files)
    frac_ambig_mol_sites = np.asarray(frac_ambig_mol_sites)

    print('\n')
    if cnts['empty'] > 0:
        print(f' => {cnts["empty"]} empty file(s)')
    if cnts['wrong_mol_type'] > 0:
        print(f' => {cnts["wrong_mol_type"]} file(s) did not contain '
              f'{molecule_type} MSAs')
    if cnts['too_long'] > 0:
        print(f' => {cnts["too_long"]} file(s) have too many sites >{msa_len}')
    if cnts['empty_rem'] > 0:
        print(f' => {cnts["empty_rem"]} file(s) have too few (<=2) sites after '
              f'removing sites with ambiguous letters.')
    if np.sum(frac_ambig_mol_sites != 0) > 0:
        perc_ambig_sites = np.round(np.mean(frac_ambig_mol_sites) * 100, 2)
        print(f' => In {np.sum(frac_ambig_mol_sites != 0)} out of '
              f'{len(alns)} MSAs {perc_ambig_sites}% sites include '
              f'ambiguous letters.')

    print(f'Loaded {len(alns)} MSAs from {n_files} files from {data_dir} '
          f'with success\n')

    if len(alns) > 0:
        stats = pd.DataFrame({'No.seqs.': get_n_seqs_per_msa(alns),
                              'No.sites': get_n_sites_per_msa(alns)}).describe()
    else:
        stats = None

    return alns, files, stats


def load_msa_reprs(path, n_alns=None):
    """Load msa representations from a directory

    :param n_alns: number of alignments
    :param path: <path/to/dir> directory containing pkl files with msa reprs.
    :return: tuple of list of msa reprs. (site-wise/MSA compositions) and
             list of MSA filenames
    """
    file_order = np.genfromtxt(os.path.join(path, 'file_order.txt'), dtype=None,
                               encoding=None)
    if n_alns is not None and n_alns != '':
        files = file_order[:n_alns]
    else:
        files = file_order
    msa_reprs = []
    for file in tqdm(files):
        msa_reprs.append(pickle_read(os.path.join(path, f'{file}.pkl')))

    return msa_reprs, files


def remove_gaps(alns):
    """Removes sites with gaps from given raw alignments (not yet encoded)

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


def encode_aln(aln, msa_len, pad='', molecule_type='protein'):
    """Transforms aligned sequences to (padded) one-hot encodings

    Trims/pads the alignment to a certain number of sites (*msa_len*)
    If sequences are > *msa_len* only the middle part of the sequence is taken
    If sequences are < *msa_len* they will be padded at both end either with
    zeros or random amino acids (according to *padding*)

    :param molecule_type: either protein or DNA sequences
    :param aln: list of amino acid sequences (strings)
    :param msa_len: number of sites (integer)
    :param pad: 'data' or else padding will use zeros
    :return: one-hot encoded alignment (3D array)
    """

    # encode sequences and limit to certain msa_len (seq taken from the middle)
    if len(aln[0]) > msa_len:
        diff = len(aln[0]) - msa_len  # overhang
        start = int(np.floor((diff / 2)))
        end = int(-np.ceil((diff / 2)))
        seqs = np.asarray([index2code(seq2index(seq[start:end],
                                                molecule_type),
                                      molecule_type).T for seq in aln])
    else:
        seqs = np.asarray([index2code(seq2index(seq, molecule_type),
                                      molecule_type).T for seq in aln])

    if len(aln[0]) < msa_len:  # padding
        pad_size = int(msa_len - len(aln[0]))
        pad_before = int(pad_size // 2)
        pad_after = (int(pad_size // 2) if pad_size % 2 == 0
                     else int(pad_size // 2 + 1))

        seqs_shape = list(seqs.shape)
        seqs_shape[2] = int(seqs_shape[2] + pad_after + pad_before)
        seqs_new = np.zeros(seqs_shape)

        if pad == 'data':  # pad with random amino acids
            seqs_new = np.asarray([index2code(np.random.randint(0,
                                                                seqs_shape[1],
                                                                seqs_shape[
                                                                    2]),
                                              molecule_type).T
                                   for _ in range(seqs_shape[0])])
        elif pad == 'gaps':
            pad_data = np.repeat('-' * int(msa_len), seqs_shape[1])
            seqs_new = np.asarray([index2code(seq2index(seq, molecule_type),
                                              molecule_type).T
                                   for seq in pad_data])

        seqs_new[:, :, pad_before:-pad_after] = seqs + 0
        return seqs_new
    else:
        return seqs


def get_sites_compositions(aln):
    """Compute site-wise compositions from one-hot encoded MSA

    :param aln: one hot coded MSA
    :return: array of site compositions (n_sites x 21)
    """
    return np.sum(aln, axis=0) / len(aln)


class TensorDataset(Dataset):
    """Empirical and simulated alignment representations and their labels

        Attributes
        ----------
        data : FloatTensor
            alignment representations (site-wise compositions)
        labels : FloatTensor
            0 for empirical, 1 for simulated
    """

    def __init__(self, data, labels):
        self.labels = torch.FloatTensor(labels)
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)


def raw_alns_prepro(data_paths, n_alns=None, msa_len=None, shuffle=False,
                    molecule_type='protein'):
    """Loading and preprocessing raw (not encoded) alignments

    Loading and preprocessing MSAs and printing statistics of number of
    sites/sequences of data collections. When simulations have multiple clusters
    statistics of clusters are printed too. Eventually shuffling sites of MSAs.

    :param msa_len: number of sites
    :param n_alns: number of alignments
    :param shuffle: shuffle sites if True
    :param molecule_type: indicate if either 'protein' or 'DNA' sequences given
    :param data_paths: <path(s)/to> empirical/simulated MSA files
                       (list of strings)
    :return: filtered filenames (2D list of strings),
             preprocessed alignments (3D list of strings),
             max. No. sites over all MSAs
    """

    n_alns = None if n_alns == '' else n_alns
    msa_len = None if msa_len == '' else msa_len

    print("Loading alignments ...")

    # load sets of multiple aligned sequences
    datasets = [os.path.basename(str(p)) for p in data_paths]
    cols = pd.MultiIndex.from_product([datasets, ['No.seqs.', 'No.sites']])
    max_n_sites_cls = -np.inf
    alns, files, stats = [], [], pd.DataFrame(columns=cols)
    for i, path in enumerate(data_paths):
        path = str(path)
        size = n_alns[i] if isinstance(n_alns, list) else n_alns

        # in case of simulations with multiple MSA clusters each cluster has dir
        sim_cl_dirs = [dir for dir in os.listdir(path)
                       if os.path.isdir(os.path.join(path, dir))]
        if len(sim_cl_dirs) > 0:  # there are multiple clusters
            sim_alns, sim_files = [], []
            cl_stats_cols = pd.MultiIndex.from_product([sim_cl_dirs,
                                                        ['No.seqs.',
                                                         'No.sites']])
            sim_stats = pd.DataFrame(columns=cl_stats_cols)
            for d in sim_cl_dirs:
                sim_data = load_alns(os.path.join(path, d), size, msa_len,
                                     molecule_type=molecule_type)
                # concat remove cluster dimension
                sim_alns += sim_data[0]
                sim_files += sim_data[1]
                # get stats from current cluster
                sim_stats[(d, 'No.sites')] = sim_data[2]['No.sites']
                sim_stats[(d, 'No.seqs.')] = sim_data[2]['No.seqs.']
            print(sim_stats)
            tmp_max_n_sites_cls = int(max(sim_stats.xs('No.sites', level=1,
                                                       axis=1).xs('max')))
            if max_n_sites_cls < tmp_max_n_sites_cls:
                max_n_sites_cls = tmp_max_n_sites_cls

            raw_data = [sim_alns, sim_files]
        else:  # empirical data set or simulations without MSA clusters
            raw_data = load_alns(path, size, msa_len,
                                 molecule_type=molecule_type)

        alns.append(raw_data[0])
        files.append(raw_data[1])

        if len(sim_cl_dirs) == 0:  # no clusters
            stats[(datasets[i], 'No.sites')] = raw_data[2]['No.sites']
            stats[(datasets[i], 'No.seqs.')] = raw_data[2]['No.seqs.']

    print(stats)
    max_n_sites = stats.xs('No.sites', level=1, axis=1).xs('max')
    # columns for simulations with clusters will be NaN
    max_n_sites = max_n_sites[~max_n_sites.isna()]
    max_n_sites = int(max(max_n_sites))
    max_n_sites = max(max_n_sites, max_n_sites_cls)

    if shuffle:  # shuffle sites/columns of alignments
        alns = shuffle_sites(alns)

    return alns, files, max_n_sites


def shuffle_sites(msa_ds):
    """Shuffle sites within MSAs of MSA data sets

    :param msa_ds: MSA data collections (3D list of strings)
    :return: data collections with shuffled sites
    """

    for i in range(len(msa_ds)):  # data set
        for j in range(len(msa_ds[i])):  # MSA
            aln = np.asarray([list(seq) for seq in msa_ds[i][j]])
            aln[:, :] = aln[:, np.random.permutation(range(aln.shape[1]))]
            msa_ds[i][j] = [''.join([aa for aa in seq]) for seq in aln]
    return msa_ds


def make_msa_reprs(alns, msa_len, pad='zeros', molecule_type='protein'):
    """Encodes alignments and generates their representations

    :param alns: preprocessed raw alignment sets (3D string list)
    :param msa_len: number of sites
    :param pad: padding type, default: zeros
    :param molecule_type: either protein or DNA sequences
    :return: alignment representations (list of site or MSA compositions)
    """

    alns_reprs = []

    print("Generating alignment representations ...")

    for alns_set in tqdm(alns):
        if msa_len == 1:
            alns_reprs.append(get_msa_compositions(alns_set,
                                                   molecule_type=molecule_type))
        else:
            alns_set_reprs = [get_sites_compositions(encode_aln(alns, msa_len,
                                                                pad,
                                                                molecule_type))
                              for alns in alns_set]
            alns_set_reprs = np.asarray(alns_set_reprs, dtype='float32')
            alns_reprs.append(alns_set_reprs)

    return alns_reprs

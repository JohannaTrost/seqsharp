import os
import subprocess

import numpy as np
from Bio import Phylo

from stats import get_nb_sites, generate_data_from_dist, get_aa_freqs, \
    nb_seqs_per_alns


def filter_trees_by_nb_seqs(in_path, tree_files, min_nb_seqs, nb_alns):
    """Returns tree files with > *min_nb_seqs* non-terminal nodes

    :param in_path: </path/to> folder containing tree files (string)
    :param tree_files: list of filenames of trees (list string)
    :param min_nb_seqs: min. number of sequences (non-terminal nodes) (integer)
    :param nb_alns: number of alignments (real alns for training) (integer)
    :return: file names of trees with appropriate size (list string)
    """

    tree_file_selection = []
    tree_file_deselection = []

    for file in tree_files:
        tree_path = in_path + '/' + file
        tree = Phylo.read(tree_path, 'newick')

        if count_species_in_tree(tree.root) >= min_nb_seqs:
            tree_file_selection.append(file)
        else:
            tree_file_deselection.append(file)

        if len(tree_file_selection) == (nb_alns * 2):  # enough trees for sim
            break

    if len(tree_file_selection) < nb_alns:
        raise ValueError(f'{len(tree_file_deselection)} files did not '
                         f'contain a minimum of {min_nb_seqs} leaves. Only '
                         f'{len(tree_file_selection)} suitable trees were '
                         f'found. Too small trees are: '
                         f'{tree_file_deselection}')

    return tree_file_selection


def enough_leaves(tree_path, min_nb_seqs):
    """Check if given tree has desired min. size"""

    tree = Phylo.read(tree_path, 'newick')
    if count_species_in_tree(tree.root) >= min_nb_seqs:
        return True
    else:
        return False


def count_species_in_tree(clade):
    """Count tree nodes"""

    if clade.name is None and clade.clades is None:
        return 0
    if clade.name is not None:
        return 1
    return np.asarray([count_species_in_tree(child)
                       for child in clade.clades]).sum()


def simulate(tree_path, alns, out_path, simulator):

    tree_files = os.listdir(tree_path)
    if len(alns) > len(tree_files):
        raise ValueError(
            f'Not enough input trees({len(tree_files)}) for {len(alns)} '
            f'simulations')

    # get lengths and frequencies from hogenom aligned sequences
    min_nb_seqs = np.min(nb_seqs_per_alns(alns))
    seq_lens = get_nb_sites(alns)
    seq_lens = generate_data_from_dist(seq_lens)
    aa_freqs = get_aa_freqs(alns, gaps=False, dict=False)

    sim_count = 0
    files_sim_fail = []
    for file in tree_files:
        fasta_out_path = out_path + '/' + \
                         file.rpartition('.')[0] + '.fasta'
        tree_in_path = tree_path + '/' + file

        if enough_leaves(tree_in_path, min_nb_seqs):
            # prepare frequency parameter
            freqs = np.array2string(aa_freqs[sim_count], separator=',')
            freqs = freqs.replace('\n ', '')[1:-1]

            bash_cmd = (f'{simulator} '
                        f'-mWAG -l{str(seq_lens[sim_count])} '
                        f'-f{freqs} -of '
                        f'< {tree_in_path} > {fasta_out_path}')
            process = subprocess.Popen(bash_cmd, shell=True,
                                       stdout=subprocess.PIPE)
            output, error = process.communicate()

            process.wait()

            if (os.path.exists(fasta_out_path) and
                    os.stat(fasta_out_path).st_size > 0):

                sim_count += 1

                print(
                    '\tSaved ' + fasta_out_path.rpartition('/')[2] +
                    ' to ' + out_path)
                print(
                    '______________________________________________'
                    '___________________________________________\n')
            else:
                if os.path.exists(fasta_out_path):
                    os.remove(fasta_out_path)
                files_sim_fail.append(fasta_out_path.rpartition('/')[2])

            if error is not None:
                print(error)

            if sim_count == len(alns):
                break

            if len(files_sim_fail) > 0:
                print(f'Simulation failed for:{files_sim_fail}\n')

            # Varify number of simulated files
            aligns_sim, _ = alns_from_fastas(out_path,
                                             min_nb_seqs, max_nb_seqs,
                                             len(alns))

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)

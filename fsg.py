"""This program automatically runs sequence alignment simulations
and allows to reformat tree and fasta files
"""

import os
import errno
import subprocess
import argparse

import numpy as np
from Bio import Phylo, SeqIO, Seq

from preprocessing import alns_from_fastas
from stats import get_aa_freqs, generate_data_from_dist


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


def count_species_in_tree(clade):
    """Count tree nodes"""

    if clade.name is None and clade.clades is None:
        return 0
    if clade.name is not None:
        return 1
    return np.asarray([count_species_in_tree(child)
                       for child in clade.clades]).sum()


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


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    sim_params = parser.add_argument_group('arguments for simulation')

    parser.add_argument('indir', type=str,
                        help='the </path/to/> input directory or file')
    parser.add_argument('outdir', type=str,
                        help='the </path/to/> output directory or file')
    group.add_argument('-f', '--format', action='store_true',
                       help='format newick trees such that they can be '
                            'passed to the Seq-Gen simulator')
    group.add_argument('-s', '--simulator', type=str, nargs=2,
                       metavar=('seq-gen path', 'hogenom fasta path'),
                       help='simulate sequences from newick trees. Requires '
                            '</path/to/> Seq-Gen directory.')
    group.add_argument('-r', '--removegaps', action='store_true',
                       help='remove column(s) with gap(s) from input fasta '
                            'file or directory and save alignment(s) without '
                            'gaps in given output file or directory')
    sim_params.add_argument('-n', '--numberseqs', type=int, nargs=2,
                            default=[4, 300],
                            metavar=('min number of sequences',
                                     'max number of sequences'),
                            help='2 integers determining minimum/maximum '
                                 'number of sequences to be simulated per '
                                 'alignmnet default: (4,300)')
    sim_params.add_argument('-a', '--numberaligns', type=int,
                            default=100,
                            help='the number of alignments to be simulated')

    args = parser.parse_args()

    in_path = args.indir
    out_path = args.outdir

    if not os.path.exists(out_path):
        if out_path.rpartition('.')[0] != '':  # is file
            os.makedirs(out_path.rpartition('.')[0])
            print(out_path.rpartition('.')[0])
        else:
            os.makedirs(out_path)

    if args.format:

        print('Adapting format to fit Gen-Seq format requirements...')

        if os.path.isfile(in_path):

            adapt_newick_format(in_path, out_path)

        elif os.path.isdir(in_path):
            tree_files = os.listdir(in_path)

            for file in tree_files:
                if file.rpartition('.')[2] == 'ph':
                    tree_out_path = out_path + '/' + file.rpartition('.')[
                        0] + '.tree'

                    adapt_newick_format(in_path + '/' + file, tree_out_path)

                else:
                    print('skipped file ' + file + ',because it is not a ".ph" '
                                                   'file.')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)

    if args.simulator:

        print('Gen-Seq is starting sequence simulation...')

        seq_gen_path, hogenom_fasta_path = args.simulator
        min_nb_seqs, max_nb_seqs = args.numberseqs if args.numberseqs else \
            (4, 300)
        nb_aligns = args.numberaligns if args.numberaligns else 100

        if not os.path.exists(seq_gen_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    seq_gen_path)

        if os.path.isfile(in_path):

            bash_cmd = seq_gen_path + '/source/seq-gen -mPAM -of < ' \
                       + in_path + ' > ' + out_path
            process = subprocess.Popen(bash_cmd, shell=True,
                                       stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)

        elif os.path.isdir(in_path):

            tree_files = os.listdir(in_path)

            seq_lens = None
            aa_freqs = None

            if os.path.exists(hogenom_fasta_path):
                alignments, _ = alns_from_fastas(hogenom_fasta_path,
                                                 min_nb_seqs,
                                                 max_nb_seqs,
                                                 nb_aligns)

                if len(alignments) <= len(tree_files):
                    # get lengths and frequences from hogenom aligned sequences
                    seq_lens = [len(algn[0]) for algn in alignments]
                    seq_lens = generate_data_from_dist(seq_lens)
                    aa_freqs = get_aa_freqs(alignments, gaps=False, dict=False)
                else:
                    raise ValueError(
                        f'Not enough input trees({len(tree_files)})'
                        f'for {len(alignments)} simulations')
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        hogenom_fasta_path)

            print(f'number of hogenom alignments used: {len(alignments)}')
            print(f'number of input trees used: {len(tree_files)}')

            i = 0  # number of simulated alns
            files_sim_fail = []
            for file in tree_files:
                if file.rpartition('.')[2] == 'tree':
                    fasta_out_path = out_path + '/' + \
                                     file.rpartition('.')[0] + '.fasta'
                    tree_in_path = in_path + '/' + file
                    if enough_leaves(tree_in_path, min_nb_seqs):
                        if seq_lens is None or aa_freqs is None:
                            bash_cmd = seq_gen_path + \
                                       '/source/seq-gen -mPAM -g20 -of < ' \
                                       + tree_in_path + ' > ' + fasta_out_path
                        else:
                            freqs = np.array2string(aa_freqs[i], separator=',')
                            freqs = freqs.replace('\n ', '')[1:-1]

                            bash_cmd = (f'{seq_gen_path}/source/seq-gen '
                                        f'-mWAG -l{str(seq_lens[i])} '
                                        f'-f{freqs} -of '
                                        f'< {tree_in_path} > {fasta_out_path}')
                        process = subprocess.Popen(bash_cmd, shell=True,
                                                   stdout=subprocess.PIPE)
                        output, error = process.communicate()

                        process.wait()

                        if os.path.exists(fasta_out_path) and \
                                os.stat(fasta_out_path).st_size > 0:
                            i += 1
                            print(
                                '\tSaved ' + fasta_out_path.rpartition('/')[2] +
                                ' to ' + out_path)
                            print(
                                '______________________________________________'
                                '___________________________________________\n')
                        else:
                            if os.path.exists(fasta_out_path):
                                os.remove(fasta_out_path)
                            files_sim_fail.append(
                                fasta_out_path.rpartition('/')[2])

                        if error is not None:
                            print(error)

                    if i == len(alignments):
                        break

                else:
                    print(f'skipped file {file},because it is not a ".tree" '
                          f'file.')

            if len(files_sim_fail) > 0:
                print(f'Simulation failed for:{files_sim_fail}\n')

            # Varify number of simulated files
            aligns_sim, _ = alns_from_fastas(out_path,
                                             min_nb_seqs, max_nb_seqs,
                                             nb_aligns)
            if seq_lens is not None:
                sim_seqs_lens = [len(align[0]) for align in aligns_sim]
                sim_seqs_lens.sort()
                seq_lens.sort()

                if len(sim_seqs_lens) == len(seq_lens):
                    print(f'Number of real alignments: {len(alignments)}')
                    print(f'Number of inputted trees: '
                          f'{len(tree_files[:len(alignments)])}')
                    print(f'Newly simulated alignments: {len(aligns_sim)}')
                else:
                    print(f'Nb. sites (simulated): {len(sim_seqs_lens)}')
                    print(f'Nb. sites (empirical): {len(seq_lens)}')
                    print(f'Number of inputted trees: '
                          f'{len(tree_files[:len(alignments)])}')
                    print(f'Newly simulated alignments: {len(aligns_sim)}')
            else:
                print(f'Number of inputted trees: {len(tree_files)}')
                print(f'Newly simulated alignments: {len(aligns_sim)}')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)

    if args.removegaps:

        print('Removing gaps ...')

        if os.path.isfile(in_path):

            remove_gaps(in_path, out_path)

        elif os.path.isdir(in_path):

            fasta_files = os.listdir(in_path)

            for file in fasta_files:

                if file.rpartition('.')[2] == 'fasta':

                    remove_gaps(in_path + '/' + file, out_path + '/' + file)

                else:
                    print('skipped file ' + file + ',because it is not a '
                                                   'fasta file')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)

    print('Process finished 0')


if __name__ == '__main__':
    main()
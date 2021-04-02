import sys, os, errno, subprocess
import argparse
import numpy as np
from Bio import Phylo
from data_preprocessing import aligns_from_fastas


def rearrange_tree(root):
    for i, clade in enumerate(root.clades):
        if i == 0 and not clade.is_terminal():
            break
        else:
            if i > 0 and not clade.is_terminal():
                tmp = root.clades[0]
                root.clades[0] = clade
                root.clades[i] = tmp
                break


def tree_remove_confidence(clade):
    clade.confidence = None
    for child in clade.clades:
        tree_remove_confidence(child)


def adapt_newick_format(in_path, out_path):
    tree = Phylo.read(in_path, 'newick')

    tree_remove_confidence(tree.root)
    rearrange_tree(tree.root)

    Phylo.write(tree, out_path, 'newick')

    print('Saved formatted tree to ' + out_path)


def count_species_in_tree(clade):
    if clade.name is None and clade.clades is None:
        return 0
    if clade.name is not None :
        return 1
    return np.asarray([count_species_in_tree(child) for child in clade.clades]).sum()


def filter_trees_by_nb_seqs(in_path, tree_files, min_nb_seqs):
    for file in tree_files:
        tree_path = in_path + '/' + file
        tree = Phylo.read(tree_path, 'newick')
        if count_species_in_tree(tree.root) < min_nb_seqs:
            tree_files.remove(file)
            print(f'Tree in {file} is not considered for simulation: not enough '
                  f'species')


def main(args):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('indir', type=str,
                        help='the </path/to/> input directory or file')
    parser.add_argument('outdir', type=str,
                        help='the </path/to/> output directory or file')
    group.add_argument('-f', '--format', action='store_true',
                       help='format newick trees such that they can be '
                            'passed to the Seq-Gen simulator')
    group.add_argument('-s', '--simulator', type=str,
                       help='simulate sequences from newick trees. Requires '
                            '</path/to/> Seq-Gen directory.')

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

        seq_gen_path = args.simulator
        if not os.path.exists(seq_gen_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    seq_gen_path)
        seq_gen_path = seq_gen_path[1:] if seq_gen_path[
                                               0] == '/' else seq_gen_path

        if os.path.isfile(in_path):

            bash_cmd = './' + seq_gen_path + '/source/seq-gen -mPAM -of < ' \
                       + in_path + ' > ' + out_path
            process = subprocess.Popen(bash_cmd, shell=True,
                                       stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)

        elif os.path.isdir(in_path):

            tree_files = os.listdir(in_path)

            # get lengths from hogenom aligned sequences
            seq_lens = None

            if os.path.exists(in_path + '/../hogenom_fasta_seqs'):
                alignments = aligns_from_fastas(in_path +
                                                    '/../hogenom_fasta_seqs',
                                                    4, 300, 100)
                # insuring equal number of sim fasta files and hogenom fasta files
                tree_files = [file for file in tree_files if file.rpartition('.')[2] == 'tree']
                filter_trees_by_nb_seqs(in_path, tree_files, 4)

                if len(alignments) <= len(tree_files):
                    seq_lens = [len(algn[0]) for algn in alignments]
                    tree_files = tree_files[:len(alignments)]
                else:
                    raise ValueError(f'{len(tree_files)} files with suitable '
                                     f'trees have been found, but there are'
                                     f' {len(alignments)} hogenom fasta files. '
                                     f'At least as many trees with a minimum of '
                                     f'4 species as alignments are required.');

            for i, file in enumerate(tree_files):
                if file.rpartition('.')[2] == 'tree':
                    fasta_out_path = out_path + '/' + \
                                     file.rpartition('.')[0] + '.fasta'
                    tree_in_path = in_path + '/' + file

                    if seq_lens is None:
                        bash_cmd = './' + seq_gen_path + \
                                   '/source/seq-gen -mPAM -g20 -of < ' \
                                   + tree_in_path + ' > ' + fasta_out_path
                    else:
                        print(f'\nbash command {seq_lens[i]} \n')
                        bash_cmd = './' + seq_gen_path + \
                                   '/source/seq-gen -mPAM -g20 -l' + \
                                   str(seq_lens[i]) + \
                                   ' -of < ' \
                                   + tree_in_path + ' > ' + \
                                   fasta_out_path

                    process = subprocess.Popen(bash_cmd, shell=True,
                                               stdout=subprocess.PIPE)
                    output, error = process.communicate()

                    if os.path.exists(fasta_out_path):
                        print('\tSaved ' + fasta_out_path.rpartition('/')[2] +
                              ' to ' + out_path)
                        print('________________________________________________'
                              '_____________________________________________\n')
                    else:
                        raise FileNotFoundError('Data for {} could not be '
                                                'simulated. Please check the '
                                                'tree file'.format(
                            fasta_out_path.rpartition(' / ')[2]))

                    if error is not None:
                        print(error)

                else:
                    print(
                        'skipped file ' + file + ',because it is not a ".tree" '
                                                 'file.')

            # Varify number of simulated files
            aligns_sim = aligns_from_fastas('data/simulated_fasta_seqs',
                                            4, 300, 100)
            if seq_lens is not None:

                if len(seq_lens) > len(aligns_sim):
                    raise ValueError('Some trees seem to have less than '
                                     'the minimum of 4 sequences. Please check '
                                     'your input files.')

                sim_seqs_lens = [len(align[0]) for align in aligns_sim]
                sim_seqs_lens.sort()
                seq_lens.sort()

                if sim_seqs_lens == seq_lens:
                    print(f'Number of real alignments: {len(alignments)}')
                    print(f'Number of inputted trees: {len(tree_files)}')
                    print(f'Newly simulated alignments: {len(aligns_sim)}')
                else:
                    raise ValueError('Sequence lengths of simulated alignments '
                                     'are not equal to real alignments sequence '
                                     'lengths. Check your inputs.')
            else:
                print(f'Number of inputted trees: {len(tree_files)}')
                print(f'Newly simulated alignments: {len(aligns_sim)}')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)

    print('Process finished 0')


if __name__ == '__main__':
    main(sys.argv[1:])

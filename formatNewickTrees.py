import sys, os, errno, subprocess
import argparse
from Bio import Phylo


def rearrange_tree(root):
    for i,clade in enumerate(root.clades):
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
                    tree_out_path = out_path + '/' + file.rpartition('.')[0] + '.tree'

                    adapt_newick_format(in_path + '/' + file, tree_out_path)

                else:
                    print('skipped file ' + file + ',because it is not a ".ph" '
                                                   'file.')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), in_path)

    if args.simulator:

        print('Gen-Seq is starting sequence simulation...')

        seq_gen_path = args.simulator
        if not os.path.exists(seq_gen_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    seq_gen_path)
        seq_gen_path = seq_gen_path[1:] if seq_gen_path[0] == '/' else seq_gen_path

        if os.path.isfile(in_path):

            bash_cmd = './' + seq_gen_path + '/source/seq-gen -mPAM -of < ' \
                       + in_path + ' > ' + out_path
            process = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
            print(error)

        elif os.path.isdir(in_path):

            tree_files = os.listdir(in_path)

            for file in tree_files:
                if file.rpartition('.')[2] == 'tree':
                    fasta_out_path = out_path + '/' + file.rpartition('.')[0] + '.fasta'
                    tree_in_path = in_path + '/' + file

                    bash_cmd = './' + seq_gen_path + '/source/seq-gen -mPAM -g20 -of < ' \
                               + tree_in_path + ' > ' + fasta_out_path
                    process = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
                    output, error = process.communicate()

                    print('\n__________________________________________________'
                          '___________________________\n')
                    print('\tSaved ' + file + ' to ' + out_path)
                    print('____________________________________________________'
                          '_________________________\n')

                    print(error)

                else:
                    print('skipped file ' + file + ',because it is not a ".tree" '
                                                   'file.')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), in_path)

    print('Process finished 0')


if __name__ == '__main__':
    main(sys.argv[1:])

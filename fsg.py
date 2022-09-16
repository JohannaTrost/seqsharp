"""This program automatically runs sequence alignment simulations
and allows to reformat tree and fasta files
"""

import os
import errno
import subprocess
import argparse

import numpy as np
from Bio import Phylo, SeqIO, Seq

from preprocessing import alns_from_fastas, aa_freq_samples
from simulation import get_leaves_count
from stats import get_aa_freqs, generate_data_from_dist, get_n_seqs_per_msa
from utils import largest_remainder_method
from tqdm import tqdm

np.random.seed(72)


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


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    sim_params = parser.add_argument_group('arguments for simulation')
    freq_params = parser.add_argument_group(
        'arguments for AA frequency samples')

    parser.add_argument('indir', type=str,
                        help='the </path/to/> input directory or file')
    parser.add_argument('outdir', type=str,
                        help='the </path/to/> output directory or file')
    group.add_argument('-f', '--format', action='store_true',
                       help='format newick trees such that they can be '
                            'passed to the simulator. '
                            'Per default files are formatted for the Seq-Gen '
                            'simulator. Use --ocaml to format for the '
                            'ocaml-simulator')
    parser.add_argument('--ocaml', action='store_true',
                        help='Indicate reformatting for simulations with the '
                             'ocaml-simulator')
    group.add_argument('-s', '--simulator', type=str, nargs=3,
                       metavar=('simulator path', 'hogenom fasta path',
                                'parameters path'),
                       help='simulate sequences from newick trees. Requires '
                            '</path/to/> seq-gen or ocaml-sim executable.')
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
    sim_params.add_argument('--profiles', type=str,
                            help="<path/to> tsv file containing AA frequency "
                                 "profiles used by Philippe's simulator")
    group.add_argument('-fs', '--freqsample', type=str, nargs='+',
                       help='directory names for MSA data sets for which '
                            'frequency samples shall be extracted. Directories'
                            ' must be present in "indir"')
    freq_params.add_argument('-l', '--levels', type=str, nargs='+',
                             help='Specify "msa", "sites" and/or "genes" for '
                                  'the level of AA frequency extraction')
    freq_params.add_argument('-p', '--proportion', type=float,
                             help='Sample size as proportion of given data set')

    args = parser.parse_args()

    in_path = args.indir
    out_path = args.outdir

    if not os.path.exists(out_path):
        if out_path.rpartition('/')[-1].rpartition('.')[0] != '':  # is file
            os.makedirs(out_path.rpartition('.')[0])
            print(out_path.rpartition('.')[0])
        # else:
        # os.makedirs(out_path) TODO

    if args.format:
        if args.ocaml:
            print('Adapting format to fit Ocaml-Sim format requirements...')

            if os.path.isfile(in_path):
                tree = Phylo.read(in_path, 'newick')
                tree_remove_confidence(tree.root)
                Phylo.write(tree, out_path, 'newick')

            elif os.path.isdir(in_path):
                tree_files = os.listdir(in_path)

                for file in tree_files:
                    tree_out_path = f'{out_path}/{file.rpartition(".")[0]}.tree'

                    tree = Phylo.read(in_path + '/' + file, 'newick')
                    tree_remove_confidence(tree.root)
                    Phylo.write(tree, tree_out_path, 'newick')
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        in_path)
        else:
            print('Adapting format to fit Gen-Seq format requirements...')

            if os.path.isfile(in_path):

                adapt_newick_format(in_path, out_path)

            elif os.path.isdir(in_path):
                tree_files = os.listdir(in_path)

                for file in tree_files:
                    tree_out_path = f'{out_path}/{file.rpartition(".")[0]}.tree'

                    adapt_newick_format(in_path + '/' + file, tree_out_path)
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        in_path)

    if args.simulator:

        print('Starting sequence simulations...')

        sim_path, hogenom_fasta_path, param_dir = args.simulator
        profile_path = args.profiles
        n_profiles = np.genfromtxt(profile_path, delimiter='\t').shape[1]
        min_nb_seqs, max_nb_seqs = args.numberseqs if args.numberseqs else (
            4, 300)
        n_alns = args.numberaligns if args.numberaligns else 100

        if not os.path.exists(sim_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    sim_path)
        if not os.path.isdir(in_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    in_path)
        if not os.path.exists(hogenom_fasta_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    hogenom_fasta_path)

        if sim_path.rpartition('/')[2] == 'simulator.exe':
            fix_param_dir = f'{sim_path.rpartition("/")[0]}/../..'

        # load alignments
        print(hogenom_fasta_path)
        alignments, fastas, lims = alns_from_fastas(hogenom_fasta_path,
                                                    n_alns=n_alns)
        # get lengths and frequences from hogenom aligned sequences
        seq_lens = [len(aln[0]) for aln in alignments]
        seq_lens = generate_data_from_dist(seq_lens)
        n_seqs = np.asarray(n_seqs(alignments))
        aa_freqs = get_aa_freqs(alignments, gaps=False, dict=False)

        # get em run "ids" from parameter directory
        estim_files = [file for file in os.listdir(param_dir)
                       if os.path.isfile(f'{param_dir}/{file}') and
                       'best' in file]
        em_runs = np.unique([file.split('/')[-1].split('.')[0].split('_')[-1]
                             for file in estim_files])
        print(f'param dir : {param_dir}')
        print(f'em runs : {em_runs}')

        files_sim_fail = []
        for em_run in em_runs:
            if not os.path.exists(f'{out_path}_{em_run}'):
                os.makedirs(f'{out_path}_{em_run}')
            # prepare cluster repartition
            if sim_path.rpartition('/')[2] == 'simulator.exe':
                if os.path.exists(f'{param_dir}/cl_weights_{em_run}.csv'):
                    cl_w = np.genfromtxt(f'{param_dir}/cl_weights_{em_run}.csv',
                                         delimiter=',')
                    cl_sizes = largest_remainder_method(cl_w, n_alns)
                    cl_assign = np.repeat(np.arange(1, len(cl_w) + 1), cl_sizes)
                    for cl in np.arange(1, len(cl_w) + 1):
                        os.mkdir(f'{out_path}_{em_run}/cl{cl}')
                else:
                    cl_assign = None

                # draw quantiles form beta for zeroing weights
                #mu = 0.4
                #var = 0.025
                #alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
                #beta = alpha * (1 / mu - 1)
                #q_zeros = np.random.beta(a=alpha, b=beta, size=n_alns)

            print(f'Simulations for parameters from run {em_run}\n')

            for i, fasta_file in tqdm(enumerate(fastas)):
                file = f'coretree_{fasta_file.strip("fasta")}.tree'

                if sim_path.rpartition('/')[2] != 'simulator.exe':
                    cl_dir = ""
                else:
                    if cl_assign is None:
                        cl_dir = ""
                    else:
                        cl_dir = f"/cl{cl_assign[i]}"

                fasta_out_path = (f'{out_path}_{em_run}{cl_dir}/'
                                  f'{file.rpartition(".")[0]}.fasta') # TODO for seq-gen

                tree_in_path = in_path + '/' + file
                if not os.path.exists(tree_in_path):
                    raise FileNotFoundError(errno.ENOENT,
                                            os.strerror(errno.ENOENT),
                                            tree_in_path)
                elif tree_in_path.rpartition('.')[2] != 'tree':
                    raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT),
                                     f'File extension for a tree needs to be '
                                     f'.tree given: {tree_in_path}')

                freqs = np.array2string(aa_freqs[i], separator=',')
                freqs = freqs.replace('\n ', '')[1:-1]

                # aas = 'ARNDCQEGHILKMFPSTWYV'
                # aa_freqs_aln = np.zeros((20, len(alignments[i][0])))
                # aln_arr = np.asarray(
                #    [list(seq) for seq in alignments[i]])
                # for j in range(aln_arr.shape[1]):
                #    for k, aa in enumerate(aas):
                #        aa_count = list(aln_arr[:, j]).count(aa)
                #        aa_freqs_aln[k, j] = aa_count / aln_arr.shape[0]

                if sim_path.rpartition('/')[2] == 'seq-gen':
                    bash_cmd = (f'{sim_path} '
                                f'-mGENERAL -l{str(seq_lens[i])} '
                                # f'-f{freqs} '
                                f'-of '
                                f'< {tree_in_path} > {fasta_out_path}')
                elif sim_path.rpartition('/')[2] == 'simulator.exe':

                    # one profile per site
                    # np.savetxt(f'{param_dir}/profile.tsv',
                    #            aa_freqs_aln, delimiter='\t')
                    # np.savetxt(f'{param_dir}/weights.csv',
                    #            np.ones((aa_freqs_aln.shape[1])) /
                    #            aa_freqs_aln.shape[1],
                    #            delimiter=',')
                    if cl_assign is not None:
                        pro_w_file = (f'cl{cl_assign[i]}_pro_weights_{em_run}'
                                      f'.csv')
                    else:
                        pro_w_file = f'pro_weights_{em_run}.csv'

                    # force sparse weights (concentrate weights)
                    # weights = np.genfromtxt(f'{param_dir}/{pro_w_file}',
                    #                       delimiter=',')
                    # weights[weights < np.quantile(weights, q_zeros[0])] = 0
                    # weights = weights / weights.sum()
                    # np.savetxt(f'{param_dir}/sparse_weights.csv', weights,
                    #           delimiter=',')
                    # pro_w_file = 'sparse_weights.csv'

                    bash_cmd = (
                        f'{sim_path} --tree {tree_in_path} '
                        f'--profiles {profile_path} '
                        f'--wag {fix_param_dir}/unif.dat '
                        f'--profile-weights {param_dir}/{pro_w_file} '
                        f'-o {fasta_out_path} '
                        f'--mu 0.0 --lambda 0.0 '
                        f'--nsites {str(seq_lens[i])}')

                process = subprocess.Popen(bash_cmd, shell=True,
                                           stdout=subprocess.PIPE)

                # print(f'Executed: \n "{bash_cmd}"')

                output, error = process.communicate()

                process.wait()

                if (os.path.exists(fasta_out_path) and
                        os.stat(fasta_out_path).st_size > 0):
                    i += 1
                    # print(
                    #    '\tSaved ' + fasta_out_path.rpartition('/')[2] +
                    #   ' to ' + out_path)
                    # print(
                    #     '______________________________________________'
                    #     '___________________________________________\n')
                else:
                    if os.path.exists(fasta_out_path):
                        os.remove(fasta_out_path)
                    files_sim_fail.append(
                        fasta_out_path.rpartition('/')[2])

                if error is not None:
                    print(error)

                if i == len(alignments):
                    break

            if len(files_sim_fail) > 0:
                print(f'Simulation failed for:{files_sim_fail}\n')

            # Varify number of simulated files
            if cl_dir != "":
                if cl_assign is not None:
                    aligns_sim = []
                    for cl in np.arange(1, len(cl_w) + 1):
                        aligns_sim += \
                            alns_from_fastas(f'{out_path}_{em_run}/cl{cl}',
                                             False, n_alns)[0]
                else:
                    aligns_sim, _, _ = alns_from_fastas(f'{out_path}_{em_run}',
                                                        False, n_alns)

                if seq_lens is not None:
                    sim_seqs_lens = [len(align[0]) for align in aligns_sim]
                    sim_seqs_lens.sort()
                    seq_lens.sort()
                    if len(sim_seqs_lens) == len(seq_lens):
                        print(f'Number of real alignments: {len(alignments)}')
                        print(f'Newly simulated alignments: {len(aligns_sim)}')
                    else:
                        print(f'Nb. sites (simulated): {len(sim_seqs_lens)}')
                        print(f'Nb. sites (empirical): {len(seq_lens)}')
                        print(f'Newly simulated alignments: {len(aligns_sim)}')
                else:
                    print(f'Newly simulated alignments: {len(aligns_sim)}')

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

    if args.freqsample:
        levels = args.levels
        n_alns = args.numberaligns
        sample_prop = args.proportion
        data_dirs = args.freqsample
        aa_freq_samples(in_path, data_dirs, sample_prop, n_alns, levels,
                        out_path)

    print('Process finished 0')


if __name__ == '__main__':
    main()

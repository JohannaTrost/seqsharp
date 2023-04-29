import os
import subprocess
import argparse

import numpy as np

from tqdm import tqdm
from seqsharp.preprocessing import load_msa
from seqsharp.utils import load_custom_distr, load_kde
from seqsharp.stats import sample_indel_params


def main():
    global gamma_shape_pdf, indel_pdf

    # ---- handling arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outpath', type=str, required=True,
                        help='Path to directory of output simulations.')
    parser.add_argument('-t', '--treepath', type=str, required=False,
                        help='Path to directory with phylogenetic tree files '
                             'in Newick format.')
    parser.add_argument('-n', '--nsim', type=int, default=None,
                        help='Number of alignments to simulate.')
    parser.add_argument('-m', '--subm', type=str, default='',
                        help='Substitution model as Alisim argument, '
                             'e.g. "WAG" or "LG+C60"')
    parser.add_argument('-g', '--gamma', type=str, default='',
                        help='Gamma model, e.g. "G4" for four rate categories '
                             'or "GC" for continuous Gamma distribution.')
    parser.add_argument('--edclpath', type=str,
                        help='Path to EDCluster repository directory to use '
                             'UDM models (for this clone '
                             'https://github.com/dschrempf/EDCluster.git).')
    parser.add_argument('-p', '--nprof', type=str, default='',
                        choices=['0004', '0008', '0016', '0032', '0064', '0128',
                                 '0192', '0256', '0512', '1024', '2048',
                                 '4096'],
                        help='Number of profiles of UDM model.')  # e.g. 0256
    parser.add_argument('-i', '--indels', action='store_true', default=False,
                        help='Simulate Indels using empirical '
                             'Indel parameters.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for the simulation folder.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (impact on shuffling trees and '
                             'sampling parameters, but not Alisim).')

    args = parser.parse_args()

    # get arguments
    out_path = args.outpath
    tree_path = args.treepath
    n_sims = args.nsim
    sub_m = args.subm
    edcl_path = args.edclpath
    n_prof = args.nprof
    gamma = args.gamma
    indels = args.indels
    suffix = f'_{args.suffix}' if args.suffix != '' else ''
    seed = args.seed

    if seed is not None:
        np.random.seed(seed)

    # make substitution model arg. string for alisim
    if n_prof != '':  # use Schrempfs UDM models
        mix_m_def = os.path.join(edcl_path, 'Distributions', 'hogenom',
                                 f'udm_hogenom_{n_prof}_lclr_iqtree.nex')
        mix_m = f'UDM{n_prof}LCLR'
        sub_m_args = f'-mdef {mix_m_def} -m {sub_m}+{mix_m}'
    else:  # just use the given model
        # can also include mixture model e.g. 'LG+C60'
        sub_m_args = f'-m {sub_m}'

    # create output directory
    sub_m_str = sub_m.lower().replace("+", "_")
    mix_m_str = f'_s{n_prof}' if n_prof != '' else n_prof
    g_str = '_' + gamma.lower() if gamma != '' else gamma
    indel_str = '_sabc' if indels else ''
    out_dir = f'alisim_{sub_m_str}{mix_m_str}{g_str}{indel_str}{suffix}'
    out_path = os.path.join(out_path, out_dir)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # filter and shuffle tree files
    trees = [t for t in os.listdir(tree_path) if not t.endswith('.log')]
    trees = np.asarray(trees)[np.random.permutation(len(trees))]
    if n_sims is not None and n_sims > len(trees):  # reuse trees
        trees = np.random.choice(trees, n_sims)

    n_sims = len(trees) if n_sims is None else n_sims  # No. simulations

    # load empirical probability density functions
    n_sites_pdf = load_custom_distr(
        os.path.join('emp_pdfs', 'n_sites_hogenom_6971.CustomPDF'))
    if gamma != '':
        gamma_shape_pdf = load_custom_distr(
            os.path.join('emp_pdfs', 'gamma_shape.CustomPDF'))
    if indels:
        indel_pdf = load_kde(os.path.join('emp_pdfs', 'indel_param_distr'))

    # sample from PDFs
    seq_lens = n_sites_pdf.draw(n_sims)
    gamma_shapes = gamma_shape_pdf.draw(n_sims) if gamma != '' else None

    n_sites_lims = (100, 10001)  # limit sequence length for indel simulation

    print(f'Start simulations for {out_path}\n')

    n_sims_succ = 0  # No of successful simulations
    files_sim_fail = []  # tree files for which simulations failed

    for i in tqdm(range(n_sims)):
        
        out_file = f'{os.path.basename(os.path.splitext(trees[i])[0])}.fa'
        out_file = os.path.join(out_path, out_file)
        
        if not os.path.exists(out_file):  # no overwrite

            tree_file = os.path.join(tree_path, trees[i])

            if gamma != '':
                gamma_arg = f'+{gamma}{"{"}{gamma_shapes[i]}{"}"}'
            else:
                gamma_arg = ''

            repeat_sim = True
            
            while repeat_sim:
                
                if indels:
                    indel_params = sample_indel_params(*indel_pdf)

                    indel_size_arg = '--indel-size POW{'
                    indel_size_arg += str(indel_params["RIM A_I"][0])
                    indel_size_arg += '/100},POW{'
                    indel_size_arg += str(indel_params["RIM A_D"][0])
                    indel_size_arg += '/100}'

                    indel_arg = f'--indel {indel_params["RIM R_I"][0]},'
                    indel_arg += f'{indel_params["RIM R_D"][0]} '
                    indel_arg += indel_size_arg

                    n_sites = indel_params['RIM RL'][0]
                else:
                    n_sites = seq_lens[i]
                    indel_arg = ''

                bash_cmd = (
                    f'iqtree2 --alisim {os.path.splitext(out_file)[0]} '
                    f'-t {tree_file} '
                    f'{sub_m_args}{gamma_arg} -mwopt -af fasta '
                    f'--seqtype AA '
                    f'--length {int(n_sites)} {indel_arg}')

                print(bash_cmd)

                process = subprocess.Popen(bash_cmd, shell=True,
                                           stdout=subprocess.PIPE)
                output, error = process.communicate()
                process.wait()

                if error is None:
                    if os.path.exists(out_file):
                        # check if No sites ok
                        sim_n_sites = len(load_msa(out_file)[0])
                        if not indels or sim_n_sites in range(*n_sites_lims):
                            repeat_sim = False
                            print(f'==> No. sites {sim_n_sites}')
                        else:
                            print(f'==> No. sites ({sim_n_sites}) '
                                  f'exceeds limits {sim_n_sites}')
                else:
                    print(error)

        if os.path.exists(out_file) and os.stat(out_file).st_size > 0:
            n_sims_succ += 1
        else:
            files_sim_fail.append(out_file)

    if len(files_sim_fail) > 0:
        print('Simulation failed for:')
        print(files_sim_fail)


if __name__ == '__main__':
    main()

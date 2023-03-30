import os
import subprocess
import argparse

import numpy as np

from tqdm import tqdm
from seqsharp.preprocessing import load_msa
from utils import load_custom_distr, load_kde
from stats import sample_indel_params

np.random.seed(34)

def main():
    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indels', action='store_true')
    parser.add_argument('-n', '--nsim', type=int)
    parser.add_argument('-m', '--subm', type=str, default='')
    parser.add_argument('-g', '--gamma', type=str, default='')
    parser.add_argument('-p', '--nprof', type=str, default='')  # e.g. 0256
    parser.add_argument('-t', '--treepath', type=str, default='')
    parser.add_argument('--postfix', type=str, default='')

    args = parser.parse_args()

    n_sites_pdf = load_custom_distr(
        'emp_pdfs/n_sites_hogenom_6971.CustomPDF')
    gamma_shape_pdf = load_custom_distr(
        'emp_pdfs/gamma_shape.CustomPDF')

    indel_distr = load_kde('emp_pdfs/indel_param_distr')

    sub_m = args.subm
    n_prof = args.nprof
    gamma = args.gamma
    name_add_on = f'_{args.postfix}'
    n_sims = args.nsim
    indels = True if args.indels else False
    tree_path = args.treepath

    if n_prof != '':
        mix_m_def = f'../../EDCluster/Distributions/hogenom/' \
                    f'udm_hogenom_{n_prof}_lclr_iqtree.nex'
        mix_m = f'UDM{n_prof}LCLR'
        sub_m_args = f'-mdef {mix_m_def} -m {sub_m}+{mix_m}'
    else:
        sub_m_args = f'-m {sub_m}'

    mix_m_str = f'_s{n_prof}' if n_prof != "" else n_prof
    gamma_str = '_' + gamma.lower() if gamma != '' else gamma
    out_path = f'../../emp_pdfs/simulations/test_samples/alisim_{sub_m.lower()}{mix_m_str}{gamma_str}{name_add_on}'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    trees = [t for t in os.listdir(tree_path) if not t.endswith('.log')]
    trees = np.asarray(trees)[np.random.permutation(len(trees))]

    seq_lens = n_sites_pdf.draw(n_sims)
    gamma_shapes = gamma_shape_pdf.draw(n_sims)

    sl_lims = (100, 10000)

    print(f'Start simulations for {out_path}\n')

    n_sims_succ = 0
    files_sim_fail = []
    sim_sls = []
    for i in tqdm(range(n_sims)):
        out_fasta = trees[i].split('.')[0]
        if out_fasta.startswith('coretree_'):
            out_fasta = out_fasta.split('coretree_')[1]

        if not os.path.exists(f'{out_path}/{out_fasta}.fa'):
                gamma_arg = f'+{gamma}{"{"}{gamma_shapes[i]}{"}"}' if gamma != '' else gamma

                repeat_sim = True
                while repeat_sim:
                    if indels:
                        indel_params = sample_indel_params(*indel_distr)

                        indel_size_arg = '--indel-size POW{'
                        indel_size_arg += str(indel_params["RIM A_I"][0])
                        indel_size_arg += '/100},'
                        indel_size_arg += 'POW{' + str(
                            indel_params["RIM A_D"][0])
                        indel_size_arg += '/100}'

                        indel_arg = f'--indel {indel_params["RIM R_I"][0]},'
                        indel_arg += f'{indel_params["RIM R_D"][0]} '
                        indel_arg += indel_size_arg

                        sl = indel_params['RIM RL'][0]
                    else:
                        sl = seq_lens[i]
                        indel_arg = ''

                    bash_cmd = (
                        f'iqtree2 --alisim {out_path}/{out_fasta} '
                        f'-t {tree_path}/{trees[i]} '
                        f'{sub_m_args}{gamma_arg} -mwopt -af fasta --seqtype AA '
                        f'--length {int(sl)} {indel_arg}')

                    print('\n' + bash_cmd)
                    process = subprocess.Popen(bash_cmd, shell=True,
                                               stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    process.wait()

                    if error is None and os.path.exists(
                            f'{out_path}/{out_fasta}.fa'):
                        # check if len ok
                        sim_sl = len(load_msa(f'{out_path}/{out_fasta}.fa')[0])
                        if not indels or sl_lims[0] <= sim_sl <= sl_lims[1]:
                            repeat_sim = False
                            print(f'==> SL {sim_sl}')
                        else:
                            print(f'sim len not within limits: {sim_sl}')
                    else:
                        print(error)

        if (os.path.exists(f'{out_path}/{out_fasta}.fa') and
                os.stat(f'{out_path}/{out_fasta}.fa').st_size > 0):
            n_sims_succ += 1
        else:
            files_sim_fail.append(out_fasta)

    if len(files_sim_fail) > 0:
        print('Simulation failed for:')
        print(files_sim_fail)


if __name__ == '__main__':
    main()

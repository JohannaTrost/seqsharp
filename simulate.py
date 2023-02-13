import os
import subprocess
from tqdm import tqdm

from utils import n_sites_pdf, gamma_shape_pdf


sub_m = 'LG'
n_prof = ''
gamma = 'GC'
name_add_on = '_gapless'

if n_prof != '':
    mix_m_def = f'../EDCluster/Distributions/hogenom/' \
                f'udm_hogenom_{n_prof}_lclr_iqtree.nex'
    mix_m = f'UDM{n_prof}LCLR'
    sub_m_args = f'-mdef {mix_m_def} -m {sub_m}+{mix_m}'
else:
    sub_m_args = f'-m {sub_m}'


tree_path = '../data/hogenom_trees_gapless'

out_path = f'../data/alisim_{sub_m.lower()}_' \
           f'{mix_m.split("LCLR")[0].lower() if n_prof != "" else ""}_' \
           f'{gamma.lower()}{name_add_on}'
while out_path.endswith('_'):
    out_path = out_path[:-1]
out_path = out_path.replace('__', '_')
os.mkdir(out_path)

trees = [t for t in os.listdir(tree_path) if not t.endswith('.log')]
n_sims = len(trees)
seq_lens = n_sites_pdf.draw(n_sims)
gamma_shapes = gamma_shape_pdf.draw(n_sims)

print(f'Start simulations for {out_path}\n')

n_sims_succ = 0
files_sim_fail = []
for i in tqdm(range(n_sims)):
    out_fasta = trees[i].split('.')[0]
    if out_fasta.startswith('coretree_'):
        out_fasta = out_fasta.split('coretree_')[1]

    seq_len = n_sites_pdf.draw()[0]
    gamma_arg = f'+{gamma}{"{"}{gamma_shapes[i]}{"}"}' if gamma != '' else gamma
    bash_cmd = (
        f'iqtree2 --alisim {out_path}/{out_fasta} '
        f'-t {tree_path}/{trees[i]} '
        f'{sub_m_args}{gamma_arg} -mwopt -af fasta --seqtype AA '
        f'--length {int(seq_lens[i])}')

    process = subprocess.Popen(bash_cmd, shell=True,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    process.wait()

    if (os.path.exists(f'{out_path}/{out_fasta}.fa') and
            os.stat(f'{out_path}/{out_fasta}.fa').st_size > 0):
        n_sims_succ += 1
    else:
        if os.path.exists(f'{out_path}/{out_fasta}'):
            os.remove(f'{out_path}/{out_fasta}')
        files_sim_fail.append(out_fasta)

    if error is not None:
        print(error)

if len(files_sim_fail) > 0:
    print('Simulation files for:')
    print(files_sim_fail)
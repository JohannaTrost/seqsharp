import os
import subprocess
from tqdm import tqdm

from utils import n_sites_pdf, gamma_shape_pdf


sub_m = 'Poisson'
n_prof = '0256'
mix_m_def = f'../../EDCluster/Distributions/hogenom' \
            f'/udm_hogenom_{n_prof}_lclr_iqtree.nex'
mix_m = f'UDM{n_prof}LCLR'

tree_path = '../../data/hogenom_trees'
out_path = f'../../data/simulations/alisim_{sub_m.lower()}_{mix_m.lower()}'
if not os.path.exists(out_path):
    os.mkdir(out_path)

trees = [t for t in os.listdir(tree_path) if t.endswith('.tree')]
n_sims = len(trees)
seq_lens = n_sites_pdf.draw(n_sims)

n_sims_succ = 0
files_sim_fail = []
for i in tqdm(range(n_sims)):
    out_fasta = trees[i].split('coretree_')[1].split('.')[0]
    seq_len = n_sites_pdf.draw()[0]
    bash_cmd = (
        f'iqtree2 --alisim {out_path}/{out_fasta} '
        f'-t {tree_path}/{trees[i]} '
        f'-mdef {mix_m_def} '
        f'-m {sub_m}+{mix_m} -mwopt -af fasta --seqtype AA '
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
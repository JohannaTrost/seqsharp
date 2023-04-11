import multiprocessing
import sys
import argparse

from matplotlib import pylab as plt

from preprocessing import raw_alns_prepro
from stats import count_mols
from utils import read_cfg_file
from hem_fcts import *
from plots import get_ylim, plot_em_learning_curves


def main():
    np.random.seed(72)

    # -------- Handling arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs=1, type=str,
                        help='Specify the <path/to/cfg.json> EM cfg file')
    parser.add_argument('--debug', action='store_true',
                        help='With debug flag the variational lower bound after'
                             ' each E- and M-step rather than only after the'
                             'M-step is traced as well as lk. dips between '
                             'two M-steps (of iteration i and iteration i+1).')
    parser.add_argument('--test', action='store_true',
                        help='Here emp_pdfs is generated using given '
                             '(in cfg by "fix_params_paths") or default '
                             'mixture and mixture component proportions to '
                             'find whether or not the EM-algo. recovers '
                             'correct estimates.')
    parser.add_argument('--random_state', nargs=1, type=int, default=72,
                        help='Set random state for parameter initialization. '
                             'Useful when number of runs is 1 e.g. when '
                             'running different EM runs on different '
                             'cluster nodes/jobs.')
    args = parser.parse_args()

    em_cfg_path = args.cfg[0]
    debug, test = args.debug, args.test
    random_state = args.random_state

    # load parameters from cfg file
    em_cfg = read_cfg_file(em_cfg_path)
    n_alns, n_iter, n_runs, n_profiles, n_clusters, n_proc = \
        list(em_cfg.values())[:6]

    # load parameters to be fixed during EM iterations
    fix_params = {}
    if ('profs' in em_cfg['fix_params_paths'].keys() and
            em_cfg['fix_params_paths']['profs'] != ''):
        fix_params['profs'] = np.genfromtxt(em_cfg['fix_params_paths']['profs'],
                                            delimiter='\t').T
    if ('pro_w' in em_cfg['fix_params_paths'].keys() and
            em_cfg['fix_params_paths']['pro_w'] != ''):
        fix_params['pro_w'] = np.genfromtxt(em_cfg['fix_params_paths']['pro_w'],
                                            delimiter=',')
    if ('cl_w' in em_cfg['fix_params_paths'].keys() and
            em_cfg['fix_params_paths']['cl_w'] != ''):
        fix_params['cl_w'] = np.genfromtxt(em_cfg['fix_params_paths']['cl_w'],
                                           delimiter=',')

    if em_cfg['save_path'] != "":
        # create necessary folders to save results
        if not os.path.exists(f'{em_cfg["save_path"]}'):
            os.mkdir(f'{em_cfg["save_path"]}')
        if not os.path.exists(f'{em_cfg["save_path"]}/lk'):
            os.mkdir(f'{em_cfg["save_path"]}/lk')  # for saving likelihoods
        if not os.path.exists(f'{em_cfg["save_path"]}/init_params'):
            os.mkdir(f'{em_cfg["save_path"]}/init_params')

    if not test:  # run full EM
        # load parameters to load empirical MSAs
        cfg = read_cfg_file(em_cfg['cfg_path_msa'])
        all_n_alns = cfg['data']['n_alignments']

        # -------- load empirical MSAs
        raw_alns, raw_fastas, cfg['data'] = raw_alns_prepro(
            [em_cfg['fasta_in_path']], cfg['data'])

        print(f'Alignments loaded in {int(time.time() - STARTTIME)}s\n')

        # sample MSAs for EM estimation from loaded empirical MSAs
        if all_n_alns > n_alns:
            sample_step_size = np.round(all_n_alns / n_alns)
            sample_inds = np.arange(0, all_n_alns, sample_step_size)
            sample_inds = sample_inds[:n_alns].astype(int)

            if len(sample_inds) < n_alns:
                sample_inds = np.concatenate(
                    (sample_inds, sample_inds[:(n_alns -
                                                len(sample_inds))] + 1))

            alns = [raw_alns[0][ind] for ind in sample_inds]
            fastas = [raw_fastas[0][ind] for ind in sample_inds]

        if em_cfg['save_path'] != '':
            np.savetxt(f'{em_cfg["save_path"]}/init_params/'
                       f'real_fastanames4estim.txt',
                       fastas, delimiter=',', fmt='%s')

        aa_counts = [count_mols([aln], 'sites').T for aln in alns]
        n_aas = aa_counts[0].shape[-1]

        optimal_lk = None  # only avail in test when true params. given

        print(f'Estimation on {len(alns)} MSAs\n')

    else:  # test EM on simulations given true parameters
        # set to true parameters to default (below) if no params specified
        true_params = {}
        if {'profs', 'pro_w', 'cl_w'}.issubset(set(fix_params.keys())):
            true_params = fix_params
        else:
            true_params['profs'] = np.asarray([[0., 0., 0.25, 0.25, 0.5, 0.],
                                               [0.05, 0.05, 0.05, 0.05, 0.4,
                                                0.4],
                                               [0.2, 0.1, 0.2, 0.15, 0.2, 0.15],
                                               [0.05, 0.05, 0.7, 0.05, 0.05,
                                                0.1]])
            true_params['pro_w'] = np.asarray([[1 / 2, 1 / 4, 1 / 8, 1 / 8],
                                               [0.05, 0.05, 0.8, 0.1],
                                               [0.2, 0.15, 0.3, 0.35]])
            true_params['cl_w'] = np.asarray([0.1, 0.6, 0.3])

        profs, pro_w, cl_w = true_params.values()

        # params for MSA generation
        n_alns, n_sites, n_seqs = 80, 40, 40
        n_aas = profs.shape[1]

        n_profiles = profs.shape[0]
        n_clusters = pro_w.shape[0]

        # -------- generate AA counts
        aa_counts = generate_alns(profs, pro_w, cl_w, n_alns, n_sites, n_seqs)

        # -------- optimal lk
        sites_pi, alns_pi = e_step(aa_counts, profs, pro_w, cl_w)
        optimal_vlb, _ = lk_lower_bound(aa_counts, profs, pro_w, cl_w,
                                        sites_pi, alns_pi)
        optimal_lk = joint_log_lk(pro_w, cl_w, profs, aa_counts)

    # -------- Initialization of parameters to be estimated
    np.random.seed(random_state)
    init_params = init_estimates(n_runs, n_clusters, n_profiles, n_alns,
                                 n_aas, equal_inits=True if test else False,
                                 true_params=[true_params['profs'],
                                              true_params['pro_w'],
                                              true_params['cl_w']]
                                 if test else None)

    # to store results
    cl_w_runs = np.zeros((n_runs, n_clusters))
    pro_w_runs = np.zeros((n_runs, n_clusters, n_profiles))
    prof_runs = np.zeros((n_runs, n_profiles, n_aas))
    loglks = np.zeros((n_runs, n_iter))
    vlbs = np.zeros((n_runs, n_iter * 2))

    try:
        print(f'Start EM after {int(time.time() - STARTTIME)}s')

        # organize runs according to number of processes
        # e.g. n_runs=6, n_proc=3 -> run 3 EMs in parallel 2 times
        # parallel_runs_inds = [[0, 1, 2], [3, 4, 5]]
        parallel_runs_inds = np.array_split(range(n_runs), n_proc)
        for runs in parallel_runs_inds:
            n_par_runs = len(runs)
            with multiprocessing.Pool(n_par_runs) as pool:
                result = pool.starmap(em, zip([aa_counts] * n_par_runs,
                                              [n_iter] * n_par_runs,
                                              init_params[runs[0]:runs[-1] + 1],
                                              [fix_params] * n_par_runs,
                                              runs,
                                              [debug] * n_par_runs,
                                              [em_cfg['save_path']] *
                                              n_par_runs))
            for proc, run in zip(range(n_par_runs), runs):
                cl_w_runs[run] = result[proc][0][2]
                pro_w_runs[run] = result[proc][0][1]
                prof_runs[run] = result[proc][0][0]
                vlbs[run] = result[proc][1]
                loglks[run] = result[proc][2]

        best_lk_run = np.argmax(vlbs[:, -1], axis=0)

        # ------ rename parameter files to indicate best parameters by vlb
        if fix_params is None or 'profs' not in fix_params.keys():
            os.rename(f'{em_cfg["save_path"]}/profiles_{best_lk_run + 1}.tsv',
                      f'{em_cfg["save_path"]}/profiles_best{best_lk_run + 1}.tsv')
        if n_clusters > 1:
            if fix_params is None or 'cl_w' not in fix_params.keys():
                os.rename(
                    f'{em_cfg["save_path"]}/cl_weights_{best_lk_run + 1}.csv',
                    f'{em_cfg["save_path"]}/cl_weights_best{best_lk_run + 1}'
                    f'.csv')
            if fix_params is None or 'pro_w' not in fix_params.keys():
                for cl in range(n_clusters):
                    os.rename(f'{em_cfg["save_path"]}/cl{cl + 1}_pro_weights_'
                              f'{best_lk_run + 1}.csv',
                              f'{em_cfg["save_path"]}/cl{cl + 1}_pro_weights_'
                              f'best{best_lk_run + 1}.csv')
        elif n_clusters == 1:
            if fix_params is None or 'pro_w' not in fix_params.keys():
                os.rename(f'{em_cfg["save_path"]}/pro_weights_{best_lk_run + 1}'
                          f'.csv', f'{em_cfg["save_path"]}/pro_weights_best'
                                   f'{best_lk_run + 1}.csv')

    except KeyboardInterrupt:
        print("Keyboard interrupt in main:")
    finally:
        print("cleaning up main")

    if test:
        # get error of true and estimated params. and cluster and profile order
        # according to minimum error between estimated and true params.
        sorted_params, errors = estimated_param_sorting(true_params['profs'],
                                                        prof_runs[best_lk_run],
                                                        true_params['pro_w'],
                                                        pro_w_runs[best_lk_run],
                                                        true_params['cl_w'],
                                                        cl_w_runs[best_lk_run])
        print(f'\nMAE profiles: {errors[0]}\nMAE profile weights: {errors[1]}\n'
              f'MAE cluster weights: {errors[2]}\n\n')
        print(f'True profiles:\n\t{true_params["profs"]}\n'
              f'Estimated profiles:\n\t{sorted_params[0]}\n\n'
              f'True profile weights:\n\t{true_params["pro_w"]}\n'
              f'Estimated profile weights:\n\t{sorted_params[1]}\n\n'
              f'True cluster weights:\n\t{true_params["cl_w"]}\n'
              f'Estimated cluster weights:\n\t{sorted_params[2]}\n')

    # **************************** SAVE RESULTS ***************************** #

    np.savetxt(f'{em_cfg["save_path"]}/lk/vlbs_{n_runs}runs_{n_iter}iter.csv',
               vlbs,
               delimiter=',')
    np.savetxt(f'{em_cfg["save_path"]}/lk/loglks_{n_runs}runs_{n_iter}iter.csv',
               loglks,
               delimiter=',')

    # **************************** PLOT RESULTS ***************************** #

    vlbs[vlbs == -np.inf] = np.min(vlbs[vlbs > -np.inf])

    # -------- plot final vlb/joint loglk for all runs
    x_axis = np.arange(1, n_runs + 1)
    titles = ['Variational lower bound (for EM)', 'Joint log-lk of MSAs']

    fig, axs = plt.subplots(2)
    for ax, lks, title, color in zip(axs, [vlbs[:, -1], loglks[:, -1]], titles,
                                     ['g', 'magenta']):
        ax.bar(x_axis, lks, tick_label=x_axis, color=color)
        ax.set_title(title)
        ax.set_xlabel('EM run')
        ax.set_ylabel('log-likelihood')
        ax.set_ylim(*get_ylim(lks, 1.5))

    plt.tight_layout()
    plt.savefig(f'{em_cfg["save_path"]}/lk/em_lk_bar.png')
    plt.close()

    # -------- plot optimization history
    if debug:
        vlbs_dips = get_dips(vlbs)
    else:
        vlbs_dips = np.zeros_like(vlbs)
        vlbs_dips[:, 1::2] = get_dips(vlbs[:, 1::2])
    # joint lk only existing after M-step
    loglks_dips = np.zeros_like(vlbs)
    loglks_dips[:, 1::2] = get_dips(loglks)

    vlbs_dips[vlbs_dips == np.inf] = np.max(vlbs_dips[vlbs_dips < np.inf])

    for lks, lk_type, d, opt_lk, dips in zip([vlbs, np.repeat(loglks, 2,
                                                              axis=1)],
                                             ['vlb loglks', 'joint loglks'],
                                             [debug, False],
                                             [optimal_lk, None],
                                             [vlbs_dips, loglks_dips]):
        plot_em_learning_curves(n_runs, n_iter, dips, lks, d, test,
                                lk_type, optimal_lk=opt_lk,
                                save_path=em_cfg["save_path"])

    print(f'Total runtime: {(time.time() - STARTTIME) / 60}min.\n')
    print(f'Results in: {em_cfg["save_path"]}')


if __name__ == '__main__':
    main()

"""Main functions to train and test a model to classify empirical and
simulated MSAs
"""

import errno
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .ConvNet import ConvNet, load_model, compute_device, activation
from .attr_methods import get_attr
from .preprocessing import TensorDataset, raw_alns_prepro, make_msa_reprs, \
    load_msa_reprs
from .plots import plot_folds, make_fig, annotate
from .utils import write_cfg_file, read_cfg_file
from .train_eval import *

import seaborn as sns

torch.cuda.empty_cache()

gc.collect()


def handle_args(parser):
    args = parser.parse_args()

    # ---------------------------- get arguments ---------------------------- #

    opts = {'val': args.validate, 'test': args.test, 'train': args.train,
            'emp_path': args.emp, 'sim_paths': args.sim,
            'result_path': (args.save or None),
            'model_path': (args.model or None), 'cfg_path': (args.cfg or None),
            'shuffle': args.shuffle,
            'attr': args.attr, 'clr': args.clr, 'n_cpus': args.ncpus}

    # -------------------- verify argument usage -------------------- #

    if (not (opts['train'] ^ opts['val'] ^ opts['test'])
            or (opts['train'] and opts['val'] and opts['test'])):
        if opts['model_path'] is None:
            parser.error('Specify --training, --test, --validate or --models.')

    if (opts['val'] or opts['test']) and not opts['model_path']:
        parser.error('--validate and --test require --modles')

    if opts['train'] and not opts['emp_path']:
        parser.error('Training requires exactly 2 datasets: one directory '
                     'containing empirical alignments and one with simulated '
                     'alignments.')

    # ------------------ verify existence of files/folders ------------------ #

    # determine cfg path from model path
    if opts['model_path'] and not opts['cfg_path']:
        if os.path.isdir(opts['model_path']):
            opts['cfg_path'] = opts['model_path'] + '/cfg.json'
        elif os.path.isfile(opts['model_path']):
            opts['cfg_path'] = os.path.dirname(opts['model_path']) + '/cfg.json'

    for path in ['emp_path', 'result_path', 'model_path', 'cfg_path']:
        if opts[path] is not None and not os.path.exists(opts[path]):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    opts[path])
    if opts['sim_paths'] is not None:
        for sim_path in opts['sim_paths']:
            if not os.path.exists(sim_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        sim_path)

    # create output folder
    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    if opts['sim_paths']:
        sim_dir = os.path.basename(opts['sim_paths'][0])
        cnn_dir = f'cnn_{sim_dir}_{str(timestamp)}'

        if opts['result_path'] is not None and not opts['model_path']:
            if not opts['result_path'].split('/')[-1].startswith('cnn_'):
                # create unique subdir for the model(s)
                opts['result_path'] += f'/{cnn_dir}'
                os.makedirs(opts['result_path'])
        elif opts['model_path'] and opts['train']:
            # create new subdir for the resumed model(s)
            opts['result_path'] = os.path.dirname(opts['model_path'])
            opts['result_path'] += f'/resume_{cnn_dir}'
            os.makedirs(opts['result_path'])

    return opts


def load_data(emp_path, sim_paths, cfg_path, model_path, shuffle):
    if sim_paths is None:
        sim_paths = []

    cfg = read_cfg_file(cfg_path)
    # parameters for loading data
    n_sites = cfg['data']['n_sites']
    n_alns = cfg['data']['n_alignments']
    molecule_type = cfg['data']['molecule_type']

    cnt_datasets = len(sim_paths)
    cnt_datasets += 1 if emp_path is not None else 0

    n_alns = n_alns if isinstance(n_alns, list) or n_alns == '' else [n_alns]
    if len(n_alns) == cnt_datasets and n_alns != '':
        n_alns = [int(x) for x in n_alns]
    elif model_path is None and n_alns != '':
        n_alns = int(n_alns[0]) if isinstance(n_alns, list) else int(n_alns)
    else:
        n_alns = None

    data_paths = [emp_path] if emp_path is not None else []
    data_paths += sim_paths

    # ------------------------- load/preprocess data ------------------------- #

    first_input_file = os.listdir(data_paths[0])[0]
    fmts = ['.fa', '.fasta', '.phy']
    correct_fmt = np.any([first_input_file.endswith(f) for f in fmts])
    if correct_fmt:
        alns, files, max_n_sites = raw_alns_prepro(
            data_paths, n_alns, n_sites, shuffle=shuffle,
            molecule_type=molecule_type)
        print('-------------------------------------------------------'
              '---------')
        if molecule_type == 'protein' and '-' not in ''.join(alns[0][0]):
            # for protein data without gaps
            for i in range(len(alns)):
                # remove first 2 sites
                alns[i] = [[seq[2:] for seq in aln] for aln in alns[i]]

        if 'input_size' in cfg['model'].keys():
            n_sites = cfg['model']['input_size']
        else:
            n_sites = max_n_sites
        reprs = make_msa_reprs(alns,
                               n_sites,
                               cfg['data']['padding'],
                               molecule_type=molecule_type)
        del alns
    elif first_input_file.endswith('.pkl'):  # msa representations is provided
        reprs = []
        for i in range(cnt_datasets):
            size = n_alns[i] if isinstance(n_alns, list) else n_alns
            reprs.append(load_msa_reprs(data_paths[i], size))

    ds_sizes = [len(ds) for ds in reprs]
    data = np.concatenate(reprs)
    labels_emp = np.zeros(len(reprs[0])) if emp_path is not None else []
    if cnt_datasets > 1 or len(labels_emp) == 0:
        labels_sim = np.ones(np.sum([len(r) for r in (reprs[1:]
                                                      if len(labels_emp) > 0
                                                      else reprs)]))
    else:
        labels_sim = []
    labels = np.concatenate((labels_emp, labels_sim))

    return data, labels, ds_sizes


def model_test(opts, in_data):

    # load config and save with timestamp
    cfg = read_cfg_file(opts['cfg_path'])
    timestamp = datetime.now()

    # get data and models
    data, labels, ds_sizes = in_data
    models = load_model(opts['model_path'])

    # determine names for data collections
    ds_names = []
    if opts['emp_path'] is not None:
        ds_names.append('emp')
    if opts['sim_paths'] is not None:
        for sim_path in opts['sim_paths']:
            ds_names.append(os.path.basename(sim_path))

    # save config with timestamp and data collection names
    cfg['comments'] += ' Test on data collections: ' + ', '.join(ds_names)
    write_cfg_file(cfg, opts['cfg_path'], timestamp=timestamp)

    # prepare dataframe to store performance (BACC, loss etc.)
    cols = [np.repeat(ds_names, 2), ['loss', 'acc'] * len(ds_names)]
    ds_res = pd.DataFrame(columns=cols)
    # indices to split arrays (data, labels) into original data collections
    ds_inds = np.split(np.arange(len(data)), np.cumsum(ds_sizes))[:-1]

    for ds_name, ds_ind in zip(ds_names, ds_inds):

        res_dict = {'loss': [], 'acc': []}
        for fold in range(len(models)):
            model = models[fold]
            test_ds = TensorDataset(data[ds_ind], labels[ds_ind])
            test_loader = DataLoader(test_ds, cfg['training']['batch_size'])

            # eval on test data collection
            with torch.no_grad():
                test_result = evaluate(model, test_loader)

            # populate results dictionary
            res_dict['loss'].append(test_result['loss'])
            if ds_name == 'emp':
                res_dict['acc'].append(test_result['acc_emp'])
            else:
                res_dict['acc'].append(test_result['acc_sim'])

        # populate MultiIndex df separating different data collections
        ds_res[(ds_name, 'loss')] = res_dict['loss']
        ds_res[(ds_name, 'acc')] = res_dict['acc']

    # print and save results
    ds_res = results2table(ds_res,
                           f'{opts["result_path"]}/test_{timestamp}.csv')
    print('\n#########################  PERFORMANCE  '
          '#########################\n')
    print(ds_res)


def validate(opts, in_data):
    timestamp = datetime.now()
    data, labels, _ = in_data
    cfg = read_cfg_file(opts['cfg_path'])
    # save cfg with timestamp
    write_cfg_file(cfg, opts['cfg_path'], timestamp=timestamp)
    bs = cfg['training']['batch_size']
    # load data and models
    models = load_model(opts['model_path'])
    # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
    kfold = StratifiedKFold(cfg['training']['n_folds'],
                            shuffle=True, random_state=42)

    res_dict = {'loss': [], 'bacc': [], 'acc_emp': [], 'acc_sim': []}
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data,
                                                            labels)):
        model = models[fold]
        val_loader = DataLoader(TensorDataset(data[val_ids],
                                              labels[val_ids]), bs)
        with torch.no_grad():
            # eval on validataion dataset
            val_result = evaluate(model, val_loader)
            for k, v in val_result.items():
                res_dict[k].append(v)

    res_df = results2table(res_dict,
                           f'{opts["result_path"]}/val_{timestamp}.csv')
    print('\n#########################  PERFORMANCE  '
          '#########################\n')
    print(res_df)


def determine_fold_lr(lr, lr_range, curr_fold, clr, train_loader,
        model_params, optimizer, result_path):
    # determine lr (either lr from cfg or lr finder)
    if isinstance(lr, list) and len(lr) > curr_fold:
        # cfg lr is list of lrs per fold
        fold_lr = lr[curr_fold]
    elif isinstance(lr, list) and len(lr) <= curr_fold and lr_range == '':
        # e.g. when resuming and the current fold was not at all trained yet
        fold_lr = lr[-1]
    elif clr or lr_range != '':  # use lr finder
        model = ConvNet(model_params).to(compute_device)
        min_lr, max_lr = find_lr_bounds(model, train_loader,
                                        optimizer,
                                        result_path if curr_fold == 0
                                        else '', lr_range,
                                        prefix=f'{curr_fold + 1}_'
                                        if curr_fold == 0 else '')
        if clr:
            fold_lr = (min_lr, max_lr)
        else:  # lr finder determined lr for non-clr
            fold_lr = max_lr
    else:
        fold_lr = lr  # single lr given in cfg

    return fold_lr


def train(opts, in_data):
    sep_line = '-------------------------------------------------------' \
               '---------'

    resume = True if opts['model_path'] else False

    timestamp = datetime.now()
    cfg = read_cfg_file(opts['cfg_path'])
    # parameters of network architecture
    model_params = cfg['model']
    # hyperparameters
    bs, epochs, lr, lr_range, opt, n_folds = cfg['training'].values()
    if opt == 'Adagrad':
        optimizer = torch.optim.Adagrad
    elif opt == 'SGD':
        optimizer = torch.optim.SGD
    elif opt == 'Adam':
        optimizer = torch.optim.Adam
    else:
        raise ValueError(
            'Please specify either Adagrad, Adam or SGD as optimizer')

    # get data and models
    data, labels, _ = in_data
    if resume:
        models = load_model(opts['model_path'], state='train')
    else:
        models = []

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\nCompute device: {compute_device}\n')
    print(f'Random seed: {seed}\n')
    print(f"{sep_line}\n\tBatch size: {bs}\n"
          f"\tLearning rate: {lr_range if lr_range != '' else lr}")

    # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
    kfold = StratifiedKFold(cfg['training']['n_folds'],
                            shuffle=True, random_state=seed)

    lrs = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data,
                                                            labels)):
        print(f'{sep_line}\nFOLD {fold + 1}\n{sep_line}')
        train_ds = TensorDataset(data[train_ids], labels[train_ids])
        val_ds = TensorDataset(data[val_ids], labels[val_ids])
        train_loader = DataLoader(train_ds, bs, shuffle=True,
                                  num_workers=opts['n_cpus'])
        val_loader = DataLoader(val_ds, bs, num_workers=opts['n_cpus'])

        if len(train_ds.data.shape) > 2:
            # if the input are site-wise compositions
            model_params['input_size'] = train_ds.data.shape[2]  # max seq len
        else:  # if the input are avg MSA compositions
            model_params['input_size'] = 1

        # load/create (pretrained) model
        if not resume or models[fold] is None:
            model = ConvNet(model_params).to(compute_device)
            start_epoch = 0
        else:
            model = models[fold]
            start_epoch = len(model.train_history['loss']) - 1

        fold_lr = determine_fold_lr(lr, lr_range, fold, opts['clr'],
                                    train_loader, model_params, optimizer,
                                    opts['result_path'])
        lrs.append(fold_lr)
        fit(fold_lr, model, train_loader, val_loader, optimizer,
            start_epoch, epochs, save=opts['result_path'], fold=fold)

        if resume:
            models[fold] = model.to('cpu')
        else:
            models.append(model.to('cpu'))

        if opts['result_path'] is not None:  # save cfg
            # add lrs per fold
            cfg['training']['lr'] = lrs if len(lrs) > 0 else lr
            # add path of original model when training was resumed
            if resume:
                cfg['original_model_path'] = cfg['results_path']
            # add path to results i.e. models, plots etc.
            cfg['results_path'] = opts['result_path']

            write_cfg_file(cfg, opts['result_path'])
            # save cfg with timestamp
            write_cfg_file(cfg, opts['result_path'], timestamp=timestamp)

    # save results regrouping folds
    print('\n#########################  PERFORMANCE  '
          '#########################\n')
    print_model_performance(models)
    if opts['result_path'] is not None:
        # save results table
        save_path = os.path.join(opts['result_path'], f'val_{timestamp}.csv')
        results2table(evaluate_folds([m.val_history for m in models],
                                     n_folds)[0], save=save_path)
        # save plot of learning curves
        make_fig(plot_folds, [models], (1, 2),
                 save=os.path.join(opts['result_path'],
                                   'folds_learning_curve.pdf'))
        print(f'\nSaved results to {opts["result_path"]}\n')
    else:
        print(
            f'\nNot saving models and evaluation plots. Please use '
            f'--save and specify a directory if you want to save '
            f'your results!\n')


def attribute(opts, in_data):
    timestamp = datetime.now()
    cfg = read_cfg_file(opts['cfg_path'])
    # get data and models
    data, labels, _ = in_data
    models = load_model(opts['model_path'])
    # choose best fold
    fold_eval = [np.min(model.val_history['loss']) for model in models]
    folds = {'best': np.argmin(fold_eval), 'worst': np.argmax(fold_eval)}

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\nCompute device: {compute_device}\n')
    print(f'Random seed: {seed}\n')
    # k-fold validator (kfold-seed ensures same fold-splits in opt. loop)
    kfold = StratifiedKFold(cfg['training']['n_folds'],
                            shuffle=True, random_state=seed)
    classes = {'emp': 0, 'sim': 1}
    val_inds = {}
    for k, v in folds.items():
        val_inds[k] = [v for fold, (_, v) in enumerate(kfold.split(data,
                                                                   labels))
                       if fold == folds[k]]
    for key, inds in val_inds.items():
        print(f'Attribution scores from fold-{folds[key] + 1}-model')
        if opts['result_path'] is not None:
            attr_path = f'{opts["result_path"]}/attr_fold{folds[key] + 1}'
            if key == 'best':
                attr_path += '_min_loss'
            elif key == 'worst':
                attr_path += '_max_loss'
            if not os.path.exists(attr_path):
                os.mkdir(attr_path)
        else:
            attr_path = ''

        # get validation data for that fold
        val_ds = TensorDataset(data[inds].copy(), labels[inds])
        # get net
        model = models[folds[key]]

        pad_mask = data[inds][0].sum(axis=1).astype(bool)  # sum over channels
        df = pd.DataFrame({'label': labels[inds][0],
                           'pred': activation(
                               model(val_ds.data[0])).detach().cpu().numpy(),
                           'seq_len': pad_mask.sum(axis=1)})  # sum over seq
        # plot predictions
        fig = sns.displot(data=df, x="pred", kind="ecdf", hue='label')
        fig.savefig(os.path.join(attr_path, 'ecdf_preds.pdf'))
        # plot correlation of seq. len. and predictions
        fig = sns.lmplot(x="pred", y="seq_len", data=df, col='label')
        fig.map_dataframe(annotate)
        fig.savefig(os.path.join(attr_path, 'corr_seq_len_preds.pdf'))

        # get attributions
        sali, ig = [], []
        for msa in val_ds.data[0]:
            sali.append(get_attr(msa, model, 'saliency'))
            ig.append(get_attr(msa, model, 'integratedgradients',
                               multiply_by_inputs=True))
        ''' In progress
        # plot site/channel importance
        plot_summary(sali, pad_mask, 'channels', None,
                     opts['molecule_type'],
                     save=f'{attr_path}/channel_attr.pdf')
        plot_summary(sali, pad_mask, 'sites', None,
                     opts['molecule_type'],
                     save=f'{attr_path}/site_attr.pdf')
        plot_summary(sali, pad_mask, 'sites', None,
                     opts['molecule_type'],
                     save=f'{attr_path}/site_attr_200.pdf',
                     max_sl=200)
        plot_summary(sali, pad_mask, 'sites', None,
                     opts['molecule_type'],
                     save=f'{attr_path}/site_attr_50.pdf',
                     max_sl=50)

        # plot individual saliency maps

        # indices of n best predicted (decreasing) and
        # n worst predicted
        # (increasing) MSAs
        n = 5
        select = np.concatenate(
            (np.arange(n), np.arange(-n, 0)))

        for l, cl in enumerate(sali.keys()):
            # validation data, sorted according to attr. maps
            msas = val_ds.data[val_ds.labels == l][
                sort_by_pred[cl]]
            fnames = files[val_ids][val_ds.labels == l][
                sort_by_pred[cl]]

            for i, sel in enumerate(select):
                score = '%.2f' % np.round(preds[cl][sel], 2)

                sal_map = sali[cl][sel][pad_mask[cl][sel]]
                ig_map = ig[cl][sel][pad_mask[cl][sel]]
                msa = msas[sel][:,
                      pad_mask[cl][sel]].detach().cpu().numpy()
                msa_id = fnames[sel].split(".")[0]
                plot_msa_attr(sal_map, ig_map, msa,
                              molecule_type,
                              save=f'{attr_path}/'
                                   f'[{i}]_{msa_id}_{score}_'
                                   f'{cl}.pdf')
        '''

import argparse

from ConvNet import load_model
from seqsharp_fcts import handle_args, load_data, validate, model_test, train
from train_eval import print_model_performance


def main():
    sep_line = '-------------------------------------------------------' \
               '---------'

    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', nargs='*', type=str,
                        help='Specify the <path/to/> directory(s) containing '
                             'simulated alignments (in fasta format)')
    parser.add_argument('--emp', type=str,
                        help='Specify the <path/to/> directory(s) containing '
                             'empirical alignments (in fasta format)')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Datasets will be used to train the neural '
                             'network (specified with --datasets option). '
                             'Requires --cfg and --datasets.')
    parser.add_argument('--test', action='store_true',
                        help='Test network on given data collections. '
                             'Requires --models and --sim')
    parser.add_argument('--validate', action='store_true',
                        help='Validate network on given data collections. '
                             'Requires --models, --emp and --sim')
    parser.add_argument('--attr', action='store_true',
                        help='Generates attribution maps using validation '
                             'emp_pdfs. '
                             ' Requires --models (only if not --training) and '
                             '--datasets')
    parser.add_argument('--clr', action='store_true',
                        help='Indicate to use cyclic learning rates '
                             '(in this case lr given in config is ignored). '
                             'Requires --training.')
    parser.add_argument('-m', '--models', type=str,
                        help='<path/to> directory with trained model(s). '
                             'These models will then be tested on a given '
                             'emp_pdfs '
                             'set. --cfg, --datasets and --test are '
                             'required for this option.')
    parser.add_argument('-c', '--cfg', type=str,
                        help='<path/to> cfg file (.json) or directory '
                             'containing: hyperparameters, emp_pdfs specific '
                             'parameters and parameters determinin the '
                             'structure of the Network. If a directory is '
                             'given, the latest modified json file will be '
                             'used')
    parser.add_argument('-s', '--save', type=str,
                        help='<path/to> directory where trained models and '
                             'result plots will be saved')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the sites of alignments/pairs in the '
                             'first directory specified')
    parser.add_argument('--molecule_type', choices=['DNA', 'protein'],
                        default='protein',
                        help='Specify if you use DNA or protein MSAs')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume training, starting from last epoch in '
                             'each fold.')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='Number of CPUs (for num_workers in DataLoader)')

    args = handle_args(parser)

    if args['model_path'] is not None and not (args['val'] or args['test'] or
                                               args['resume']):
        print_model_performance(load_model(args['model_path']))
    elif args['test']:
        model_test(args)
    else:
        in_data = load_data(args['emp_path'], args['sim_paths'],
                            args['cfg_path'], args['model_path'],
                            args['shuffle'])
        if args['val']:
            validate(args, in_data)
        elif args['train'] or args['resume']:
            train(args, in_data)
        # if args['attr']:
        # attribute(args, in_data)


if __name__ == '__main__':
    main()

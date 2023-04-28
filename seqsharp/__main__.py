import argparse

from .ConvNet import load_model
from .seqsharp_fcts import handle_args, load_data, validate, model_test, train
from .train_eval import print_model_performance


def main():
    sep_line = '-------------------------------------------------------' \
               '---------'

    # -------------------- handling arguments -------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', nargs='*', type=str,
                        help='Specify the <path/to/> directory(s) containing '
                             'simulated alignments (in fasta or phylip format)')
    parser.add_argument('--emp', type=str,
                        help='Specify the <path/to/> directory containing '
                             'empirical alignments (in fasta or phylip format)')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Data collections will be used to train the neural'
                             ' network (specified with --emp and --sim option).'
                             ' Requires --cfg or --model. If a pretrained '
                             'model is given training will be resumed.')
    parser.add_argument('--test', action='store_true',
                        help='Test network on given data collections. '
                             'Requires --models and --sim')
    parser.add_argument('--validate', action='store_true',
                        help='K-fold cross validation with pretrained model.'
                             'Requires --models, --emp and --sim')
    parser.add_argument('--attr', action='store_true',
                        help='Generates attribution maps using validation '
                             'data. '
                             ' Requires --models (only if not --train) and '
                             '--sim')
    parser.add_argument('--clr', action='store_true',
                        help='Use cyclic learning rates (in this case lr given '
                             'in config is ignored). Requires --train.')
    parser.add_argument('-m', '--model', type=str,
                        help='<path/to> directory with pretrained model(s) '
                             '(one per fold).')
    parser.add_argument('-c', '--cfg', type=str,
                        help='<path/to> cfg file (.json) or directory '
                             'containing: hyperparameters for training, '
                             'data specific parameters and parameters for the '
                             'network architecture. Is not required when '
                             '--model is given.')
    parser.add_argument('-s', '--save', type=str,
                        help='<path/to> directory where trained models and '
                             'result plots/tables will be saved.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the sites of alignments in the '
                             'data collections.')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='Number of CPUs to be used.')

    args = handle_args(parser)

    if args['model_path'] is not None and not (args['val'] or args['test']
                                               or args['train']):
        print_model_performance(load_model(args['model_path']))
    else:
        in_data = load_data(args['emp_path'], args['sim_paths'],
                            args['cfg_path'], args['model_path'],
                            args['shuffle'])
        if args['val']:
            validate(args, in_data)
        elif args['train']:
            train(args, in_data)
        elif args['test']:
            model_test(args, in_data)
        # if args['attr']:
        # attribute(args, in_data)


if __name__ == '__main__':
    main()

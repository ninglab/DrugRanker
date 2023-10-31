import argparse

def override_args(args):
    if args.gnn is None:
        args.message_steps = None
        args.pooling = None
        if args.feature_gen is None:
            raise Exception('input features must be provided for models without GNNs')

    return args

def parse_args(args = None):
    parser = argparse.ArgumentParser(
        description = 'Training and Testing Compound Rank Net',
        usage = 'run.py [<args>] [-h | --help]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # General
    parser.add_argument('--cuda', action = 'store_true', help = 'use GPU')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--only_fold', type=int, default=-1)
    parser.add_argument('--setup', type=str, default='LCO', choices=['LCO', 'LRO'])

    # Model and features
    parser.add_argument('--model', type=str, default='listall',
                        choices=['pairpushc', 'listone', 'listall'])
    parser.add_argument('--gnn', type=str,
                        choices=['dmpn'])
    parser.add_argument('-fgen', '--feature_gen', default='morgan_count', choices=['morgan', 'morgan_count',
                        'morgan_tanimoto_bioassay', 'rdkit_2d', 'rdkit_2d_normalized'],
                        help='Without `use_features_only`, this will concat molecule level features to learned gnn emb')
    #parser.add_argument('-fonly', '--use_features_only', action='store_true', help='use only features for baselines')

    # Training and testing
    parser.add_argument('--do_train', action='store_true', help='Train the model?')
    parser.add_argument('--do_comb_eval', action='store_true', help='Evaluating on both train+test drugs in 1st setting')
    #parser.add_argument('--do_test', action='store_true', help='Test the model?')
    parser.add_argument('--do_train_eval', action='store_true', help='Evaluating on training data')
    
    # Data
    parser.add_argument('--data_path', type=str, help='Path to the cell,drug,AUC list')
    parser.add_argument('--smiles_path', type=str, help='Path to the drug-smiles mapping')
    parser.add_argument('--splits_path', type=str, help='Path to the CV split indices')
    parser.add_argument('--genexp_path', type=str, help='Path to the gene expression data',
                        default='/fs/ess/PCON0041/Vishal/DrugRank/data/CCLE/CCLE_expression.csv')

    # Model architecture
    parser.add_argument('-mol_outd', '--mol_out_size', default=50, type=int)
    parser.add_argument('-ae_ind', '--ae_in_size', default=19177, type=int)
    parser.add_argument('-ae_outd', '--ae_out_size', default=128, type=int)
    parser.add_argument('-attn_d', '--attn_dim', default=25, type=int)
    parser.add_argument('-T', '--message_steps', default=3, type=int)
    parser.add_argument('-pool', '--pooling', type=str, default='sum', help='Pooling', \
                    choices=['sum', 'mean', 'max', 'self-attention', 'v-attention', 'all-concat', \
                    'hier-attention', 'hier-sum', 'hier-cat-sum', 'hier-cat-attention', 'hier-cat-v-attention'])
    parser.add_argument('--atom_messages', action = 'store_true')
    parser.add_argument('--update_emb', default='None',
                    choices=['cell-attention', 'list-attention', 'cell+list-attention'], \
                    help='how to convolve comp embeddings to create context vector')
    parser.add_argument('--agg_emb', default='sum', choices=['self', 'concat', 'sum'],
                     help='how to update comp embeddings from context vector')
    parser.add_argument('-score', '--scoring', default='linear', choices=['linear', 'mlp'], help='Scoring fn')

    # Training hyperparameters
    parser.add_argument('-gstep', '--gradient_steps', default=16, type=int, help='Gradient accumulation every `gstep` steps')
    parser.add_argument('-sur', '--surrogate', default='tcbb', choices=['logistic', 'tcbb'], type=str)
    parser.add_argument('-reg', '--regularization', default=None, type=float, help='Norm regularization')
    parser.add_argument('-drop', '--dropout', default=0, type=float, help='Random dropouts')
    parser.add_argument('-del', '--delta', default=5, type=float, \
                    help='Percentile to decide sensitivity threshold for creating ranked pairs')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-e', '--max_iter', default=100, type=int, help='Num of epochs')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch size')
    #parser.add_argument('-clip', '--grad_clip', default=100, type=int, help='Allow gradient clipping')

    # Loss hyperparameters
    parser.add_argument('-npair', '--num_pairs', default=0, type=int, help='Number of pairs per sensitive drug per cell line')
    parser.add_argument('-mix', '--mixture', action='store_true', help='Whether to input ordered or mix ordered pairs')
    parser.add_argument('-A', '--alpha', default=0.5, type=float, help='Trade-off for TCBB loss (equ 6)')
    parser.add_argument('-B', '--beta', default=0.1, type=float, help='Trade-off for TCBB loss (equ 6)')
    #parser.add_argument('-G', '--gamma', default=0.5, type=float, help='Trade-off for margin-based loss in clustering')
    parser.add_argument('-M', default=1, type=float, help='Softmax temperature')
    parser.add_argument('-sample_list', default=0, help='sample list for each query in listwise ranking') # type: ignore

    # Pretrained models
    parser.add_argument('--pretrained_ae', action='store_true', help='whether using pretrained cell line AE')
    parser.add_argument('-ae_path', '--trained_ae_path', type=str, help='Checkpoint path for the AE model')

    # Saving and logging
    parser.add_argument('--log_steps', default=5, type=int, help='log evaluation results every 5 epochs')
    parser.add_argument('--save_path', default='../tmp/', type=str)
    parser.add_argument('--checkpointing', action='store_true', help='Whether to save model every `log_steps` epochs')
    
    # Misc
    #parser.add_argument('-cluster', '--cluster', action='store_true', help='additionally optimize hinge loss')
    parser.add_argument('-classp', '--classify_pairs', action='store_true', help='additionally classify if two compounds in the pair has same class')
    parser.add_argument('-classc', '--classify_cmp', action='store_true', help='additionally classify if comp is +/-')
    parser.set_defaults(do_train=True, cuda=True)

    #parser.set_defaults(data_path='data/ctrpv2/LRO/aucs.txt',
    #                    smiles_path='data/ctrpv2/cmpd_smiles.txt',
    #                    splits_path='data/ctrpv2/LRO/',
    #                    feature_gen='morgan_count',
    #                    gnn='dmpn',
    #                    setup='LRO',
    #                    model='pairpushc', num_pairs=10)
    #parser.set_defaults(pretrained_ae=True)
    #parser.set_defaults(trained_ae_path='/fs/ess/PCON0041/Vishal/DrugRank/expts/ae/LRO/all_bs_64_outd_128/model.pt')
    
    temp = parser.parse_args(args)
    args = override_args(temp)
    return args

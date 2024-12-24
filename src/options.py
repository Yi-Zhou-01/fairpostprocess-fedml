import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of global training FL rounds for FedAvg")
    parser.add_argument('--num_users', type=int, default=100,
                        help="total number of clients")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients used in each FL round')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="local training epochs on each clients per global round")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for FedAvg')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='plain', help='model name')
    # parser.add_argument('--kernel_num', type=int, default=9,
    #                     help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to \
    #                     use for convolution')
    # parser.add_argument('--num_channels', type=int, default=1, help="number \
    #                     of channels of imgs")
    # parser.add_argument('--norm', type=str, default='batch_norm',
    #                     help="batch_norm, layer_norm, or None")
    # parser.add_argument('--num_filters', type=int, default=32,
    #                     help="number of filters for conv nets -- 32 for \
    #                     mini-imagenet, 64 for omiglot.")
    # parser.add_argument('--max_pool', type=str, default='True',
    #                     help="Whether use max pooling rather than \
    #                     strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='adult', \
                        help="dataset name")
    parser.add_argument('--num_classes', type=int, default=10, \
                        help="number of classes (clients)")
    

    parser.add_argument('--optimizer', type=str, default='sgd', help=" \
                        training optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--partition_idx', type=int, default=1, help='partition file path')

    # Choose which fl algorithm to perform
    parser.add_argument('--fl_new', type=bool, default=True, help='fl algorithm: new')

    parser.add_argument('--fl_fairfed', type=bool, default=True, help='fl algorithm: FairFed')

    parser.add_argument('--idx', type=int, help='Experiment index.')

    parser.add_argument('--beta', type=float, default=0.3, help='Beta parameter for Fairfed, i.e. Fairfed fairness budget.')
    
    parser.add_argument('--local_test_ratio', type=float, default=0.2, help='Local test set ratio')

    parser.add_argument('--plot_tpfp', type=bool, default=True, help='plot tpr and fpr')

    parser.add_argument('--gpu', default=None, help='Default no gpu, if gpu: pass True')

    parser.add_argument('--fairfed_ep', type=int, default=20, help='Global training round for FairFed')
    
    parser.add_argument('--debias', type=str, default="ppft", help='Local debias approaches for new fl_new: "pp" for post-processing and "ft" for final layer fine-tuning')
    
    parser.add_argument('--ft_alpha', type=float, default=1.0, help='parameter alpha used to calculate loss in final layer fine-tuning: loss = ft_alpha2 * loss + ft_alpha * loss_fairness')
    
    parser.add_argument('--ft_alpha2', type=float, default=1.0, help='parameter alpha2 used to calculate loss in final layer fine-tuning: loss = ft_alpha2 * loss + ft_alpha * loss_fairness')
    
    parser.add_argument('--ft_lr', type=float, default=5e-3, help='learning rate for final layer fine tuning')
    
    parser.add_argument('--ft_ep', type=int, default=5, help='number of final layer fine-tuning epochs')

    parser.add_argument('--ft_bs', type=int, default=256, help='training batch size for final layer fine-tuning')
    
    parser.add_argument('--local_split', type=str, default="", help='File path if use saved local client train/test split; Empty if want a newly generated one.')

    parser.add_argument('--platform', type=str, default="", help='The platform that the code will run on, choose from "local", "kaggle", "colab", "azure"')

    parser.add_argument('--rep', type=int, default=0, help='run exp with the same setting with # of rep')

    # parser.add_argument('--crop', type=int, default=0, help='run exp with the same setting with # of rep')
    
    # parser.add_argument('--example_folder', default=None, help='Example folder for collecting results')

    # parser.add_argument('--rounds_ls', nargs="*", type=int, default=[], help='Example folder for collecting results')

    parser.add_argument('--save_avg_model', type=str, default=None, help='Example folder for collecting results')

    parser.add_argument('--use_saved_model', type=str, default="", help='Example folder for collecting results')

    parser.add_argument('--fair_rep', type=bool, default=None, help='Example folder for collecting results')

    parser.add_argument('--lbd', type=float, default=1.0, help='Parameter for fair representation')

    parser.add_argument('--threshold', type=float, default=0.5, help='The classification threshold, default=0.5')

    parser.add_argument('--ba', type=int, default=0, help='if use balanced accuracy or normal accuracy. 0: normal accuracy, 1: balanced accuracy')

    args = parser.parse_args()
    return args


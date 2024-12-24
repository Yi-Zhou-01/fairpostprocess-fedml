import copy
import torch

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



def get_fpr_diff(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / (torch.sum(a) + 1e-5) 
                    - torch.sum(p * (1 - y) * (1 - a)) / (torch.sum(1 - a) + 1e-5))
    return fpr

def get_tpr_diff(p, y, a):
    tpr = torch.abs(torch.sum(p * y * a) / (torch.sum(a) + 1e-5) 
                - torch.sum(p * y * (1 - a)) / (torch.sum(1 - a) + 1e-5))
    
    return tpr

def equalized_odds_diff(p, y, a):
    return torch.mean(torch.tensor([get_fpr_diff(p, y, a), get_tpr_diff(p, y, a)]))





def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def check_train_test_split(num_clients, pred_train_dic, pred_test_dic, save_dir=None):

    lines = []
    for i in range(num_clients):

        # Calculate tt;  tf; ff; ft
        # tt: Y=1 A=0
        test_len = len(pred_test_dic["labels"][i])
        train_len = len(pred_train_dic["labels"][i])

        YA_11 = ((pred_test_dic["labels"][i]) * (pred_test_dic["s_attr"][i]) )
        YA_10 = ((pred_test_dic["labels"][i]) * (1-pred_test_dic["s_attr"][i]) )
        YA_00 = ((1-pred_test_dic["labels"][i]) * (1-pred_test_dic["s_attr"][i]) )
        YA_01 = ((1-pred_test_dic["labels"][i]) * (pred_test_dic["s_attr"][i]) )

        YA_11_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["s_attr"][i]) )
        YA_10_tr = ((pred_train_dic["labels"][i]) * (1-pred_train_dic["s_attr"][i]) )
        YA_00_tr = ((1-pred_train_dic["labels"][i]) * (1-pred_train_dic["s_attr"][i]) )
        YA_01_tr = ((1-pred_train_dic["labels"][i]) * (pred_train_dic["s_attr"][i]) )

        tp_p = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_fedavg"][i]) * (pred_test_dic["s_attr"][i]) )
        tp_unp = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_fedavg"][i]) * (1-pred_test_dic["s_attr"][i]) )
        tp_p_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_fedavg"][i]) * (pred_train_dic["s_attr"][i]) )
        tp_unp_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_fedavg"][i]) * (1-pred_train_dic["s_attr"][i]) )
        
        tp_p_ = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_pp"][i]) * (pred_test_dic["s_attr"][i]) )
        tp_unp_ = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_pp"][i]) * (1-pred_test_dic["s_attr"][i]) )
        tp_p_tr_ = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_pp"][i]) * (pred_train_dic["s_attr"][i]) )
        tp_unp_tr_ = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_pp"][i]) * (1-pred_train_dic["s_attr"][i]) )
       



        # Number of prediction: 0 -> 1
        # Test Set
        flip_01 = (1 - pred_test_dic["pred_labels_fedavg"][i]) * (pred_test_dic["pred_labels_pp"][i])
        flip_10 = (pred_test_dic["pred_labels_fedavg"][i]) * (1 - pred_test_dic["pred_labels_pp"][i])
        
        flip_01_p = flip_01 * (pred_test_dic["s_attr"][i])
        flip_01_unp = flip_01 * (1-pred_test_dic["s_attr"][i])

        flip_10_p = flip_10 * (pred_test_dic["s_attr"][i])
        flip_10_unp = flip_10 * (1 - pred_test_dic["s_attr"][i])

        flip_01_p_Y1 = flip_01_p * (pred_test_dic["labels"][i])
        flip_01_unp_Y1 = flip_01_p * (pred_test_dic["labels"][i])

        flip_10_p_Y0 = flip_10_p * (1 -pred_test_dic["labels"][i])
        flip_10_unp_Y0 = flip_10_unp * (1 - pred_test_dic["labels"][i])


        lines.append("******** Client #{} ********".format(i+1))
        lines.append("           Test (%)   -   Train (%)")
        lines.append("#Samples:  {}        -    {}".format(test_len, train_len))
        lines.append("YA_11:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_11), sum(YA_11)/test_len, sum(YA_11_tr),sum(YA_11_tr)/train_len ))
        lines.append("YA_00:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_00), sum(YA_00)/test_len, sum(YA_00_tr),sum(YA_00_tr)/train_len))
        lines.append("YA_10:     {:n} ({:.0%})      {:n} ({:.0%})".format(sum(YA_10), sum(YA_10)/test_len, sum(YA_10_tr),sum(YA_10_tr)/train_len ))
        lines.append("YA_01:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_01), sum(YA_01)/test_len, sum(YA_01_tr),sum(YA_01_tr)/train_len ))
        lines.append("-")
        lines.append("flip_01_p:  {:n}, Y=1: {:n}".format(sum(flip_01_p), sum(flip_01_p_Y1)))
        lines.append("flip_01_up: {:n}, Y=1: {:n}".format(sum(flip_01_unp), sum(flip_01_unp_Y1)))
        lines.append("flip_10_p:  {:n}, Y=0: {:n}".format(sum(flip_10_p), sum(flip_10_p_Y0)))
        lines.append("flip_10_up: {:n}, Y=0: {:n}".format(sum(flip_10_unp), sum(flip_10_unp_Y0)))
        lines.append("-")
        lines.append("Before post-processing ...")
        lines.append("             Test (%)    -    Train (%)")
        try:
            lines.append("TPR  p:     {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_p),sum(YA_11), sum(tp_p)/sum(YA_11),\
                                                                                        sum(tp_p_tr),sum(YA_11_tr),sum(tp_p_tr)/sum(YA_11_tr) ))
            lines.append("TPR  unp:   {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_unp), sum(YA_10), sum(tp_unp)/sum(YA_10),\
                                                                                        sum(tp_unp_tr),sum(YA_10_tr), sum(tp_unp_tr)/sum(YA_10_tr) ))
            lines.append("TPR Diff         ({:.0%})          ({:.0%})".format(sum(tp_unp)/sum(YA_10)-sum(tp_p)/sum(YA_11),\
                                                                            sum(tp_unp_tr)/sum(YA_10_tr)-sum(tp_p_tr)/sum(YA_11_tr) ))
            lines.append("-")   
            lines.append("After post-processing ...")
            lines.append("             Test (%)    -    Train (%)")
            lines.append("TPR  p:     {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_p_),sum(YA_11), sum(tp_p_)/sum(YA_11),\
                                                                                        sum(tp_p_tr_),sum(YA_11_tr),sum(tp_p_tr_)/sum(YA_11_tr) ))
            lines.append("TPR  unp:   {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_unp_), sum(YA_10), sum(tp_unp_)/sum(YA_10),\
                                                                                        sum(tp_unp_tr_),sum(YA_10_tr), sum(tp_unp_tr_)/sum(YA_10_tr) ))
            lines.append("TPR Diff         ({:.0%})          ({:.0%})".format(sum(tp_unp_)/sum(YA_10)-sum(tp_p_)/sum(YA_11),\
                                                                         sum(tp_unp_tr_)/sum(YA_10_tr)-sum(tp_p_tr_)/sum(YA_11_tr) ))
        except:
            lines.append("Scalar error!!")
        lines.append("-")   

    
    for line in lines:
        print(line)
    
    if save_dir is not None:
        with open(save_dir + '/print_stats.txt', 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    # with open('somefile.txt', 'a') as the_file:
    # the_file.write('Hello\n')



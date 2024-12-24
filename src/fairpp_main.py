
import os
import time
import pickle
import numpy as np
import pandas as pd
import json
import torch
from tensorboardX import SummaryWriter

import options
import dataset

from utils import exp_details
import update
import plot
import pickle
import utils
import models
import algorithm
import options
import dataset

# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
# from aif360.algorithms.postprocessing import EqOddsPostprocessing



def main():
    start_time = time.time()

    args = options.args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'
    print("Using devide: ", device)
 

    # Create folder to save training info
    if args.platform=="kaggle":
        save_to_root = "/kaggle/working"
    elif args.platform=="colab":
        save_to_root = "/content/drive/MyDrive/Fair_FL_new/save"
    elif args.platform=="azure":
        save_to_root = os.getcwd() + '/save'
    else:
        save_to_root =  os.getcwd() + '/save'

    all_fl = ""
    if args.fl_new:
        all_fl = all_fl + "new"
    if args.fl_fairfed:
        all_fl = all_fl + "fairfed"    
    statistics_dir = save_to_root+'/statistics/{}/{}_{}_{}_{}_frac{}_client{}_lr{}_ftlr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}_bs{}_ftbs{}_fta_{}{}_{}'.\
        format(args.idx, all_fl, args.debias, args.dataset, args.model, args.frac, args.num_users,
               args.lr, args.ft_lr, args.partition_idx, args.beta, args.epochs, args.local_ep, args.fairfed_ep, args.ft_ep, args.local_bs, args.ft_bs, args.ft_alpha,args.ft_alpha2, args.rep)    # <------------- iid tobeadded

    os.makedirs(statistics_dir, exist_ok=True)


    # define paths
    logger = SummaryWriter(statistics_dir + '/logs')

    with open(statistics_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    print("time: ", int(time.time() - start_time), int(time.time()))
    print("Getting dataset ... ")
    train_dataset, test_dataset, user_groups = dataset.get_dataset(args)

    print("time: ", int(time.time() - start_time), int(time.time()))


    # Split train/test for all local dataset
    print("Getting train/test split ... ")
    if args.local_split == "":
        local_set_ls = []
        for i in range(args.num_users):
            local_idxs = user_groups[i]
            # print("local_idxs: ", len(local_idxs))
            # print(local_idxs)
            # print(local_idxs)
            local_dataset = update.LocalDataset(train_dataset, local_idxs, test_ratio=args.local_test_ratio)
            local_set_ls.append(local_dataset)
            print("New local train/test split generated for Client {}: Train: {} | Test: {} | Total: {}".format(
                i, len(local_dataset.train_set_idxs), len(local_dataset.test_set_idxs), len(local_idxs)))
    else:
        with open(args.local_split, 'rb') as inp:
            local_set_ls = pickle.load(inp)
            print("Using saved local train/test split in: ", args.local_split)

    if args.fair_rep:
        train_dataset_rep = dataset.fair_rep_dataset(train_dataset, local_set_ls, args.lbd)
        # train_dataset = train_dataset_rep
    
    # BUILD MODEL
    print("args.use_saved_model: ", args.use_saved_model)
    img_size = train_dataset[0][0].shape
    if args.use_saved_model != "":
        global_model  = torch.load(args.use_saved_model)
        print("Using saved FedAvg model: ", args.use_saved_model)
    else:
        global_model=models.get_model(args, img_size=img_size)


    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []


    stat_keys = []
    set_split = ["train", "test"]
    local_metrics = ["acc", "eod"]

    if args.fl_new:
        stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in local_metrics]
    # if args.fl_avg:
        stat_keys += [ss+"_"+lm+"_"+"fedavg" for ss in set_split for lm in local_metrics]
        if args.fair_rep:
            stat_keys += [ss+"_"+lm+"_"+"new_rep" for ss in set_split for lm in local_metrics]
            stat_keys += [ss+"_"+lm+"_"+"fedavg_rep" for ss in set_split for lm in local_metrics]
        if args.plot_tpfp:
            stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in  ["tpr", "fpr"]]
            stat_keys += [ss+"_"+lm+"_"+"fedavg" for ss in set_split for lm in  ["tpr", "fpr"]]
            if args.fair_rep:
                stat_keys += [ss+"_"+lm+"_"+"fedavg_rep" for ss in set_split for lm in  ["tpr", "fpr"]]
    if args.fl_fairfed:
        stat_keys += [ss+"_"+lm+"_"+"fairfed" for ss in set_split for lm in local_metrics]
        stat_keys += [ss+"_"+lm+"_"+"fairfed_rep" for ss in set_split for lm in local_metrics]
        stat_keys += [ss+"_"+lm+"_"+"fairfed" for ss in set_split for lm in  ["tpr", "fpr"]]
        stat_keys += [ss+"_"+lm+"_"+"fairfed_rep" for ss in set_split for lm in  ["tpr", "fpr"]]


    stat_dic = {k: np.zeros(args.num_users) for k in stat_keys}
    pred_train_dic = {}
    pred_test_dic = {}

    print("time: ", int(time.time() - start_time), int(time.time()))
    time_point_1 = time.time()
    if args.fl_new and ( args.use_saved_model == "") and (args.epochs != 0):
        print("********* Start FedAvg/New Training **********")

        global_model = algorithm.fedavg_train(args, global_model, local_set_ls, train_dataset, statistics_dir, user_groups, logger)

        global_model, stat_dic, pred_train_dic, pred_test_dic = algorithm.fedavg_inference(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger)
        
        print("check stat_dic: ")
        print(stat_dic["train_acc_fedavg"])

        if args.fair_rep:
            print("********* Start [Fair Rep] FedAvg/New Training **********")
            global_model_rep = algorithm.fedavg_train(args, global_model, local_set_ls, train_dataset_rep, statistics_dir, user_groups, logger)

            global_model_rep, stat_dic, pred_train_dic_rep, pred_test_dic_rep = algorithm.fedavg_inference(args, global_model_rep, local_set_ls, train_dataset_rep, stat_dic, user_groups, logger, fair_rep=True)

        # Save trained model
        if args.save_avg_model:
            torch.save(global_model, statistics_dir+"/fedavg_model.pt")
            print("FedAvg trained model saved in: ", (statistics_dir+"/fedavg_model.pt"))

        print("time: ", int(time.time() - start_time), int(time.time()))
        time_point_2 =  time.time()
        # Post-processing approach
        if "pp" in args.debias:
            # Apply post-processing locally at each client:
            print("******** Start post-processing ******** ")
            pred_train_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 
            pred_test_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 

            stat_dic, pred_train_dic, pred_test_dic = algorithm.post_processing(args, stat_dic, pred_train_dic, pred_test_dic, fair_rep=False)
           
            time_point_3 =  time.time()
            if args.fair_rep:
                pred_train_dic_rep['pred_labels_pp'] = []
                pred_test_dic_rep['pred_labels_pp'] = []
                stat_dic, pred_train_dic_rep, pred_test_dic_rep = algorithm.post_processing(args, stat_dic, pred_train_dic_rep, pred_test_dic_rep, fair_rep=True)

        print("time: ", int(time.time() - start_time), int(time.time()))
        # Apply final-layer fine-tuning
    ft_keys = ['test_acc_new_ft', 'test_eod_new_ft', 'test_tpr_new_ft','test_fpr_new_ft', \
                    'train_acc_new_ft','train_eod_new_ft','train_tpr_new_ft', 'train_fpr_new_ft']
    for k in ft_keys:
        stat_dic[k] = np.zeros(args.num_users) 
    
    time_point_4 =  time.time()

    if ("ft" in args.debias) and  (args.ft_ep != 0):
        
        print("******** Final layer fine-tuning ******** ")
        
        stat_dic = algorithm.fine_tuning(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir, fair_rep=False)
    time_point_5 =  time.time()

    
    print("time: ", int(time.time() - start_time), int(time.time()))
    if args.fl_fairfed and (args.fairfed_ep != 0):
        print("********* Start FairFed Training **********")

        print("time: ", int(time.time() - start_time), int(time.time()))

        # Reinitialize model
        img_size = train_dataset[0][0].shape
        global_model=models.get_model(args, img_size=img_size)

        global_model, stat_dic = algorithm.fairfed_train(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir)

        if args.fair_rep:
            print("********* Start [Fair Rep] FairFed Training **********")
            img_size = train_dataset_rep[0][0].shape
            global_model_rep=models.get_model(args, img_size=img_size)
            global_model_rep, stat_dic = algorithm.fairfed_train(args, global_model_rep, local_set_ls, train_dataset_rep, stat_dic, user_groups, logger, statistics_dir, fair_rep=True)
         
    time_point_6=time.time()

    with open(statistics_dir +"/time.txt", "a") as w_file:
        try:
            w_file.write("******** FedAvg ********\n")
            w_file.write(str(time_point_2-time_point_1) + "\n")
            w_file.write("******** FedAvg + PP ********\n")
            w_file.write(str(time_point_3-time_point_1) + "\n")           
            w_file.write("******** FedAvg + FT ********\n")
            w_file.write(str((time_point_2-time_point_1) + (time_point_5-time_point_4)) + "\n")    
            w_file.write("******** FairFed ********\n")
            w_file.write(str(time_point_6-time_point_5) + "\n")    
        except:
            print("No time records available")    

    print("time: ", int(time.time() - start_time), int(time.time()))
    print("Start saving...")
    stat_df = pd.DataFrame(stat_dic)
    stat_df.to_csv(statistics_dir + "/stats.csv")

    with open(statistics_dir+'/client_datasets.pkl', 'wb') as outp:
        pickle.dump(local_set_ls, outp, pickle.HIGHEST_PROTOCOL)
    
    with open(statistics_dir+'/pred_train_dic.pkl', 'wb') as outp:
        pickle.dump(pred_train_dic, outp, pickle.HIGHEST_PROTOCOL)
    
    with open(statistics_dir+'/pred_test_dic.pkl', 'wb') as outp:
        pickle.dump(pred_test_dic, outp, pickle.HIGHEST_PROTOCOL)
    
    print("Exp stats saved in dir: ", statistics_dir)

    fig_title = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)
    plot_file_all = statistics_dir + "/all_acc_eod_plot.png"
    
    plot.plot_multi_exp(stat_dic, args, new=args.fl_new, plot_tpfp=args.plot_tpfp, 
                        plot_fed = (args.fairfed_ep != 0),
                        plot_ft = (args.ft_ep != 0),
                        title=fig_title, save_to=plot_file_all)

    # Saving the objects train_loss and train_accuracy:
    file_name = statistics_dir + '/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print(file_name, " saved!")

    print('\n Total Run Time: {0:0.4f} s'.format(time.time()-start_time))

    # Check statistics of training set
    utils.check_train_test_split(args.num_users, pred_train_dic, pred_test_dic, save_dir=statistics_dir)




if __name__ == '__main__':
    main()
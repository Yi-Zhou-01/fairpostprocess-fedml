
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset

import h5py
# import cv2
from torchvision import transforms
from PIL import Image
import sys
import fair_var


class AdultDataset(Dataset):

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):
        self.target = "income"
        self.s_attr = "sex_1"
        self.name = "adult"

        if X is None:
            if df is not None:
                self.df = df
            else:
                self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            
            if crop:
                self.df = self.df[:crop]

            if subset:
                self.df = self.df[self.df.index.isin(subset)]

            self.X = self.df.drop([self.target], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = self.df[self.target].to_numpy().astype(np.float32)
            self.a = self.df[self.s_attr].to_numpy().astype(np.float32)

        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
    
    def standardlize_X(self, X_data):
        # Define the columns to standardize
        columns_to_standardize = list(range(25, len(self.X[0])))
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data


class CompasDataset(Dataset):
    """Compas dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):

        self.target = "two_year_recid"
        self.s_attr = "race"
        self.name = "compas"

        if X is None:
            if df is not None:
                self.df = df
            else:
                self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            
            if crop:
                self.df = self.df[:crop]

            if subset:
                self.df = self.df[self.df.index.isin(subset)]
            
            self.X = self.df.drop([self.target, ], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = self.df[self.target].to_numpy().astype(np.float32)
            self.a = self.df[self.s_attr].to_numpy().astype(np.float32)
        
        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()


        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
        
    
    def standardlize_X(self, X_data):
        # # Define the columns to standardize
        columns_to_standardize = [5]
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data


class CompasBinaryDataset(Dataset):
    """Compas dataset with binary attribute: race."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):

        self.target = "two_year_recid"
        self.s_attr = "race"
        self.name = "compas-binary"

        if X is None:
            if df is not None:
                self.df = df
            else:
                self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            
            if crop:
                self.df = self.df[:crop]

            if subset:
                self.df = self.df[self.df.index.isin(subset)]
            
            self.X = self.df.drop([self.target, ], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = self.df[self.target].to_numpy().astype(np.float32)
            self.a = self.df[self.s_attr].to_numpy().astype(np.float32)
        
        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
        
    
    def standardlize_X(self, X_data):
        # # Define the columns to standardize
        columns_to_standardize = [5]
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data


class WCLDDataset(Dataset):
    """WCLD dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):

        self.target = "recid_180d"
        self.s_attr = "sex"
        self.name = "wcld"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            if crop:
                df = df[:crop]
            if subset:
                df = df[df.index.isin(subset)]
        
            self.X = df.drop([self.target, self.s_attr], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
        

        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
    
    def standardlize_X(self, X_data):
        # Define the columns to standardize
        # columns_to_standardize = [26, 27, 28, 29, 30, 31]
        columns_to_standardize = list(range(9)) # standardize all
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data


class PTBDataset(Dataset):
    """PTB-xl dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, platform=None, df=None, crop=None, subset=None, traces=True):
        self.target = "NORM"
        # self.s_attr = "sex"
        self.s_attr = "age>60"
        self.name = "ptb-xl"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)#[:1000] #.drop("Unnamed: 0", axis=1)
                columns = ["record_id", "ecg_id","patient_id","age","sex", "NORM", "age>60"]
                df = df.loc[:, df.columns.isin(columns)]
            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]
            
            if traces:
                if platform=="kaggle":
                    path_to_traces = "/kaggle/input/ptb-xl/ptbxl_all_clean_new_100hz.hdf5"
                elif platform=="azure":
                    path_to_traces = os.getcwd() + "/data/ptb-xl/ptbxl_all_clean_new_100hz.hdf5"
                else:
                    path_to_traces = os.getcwd() + "/data/ptb-xl/ptbxl_all_clean_new_100hz.hdf5"
                f = h5py.File(path_to_traces, 'r')
                self.X = np.array(f["tracings"][:])#[:1000] 
            else:
                self.X = df["record_id"].to_numpy().astype(np.float32)
            

            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)


        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]




class NIHDataset(Dataset):
    """NIH Chest X-Ray dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, platform=None, df=None, transform=None, crop=None, subset=None, traces=True):

        self.target = "Disease"
        self.s_attr = "Patient Gender"
        self.name = "nih-chest"
        self.transform = transform

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)#[:1000] #.drop("Unnamed: 0", axis=1)
                columns = ["Image Index", "Patient Gender","Disease","Multi_label", "folder_name", "kaggle_path"]
                df = df.loc[:, df.columns.isin(columns)]

            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]
            
            if traces:
                if platform=="kagle":
                    self.path_to_traces = "/kaggle/input/data"
                else:
                    self.path_to_traces =  os.getcwd() + "/data/nih-chest/png"
  
            else:
                self.path_to_traces = None
            
            self.X = df["Image Index"].to_numpy()
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
            self.kaggle_path = df["kaggle_path"].to_numpy()

        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            print("list index ")

        img = Image.open(self.kaggle_path[idx]).convert('RGB')


        if self.transform:
            img = self.transform(img)

        if s_att:
            return [img, self.y[idx], self.a[idx]]
        else:
            return [img, self.y[idx]]


class NIHDataset2(Dataset):
    """NIH dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, platform="", transform=None, df=None, crop=None, subset=None, traces=True):
        """Initializes instance of class Compas Dataset.
        """
        self.target = "Disease"
        self.s_attr = "Patient Gender"
        self.name = "nih-chest"
        self.transform = transform

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)#[:1000] #.drop("Unnamed: 0", axis=1)
                # print("self.df: ", self.df[:5])
                columns = ["Image Index", "Patient Gender","Disease","Multi_label", "folder_name"]
                df = df.loc[:, df.columns.isin(columns)]
            
            if platform == "":
                crop = 500

            if subset:
                df = df[df.index.isin(subset)]

            if traces:
                if platform=="kaggle":
                    # self.path_to_traces = "/kaggle/input/nih-chest/nih_chest_100_256_rgb_xx3_int_h5.hdf5"
                    self.path_to_traces = "/kaggle/input/nih-chest/nih_chest_100_256_gray_xx3_int_h5.hdf5"
                elif platform=="colab":
                    self.path_to_traces = "/content/drive/MyDrive/Fair_FL_new/data/nih-chest/nih_chest_100%_256_gray_xx3_int_h5.hdf5"
                elif platform=="azure":
                    self.path_to_traces =  os.getcwd() + "/data/nih-chest/nih_chest_100%_256_gray_xx3_int_h5.hdf5"
                else:
                    self.path_to_traces =  os.getcwd() + "/data/nih-chest/nih_chest_10%_256_gray_xx3_int_h5.hdf5"
                
                print("getting f...")
                f = h5py.File(self.path_to_traces, 'r')
                self.X = np.array(f["images"][:]) #.astype(np.float32) #[:1000] 
                print("shape self.X: ", self.X.shape)
                print("sys.getsizeof: ", sys.getsizeof(self.X))
            else:
                self.path_to_traces = None
            
            if crop:
                df = df[:crop]
                self.X = self.X[:crop]
            
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)

        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            print("list index ")
        
        img = self.X[idx].astype(np.float32)/255
        img = np.repeat(img, 3, axis=-1)

        if self.transform:
            img = self.transform(img)

        if s_att:
            return [img, self.y[idx], self.a[idx]]
        else:
            return [img, self.y[idx]]



class NIHEffDataset(Dataset):
    """NIH dataset with target label = Effusion."""

    def __init__(self, csv_file, X=None, y=None, a=None, platform="", transform=None, df=None, crop=None, subset=None, traces=True):
        self.target = "Disease"
        # self.s_attr = "age>50"
        self.s_attr = "Patient Gender"
        self.name = "nih-chest-eff"
        self.transform = transform

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)
                columns = ["Image Index", "Patient Gender","Disease","Multi_label", "folder_name", "age>50", "age>60"]
                df = df.loc[:, df.columns.isin(columns)]

            if subset:
                df = df[df.index.isin(subset)]

            if traces:
                if platform=="kaggle":
                    self.path_to_traces = "/kaggle/input/nih-chest/nih_chest_100_256_gray_xx3_int_h5.hdf5"
                elif platform=="colab":
                    self.path_to_traces = "/content/drive/MyDrive/Fair_FL_new/data/nih-chest/nih_chest_100%_256_gray_xx3_int_h5.hdf5"
                elif platform=="azure":
                    self.path_to_traces =  os.getcwd() + "/data/nih-chest-eff/nih_chest_eff_256_gray_xx3_int_h5.hdf5"
                else:
                    self.path_to_traces =  os.getcwd() + "/data/nih-chest/nih_chest_eff_256_gray_xx3_int_h5.hdf5"
                
                print("getting f...")
                self.X = np.zeros(len(df))
                print("shape self.X: ", self.X.shape)
                print("sys.getsizeof: ", sys.getsizeof(self.X))
            else:
                self.path_to_traces = None
            
            if crop:
                df = df[:crop]
                self.X = self.X[:crop]
            
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
 

        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            print("list index ")
     
        img = self.X[idx].astype(np.float32)/255
        img = np.repeat(img, 3, axis=-1)
  
        if self.transform:
            img = self.transform(img)

        if s_att:
            return [img, self.y[idx], self.a[idx]]
        else:
            return [img, self.y[idx]]


def fair_rep_dataset(dataset, local_set_ls, lbd):
    dataset_df = dataset.df
    
    new_df_ls = []
    
    for i in range(len(local_set_ls)):
        dataset_df_train = dataset_df[dataset_df.index.isin(local_set_ls[i].train_set_idxs)]
        dataset_df_test = dataset_df[dataset_df.index.isin(local_set_ls[i].test_set_idxs)]
    
        feature_list = list(dataset_df.columns)
        feature_list.remove(dataset.target)

        protect_list = [dataset.s_attr]
        outcome = dataset.target

        df_train_tmp = pd.DataFrame()
        df_test_tmp = pd.DataFrame()
        for col in feature_list:
            if col == dataset.s_attr: continue
            df_train_tmp[col] = fair_var.gen_latent_nonparam_regula(dataset_df_train[feature_list], protect_list, col, lbd)
            df_test_tmp[col] = fair_var.gen_latent_nonparam_regula(dataset_df_test[feature_list], protect_list, col, lbd)
    
        for column in protect_list:
            df_train_tmp[column] = dataset_df_train[column].values
            df_test_tmp[column] = dataset_df_test[column].values
            
        df_train_tmp[outcome] = dataset_df_train[outcome].values
        df_test_tmp[outcome] = dataset_df_test[outcome].values
        
        df_train_tmp.index = dataset_df_train.index
        df_test_tmp.index = dataset_df_test.index
        
        print(df_train_tmp[:5])
        new_df_ls.append(df_train_tmp)
        new_df_ls.append(df_test_tmp)
    
    converted_df = pd.concat(new_df_ls, sort=False).sort_index()

    if dataset.name == "adult":
        fair_dataset = AdultDataset(csv_file="", df=converted_df)
    elif dataset.name == "compas-binary":
        fair_dataset = CompasBinaryDataset(csv_file="", df=converted_df)
    elif dataset.name == "compas":
        fair_dataset = CompasDataset(csv_file="", df=converted_df)

    return fair_dataset



def get_bld_dataset_w_pred(a, pred_labels):

    new_df = pd.DataFrame()
    new_df["a"] = list(a)
    new_df["y"] =  list(pred_labels)
    bld_set = BinaryLabelDataset(df=new_df, label_names=["y"], protected_attribute_names=["a"])

    return bld_set



def get_partition(platform, p_idx, dataset="adult"):

    if dataset == "nih-chest-h5":
        dataset = "nih-chest"

    if platform=="kaggle":
        path_root = "/kaggle/input/" + dataset + '/partition/' + str(p_idx)
    elif platform=="colab":
        path_root = "/content/drive/MyDrive/Fair_FL_new/data/" + dataset + '/partition/' + str(p_idx)
    elif platform=="azure":
        path_root = 'data/' + dataset + '/partition/' + str(p_idx)
    else:
        path_root =  os.getcwd() + "/data/" + dataset + '/partition/' + str(p_idx)
    file_ls = os.listdir(path_root)
    partition_file_ls = [file for file in file_ls if '.npy' in file]
    partition_file = path_root + '/' + partition_file_ls[0]

    return partition_file


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.platform=="kaggle":
        data_path = "/kaggle/input"
    elif args.platform=="colab":
        data_path = "/content/drive/MyDrive/Fair_FL_new/data"
    elif args.platform=="azure":
        data_path = os.getcwd()+"/data"
    else:
        data_path = os.getcwd()+"/data"


    if  args.dataset == 'adult':

        csv_file_train = data_path+'/adult/adult_all_33col.csv'
        csv_file_test =  data_path+'/adult/adult_all_33col_20test_0.csv'
        csv_file_val =  data_path+'/adult/adult_all_33col_10val_0.csv'

        train_dataset = AdultDataset(csv_file_train)
        test_dataset = AdultDataset(csv_file_test)
        partition_file = get_partition(args.platform, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    
    elif args.dataset == 'compas':

        csv_file_train =data_path+'/compas/compas_encoded_all_new_encoded.csv'

        train_dataset = CompasDataset(csv_file_train)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.platform, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    elif args.dataset == 'compas-binary':

        csv_file_train =data_path+'/compas-binary/compas_encoded_all_new_encoded_binary.csv'

        train_dataset = CompasBinaryDataset(csv_file_train)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.platform, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()

    elif args.dataset == 'wcld':

        csv_file_train =data_path+'/wcld/wcld_60000.csv'

        train_dataset = WCLDDataset(csv_file_train)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.platform, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    elif args.dataset == 'ptb-xl':

        csv_file_train = data_path+'/ptb-xl/ptbxl_all_clean_new_2.csv'

        train_dataset = PTBDataset(csv_file_train, platform=args.platform)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.platform, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    
    elif args.dataset == 'nih-chest-h5':

        csv_file_train = data_path+'/nih-chest/nih_chest_all_clean.csv'

        nih_mean = [0.485, 0.456, 0.406] 
        nih_std = [0.229, 0.224, 0.225]
        pretrained_size = 256
    
        nih_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[nih_mean[0], nih_mean[1], nih_mean[2]],
                                                                  std=[nih_std[0], nih_std[1], nih_std[2]])
                                           ])

        if args.crop != 0:
            train_dataset = NIHDataset2(csv_file_train, platform=args.platform, transform=nih_transform, crop=args.crop)
        else:
            train_dataset = NIHDataset2(csv_file_train, platform=args.platform, transform=nih_transform)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition( platform=args.platform, p_idx=args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()


    elif args.dataset == 'nih-chest-eff':

        csv_file_train = data_path+'/nih-chest-eff/nih_chest_all_clean_eff.csv'

        nih_mean = [0.485, 0.456, 0.406] 
        nih_std = [0.229, 0.224, 0.225]
        pretrained_size = 256
       
        nih_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[nih_mean[0], nih_mean[1], nih_mean[2]],
                                                                  std=[nih_std[0], nih_std[1], nih_std[2]])
                                           ])

        if args.crop != 0:
            train_dataset = NIHEffDataset(csv_file_train, platform=args.platform, transform=nih_transform, crop=args.crop)
        else:
            train_dataset = NIHEffDataset(csv_file_train, platform=args.platform, transform=nih_transform)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition( platform=args.platform, p_idx=args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()


    elif args.dataset == 'nih-chest':
        # if not args.kaggle:
        #     data_path =  data_path
        csv_file_train = data_path+'/nih-chest/nih_chest_all_clean.csv'

        nih_mean = [0.485, 0.456, 0.406] 
        nih_std = [0.229, 0.224, 0.225]
        pretrained_size = 256

        nih_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(pretrained_size, antialias=True),
                                        transforms.Normalize(mean=[nih_mean[0], nih_mean[1], nih_mean[2]],
                                                                std=[nih_std[0], nih_std[1], nih_std[2]])
                                        ])

        if args.crop != 0:
            train_dataset = NIHDataset(csv_file_train, platform=args.platform, transform=nih_transform, crop=args.crop)
        else:
            train_dataset = NIHDataset(csv_file_train,platform=args.platform, transform=nih_transform)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(platform=args.platform,  p_idx=args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()


    return train_dataset, test_dataset, user_groups



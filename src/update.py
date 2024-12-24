import torch
from torch.utils.data import DataLoader, Dataset
import dataset
import pandas as pd
import utils
import time
import copy
from sklearn.model_selection import train_test_split
import numpy as np

# from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
# from aif360.datasets import BinaryLabelDataset



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
 
        image, label, s_attr = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label), torch.as_tensor(s_attr)


class BatchDataloader:
    def __init__(self, *tensors, bs=1, mask=None):
        nonzero_idx, = np.nonzero(mask)
        self.tensors = tensors
        self.batch_size = bs
        self.mask = mask
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)+1
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.start = end
        self.sum += sum(batch_mask)
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count


class LocalDataset(object):
    def __init__(self, global_dataset, local_idxs, test_ratio=0.2):
        
        self.local_idxs = np.asarray(list(local_idxs))
        self.test_ratio = test_ratio
        self.target_label = global_dataset.target
        self.s_attr =  global_dataset.s_attr

        self.train_set_idxs, self.test_set_idxs, self.val_set_idxs  =  \
            self.train_test_split(global_dataset.name, global_dataset.X[(self.local_idxs)], \
                                  global_dataset.y[(self.local_idxs)], global_dataset.a[(self.local_idxs)])
        
        self.size = len(self.local_idxs)

    # Return df
    def train_test_split(self, name, X, y, a):

        if name == "ptb-xl" or "nih-chest":
            dummy_X = np.array(range(len(X)))
            X_train, X_test, y_train, y_test, a_train, a_test  = train_test_split(pd.DataFrame(dummy_X), pd.DataFrame(y), pd.DataFrame(a), test_size=self.test_ratio, stratify=y)
            X_val = X_test
            train_set_idxs =  list(X_train.index)
            test_set_idxs = list(X_test.index)
            val_set_idxs = list(X_val.index)

            X_train =np.squeeze(X[X_train])
            X_test = np.squeeze(X[X_test])
            X_val =  np.squeeze(X[X_val])
            
            
        else:
            X_train, X_test, y_train, y_test, a_train, a_test  = train_test_split(pd.DataFrame(X), pd.DataFrame(y), pd.DataFrame(a), test_size=self.test_ratio, stratify=y)
            X_val = X_test
            train_set_idxs =  list(X_train.index)
            test_set_idxs = list(X_test.index)
            val_set_idxs = list(X_val.index)

        return self.local_idxs[train_set_idxs], self.local_idxs[test_set_idxs], self.local_idxs[val_set_idxs]



def get_mask_from_idx(data_size, train_idxs):

    mask = np.zeros(data_size, dtype=np.int8)
    for idx in train_idxs:
        mask[idx] = 1
    return mask


class LocalUpdate(object):
    def __init__(self, args, split_idxs, dataset, idxs, logger,local_dataset=None):
        self.args = args
        self.logger = logger
        self.local_dataset = local_dataset
   
        self.trainloader, self.validloader, self.testloader = self.split_w_idxs(dataset, split_idxs, args.local_bs, args.dataset)
        self.ft_trainloader, _, _ = self.split_w_idxs(dataset, split_idxs, args.ft_bs, args.dataset)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.dataset = dataset

    def split_w_idxs(self, dataset, idxs, batch_size, dataset_name=""):
        train_idxs, test_idxs, val_idxs = idxs
        test_bs = batch_size

        trainloader = DataLoader(DatasetSplit(dataset, train_idxs),
                                batch_size=batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, val_idxs),
                                batch_size=test_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, test_idxs),
                                batch_size=test_bs, shuffle=False)
        
        return trainloader, validloader, testloader
    

    def update_final_layer(self, model,global_round, client_idx=-1):
        model.train()
        model.set_grad(False)
        epoch_loss = []

        optimizer = torch.optim.SGD(model.final_layer.parameters(), lr=self.args.ft_lr,
                                        momentum=0.9, weight_decay=5e-4)
 
        criterion = torch.nn.BCELoss().to(self.device)
 
        lowest_loss=1000
        # best_model
        for iter in range(self.args.ft_ep):
            batch_loss = []
            batch_loss_fairness = []
            batch_loss_1 = []
            for batch_idx, (images, labels, a) in enumerate(self.ft_trainloader):
                images, labels, a = images.to(self.device), labels.to(self.device), a.to(self.device)
                optimizer.zero_grad()  

                outputs = model(images).squeeze()
                loss_1 = criterion(outputs, labels)
            
                pred_labels = (outputs > 0.5).to(torch.float32)
                eod_loss = utils.equalized_odds_diff(pred_labels, labels, a)

                loss = loss_1*self.args.ft_alpha2 + self.args.ft_alpha * eod_loss
     
                loss.backward(retain_graph=True)
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
              
                batch_loss_fairness.append(eod_loss.item())
                batch_loss.append(loss.item())
                batch_loss_1.append(loss_1.item())

            print('{} | # {} | Global Round : {} | Local Epoch : {} | Loss: {:.6f}  L1|EOD:  {:.6f} | {:.6f}'.format(
                   int(time.time()), client_idx, global_round, iter, sum(batch_loss)/len(batch_loss), sum(batch_loss_1)/len(batch_loss_1), sum(batch_loss_fairness)/len(batch_loss_fairness)))
    
            if sum(batch_loss)/len(batch_loss) < lowest_loss:
                best_model = copy.deepcopy(model)
                lowest_loss = sum(batch_loss)/len(batch_loss)

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return best_model.state_dict(), sum(epoch_loss) / len(epoch_loss), np.asarray(epoch_loss)


    def update_weights(self, model, global_round, client_idx=-1):
        # Set mode to train model
        start_time = time.time()
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                
                log_probs = model(images).squeeze()
                log_probs = log_probs.reshape(-1)
                loss = self.criterion(log_probs, labels)
                
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    if batch_idx % 100 == 0 and (iter<10 or iter%10 ==0):
                    # if self.args.local_ep <= 10 or (self.args.local_ep <=100 and self.args.local_ep % 10 == 0) or (self.args.local_ep % 50 == 0):
                        # print("time: ", int(time.time() - start_time), int(time.time()))
                        print('{} | # {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                            int(time.time()), client_idx, global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), batch_loss


    def inference_w_fairness(self, model, set="test", fairness_metric=["eod"], client_idx=-1):
        """ Returns the inference accuracy and loss of local test data (?). 
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        all_y = np.array([])
        all_a = np.array([])
        all_pred = np.array([])

        if set == "test":
            loader = self.testloader
        elif set == "val":
            loader = self.testloader
        else:
            loader = self.trainloader

        for batch_idx, (images, labels, a) in enumerate(loader):
        # for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels, a = images.to(self.device), labels.to(self.device),  a.to(self.device)

            outputs = model(images).squeeze()
            outputs = outputs.reshape(-1)

            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            pred_labels = (outputs > self.args.threshold).to(torch.float32)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            all_y = np.append(all_y, labels.detach().cpu().numpy())
            all_a = np.append(all_a, a.detach().cpu().numpy())
            all_pred = np.append(all_pred, pred_labels.detach().cpu().numpy())

            if self.args.verbose and (batch_idx % 10 == 0):
                    if batch_idx % 100 == 0:
                        print('{} | # {} | Global Round : .. | Local Epoch : .. | [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                            int(time.time()), client_idx, batch_idx * len(images),
                            len(loader.dataset),
                            100. * batch_idx / len(self.trainloader), batch_loss.item()))
        

        train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(all_a, all_pred)
        original_bld = dataset.get_bld_dataset_w_pred(all_a, all_y)
                    
        privileged_groups = [{"a": 1}]
        unprivileged_groups = [{"a": 0}]
        cm_pred_train = ClassificationMetric(original_bld, train_bld_prediction_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

        if self.args.ba == 1:
            accuracy = (cm_pred_train.specificity() + cm_pred_train.sensitivity())/2
        else:
            accuracy = cm_pred_train.accuracy()
        # print("accuracy: ", accuracy.size, sys.getsizeof(accuracy))
        local_fairness = {}
        if "eod" in fairness_metric:
            local_fairness["eod"] = (cm_pred_train.equalized_odds_difference())
            # local_fairness["eod"] = (cm_pred_train.average_abs_odds_difference())
        if True:
            local_fairness["tpr"] = (cm_pred_train.true_positive_rate_difference())
            local_fairness["fpr"] = (cm_pred_train.false_positive_rate_difference())

        accuracy = correct/total
        return accuracy, loss, all_y, all_a, all_pred, local_fairness


def get_global_fairness_new(args, local_a_ls, local_y_ls, prediction_ls, metric="eod", set="train"):

    all_a = np.concatenate(local_a_ls).ravel()
    all_y =  np.concatenate(local_y_ls).ravel()
    all_prediction = np.concatenate(prediction_ls).ravel()

    
    original_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_y)
    prediction_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_prediction)

    privileged_groups = [{"a": 1}]
    unprivileged_groups = [{"a": 0}]

    cm_pred = ClassificationMetric(original_bld_dataset, prediction_bld_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    if args.ba == 1:
        accuracy = (cm_pred.specificity() + cm_pred.sensitivity())/2
    else:
        accuracy = cm_pred.accuracy()
    if metric == "eod":
        fairness = cm_pred.equalized_odds_difference()    

    return accuracy, fairness
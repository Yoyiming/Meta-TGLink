import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
import torch.nn as nn
import ot
import random

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)


class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self, TF_set=None, direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if not direction:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
                
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()

        return adj


class FewShotDataset_tf(Dataset):
    def __init__(self, data, k_shot, k_query, n_way, num_classes=2):
        """
        Args:
            data (pd.DataFrame): The input data containing columns ['TF', 'Target', 'Label'].
            k_shot (int): Number of samples for the support set.
            k_query (int): Number of samples for the query set.
            n_way (int): Number of different TFs to sample for each class.
            num_classes (int): Number of classes (default is 2 for binary classification).
        """
        self.data = data
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        self.num_classes = num_classes
        self.batch_data = []
        self.tf_groups = self.data.groupby('TF')
        self.tf_list = list(self.tf_groups.groups.keys())
        self.__create_batch_data()

    def _sample_data_balanced(self, tf_data, remaining_samples_needed):
        support_set = []
        query_set = []

        tf_data = tf_data.sample(frac=1).reset_index(drop=True)
        tf_data_1 = tf_data[tf_data['Label'] == 1]
        tf_data_0 = tf_data[tf_data['Label'] == 0]
        remaining_samples_num = tf_data.shape[0]
        while remaining_samples_num >= remaining_samples_needed:
            support_samples_1 = tf_data_1.sample(n=self.k_shot // 2)
            support_samples_0 = tf_data_0.sample(n=self.k_shot // 2)
            support_samples = pd.concat([support_samples_1, support_samples_0])

            tf_data_1 = tf_data_1.drop(support_samples_1.index)
            tf_data_0 = tf_data_0.drop(support_samples_0.index)

            query_samples_1 = tf_data_1.sample(n=self.k_query // 2)
            query_samples_0 = tf_data_0.sample(n=self.k_query // 2)
            query_samples = pd.concat([query_samples_1, query_samples_0])

            tf_data_1 = tf_data_1.drop(query_samples_1.index)
            tf_data_0 = tf_data_0.drop(query_samples_0.index)

            support_set.append(support_samples.values.tolist())
            query_set.append(query_samples.values.tolist())

            remaining_samples_num -= remaining_samples_needed
        return np.array(support_set), np.array(query_set)

    def __create_batch_data(self):

        if len(self.tf_list) < self.n_way:
            raise ValueError(
                f"Not enough TF types. Required: {self.n_way}, but got: {len(self.tf_list)}")

        sampled_tf_list = random.sample(self.tf_list, self.n_way)
        tf_data = self.data[self.data['TF'].isin(sampled_tf_list)]

        remaining_samples_needed = self.k_shot + self.k_query
        if tf_data.shape[0] >= remaining_samples_needed:
            support, query = self._sample_data_balanced(tf_data, remaining_samples_needed)
        else:
            raise ValueError(
                f"Not enough samples for the support and query set. Required: {remaining_samples_needed}, but got: {tf_data.shape[0]}")

        self.batch_data.append([support, query])

    def __getitem__(self, idx):
        spt, qry = self.batch_data[idx]
        return spt, qry

    def __len__(self):
        return len(self.batch_data)


class FewShotDataset(Dataset):
    def __init__(self, data, k_shot, k_query, n_way, num_classes=2):
        """
        Args:
            data (pd.DataFrame): The input data containing columns ['TF', 'Target', 'Label'].
            k_shot (int): Number of samples for the support set.
            k_query (int): Number of samples for the query set.
            n_way (int): Number of different TFs to sample for each class.
            num_classes (int): Number of classes (default is 2 for binary classification).
        """
        self.data = data
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        self.num_classes = num_classes
        self.batch_data = []
        self.tf_groups = self.data.groupby('TF')
        self.tf_list = list(self.tf_groups.groups.keys())
        self.__create_batch_data()

    def _sample_data(self, tf_data, remaining_samples_needed):
        support_set = []
        query_set = []

        tf_data = tf_data.sample(frac=1).reset_index(drop=True)
        remaining_samples_num = tf_data.shape[0]
        while remaining_samples_num >= remaining_samples_needed:

            support_samples = tf_data.sample(n=self.k_shot)
            remaining_tf_data = tf_data.drop(support_samples.index)
            query_samples = remaining_tf_data.sample(n=self.k_query)

            support_set.append(support_samples.values.tolist())
            query_set.append(query_samples.values.tolist())

            remaining_samples_num -= remaining_samples_needed

            tf_data = tf_data.drop(support_samples.index)
            tf_data = tf_data.drop(query_samples.index)

        return np.array(support_set), np.array(query_set)

    def __create_batch_data(self):
        if len(self.tf_list) < self.n_way:
            raise ValueError(
                f"Not enough TF types. Required: {self.n_way}, but got: {len(self.tf_list)}")

        sampled_tf_list = random.sample(self.tf_list, self.n_way)
        tf_data = self.data[self.data['TF'].isin(sampled_tf_list)]

        remaining_samples_needed = self.k_shot + self.k_query
        if tf_data.shape[0] >= remaining_samples_needed:
            support, query = self._sample_data(tf_data, remaining_samples_needed)
        else:
            raise ValueError(
                f"Not enough samples for the support and query set. Required: {remaining_samples_needed}, but got: {tf_data.shape[0]}")

        self.batch_data.append([support, query])

    def __getitem__(self, idx):
        spt, qry = self.batch_data[idx]
        return spt, qry

    def __len__(self):
        return len(self.batch_data)


class MetaDataset(Dataset):
    def __init__(self, train_set, flag=False, shot=5, task_num=5, batchsz=None):
        super(MetaDataset, self).__init__()
        self.train_set = train_set
        self.flag = flag
        self.shot = shot
        self.batchsz = batchsz
        self.batch_data = []
        self.task_num = task_num
        self._create_batch_data()

    def _create_batch_data(self):

        tf_set = np.unique(self.train_set[:, 0])
        tf_num = len(tf_set)
        task_num = self.task_num
        # 计算需要重复抽取的TF数量
        if tf_num % task_num != 0:
            repeat_num = task_num - (tf_num % task_num)
        else:
            repeat_num = 0
        # 将TF列表转为numpy数组,方便打乱顺序
        tf_list = np.array(list(tf_set))
        np.random.shuffle(tf_list)
        # 计算需要循环的次数
        if self.batchsz is None:
            self.batchsz = (tf_num + repeat_num) // task_num
        for b in range(self.batchsz):
            # 选择当前batch的TF
            start_idx = b * task_num
            end_idx = start_idx + task_num
            if end_idx > tf_num:
                end_idx = tf_num
            selected_tf = tf_list[start_idx:end_idx]

            if len(selected_tf) < task_num:
                remaining_tfs = task_num - len(selected_tf)
                select_index = np.random.choice(tf_num, remaining_tfs, replace=False)
                selected_tf = np.concatenate((selected_tf, tf_list[select_index]))

            support_x = []
            support_y = []
            query_x = []
            query_y = []

            tf_support_x = []
            tf_support_y = []
            tf_query_x = []
            tf_query_y = []
            tf_data = []

            for tf in selected_tf:
                tf_data.append(self.train_set[self.train_set[:, 0] == tf])
            tf_data = np.concatenate(tf_data, axis=0)

            negative_index = np.where(tf_data[:, -1] == 0)[0]
            negative_idx = np.random.choice(negative_index, self.shot, replace=False)
            negative_index = list(set(negative_index) - set(negative_idx))

            negative_spt_set = tf_data[negative_idx, :2]
            negative_spt_label = tf_data[negative_idx, -1]
            # 剩下的作为query
            negative_qry_set = tf_data[negative_index, :2]
            negative_qry_label = tf_data[negative_index, -1]

            tf_support_x.extend(negative_spt_set)
            tf_support_y.extend(negative_spt_label)
            tf_query_x.extend(negative_qry_set)
            tf_query_y.extend(negative_qry_label)

            positive_index = np.where(tf_data[:, -1] == 1)[0]
            positive_idx = np.random.choice(positive_index, self.shot, replace=False)
            positive_index = list(set(positive_index) - set(positive_idx))
            positive_spt_set = tf_data[positive_idx, :2]
            positive_spt_label = tf_data[positive_idx, -1]
            # 剩下的作为query
            positive_qry_set = tf_data[positive_index, :2]
            positive_qry_label = tf_data[positive_index, -1]

            tf_support_x.extend(positive_spt_set)
            tf_support_y.extend(positive_spt_label)
            tf_query_x.extend(positive_qry_set)
            tf_query_y.extend(positive_qry_label)

            support_x.append(tf_support_x)
            support_y.append(tf_support_y)
            query_x.append(tf_query_x)
            query_y.append(tf_query_y)

            support_x = np.array(support_x).reshape(-1, 2)
            support_y = np.array(support_y).reshape(-1, 1)
            query_x = np.array(query_x).reshape(-1, 2)
            query_y = np.array(query_y).reshape(-1, 1)
            # 一个batch的数据
            self.batch_data.append([support_x, support_y, query_x, query_y])


    def __getitem__(self, idx):
        x_spt, y_spt, x_qry, y_qry = self.batch_data[idx]
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return len(self.batch_data)


# class MetaDataset(Dataset):
#     def __init__(self, train_set, flag=False, shot=5, task_num=5, batchsz=None):
#         super(MetaDataset, self).__init__()
#         self.train_set = train_set
#         self.flag = flag
#         self.shot = shot
#         self.batchsz = batchsz
#         self.batch_data = []
#         self.task_num = task_num
#         self._create_batch_data()
#
#     def _create_batch_data(self):
#
#         tf_set = np.unique(self.train_set[:, 0])
#         tf_num = len(tf_set)
#         task_num = self.task_num
#         # 计算需要重复抽取的TF数量
#         repeat_num = task_num - (tf_num % task_num)
#         # 将TF列表转为numpy数组,方便打乱顺序
#         tf_list = np.array(list(tf_set))
#         np.random.shuffle(tf_list)
#         # 计算需要循环的次数
#         if self.batchsz is None:
#             self.batchsz = (tf_num + repeat_num) // task_num
#         for b in range(self.batchsz):
#             # 选择当前batch的TF
#             start_idx = b * task_num
#             end_idx = start_idx + task_num
#             selected_tf = tf_list[start_idx:end_idx]
#
#             # 如果选择的TF数量不足task_num,则从头开始补充
#             if len(selected_tf) < task_num:
#                 remaining_tfs = task_num - len(selected_tf)
#                 select_index = np.random.choice(tf_num, remaining_tfs, replace=False)
#                 selected_tf = np.concatenate((selected_tf, tf_list[select_index]))
#             support_x = []
#             support_y = []
#             query_x = []
#             query_y = []
#
#             tf_support_x = []
#             tf_support_y = []
#             tf_query_x = []
#             tf_query_y = []
#             tf_data = []
#             for tf in selected_tf:
#                 tf_data.append(self.train_set[self.train_set[:, 0] == tf])
#             tf_data = np.concatenate(tf_data, axis=0)
#
#             negative_index = np.where(tf_data[:, -1] == 0)[0]
#             negative_idx = np.random.choice(negative_index, self.shot, replace=False)
#             negative_index = list(set(negative_index) - set(negative_idx))
#
#             negative_spt_set = tf_data[negative_idx, :2]
#             negative_spt_label = tf_data[negative_idx, -1]
#             # 剩下的作为query
#             negative_qry_set = tf_data[negative_index, :2]
#             negative_qry_label = tf_data[negative_index, -1]
#
#             tf_support_x.extend(negative_spt_set)
#             tf_support_y.extend(negative_spt_label)
#             tf_query_x.extend(negative_qry_set)
#             tf_query_y.extend(negative_qry_label)
#
#             positive_index = np.where(tf_data[:, -1] == 1)[0]
#             positive_idx = np.random.choice(positive_index, self.shot, replace=False)
#             positive_index = list(set(positive_index) - set(positive_idx))
#             positive_spt_set = tf_data[positive_idx, :2]
#             positive_spt_label = tf_data[positive_idx, -1]
#             # 剩下的作为query
#             positive_qry_set = tf_data[positive_index, :2]
#             positive_qry_label = tf_data[positive_index, -1]
#
#             tf_support_x.extend(positive_spt_set)
#             tf_support_y.extend(positive_spt_label)
#             tf_query_x.extend(positive_qry_set)
#             tf_query_y.extend(positive_qry_label)
#
#             support_x.append(tf_support_x)
#             support_y.append(tf_support_y)
#             query_x.append(tf_query_x)
#             query_y.append(tf_query_y)
#
#             support_x = np.array(support_x).reshape(-1, 2)
#             support_y = np.array(support_y).reshape(-1, 1)
#             query_x = np.array(query_x).reshape(-1, 2)
#             query_y = np.array(query_y).reshape(-1, 1)
#             # 一个batch的数据
#             self.batch_data.append([support_x, support_y, query_x, query_y])
#
#
#     def __getitem__(self, idx):
#         x_spt, y_spt, x_qry, y_qry = self.batch_data[idx]
#         return x_spt, y_spt, x_qry, y_qry
#
#     def __len__(self):
#         return len(self.batch_data)


# class MetaDataset(Dataset):
#     def __init__(self, train_set, flag=False, shot=5, task_num=1, batchsz=None):
#         super(MetaDataset, self).__init__()
#         self.train_set = train_set
#         self.flag = flag
#         self.shot = shot
#         self.batchsz = batchsz
#         self.batch_data = []
#         self.task_num = task_num
#         self._create_batch_data()
#
#     def _create_batch_data(self):
#
#         tf_set = np.unique(self.train_set[:, 0])
#         tf_num = len(tf_set)
#         task_num = self.task_num
#
#         # 将TF列表转为numpy数组,方便打乱顺序
#         tf_list = np.array(list(tf_set))
#         np.random.shuffle(tf_list)
#         # 计算需要循环的次数
#         if self.batchsz is None:
#             self.batchsz = tf_num
#         for b in range(self.batchsz):
#             # 选择当前batch的TF
#             selected_tf = tf_list[b]
#
#             # 如果选择的TF数量不足task_num,则从头开始补充
#             # if len(selected_tf) < task_num:
#             #     remaining_tfs = task_num - len(selected_tf)
#             #     select_index = np.random.choice(tf_num, remaining_tfs, replace=False)
#             #     selected_tf = np.concatenate((selected_tf, tf_list[select_index]))
#             support_x = []
#             support_y = []
#             query_x = []
#             query_y = []
#
#             tf_support_x = []
#             tf_support_y = []
#             tf_query_x = []
#             tf_query_y = []
#
#             tf_data = []
#             tf_data.append(self.train_set[self.train_set[:, 0] == selected_tf])
#             tf_data = np.concatenate(tf_data, axis=0)
#
#             negative_index = np.where(tf_data[:, -1] == 0)[0]
#             if len(negative_index) < self.shot:
#                 negative_idx = np.random.choice(negative_index, self.shot, replace=True)
#             else:
#                 negative_idx = np.random.choice(negative_index, self.shot, replace=False)
#             negative_index = list(set(negative_index) - set(negative_idx))
#
#             negative_spt_set = tf_data[negative_idx, :2]
#             negative_spt_label = tf_data[negative_idx, -1]
#             # 剩下的作为query
#             negative_qry_set = tf_data[negative_index, :2]
#             negative_qry_label = tf_data[negative_index, -1]
#
#             tf_support_x.extend(negative_spt_set)
#             tf_support_y.extend(negative_spt_label)
#             tf_query_x.extend(negative_qry_set)
#             tf_query_y.extend(negative_qry_label)
#
#             positive_index = np.where(tf_data[:, -1] == 1)[0]
#             positive_idx = np.random.choice(positive_index, self.shot, replace=False)
#             positive_index = list(set(positive_index) - set(positive_idx))
#             positive_spt_set = tf_data[positive_idx, :2]
#             positive_spt_label = tf_data[positive_idx, -1]
#             # 剩下的作为query
#             positive_qry_set = tf_data[positive_index, :2]
#             positive_qry_label = tf_data[positive_index, -1]
#
#             tf_support_x.extend(positive_spt_set)
#             tf_support_y.extend(positive_spt_label)
#             tf_query_x.extend(positive_qry_set)
#             tf_query_y.extend(positive_qry_label)
#
#             support_x.append(tf_support_x)
#             support_y.append(tf_support_y)
#             query_x.append(tf_query_x)
#             query_y.append(tf_query_y)
#
#             support_x = np.array(support_x).reshape(-1, 2)
#             support_y = np.array(support_y).reshape(-1, 1)
#             query_x = np.array(query_x).reshape(-1, 2)
#             query_y = np.array(query_y).reshape(-1, 1)
#             # 一个batch的数据
#             self.batch_data.append([support_x, support_y, query_x, query_y])
#
#     def __getitem__(self, idx):
#         x_spt, y_spt, x_qry, y_qry = self.batch_data[idx]
#         return x_spt, y_spt, x_qry, y_qry
#
#     def __len__(self):
#         return len(self.batch_data)


class scRNADataset_Weight(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset_Weight, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)


    def Adj_Generate(self,TF_set,direction=False, loop=False):

        adj = sp.lil_matrix((self.num_gene, self.num_gene), dtype=np.float32)


        for pos in self.train_set:

            tf = int(pos[0])
            target = int(pos[1])
            weight = pos[2]

            if direction == False:
                    adj[tf, target] = weight
                    adj[target, tf] = weight

            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.eye(self.num_gene, dtype=np.float32)

        adj = adj.tocsr()

        return adj


class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T

    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj_generate(data, num_gene, direction=False, loop=False):
    adj = sp.dok_matrix((num_gene, num_gene), dtype=np.float32)

    for pos in data:
        tf = pos[0]
        target = pos[1]

        if not direction:
            if pos[-1] == 1:
                adj[tf, target] = 1.0
                adj[target, tf] = 1.0

        else:
            if pos[-1] == 1:
                adj[tf, target] = 1.0
                adj[target, tf] = 1.0

    if loop:
        adj = adj + sp.eye(num_gene, dtype=np.float32)

    adj = adj.tocsr()
    adj = adj2saprse_tensor(adj)
    return adj


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(values).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)

    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)

    return AUC, AUPR, AUPR_norm


def auc_evaluate(pred_list, label_list):
    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    presicion, recall, thresholds = metrics.precision_recall_curve(label_list, pred_list, pos_label=1)
    aupr = metrics.auc(recall, presicion)
    return auc, aupr


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr


def dataset_integration(expression1, expression2, train1, train2, target1, target2):
    train1['TF'] = train1['TF'].map(target1.set_index('index')['Gene'])
    train1['Target'] = train1['Target'].map(target1.set_index('index')['Gene'])
    train2['TF'] = train2['TF'].map(target2.set_index('index')['Gene'])
    train2['Target'] = train2['Target'].map(target2.set_index('index')['Gene'])

    expression = pd.concat([expression1, expression2], axis=0)
    gene_indices1 = list(range(expression1.shape[0]))
    gene_df1 = pd.DataFrame({'Gene': expression1.index, 'index': gene_indices1})
    gene_indices2 = list(range(expression1.shape[0], expression2.shape[0] + expression1.shape[0]))
    gene_df2 = pd.DataFrame({'Gene': expression2.index, 'index': gene_indices2})

    train1['TF'] = train1['TF'].map(gene_df1.set_index('Gene')['index'])
    train1['Target'] = train1['Target'].map(gene_df1.set_index('Gene')['index'])
    train2['TF'] = train2['TF'].map(gene_df2.set_index('Gene')['index'])
    train2['Target'] = train2['Target'].map(gene_df2.set_index('Gene')['index'])

    train = pd.concat([train1, train2], axis=0)

    return expression, train


def compute_mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


def compute_page_ranks(A, device, damping_factor=0.85, max_iter=100, tol=1e-6):
    A_dense = A.to_dense().to(device)
    num_nodes = A.size(0)
    page_ranks = torch.ones(num_nodes, device=A_dense.device) / num_nodes

    for _ in range(max_iter):
        prev_page_ranks = page_ranks.clone()

        out_degrees = A_dense.sum(dim=1)
        out_degrees[out_degrees == 0] = 1
        page_ranks = (1 - damping_factor) / num_nodes + damping_factor * torch.matmul(A_dense,
                                                                                      prev_page_ranks / out_degrees)
        page_ranks[out_degrees == 0] = 1 / num_nodes

        if torch.allclose(page_ranks.half(), prev_page_ranks.half(), atol=tol):
            break

    return page_ranks
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class MetaDataset_balanced(Dataset):
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


class MetaDataset(Dataset):
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from utils import compute_page_ranks
import copy
import time
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNLayer(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, device, decoder_type):
        super(GNNLayer, self).__init__()
        self.device = device
        self.decoder_type = decoder_type

        # self.ConvLayer1 = GATConv(input_dim, hidden1_dim, heads=num_head1, concat=(self.reduction == 'concate'),
        #                           dropout=alpha)
        # self.ConvLayer2 = GATConv(self.hidden1_dim, hidden2_dim, heads=num_head2, concat=False,
        #                           dropout=alpha)

        self.ConvLayer1 = GCNConv(input_dim, hidden1_dim, bias=False)
        self.ConvLayer2 = GCNConv(hidden1_dim, hidden2_dim, bias=False)
        
        # self.ConvLayer1 = SAGEConv(input_dim, hidden1_dim, aggr='mean')
        # self.ConvLayer2 = SAGEConv(hidden1_dim, hidden2_dim, aggr='mean')

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        if self.decoder_type == 'MLP':
            self.linear = nn.Linear(2 * output_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.ConvLayer1.reset_parameters()
        self.ConvLayer2.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    def encode(self, x, edge_index):
        x = self.ConvLayer1(x, edge_index)
        x = F.elu(x)
        out = self.ConvLayer2(x, edge_index)
        return out

    def decode(self, tf_embed, target_embed):
        if self.decoder_type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1, 1)
            return prob
        elif self.decoder_type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(f'{self.decoder_type} is not available')

    def forward(self, x, edge_index, train_sample):
        edge_index = edge_index.coalesce()
        embed = self.encode(x, edge_index)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred


class AttentionSampler:
    def __init__(self, k, alpha, device):
        super(AttentionSampler, self).__init__()
        self.k = k
        self.alpha = alpha
        self.device = device

    def sample(self, X_corr, A):
        A_dense = A.to_dense().to(self.device)

        similarity = X_corr + self.alpha * torch.matmul((A_dense + torch.eye(A_dense.shape[0]).to(self.device)), X_corr)
        similarity.fill_diagonal_(-float('inf'))
        _, topk_indices = torch.topk(similarity, self.k, dim=-1)
        return topk_indices


class PageRankPositionalEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PageRankPositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, page_ranks):
        pre = self.mlp(page_ranks.unsqueeze(-1))
        return pre


class DegreePositionalEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DegreePositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, degrees):
        de = self.mlp(degrees.unsqueeze(-1))
        return de


class PositionalEncoder(nn.Module):
    def __init__(self, pos_dim, device):
        super(PositionalEncoder, self).__init__()
        self.degree_enc = DegreePositionalEncoding(1, pos_dim)
        self.pagerank_enc = PageRankPositionalEncoding(1, pos_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * pos_dim, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
        )
        self.page_ranks = None
        self.degrees = None
        self.device = device
        self._initialize_weights()

    def initialize_positional(self, A, device):
        self.page_ranks = compute_page_ranks(A, device)
        A_dense = A.to_dense().to(device)
        self.degrees = A_dense.sum(dim=-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, X, node_list):
        degree_enc = self.degree_enc(self.degrees[node_list])
        pagerank_enc = self.pagerank_enc(self.page_ranks[node_list])
        X = self.mlp(torch.cat([X, degree_enc, pagerank_enc], dim=-1))
        return X


class TGLink(nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, num_heads, device, decoder_type, num_layers=1):

        super(TGLink, self).__init__()

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True), num_layers=num_layers)

        self.gnn_layers = GNNLayer(d_model, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, device, decoder_type)

    def forward(self, X, X_aggregated, A, train_sample, node_list):
        X_transformed = self.transformer_encoder(X_aggregated)
        X_sample = X_transformed[:, 0, :].clone()
        # 将X中的节点特征转换为transformer后的特征
        X = X.clone()
        X[node_list, :] = X_sample
        pred = self.gnn_layers(X, A, train_sample)

        return pred


class Meta_TGLink(nn.Module):
    def __init__(self, input_dim, pos_dim, d_model, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, num_heads, k, alpha, device, decoder_type, num_layers=1):

        super(Meta_TGLink, self).__init__()
        self.attention_sampler = AttentionSampler(k, alpha, device)
        self.tglink = TGLink(input_dim, d_model, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, num_heads, device, decoder_type, num_layers)

        self.positional_encoder = PositionalEncoder(pos_dim, device)
        self.k = k
        self.top_k_indices = None

    def initialize_top_k_indices(self, X, A):
        self.top_k_indices = self.attention_sampler.sample(X, A)

    def neighbor_aggregate(self, X, X_encoded, k, top_k_indices, node_list):
        N, input_dim = X_encoded.size()
        top_k_indices = top_k_indices[node_list, :]
        # 获取每个节点的相关节点数量
        k = int(k)
        out = X_encoded.new_zeros((N, k + 1, input_dim))
        # 遍历每个节点
        for i in range(N):
            neighbors = top_k_indices[i, :]
            out[i, 0] = X_encoded[i, :]
            out[i, 1:] = X[neighbors, :]
        return out

    def forward(self, X, A, train_sample):
        node_list = list(set(train_sample[:, 0].tolist() + train_sample[:, 1].tolist()))
        # 根据node_list获取X的子矩阵，其余的不参与position encode和transformer encode
        X_sample = X[node_list, :]
        X_encoded = self.positional_encoder(X_sample, node_list)
        X_aggregated = self.neighbor_aggregate(X, X_encoded, self.k, self.top_k_indices, node_list)
        pred = self.tglink(X, X_aggregated, A, train_sample, node_list)
        return pred

    def adapt(self, X, A, x_spt, y_spt, update_step=1, lr=0.0001, decay=None):

        self.train()
        for _ in range(update_step):
            self.zero_grad()

            y_hat = self(X, A, x_spt)
            y_hat = torch.sigmoid(y_hat)

            loss = F.binary_cross_entropy(y_hat, y_spt)

            model_params = [param for param in self.parameters()]

            grads = torch.autograd.grad(loss, model_params, create_graph=False, allow_unused=False)
            with torch.no_grad():
                for param, grad in zip(model_params, grads):
                    if grad is not None:
                        param.data -= lr * grad.data

            if decay is not None:
                lr *= decay
                
        self.eval()

    def clone(self):
        return copy.deepcopy(self)


import pandas as pd
import numpy as np
import random
import os
import time
import argparse
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model import Meta_TGLink
from utils import auc_evaluate, adj_generate, normalize
from dataset import MetaDataset_balanced


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epoch.')
parser.add_argument('--batch_size', type=int, default=64, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--sample', type=str, default='sample2', help='sample')
parser.add_argument('--train_cell', type=str, default='A549', help='sample')
parser.add_argument('--relevant_neighbor', type=int, default=15, help='the nunmber of relevant neighbors')
parser.add_argument('--k_shot', type=int, default=10, help='the number of samples for the support set')
parser.add_argument('--k_query', type=int, default=30, help='the number of samples for the query set')
parser.add_argument('--svd_dim', type=int, default=200, help='the number of svd dimension')
parser.add_argument('--alpha', type=float, default=0.5, help='the alpha value for the relevant matrix')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--few-shot', type=bool, default=True, help='whether to perform few-shot testing')


args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = args.device

k = args.relevant_neighbor
k_shot = args.k_shot
k_query = args.k_query
svd_dim = args.svd_dim

cell_lines = [args.train_cell]
cell_lines_train = args.train_cell

train_data_loaders = {}
val_data_loaders = {}
test_data_loaders = {}

train_data_path = f'cell_line_dataset/oe/{cell_lines_train}/{args.sample}_tf_train.csv'
val_data_path = f'cell_line_dataset/oe/{cell_lines_train}/{args.sample}_tf_val.csv'
test_data_path = f'cell_line_dataset/oe/{cell_lines_train}/{args.sample}_tf_test.csv'

for data_path, data_loaders in zip([train_data_path, val_data_path, test_data_path],
                                    [train_data_loaders, val_data_loaders, test_data_loaders]):
    data = pd.read_csv(data_path)
    n_way = data['TF'].unique().shape[0]
    if data_path == test_data_path or data_path == val_data_path:
        dataset = MetaDataset_balanced(data, k_shot, data.shape[0] - k_shot, n_way)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        data_loaders[cell_lines_train] = dataloader
    else:
        dataset = MetaDataset_balanced(data, k_shot, k_query, n_way)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        data_loaders[cell_lines_train] = dataloader

expression_matrices = {}

data_path = f'expression matrix/oe/{cell_lines_train}/limma_expression_matrix.csv'
expression_matrix = pd.read_csv(data_path, index_col=0, header=0).values.astype(np.float32)
expression_matrices[cell_lines_train] = normalize(expression_matrix)

corrs = {}

expression_matrix = expression_matrices[cell_lines_train]
corrs[cell_lines_train] = torch.corrcoef(torch.from_numpy(expression_matrix)).to(device)

svd = TruncatedSVD(n_components=svd_dim, n_iter=7, random_state=seed)
expression_matrices[cell_lines_train] = torch.from_numpy(svd.fit_transform(expression_matrices[cell_lines_train])).to(device)

model = Meta_TGLink(input_dim=svd_dim,
              pos_dim=svd_dim,
              d_model=svd_dim,
              hidden_dim=4*svd_dim,
              hidden_dim1=128,
              hidden_dim2=64,
              hidden_dim3=32,
              output_dim=16,
              num_heads=8,
              k=k,
              alpha=args.alpha,
              device=device,
              decoder_type='dot',
              ).to(device)

meta_optimizer = Adam(model.parameters(), lr=args.lr)

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

best_auc = 0
best_epoch = 0

train_loader = train_data_loaders[cell_lines_train]
val_loader = val_data_loaders[cell_lines_train]

feature = expression_matrices[cell_lines_train]
X_corr = corrs[cell_lines_train]

adj = adj_generate(train_loader.dataset.data.values, feature.shape[0])
adj = adj.to(device)

model.initialize_top_k_indices(X_corr, adj)
model.positional_encoder.initialize_positional(adj, device)

for epoch in range(args.epochs):
    meta_train_loss = 0.0

    for step, (spt_set, qry_set) in enumerate(train_loader):
        task_num = len(spt_set[0])

        for task in range(task_num):
            spt_data = spt_set[0, task]
            qry_data = qry_set[0, task]

            spt_data = spt_data.to(device)
            qry_data = qry_data.to(device)

            model.train()
            model.adapt(feature, adj, spt_data[:, :-1], spt_data[:, -1].view(-1, 1).float(), update_step=1)

            query_pred = model(feature, adj, qry_data[:, :-1])
            query_pred = torch.sigmoid(query_pred)
            query_loss = F.binary_cross_entropy(query_pred, qry_data[:, -1].view(-1, 1).float())

            meta_train_loss += query_loss.item()
            
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()
            
    new_model = model.clone()
    y_true, y_pred = [], []
    for step, (spt_set, qry_set) in enumerate(val_loader):
        
        task_num = len(spt_set[0])

        for task in range(task_num):

            spt_data = spt_set[0, task]
            qry_data = qry_set[0, task]

            adj_data = torch.concat([spt_data, torch.tensor(train_loader.dataset.data.values)], dim=0)
            adj = adj_generate(adj_data, feature.shape[0])
            adj = adj.to(device)

            new_model.initialize_top_k_indices(X_corr, adj)
            new_model.positional_encoder.initialize_positional(adj, device)

            spt_data = spt_data.to(device)
            qry_data = qry_data.to(device)

            new_model.adapt(feature, adj, spt_data[:, :-1], spt_data[:, -1].view(-1, 1).float(), update_step=1)

            new_model.eval()
            with torch.no_grad():
                query_pred = new_model(feature, adj, qry_data[:, :-1])
                query_pred = torch.sigmoid(query_pred)

                y_true.append(qry_data[:, -1].cpu().numpy())
                y_pred.append(query_pred.cpu().numpy())

    del new_model
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc, aupr = auc_evaluate(y_pred, y_true)

    if auc > best_auc:
        best_auc = auc
        best_epoch = epoch
        torch.save(model.state_dict(), model_path + '/FewshotTransGAT_tf' + '_' + cell_lines_train + '_' + args.sample + '_' + str(k) +'.pt')

    print('Epoch:', epoch + 1, '| Meta-Train Loss:', meta_train_loss, '| AUC:', auc, '| AUPRC:', aupr)
print('Best AUC:', best_auc, 'Best Epoch:', best_epoch + 1)

# Test stage
model.load_state_dict(torch.load(model_path + '/FewshotTransGAT_tf' + '_' + cell_lines_train + '_' + args.sample + '_' + str(k) +'.pt'))

test_loader = test_data_loaders[cell_lines_train]
feature = expression_matrices[cell_lines_train]
X_corr = corrs[cell_lines_train]
new_model = model.clone()

y_true, y_pred = [], []
for step, (spt_set, qry_set) in enumerate(test_loader):

    task_num = len(spt_set[0])
    for task in range(task_num):
        spt_data = spt_set[0, task]
        qry_data = qry_set[0, task]

        ########## Few-shot ############
        if args.few_shot:
            adj_data = torch.concat([spt_data, torch.tensor(train_loader.dataset.data.values)], dim=0)
            adj = adj_generate(adj_data, feature.shape[0])
            adj = adj.to(device)

            new_model.initialize_top_k_indices(X_corr, adj)
            new_model.positional_encoder.initialize_positional(adj, device)

            spt_data = spt_data.to(device)
            qry_data = qry_data.to(device)

            new_model.adapt(feature, adj, spt_data[:, :-1], spt_data[:, -1].view(-1, 1).float(), update_step=1)

            new_model.eval()
            with torch.no_grad():
                query_pred = new_model(feature, adj, qry_data[:, :-1])
                query_pred = torch.sigmoid(query_pred)

                y_true.append(qry_data[:, -1].cpu().numpy())
                y_pred.append(query_pred.cpu().numpy())
        ########## Cold-Start ############
        else:
            adj = adj_generate(train_loader.dataset.data.values, feature.shape[0])
            adj = adj.to(device)

            new_model.initialize_top_k_indices(X_corr, adj)
            new_model.positional_encoder.initialize_positional(adj, device)

            spt_data = spt_data.to(device)
            qry_data = qry_data.to(device)

            qry_data = torch.cat([spt_data, qry_data], dim=0)
            new_model.eval()
            with torch.no_grad():
                query_pred = new_model(feature, adj, qry_data[:, :-1])
                query_pred = torch.sigmoid(query_pred)

                y_true.append(qry_data[:, -1].cpu().numpy())
                y_pred.append(query_pred.cpu().numpy())

del new_model

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
auc, aupr = auc_evaluate(y_pred, y_true)

print('Test Cell Line:', cell_lines_train, '| AUC:', auc, '| AUPRC:', aupr)
save_dir = f'Meta_Result/Meta_TGLink/oe/{args.train_cell}/{args.sample}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.few_shot:
    np.savetxt(f'{save_dir}/tf_few-shot_{k}.txt', np.array([auc, aupr]), fmt='%.4f')
else:
    np.savetxt(f'{save_dir}/tf_zero-shot_{k}.txt', np.array([auc, aupr]), fmt='%.4f')

print('k: {} | sample:{}'.format(k, args.sample))

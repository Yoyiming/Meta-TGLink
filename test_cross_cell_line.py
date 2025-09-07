import pandas as pd
import numpy as np
import random
import os
import time
import argparse
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
import torch
from model import Meta_TGLink
from utils import auc_evaluate, adj_generate, normalize
from dataset import MetaDataset_balanced
import time


parser = argparse.ArgumentParser()
parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--sample', type=str, default='sample1', help='sample')
parser.add_argument('--train_cell', type=str, default='PC3', help='cell line name')
parser.add_argument('--device', type=str, default='cuda:3', help='device')
parser.add_argument('--relevant_neighbor', type=int, default=15, help='the nunmber of relevant neighbors')
parser.add_argument('--k_shot', type=int, default=10, help='the number of samples for the support set')
parser.add_argument('--k_query', type=int, default=30, help='the number of samples for the query set')
parser.add_argument('--svd_dim', type=int, default=200, help='the number of svd dimension')
parser.add_argument('--alpha', type=float, default=0.5, help='the alpha value for the relevant matrix')

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
cell_lines_train = args.train_cell
cell_lines = ['A549', 'A375', 'HEK293T', 'PC3']

train_data_loaders = {}
test_data_loaders = {}
for cell_line in cell_lines:
    train_data_path = f'cell_line_dataset/oe/{cell_line}/{args.sample}_train.csv'
    test_data_path = f'cell_line_dataset/oe/{cell_line}/{args.sample}_test.csv'

    for data_path, data_loaders in zip([train_data_path, test_data_path],
                                       [train_data_loaders, test_data_loaders]):
        data = pd.read_csv(data_path)
        n_way = data['TF'].unique().shape[0]
        if data_path == test_data_path:
            # Using all remaining data for testing
            dataset = MetaDataset_balanced(data, k_shot, data.shape[0] - k_shot, n_way)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            data_loaders[cell_line] = dataloader
        else:
            dataset = MetaDataset_balanced(data, k_shot, k_query, n_way)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            data_loaders[cell_line] = dataloader

expression_matrices = {}
corrs = {}
for cell_line in cell_lines:
    data_path = f'expression matrix/oe/{cell_line}/limma_expression_matrix.csv'
    expression_matrix = pd.read_csv(data_path, index_col=0, header=0).values.astype(np.float32)
    expression_matrix = normalize(expression_matrix)

    corrs[cell_line] = torch.corrcoef(torch.from_numpy(expression_matrix)).to(device)

    svd = TruncatedSVD(n_components=svd_dim, n_iter=7, random_state=seed)
    expression_matrices[cell_line] = torch.from_numpy(svd.fit_transform(expression_matrix)).to(device)

# Define model
model = Meta_TGLink(input_dim=svd_dim,
              pos_dim=svd_dim,
              d_model=svd_dim,
              hidden_dim=512,
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

# Prepare feature and adjacency matrix
feature = expression_matrices[cell_lines_train]
X_corr = corrs[cell_lines_train]

train_loader = train_data_loaders[cell_lines_train]
adj = adj_generate(train_loader.dataset.data.values, feature.shape[0])
adj = adj.to(device)

# Load model
model_path = 'model'
if not os.path.exists(model_path):
    raise FileNotFoundError('Model path not found, model path: ' + model_path)
model.load_state_dict(torch.load(model_path + '/Meta_TGLink' + '_' + cell_lines_train + '_' + args.sample + '_' + str(k) +'.pt'))

# Begin to test
start_time = time.time()

for cell_line in cell_lines:
    test_loader = test_data_loaders[cell_line]
    feature = expression_matrices[cell_line]
    X_corr = corrs[cell_line]
    new_model = model.clone()

    y_true, y_pred = [], []
    if cell_line == cell_lines_train:
        for step, (spt_set, qry_set) in enumerate(test_loader):

            new_model.initialize_top_k_indices(X_corr, adj)
            new_model.positional_encoder.initialize_positional(adj, device)

            # zero-shot
            task_num = len(spt_set[0])

            for task in range(task_num):
                spt_data = spt_set[0, task]
                qry_data = qry_set[0, task]
                test_data = torch.concat([spt_data, qry_data], dim=0).to(device)

                new_model.eval()
                with torch.no_grad():
                    query_pred = new_model(feature, adj, test_data[:, :-1])
                    query_pred = torch.sigmoid(query_pred)

                    y_true.append(test_data[:, -1].cpu().numpy())
                    y_pred.append(query_pred.cpu().numpy())

        end_time = time.time()
        test_time = end_time - start_time
        print(f'Test time: {test_time:.2f} seconds')

        del new_model
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        auc, aupr = auc_evaluate(y_pred, y_true)

        print('Zero shot testing | Cell Line:', cell_line, '| AUC:', auc, '| AUPRC:', aupr)
        save_dir = f'Meta_Result/Meta_TGLink/oe/{args.train_cell}/{args.sample}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(f'{save_dir}/zero-shot_{k}.txt', np.array([auc, aupr]), fmt='%.4f')

    else:
        for step, (spt_set, qry_set) in enumerate(test_loader):
            # few-shot
            task_num = len(spt_set[0])
            auc_list, aupr_list = [[] for i in range(task_num)], [[] for i in range(task_num)]
            for task in range(task_num):
                spt_data = spt_set[0, task]
                qry_data = qry_set[0, task]
                spt_adj = adj_generate(spt_data, feature.shape[0])
                spt_data = spt_data.to(device)
                qry_data = qry_data.to(device)
                spt_adj = spt_adj.to(device)

                new_model.initialize_top_k_indices(X_corr, spt_adj)
                new_model.positional_encoder.initialize_positional(spt_adj, device)
                new_model.adapt(feature, spt_adj, spt_data[:, :-1], spt_data[:, -1].view(-1, 1).float(), update_step=5, lr=args.inner_lr)

                new_model.eval()
                with torch.no_grad():
                    query_pred = new_model(feature, spt_adj, qry_data[:, :-1])
                    query_pred = torch.sigmoid(query_pred)

                    y_true.append(qry_data[:, -1].cpu().numpy())
                    y_pred.append(query_pred.cpu().numpy())
                    
                # Export support data and query data
                # spt_data = spt_data.cpu().numpy()
                # qry_data = qry_data.cpu().numpy()
                # spt_data = pd.DataFrame(spt_data, columns=['TF', 'Target', 'Label'])
                # qry_data = pd.DataFrame(qry_data, columns=['TF', 'Target', 'Label'])
                # spt_data.to_csv(f'cell_line_dataset/oe/{cell_line}/{args.sample}_spt.csv', index=False)
                # qry_data.to_csv(f'cell_line_dataset/oe/{cell_line}/{args.sample}_qry.csv', index=False)

        del new_model
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        auc, aupr = auc_evaluate(y_pred, y_true)

        print('Few shot testing | Cell Line:', cell_line, '| AUC:', auc, '| AUPRC:', aupr)
        save_dir = f'Meta_Result/Meta_TGLink/oe/{args.train_cell}/{args.sample}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(f'{save_dir}/few-shot_{cell_line}_{k}.txt', np.array([auc, aupr]), fmt='%.4f')

print('Testing config | k: {} | sample:{}'.format(k, args.sample))
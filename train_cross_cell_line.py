import pandas as pd
import numpy as np
import random
import glob
import os
import time
import argparse
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Meta_TGLink
import scipy.sparse as sp
import matplotlib.pyplot as plt
from utils import adj2saprse_tensor, auc_evaluate, adj_generate, normalize
from dataset import MetaDataset_balanced
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Meta-TGLink')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=8, help='Random seed')
    parser.add_argument('--sample', type=str, default='sample1', help='Sample name')
    parser.add_argument('--train_cell', type=str, default='PC3', help='Cell line for training')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device for training')
    parser.add_argument('--relevant_neighbor', type=int, default=15, help='Relevant neighbor number')
    parser.add_argument('--k_shot', type=int, default=10, help='Number of shot in support set and query set')
    parser.add_argument('--k_query', type=int, default=30, help='Number of query in support set')
    parser.add_argument('--svd_dim', type=int, default=200, help='SVD dimension')
    parser.add_argument('--alpha', type=float, default=0.5, help='The alpha value for the relevant matrix')
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    Setting random seed for reproducibility
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_loaders(args: argparse.Namespace) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare training, validation, and test data loaders
    """
    train_data_loaders = {}
    val_data_loaders = {}

    cell_line = args.train_cell
    train_data_path = f'cell_line_dataset/oe/{cell_line}/{args.sample}_train.csv'
    val_data_path = f'cell_line_dataset/oe/{cell_line}/{args.sample}_val.csv' 

    # Load training data
    train_data = pd.read_csv(train_data_path)
    train_n_way = train_data['TF'].unique().shape[0]
    train_dataset = MetaDataset_balanced(train_data, args.k_shot, args.k_query, train_n_way)
    train_data_loaders[cell_line] = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Load validation data
    val_data = pd.read_csv(val_data_path)
    val_n_way = val_data['TF'].unique().shape[0]
    val_dataset = MetaDataset_balanced(val_data, args.k_shot, args.k_query, val_n_way)
    val_data_loaders[cell_line] = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_data_loaders, val_data_loaders


def prepare_features(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    """
    Prepare feature matrices and correlation matrices
    """
    expression_matrices = {}
    corrs = {}
    cell_line = args.train_cell

    # Normalize expression matrix
    data_path = f'expression matrix/oe/{cell_line}/limma_expression_matrix.csv'
    expression_matrix = pd.read_csv(data_path, index_col=0, header=0).values.astype(np.float32)
    expression_matrix = normalize(expression_matrix)

    # Compute correlation matrix
    corrs[cell_line] = torch.corrcoef(torch.from_numpy(expression_matrix)).to(args.device)

    # Dimensionality reduction using SVD
    svd = TruncatedSVD(n_components=args.svd_dim, n_iter=7, random_state=args.seed)
    expression_matrices[cell_line] = torch.from_numpy(svd.fit_transform(expression_matrix)).to(args.device)

    return expression_matrices, corrs


def train_model(args: argparse.Namespace, train_data_loaders: Dict, val_data_loaders: Dict,
                expression_matrices: Dict, corrs: Dict) -> torch.nn.Module:
    """
    Train the model and return the best model
    """
    cell_line = args.train_cell
    device = args.device

    # Initialize model
    model = Meta_TGLink(
        input_dim=args.svd_dim,
        pos_dim=args.svd_dim,
        d_model=args.svd_dim,
        hidden_dim=512,
        hidden_dim1=128,
        hidden_dim2=64,
        hidden_dim3=32,
        output_dim=16,
        num_heads=8,
        k=args.relevant_neighbor,
        alpha=args.alpha,
        device=device,
        decoder_type='dot',
    ).to(device)

    meta_optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(meta_optimizer, T_max=20, eta_min=0.0001)

    model_path = 'model'
    os.makedirs(model_path, exist_ok=True)

    best_auc = 0.0
    best_epoch = 0

    train_loader = train_data_loaders[cell_line]
    val_loader = val_data_loaders[cell_line]
    feature = expression_matrices[cell_line]
    X_corr = corrs[cell_line]

    # Generate training adjacency matrix
    train_adj = adj_generate(train_loader.dataset.data.values, feature.shape[0])
    train_adj = train_adj.to(device)

    model.initialize_top_k_indices(X_corr, train_adj)
    model.positional_encoder.initialize_positional(train_adj, device)

    print(f'Begin to train, Cell line: {cell_line}, Sample name: {args.sample}')
    
    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(args.epochs):
        meta_train_loss = 0.0
        model.train()

        for step, (spt_set, qry_set) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            task_num = len(spt_set[0])
            for task in range(task_num):
                spt_data = spt_set[0, task]
                qry_data = qry_set[0, task]

                spt_adj = adj_generate(spt_data, feature.shape[0])
                spt_adj = spt_adj.to(device)

                spt_data = spt_data.to(device)
                qry_data = qry_data.to(device)

                model.adapt(feature, spt_adj, spt_data[:, :-1], spt_data[:, -1].view(-1, 1).float(), update_step=1)

                query_pred = model(feature, spt_adj, qry_data[:, :-1])
                query_pred = torch.sigmoid(query_pred)
                query_loss = F.binary_cross_entropy(query_pred, qry_data[:, -1].view(-1, 1).float())

                meta_train_loss += query_loss.item()

                meta_optimizer.zero_grad()
                query_loss.backward()
                meta_optimizer.step()
                scheduler.step()

        model.eval()
        new_model = model.clone()
        y_true, y_pred = [], []

        with torch.no_grad():
            for _, (spt_set, qry_set) in enumerate(val_loader):
                new_model.initialize_top_k_indices(X_corr, train_adj)
                new_model.positional_encoder.initialize_positional(train_adj, device)
                task_num = len(spt_set[0])

                for task in range(task_num):
                    spt_data = spt_set[0, task]
                    qry_data = qry_set[0, task]
                    val_data = torch.concat([spt_data, qry_data], dim=0).to(device)
                    spt_data = spt_data.to(device)
                    qry_data = qry_data.to(device)

                    query_pred = new_model(feature, train_adj, val_data[:, :-1])
                    query_pred = torch.sigmoid(query_pred)

                    y_true.append(val_data[:, -1].cpu().numpy())
                    y_pred.append(query_pred.cpu().numpy())

        # Delete the adapted model to prevent data leakage
        del new_model
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        auc, aupr = auc_evaluate(y_pred, y_true)

        # Save the best model
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_path, f'Meta_TGLink_{cell_line}_{args.sample}_{args.relevant_neighbor}.pt'))

        print(f'Epoch: {epoch + 1} | Meta-Train Loss: {meta_train_loss:.4f} | AUC: {auc:.4f} | AUPRC: {aupr:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    print(f'Best AUC: {best_auc:.4f}, in epoch {best_epoch + 1}')
    print(f'Relevant Neighbors: {args.relevant_neighbor} | Sample: {args.sample}')

    return model


def main() -> None:
    """
    Main function to execute training
    """
    args = parse_args()
    set_random_seed(args.seed)
    train_data_loaders, val_data_loaders = prepare_data_loaders(args)
    expression_matrices, corrs = prepare_features(args)
    _ = train_model(args, train_data_loaders, val_data_loaders, expression_matrices, corrs)


if __name__ == '__main__':
    main()

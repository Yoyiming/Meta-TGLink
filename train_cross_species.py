import pandas as pd
import numpy as np
import random
import os
import time
import argparse
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import Meta_TGLink
from utils import auc_evaluate, adj_generate, normalize
from dataset import MetaDataset
from tqdm import tqdm
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Meta-TGLink Cross-Species Training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=8, help='Random seed')
    parser.add_argument('--sample', type=str, default='sample2', help='Sample name')
    parser.add_argument('--train_cell', type=str, default='mESC', help='Cell line name for training')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device for training')
    parser.add_argument('--relevant_neighbor', type=int, default=15, help='Number of relevant neighbors')
    parser.add_argument('--k_shot', type=int, default=10, help='Number of samples for support set')
    parser.add_argument('--k_query', type=int, default=30, help='Number of samples for query set')
    parser.add_argument('--svd_dim', type=int, default=200, help='SVD dimension for feature reduction')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for relevant matrix')
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_loaders(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Prepare data loaders for training, validation and testing
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Tuple of three dictionaries containing train, validation and test data loaders
    """
    train_data_loaders = {}
    val_data_loaders = {}
    cell_line = args.train_cell
    data_dir = Path(f'cell_line_dataset/Specific/{cell_line} 1000')
    
    # Define data paths
    train_data_path = data_dir / f'{args.sample}_train.csv'
    val_data_path = data_dir / f'{args.sample}_val.csv' 
    
    # Check if data files exist
    if not train_data_path.exists():
        raise FileNotFoundError(f"Train data file not found: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data file not found: {val_data_path}")
    
    for data_path, data_loaders in zip([train_data_path, val_data_path],
                                       [train_data_loaders, val_data_loaders]):
        data = pd.read_csv(data_path)
        n_way = data['TF'].unique().shape[0]
        if data_path == test_data_path:
            # Use all remaining data for testing
            dataset = MetaDataset(data, args.k_shot, data.shape[0] - args.k_shot, n_way)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            data_loaders[cell_line] = dataloader
        else:
            dataset = MetaDataset(data, args.k_shot, args.k_query, n_way)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            data_loaders[cell_line] = dataloader
    
    return train_data_loaders, val_data_loaders


def prepare_features(args: argparse.Namespace) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Prepare expression matrices and correlation matrices
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Tuple of two dictionaries containing expression matrices and correlation matrices
    """
    expression_matrices = {}
    corrs = {}
    cell_line = args.train_cell
    
    # Load and process expression matrix
    data_path = Path(f'expression matrix/Specific Dataset/{cell_line}/TFs+1000/BL--ExpressionData.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Expression matrix file not found: {data_path}")
    
    expression_matrix = pd.read_csv(data_path, index_col=0, header=0).values.astype(np.float32)
    expression_matrix = normalize(expression_matrix)
    
    # Compute correlation matrix
    corrs[cell_line] = torch.corrcoef(torch.from_numpy(expression_matrix)).to(args.device)
    
    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=args.svd_dim, n_iter=7, random_state=args.seed)
    expression_matrices[cell_line] = torch.from_numpy(svd.fit_transform(expression_matrix)).to(args.device)
    
    return expression_matrices, corrs


def initialize_model(args: argparse.Namespace, input_dim: int) -> Meta_TGLink:
    """
    Initialize the Meta-TGLink model
    
    Args:
        args: Parsed command line arguments
        input_dim: Input dimension for the model
    
    Returns:
        Initialized Meta_TGLink model
    """
    model = Meta_TGLink(
        input_dim=input_dim,
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
        device=args.device,
        decoder_type='dot',
    ).to(args.device)
    return model


def train_model(
    args: argparse.Namespace,
    model: Meta_TGLink,
    train_loader: DataLoader,
    feature: torch.Tensor,
    X_corr: torch.Tensor,
    adj: torch.Tensor,
    meta_optimizer: Adam
) -> float:
    """
    Train the model for one epoch
    
    Args:
        args: Parsed command line arguments
        model: Meta-TGLink model
        train_loader: Training data loader
        feature: Expression matrix features
        X_corr: Correlation matrix
        adj: Adjacency matrix
        meta_optimizer: Meta optimizer
    
    Returns:
        Average meta training loss for the epoch
    """
    meta_train_loss = 0.0
    model.train()
    
    for step, (spt_set, qry_set) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        task_num = len(spt_set[0])
        for task in range(task_num):
            spt_data = spt_set[0, task]
            qry_data = qry_set[0, task]
            spt_adj = adj_generate(spt_data, feature.shape[0])
            spt_data = spt_data.to(args.device)
            qry_data = qry_data.to(args.device)
            spt_adj = spt_adj.to(args.device)

            # Adapt model to the support set
            model.adapt(
                feature,
                spt_adj,
                spt_data[:, :-1],
                spt_data[:, -1].view(-1, 1).float(),
                update_step=1,
                lr=args.lr
            )

            # Forward pass on query set
            query_pred = model(feature, spt_adj, qry_data[:, :-1])
            query_pred = torch.sigmoid(query_pred)
            query_loss = F.binary_cross_entropy(query_pred, qry_data[:, -1].view(-1, 1).float())

            meta_train_loss += query_loss.item()
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()
    
    return meta_train_loss / len(train_loader)


def validate_model(
    args: argparse.Namespace,
    model: Meta_TGLink,
    val_loader: DataLoader,
    feature: torch.Tensor,
    X_corr: torch.Tensor,
    adj: torch.Tensor
) -> Tuple[float, float]:
    """
    Validate the model
    
    Args:
        args: Parsed command line arguments
        model: Meta-TGLink model
        val_loader: Validation data loader
        feature: Expression matrix features
        X_corr: Correlation matrix
        adj: Adjacency matrix
    
    Returns:
        Tuple of AUC and AUPRC scores
    """
    new_model = model.clone()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for step, (spt_set, qry_set) in enumerate(tqdm(val_loader, desc="Validation")):
            new_model.initialize_top_k_indices(X_corr, adj)
            new_model.positional_encoder.initialize_positional(adj, args.device)
            task_num = len(spt_set[0])

            for task in range(task_num):
                spt_data = spt_set[0, task]
                qry_data = qry_set[0, task]
                val_data = torch.concat([spt_data, qry_data], dim=0).to(args.device)

                new_model.eval()
                query_pred = new_model(feature, adj, val_data[:, :-1])
                query_pred = torch.sigmoid(query_pred)

                y_true.append(val_data[:, -1].cpu().numpy())
                y_pred.append(query_pred.cpu().numpy())
    
    del new_model
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc, aupr = auc_evaluate(y_pred, y_true)
    
    return auc, aupr


def main() -> None:
    """
    Main function to run cross-species training
    """
    args = parse_args()
    set_random_seed(args.seed)
    
    # Prepare data loaders
    train_data_loaders, val_data_loaders, _ = prepare_data_loaders(args)
    
    # Prepare features
    expression_matrices, corrs = prepare_features(args)
    
    # Initialize model
    cell_line = args.train_cell
    feature = expression_matrices[cell_line]
    model = initialize_model(args, feature.shape[1])
    
    # Setup optimizer
    meta_optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Create model directory
    model_path = Path('model')
    model_path.mkdir(exist_ok=True)
    
    # Prepare adjacency matrix
    train_loader = train_data_loaders[cell_line]
    val_loader = val_data_loaders[cell_line]
    X_corr = corrs[cell_line]
    adj = adj_generate(train_loader.dataset.data.values, feature.shape[0])
    adj = adj.to(args.device)
    
    # Initialize model parameters
    model.initialize_top_k_indices(X_corr, adj)
    model.positional_encoder.initialize_positional(adj, args.device)
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train for one epoch
        avg_meta_train_loss = train_model(args, model, train_loader, feature, X_corr, adj, meta_optimizer)
        
        # Validate
        auc, aupr = validate_model(args, model, val_loader, feature, X_corr, adj)
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                model_path / f'Meta_TGLink_hm{cell_line}_{args.sample}_{args.relevant_neighbor}.pt'
            )
        
        print(f'Epoch: {epoch + 1} | Meta-Train Loss: {avg_meta_train_loss:.4f} | AUC: {auc:.4f} | AUPRC: {aupr:.4f}')
    
    print(f'Best AUC: {best_auc:.4f} | Best Epoch: {best_epoch + 1}')


if __name__ == '__main__':
    main()

